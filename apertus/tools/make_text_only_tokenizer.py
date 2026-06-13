import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Pattern, Set


_TOKEN_PATTERNS: List[Pattern[str]] = [
    # Tokens like <|system_start|>
    re.compile(r"<\|[^|]+\|>"),
    re.compile(r"<SPECIAL_\\d+>"),
]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _extract_tokens_from_template(template_text: str) -> Set[str]:
    out: Set[str] = set()
    for pat in _TOKEN_PATTERNS:
        out.update(pat.findall(template_text))
    return out


def _add_if_str_set(dst: Set[str], v: Any) -> None:
    if isinstance(v, str) and v:
        dst.add(v)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Create a text-only tokenizer view from a multimodal tokenizer directory by pruning the massive "
            "added-token tables (tokenizer.json added_tokens + tokenizer_config.json added_tokens_decoder). "
            "This is intended for runs that never use multimodal inputs/outputs."
        )
    )
    ap.add_argument("--source-dir", required=True, help="Path to the multimodal tokenizer directory")
    ap.add_argument("--output-dir", required=True, help="Path to write the text-only tokenizer directory")
    ap.add_argument(
        "--keep-added-from",
        default=None,
        help=(
            "Optional reference tokenizer directory (e.g. Apertus-8B-Instruct-2509 snapshot). "
            "All reference added tokens that also exist in the source tokenizer will be kept. "
            "This is useful when the multimodal model is continued from a text-only checkpoint and tokenizer "
            "alignment matters for those tokens."
        ),
    )
    ap.add_argument(
        "--force-eos-token",
        default=None,
        help=(
            "Override eos_token in tokenizer_config.json and special_tokens_map.json (string value). "
            "Example: --force-eos-token '<|assistant_end|>'."
        ),
    )
    ap.add_argument(
        "--force-default-system-prompt",
        action="store_true",
        help=(
            "Rewrite chat_template.jinja to use a `default_system_prompt` variable in the no-system-message path. "
            "The current prompt is kept as the default and the older prompt is left as a comment."
        ),
    )
    ap.add_argument(
        "--generation-config-from",
        default=None,
        help="Optional path to a HF generation config JSON to write as generation_config.json in the output dir.",
    )
    ap.add_argument(
        "--generation-config-json",
        default=None,
        help="Optional inline JSON string for generation_config.json (overrides --generation-config-from).",
    )
    ap.add_argument(
        "--extra-token",
        action="append",
        default=[],
        help="Extra token string to keep (repeatable). Example: --extra-token '<|image|>'",
    )
    args = ap.parse_args()

    src = Path(args.source_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok_json_path = src / "tokenizer.json"
    tok_cfg_path = src / "tokenizer_config.json"
    st_map_path = src / "special_tokens_map.json"
    template_path = src / "chat_template.jinja"

    if not tok_json_path.exists():
        raise FileNotFoundError(f"Missing {tok_json_path}")
    if not tok_cfg_path.exists():
        raise FileNotFoundError(f"Missing {tok_cfg_path}")
    if not st_map_path.exists():
        raise FileNotFoundError(f"Missing {st_map_path}")

    tok_json = _read_json(tok_json_path)
    tok_cfg = _read_json(tok_cfg_path)
    st_map = _read_json(st_map_path)

    keep: Set[str] = set(args.extra_token or [])
    # Keep explicit tool-output sentinels if present in the source tokenizer/template.
    # These are used by the tools-fixed chat template to delimit tool outputs.
    keep.update({"<|tool_output_start|>", "<|tool_output_end|>"})

    # Optionally keep all added tokens from a reference tokenizer that also exist in the source.
    # (We do an intersection to avoid introducing token ids that the model may not have embeddings for.)
    ref_dir = Path(args.keep_added_from) if args.keep_added_from else None
    ref_added_missing_in_source: Set[str] = set()
    if ref_dir is not None:
        ref_tok_path = ref_dir / "tokenizer.json"
        if not ref_tok_path.exists():
            raise FileNotFoundError(f"--keep-added-from missing {ref_tok_path}")
        ref_tok = _read_json(ref_tok_path)
        ref_added = ref_tok.get("added_tokens") or []
        ref_contents = {
            t.get("content")
            for t in ref_added
            if isinstance(t, dict) and isinstance(t.get("content"), str)
        }
        src_added_all = tok_json.get("added_tokens") or []
        src_contents = {
            t.get("content")
            for t in src_added_all
            if isinstance(t, dict) and isinstance(t.get("content"), str)
        }
        keep.update({c for c in ref_contents if isinstance(c, str) and c in src_contents})
        ref_added_missing_in_source = {c for c in ref_contents if isinstance(c, str) and c not in src_contents}

    # Always keep the standard special tokens used by the model/tokenizer.
    for k in ("bos_token", "eos_token", "pad_token", "unk_token"):
        _add_if_str_set(keep, tok_cfg.get(k))
        _add_if_str_set(keep, st_map.get(k))

    # Keep sft sequence markers if present.
    for k in ("sft_user_begin_sequence", "sft_assistant_begin_sequence", "sft_eot_token"):
        v = tok_cfg.get(k)
        if isinstance(v, list):
            for item in v:
                _add_if_str_set(keep, item)
        else:
            _add_if_str_set(keep, v)

    # If a chat template file exists, keep any explicit token literals referenced by it.
    if template_path.exists():
        keep.update(_extract_tokens_from_template(template_path.read_text(encoding="utf-8")))

    # Keep any tokens explicitly listed as additional_special_tokens only if they are referenced in the template.
    # (The source tokenizer may carry thousands of modality-code tokens here, which is exactly what we want to drop.)
    additional_special = tok_cfg.get("additional_special_tokens")
    if isinstance(additional_special, list):
        keep_additional = [t for t in additional_special if isinstance(t, str) and t in keep]
    else:
        keep_additional = []

    # Prune tokenizer.json added_tokens.
    src_added = tok_json.get("added_tokens") or []
    kept_added = []
    for entry in src_added:
        if not isinstance(entry, dict):
            continue
        content = entry.get("content")
        if isinstance(content, str) and content in keep:
            kept_added.append(entry)

    tok_json["added_tokens"] = kept_added

    # Prune tokenizer_config.json added_tokens_decoder.
    src_decoder = tok_cfg.get("added_tokens_decoder")
    kept_decoder: dict[str, dict] = {}
    if isinstance(src_decoder, dict):
        for tok_id, meta in src_decoder.items():
            if not isinstance(meta, dict):
                continue
            content = meta.get("content")
            if isinstance(content, str) and content in keep:
                kept_decoder[tok_id] = meta
    tok_cfg["added_tokens_decoder"] = kept_decoder
    tok_cfg["added_tokens_count"] = len(kept_decoder)
    tok_cfg["additional_special_tokens"] = keep_additional

    # Reduce special_tokens_map.json to the standard fields (and kept additional specials, if any).
    new_st_map = {}
    for k in ("bos_token", "eos_token", "pad_token", "unk_token"):
        v = st_map.get(k)
        if isinstance(v, dict):
            v = v.get("content")
        if isinstance(v, str) and v:
            new_st_map[k] = v
    if keep_additional:
        new_st_map["additional_special_tokens"] = keep_additional

    # Optionally force eos token to align with a reference tokenizer / generation config.
    if args.force_eos_token:
        tok_cfg["eos_token"] = str(args.force_eos_token)
        new_st_map["eos_token"] = str(args.force_eos_token)

    # Write out the slimmed files.
    _write_json(out / "tokenizer.json", tok_json)
    _write_json(out / "tokenizer_config.json", tok_cfg)
    _write_json(out / "special_tokens_map.json", new_st_map)

    # Copy template for downstream chat formatting if present.
    if template_path.exists():
        shutil.copy2(template_path, out / template_path.name)
        if args.force_default_system_prompt:
            # Patch the copied template to centralize the default system prompt,
            # but keep it intentionally blank (caller should provide system messages).
            tpath = out / template_path.name
            lines = tpath.read_text(encoding="utf-8", errors="ignore").splitlines()

            # 1) Ensure `default_system_prompt` is defined right after end_system_token.
            if not any("default_system_prompt" in ln for ln in lines):
                anchor = "{%- set end_system_token = '<|system_end|>' -%}"
                for i, ln in enumerate(lines):
                    if ln.strip() == anchor:
                        lines.insert(i + 1, "{%- set default_system_prompt = '' -%}")
                        break

            # 2) Replace ONLY the *no-system-message* default branch to use the variable.
            # Do not touch the branch that renders an explicit system message from `messages[0]`.
            target = "{{ system_token + default_system_prompt + end_system_token }}"
            if_anchor = "{%- if messages and messages[0].role == 'system' -%}"
            else_anchor = "{%- else -%}"
            try:
                start = next(i for i, ln in enumerate(lines) if ln.strip() == if_anchor)
            except StopIteration:
                start = None
            if start is not None:
                # Find the matching else branch for that top-level if.
                try:
                    else_idx = next(i for i in range(start + 1, len(lines)) if lines[i].strip() == else_anchor)
                except StopIteration:
                    else_idx = None
                if else_idx is not None:
                    # Replace the first template line after the else that renders the default system content.
                    for j in range(else_idx + 1, min(else_idx + 10, len(lines))):
                        ln = lines[j].strip()
                        if not ln:
                            continue
                        if ln.startswith("{{") and "system_token" in ln and "end_system_token" in ln:
                            lines[j] = "    " + target if lines[j].startswith("    ") else target
                            break

            tpath.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Optionally write generation_config.json into the derived tokenizer dir.
    if args.generation_config_json or args.generation_config_from:
        gen_path = out / "generation_config.json"
        if args.generation_config_json:
            gen_obj = json.loads(args.generation_config_json)
        else:
            src_gen = Path(args.generation_config_from)
            gen_obj = _read_json(src_gen)
        _write_json(gen_path, gen_obj)

    # Copy other tiny metadata files if present.
    for name in ("NOTES.md",):
        p = src / name
        if p.exists():
            shutil.copy2(p, out / name)

    # Print a small summary (sizes in bytes).
    def sz(p: Path) -> int:
        return p.stat().st_size

    print("kept_tokens", len(keep))
    print("added_tokens_kept", len(kept_added), "of", len(src_added))
    print("added_tokens_decoder_kept", len(kept_decoder), "of", (len(src_decoder) if isinstance(src_decoder, dict) else 0))
    if ref_dir is not None:
        print("reference_added_missing_in_source", len(ref_added_missing_in_source))
        if ref_added_missing_in_source:
            print("reference_missing_examples", sorted(list(ref_added_missing_in_source))[:20])
    print("out_files:")
    for p in sorted(out.iterdir()):
        if p.is_file():
            print(" ", p.name, sz(p))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
