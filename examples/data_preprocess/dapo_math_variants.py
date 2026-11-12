# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prepare pinned DAPO-Math training variants and ad-hoc prompt/dedup transforms.

Supported presets (`--variant`):
- ``dapo_math_filter_dedup_boxed`` (default): Pinned ``Jiawei415/DPAO_filter``
  (12,691 rows; SimpleTIR initial-model accuracy filtering in ``aaabiao/dapo_filter``
  followed by first-occurrence ``prompt`` deduplication), rewriting only
  ``data_source: "dapo_filter" -> "math_dapo"`` so ``default_compute_score`` routes
  every row to ``math_dapo.compute_score``.
- ``dapo_math_17k_dedup``: Pinned ``BytedTsinghua-SIA/DAPO-Math-17k`` (1,791,700 raw
  rows = 100 exact repeats of 17,917 UUIDs) deduplicated by ``extra_info.index`` with
  strict row-equality verification, preserving the native ``Answer: $Answer`` format
  (17,917 rows; or 17,398 rows with ``--dedupe-prompts``).
- ``dapo_math_17k_boxed``: Same pinned ``BytedTsinghua-SIA/DAPO-Math-17k`` source,
  deduplicated and converted from the ``Answer: $Answer`` wrapper to the ``\\boxed{}``
  instruction suffix while preserving the exact mathematical problem body.
- ``custom``: Ad-hoc preprocessing on any local ``--input-parquet`` (supports
  ``--prompt-format {keep,boxed}``, ``--dedupe {none,index,prompt}``, and
  ``--target-data-source``).

Validation benchmarks (e.g. AIME) are intentionally kept separate from this script.

    python examples/data_preprocess/dapo_math_variants.py \
        --variant dapo_math_filter_dedup_boxed \
        --output-dir ~/data/dapo_math_filter_dedup_boxed
"""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score import default_compute_score

DAPO_ANSWER_PREFIX = (
    "Solve the following math problem step by step. The last line of your response should be of the form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
)
DAPO_ANSWER_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'
BOXED_SUFFIX = "\n\nLet's think step by step and output the final answer within \\boxed{}."
TARGET_DATA_SOURCE = "math_dapo"

PINNED_SOURCES = {
    "Jiawei415/DPAO_filter": {
        "repo_id": "Jiawei415/DPAO_filter",
        "revision": "d4bb0f1f83c32cfa06424eee48bf0e4b90990ebd",
        "filename": "train/dapo_filter.parquet",
        "sha256": "47a9806fdb10be56336651384f0c1a7d2c7de6be2883b1b64a7d559bf6f9e39c",
        "bytes": 2428814,
        "raw_rows": 12691,
        "raw_data_source": "dapo_filter",
        "license": None,
        "license_status": "Not declared in the pinned Hugging Face dataset metadata; no usage rights inferred",
        "repo_url": "https://huggingface.co/datasets/Jiawei415/DPAO_filter/tree/d4bb0f1f83c32cfa06424eee48bf0e4b90990ebd",
    },
    "BytedTsinghua-SIA/DAPO-Math-17k": {
        "repo_id": "BytedTsinghua-SIA/DAPO-Math-17k",
        "revision": "65877096c24ffa7abc4e4fa5edb95cf3413a5674",
        "filename": "data/dapo-math-17k.parquet",
        "sha256": "534375d6bb8630d22ab46a56e11f2ffec1d288d8f7d04099bc82d68948705941",
        "bytes": 299363855,
        "raw_rows": 1791700,
        "raw_data_source": "math_dapo",
        "license": "apache-2.0",
        "license_status": "Declared as apache-2.0 in the pinned Hugging Face dataset card",
        "repo_url": (
            "https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/tree/"
            "65877096c24ffa7abc4e4fa5edb95cf3413a5674"
        ),
    },
}

VARIANT_PRESETS = {
    "dapo_math_filter_dedup_boxed": {
        "source_key": "Jiawei415/DPAO_filter",
        "dedupe": "none",
        "prompt_format": "keep",
        "expected_rows": 12691,
        "lineage": {
            "upstream_base_dataset": "BytedTsinghua-SIA/DAPO-Math-17k",
            "difficulty_filtered_predecessor": {
                "repo_id": "aaabiao/dapo_filter",
                "repo_url": "https://huggingface.co/datasets/aaabiao/dapo_filter",
                "rows": 13044,
                "description": (
                    "Curated in the SimpleTIR setup (arXiv:2509.02479) by filtering DAPO-Math-17k according "
                    "to initial-model accuracy to discard tasks with overly high or overly low solve rates, "
                    "formatted with the boxed answer instruction."
                ),
            },
            "prompt_deduplicated_source": {
                "repo_id": "Jiawei415/DPAO_filter",
                "rows": 12691,
                "description": (
                    "First-occurrence prompt deduplication of aaabiao/dapo_filter (removing 353 "
                    "duplicate-prompt entries while preserving all 12,691 unique rows and order verbatim)."
                ),
            },
        },
    },
    "dapo_math_17k_dedup": {
        "source_key": "BytedTsinghua-SIA/DAPO-Math-17k",
        "dedupe": "index",
        "prompt_format": "keep",
        "expected_rows": 17917,
        "expected_prompt_dedup_rows": 17398,
        "lineage": {
            "upstream_base_dataset": "BytedTsinghua-SIA/DAPO-Math-17k",
            "description": (
                "Exact index deduplication of the 1,791,700-row release (100 exact copies per UUID -> "
                "17,917 unique UUID rows, or 17,398 unique problem texts when --dedupe-prompts is enabled)."
            ),
        },
    },
    "dapo_math_17k_boxed": {
        "source_key": "BytedTsinghua-SIA/DAPO-Math-17k",
        "dedupe": "index",
        "prompt_format": "boxed",
        "expected_rows": 17917,
        "expected_prompt_dedup_rows": 17398,
        "lineage": {
            "upstream_base_dataset": "BytedTsinghua-SIA/DAPO-Math-17k",
            "description": (
                "Exact index deduplication of BytedTsinghua-SIA/DAPO-Math-17k followed by stripping the "
                "Answer-prefix/suffix wrapper and appending the standard \\boxed{} instruction suffix."
            ),
        },
    },
}


def digest(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file in streaming chunks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value) -> str:
    """Serialize nested row values deterministically for semantic hashing."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def prompt_dedupe_key(prompt) -> str:
    """Compute deterministic JSON key matching Jiawei415/DPAO_filter dedupe_downsample.py."""
    return json.dumps(prompt, ensure_ascii=True, sort_keys=True)


def fetch_source(source: dict, local_dir: Path) -> Path:
    """Download a pinned Hugging Face dataset file and verify its byte size and SHA-256."""
    path = Path(
        hf_hub_download(
            repo_id=source["repo_id"],
            repo_type="dataset",
            revision=source["revision"],
            filename=source["filename"],
            local_dir=local_dir,
        )
    )
    if path.stat().st_size != source["bytes"] or digest(path) != source["sha256"]:
        raise ValueError(f"Source size/hash mismatch: {source['repo_id']}@{source['revision']}:{source['filename']}")
    return path


def iter_parquet(path: Path):
    """Yield Python row dicts from a Parquet file in streaming batches."""
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
        yield from batch.to_pylist()


def extract_problem_body(content: str) -> str:
    """Extract mathematical problem body from either DAPO Answer-wrapped or boxed/plain prompt text."""
    if content.endswith(BOXED_SUFFIX):
        body = content[: -len(BOXED_SUFFIX)]
    elif content.startswith(DAPO_ANSWER_PREFIX) and content.endswith(DAPO_ANSWER_SUFFIX):
        body = content[len(DAPO_ANSWER_PREFIX) : -len(DAPO_ANSWER_SUFFIX)]
    else:
        body = content.strip()
    if not body.strip() or "\x00" in content:
        raise ValueError("Invalid empty or NUL-corrupted mathematical prompt text")
    return body


def format_prompt_content(content: str, prompt_format: str) -> str:
    """Validate prompt text and optionally convert it to the standard boxed instruction format."""
    if prompt_format == "keep":
        extract_problem_body(content)
        return content
    if prompt_format == "boxed":
        if content.endswith(BOXED_SUFFIX):
            extract_problem_body(content)
            return content
        body = extract_problem_body(content)
        return body + BOXED_SUFFIX
    raise ValueError(f"Unsupported prompt_format: {prompt_format!r}")


def verify_reward(rows: list[dict], *, check_boxed: bool) -> None:
    """Ensure every prepared row dispatches cleanly and scores correct/incorrect answers as expected."""
    for row in rows:
        answer = row["reward_model"]["ground_truth"]
        probe = f"\\boxed{{{answer}}}" if check_boxed else f"Answer: {answer}"
        wrong_probe = r"\boxed{INCORRECT-SENTINEL}" if check_boxed else "Answer: INCORRECT-SENTINEL"
        correct = default_compute_score(row["data_source"], probe, answer)
        wrong = default_compute_score(row["data_source"], wrong_probe, answer)
        if correct["acc"] is not True or correct["score"] != 1.0 or wrong["acc"] is not False or wrong["score"] != -1.0:
            raise ValueError(f"Unexpected reward dispatch or answer normalization for index={row['extra_info']}")


def write_parquet(rows: list[dict], path: Path) -> dict:
    """Write rows to Parquet, verify lossless round-trip, and return artifact provenance."""
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    if pq.read_table(path).to_pylist() != rows:
        raise ValueError("Parquet roundtrip changed prepared rows")
    semantic = hashlib.sha256()
    for row in rows:
        semantic.update((canonical(row) + "\n").encode())
    return {
        "path": str(path),
        "rows": len(rows),
        "bytes": path.stat().st_size,
        "sha256": digest(path),
        "canonical_rows_sha256": semantic.hexdigest(),
    }


def prepare_variant_rows(
    raw_rows,
    *,
    dedupe: str = "none",
    dedupe_prompts: bool = False,
    prompt_format: str = "keep",
    expected_raw_rows: int | None = None,
    expected_rows: int | None = None,
    expected_raw_data_source: str | None = None,
    target_data_source: str = TARGET_DATA_SOURCE,
) -> list[dict]:
    """Validate, deduplicate, and transform DAPO-Math rows into a verl-ready training split."""
    if dedupe not in ("none", "index", "prompt"):
        raise ValueError(f"Unsupported dedupe mode: {dedupe!r}")

    by_index: dict[str, dict] = {}
    deduped_raw: list[dict] = []
    raw_count = 0

    for row in raw_rows:
        raw_count += 1
        if expected_raw_data_source is not None and row.get("data_source") != expected_raw_data_source:
            raise ValueError(f"Expected raw data_source={expected_raw_data_source!r}, got {row.get('data_source')!r}")

        extra_info = row.get("extra_info")
        index = extra_info.get("index") if isinstance(extra_info, dict) else None
        if not isinstance(index, str) or not index.strip():
            raise ValueError("Each training row must preserve a non-empty string extra_info.index")

        if dedupe == "index":
            if index in by_index:
                if row != by_index[index]:
                    raise ValueError(f"Conflicting rows share extra_info.index={index}")
                continue
            by_index[index] = row
        elif index in by_index and dedupe == "none":
            raise ValueError(f"Duplicate extra_info.index detected: {index}")
        else:
            by_index[index] = row

        deduped_raw.append(row)

    if expected_raw_rows is not None and raw_count != expected_raw_rows:
        raise ValueError(f"Expected {expected_raw_rows} raw rows, got {raw_count}")
    if not deduped_raw:
        raise ValueError("Training dataset cannot be empty")

    apply_prompt_dedupe = dedupe == "prompt" or dedupe_prompts
    seen_prompts: set[str] = set()
    prepared: list[dict] = []

    for row in deduped_raw:
        prompt = row.get("prompt")
        if (
            not isinstance(prompt, list)
            or len(prompt) != 1
            or prompt[0].get("role") != "user"
            or not isinstance(prompt[0].get("content"), str)
        ):
            raise ValueError("Each training row must have a single user prompt message")

        if expected_raw_data_source == "dapo_filter" and not prompt[0]["content"].endswith(BOXED_SUFFIX):
            raise ValueError("Each dapo_filter row must already end with the boxed suffix")

        new_content = format_prompt_content(prompt[0]["content"], prompt_format)
        new_prompt = [{**prompt[0], "content": new_content}]

        prompt_key = prompt_dedupe_key(new_prompt)
        if prompt_key in seen_prompts:
            if apply_prompt_dedupe:
                continue
            if dedupe == "none":
                raise ValueError(f"Duplicate prompt detected at index={row['extra_info']['index']}")
        seen_prompts.add(prompt_key)

        reward_model = row.get("reward_model")
        if (
            not isinstance(reward_model, dict)
            or not isinstance(reward_model.get("style"), str)
            or not reward_model["style"].startswith("rule")
            or not isinstance(reward_model.get("ground_truth"), str)
            or not reward_model["ground_truth"].strip()
        ):
            raise ValueError("Each training row must specify a non-empty rule-based ground_truth string")

        updated = deepcopy(row)
        updated["data_source"] = target_data_source
        if new_content != prompt[0]["content"]:
            updated["prompt"] = new_prompt
        else:
            if {k: v for k, v in updated.items() if k != "data_source"} != {
                k: v for k, v in row.items() if k != "data_source"
            }:
                raise ValueError("Unexpected mutation outside data_source")
        prepared.append(updated)

    if expected_rows is not None and len(prepared) != expected_rows:
        raise ValueError(f"Expected {expected_rows} prepared rows, got {len(prepared)}")

    check_boxed = all(r["prompt"][0]["content"].endswith(BOXED_SUFFIX) for r in prepared)
    verify_reward(prepared, check_boxed=check_boxed)
    return prepared


def main() -> None:
    """Parse CLI arguments, prepare the selected DAPO-Math variant, and write Parquet + manifest."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--variant",
        choices=(*VARIANT_PRESETS.keys(), "custom"),
        default="dapo_math_filter_dedup_boxed",
        help="Preset DAPO-Math variant to prepare, or 'custom' for ad-hoc local Parquet transforms.",
    )
    parser.add_argument(
        "--output-dir",
        "--local_save_dir",
        dest="output_dir",
        type=Path,
        required=True,
        help="Local directory where train.parquet and manifest.json will be written.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Optional offline checked source cache (<source-dir>/<publisher>--<dataset>/<filename>).",
    )
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=None,
        help="Local input Parquet path (required when --variant=custom).",
    )
    parser.add_argument(
        "--prompt-format",
        choices=("keep", "boxed"),
        default=None,
        help="Override prompt formatting ('keep' preserves prompt; 'boxed' converts to \\boxed{} suffix).",
    )
    parser.add_argument(
        "--dedupe",
        choices=("none", "index", "prompt"),
        default=None,
        help="Override deduplication mode for --variant=custom (default: index).",
    )
    parser.add_argument(
        "--dedupe-prompts",
        action="store_true",
        help="Additionally deduplicate identical prompts across distinct UUIDs (e.g. 17,917 -> 17,398 on 17k).",
    )
    parser.add_argument(
        "--target-data-source",
        type=str,
        default=TARGET_DATA_SOURCE,
        help="Target data_source written into prepared rows (default: math_dapo).",
    )
    parser.add_argument(
        "--hdfs-dir",
        "--hdfs_dir",
        dest="hdfs_dir",
        type=str,
        default=None,
        help="Optional HDFS destination directory to mirror prepared outputs.",
    )
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    if args.variant == "custom":
        if args.input_parquet is None:
            raise ValueError("--input-parquet is required when --variant=custom")
        source_path = args.input_parquet.expanduser().resolve()
        source_meta = {
            "path": str(source_path),
            "bytes": source_path.stat().st_size,
            "sha256": digest(source_path),
        }
        dedupe_mode = args.dedupe or "index"
        prompt_format = args.prompt_format or "boxed"
        prepared_rows = prepare_variant_rows(
            iter_parquet(source_path),
            dedupe=dedupe_mode,
            dedupe_prompts=args.dedupe_prompts,
            prompt_format=prompt_format,
            target_data_source=args.target_data_source,
        )
        lineage = {"custom_input": str(source_path)}
    else:
        if args.input_parquet is not None:
            raise ValueError("--input-parquet can only be used with --variant=custom (use --source-dir for presets)")
        preset = VARIANT_PRESETS[args.variant]
        source_meta = PINNED_SOURCES[preset["source_key"]]
        directory = source_meta["repo_id"].replace("/", "--")
        if args.source_dir is not None:
            source_path = args.source_dir.expanduser().resolve() / directory / source_meta["filename"]
            if source_path.stat().st_size != source_meta["bytes"] or digest(source_path) != source_meta["sha256"]:
                raise ValueError(f"Offline source size/hash mismatch: {source_path}")
        else:
            source_path = fetch_source(source_meta, output / "raw" / directory)

        dedupe_mode = args.dedupe or preset["dedupe"]
        prompt_format = args.prompt_format or preset["prompt_format"]
        expected_rows = (
            preset.get("expected_prompt_dedup_rows", preset["expected_rows"])
            if args.dedupe_prompts
            else preset["expected_rows"]
        )
        prepared_rows = prepare_variant_rows(
            iter_parquet(source_path),
            dedupe=dedupe_mode,
            dedupe_prompts=args.dedupe_prompts,
            prompt_format=prompt_format,
            expected_raw_rows=source_meta["raw_rows"],
            expected_rows=expected_rows,
            expected_raw_data_source=source_meta["raw_data_source"],
            target_data_source=args.target_data_source,
        )
        lineage = preset["lineage"]

    train_artifact = write_parquet(prepared_rows, output / "train.parquet")

    manifest = {
        "variant": args.variant,
        "schema_version": 1,
        "source": source_meta,
        "lineage": lineage,
        "transform": {
            "dedupe": dedupe_mode,
            "dedupe_prompts": args.dedupe_prompts,
            "prompt_format": prompt_format,
            "target_data_source": args.target_data_source,
            "boxed_suffix": BOXED_SUFFIX,
        },
        "validation_policy": (
            "Training preparation only. Do not use test/aime24.parquet or test/aime25.parquet from "
            "Jiawei415/DPAO_filter (both files have identical bytes/SHA-256); prepare validation splits separately."
        ),
        "reward": {
            "dispatcher": "verl.utils.reward_score.default_compute_score",
            "scorer": "verl.utils.reward_score.math_dapo.compute_score",
        },
        "artifacts": {"train": train_artifact},
        "generator": {
            "path": "examples/data_preprocess/dapo_math_variants.py",
            "sha256": digest(__file__),
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        if copy(src=str(output), dst=args.hdfs_dir) is False:
            raise RuntimeError(f"Failed to copy prepared outputs from {output} to {args.hdfs_dir}")

    print(
        json.dumps(
            {
                "variant": args.variant,
                "manifest": str(manifest_path),
                "rows": train_artifact["rows"],
                "artifacts": {"train": train_artifact},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
