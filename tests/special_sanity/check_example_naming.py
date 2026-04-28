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

"""Enforce the canonical naming convention for example run scripts.

The current convention (see ``examples/README.md``) is::

    run_<model>_<train-backend>.sh

Where ``<train-backend>`` is one of ``fsdp``, ``fsdp2``, ``megatron``,
``mindspeed``, ``automodel`` or ``veomni``, and **must be the last
underscore-separated token before** ``.sh`` — nothing follows it. The legacy
convention used to embed the inference backend (``vllm``/``sglang``/``trtllm``),
platform tokens (``_npu``/``_amd``), machine-type tokens (``_gb200``,
``_blackwell``), quantization variants (``_fp8``), and ad-hoc trailing
suffixes into the filename. All of those are now merged into a single
canonical script and selected at runtime via env-var toggles
(``INFER_BACKEND``, ``DEVICE``, ``MACHINE``, ``QUANT``, ...).

Usage::

    python3 tests/special_sanity/check_example_naming.py
    python3 tests/special_sanity/check_example_naming.py --root examples \
        --ignore-dirs examples/data_preprocess examples/profile

Exits with status 1 (and prints offending paths) if any script under
``--root`` violates the convention.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Tokens that may NOT appear as a word in the filename. They used to be
# embedded as ``_vllm_fsdp``/``_sglang_fsdp``/``_trtllm_fsdp``/``_npu``/
# ``_gb200``/``_fp8`` etc.; the active convention exposes them via env-var
# toggles instead.
FORBIDDEN_TOKENS = (
    # Inference backends.
    "vllm",
    "sglang",
    "trtllm",
    # Platform / accelerator vendors.
    "npu",
    "amd",
    # Specific GPU machine types — selected at runtime via ``MACHINE`` env
    # var, never embedded in filenames.
    "gb200",
    "b200",
    "blackwell",
    # Quantization variants.
    "fp8",
)

# Recognised train-backend / engine markers. The filename must end with
# ``_<one of these>.sh`` (i.e. the train-backend is the LAST underscore-
# separated token). Generation-only scripts that do not run a trainer are
# listed in ``DEFAULT_IGNORE_FILES`` instead.
ALLOWED_BACKENDS = (
    "fsdp",
    "fsdp2",
    "megatron",
    "mindspeed",
    "automodel",
    "veomni",
)

# Directories whose scripts have their own conventions and are exempt from
# the train-backend rule. Paths are interpreted relative to the repo root.
DEFAULT_IGNORE_DIRS = (
    "examples/data_preprocess",
    "examples/profile",
    "examples/tutorial",
    "examples/vllm_omni",
    # MTP scripts encode async/multinode dispatch in the filename; they pre-
    # date this convention. Migrate separately rather than block this hook.
    "examples/mtp_trainer",
)

# Individual files that are exempt. Use sparingly and document why.
DEFAULT_IGNORE_FILES = (
    # Inference-only entry point — no train-backend in the name.
    "examples/generation/run_deepseek_llm_7b.sh",
    # Diffusion (flow-grpo) example using the vllm_omni rollout; the ``omni``
    # token in the name refers to that rollout variant rather than a train
    # backend, and this is the only example in flowgrpo_trainer/.
    "examples/flowgrpo_trainer/run_qwen_image_omni_lora.sh",
    # Sibling of run_qwen2_5_7b_fsdp.sh that adds a multi-rollout-server
    # dispatch flavour the canonical script does not yet expose. Migrate by
    # folding ``_multi_rs`` into a ROLLOUT_SERVER env-var toggle rather than
    # in this PR.
    "examples/rollout_correction/run_qwen2_5_7b_fsdp_multi_rs.sh",
)


def _split_tokens(stem: str) -> list[str]:
    """Split a script stem on underscores, dropping the leading ``run`` token."""
    parts = stem.split("_")
    if parts and parts[0] == "run":
        parts = parts[1:]
    return parts


def _is_ignored(path: Path, repo_root: Path, ignore_dirs: tuple[str, ...], ignore_files: tuple[str, ...]) -> bool:
    rel = path.relative_to(repo_root).as_posix()
    if rel in ignore_files:
        return True
    return any(rel == d or rel.startswith(d.rstrip("/") + "/") for d in ignore_dirs)


def check_filename(path: Path, display: str | None = None) -> list[str]:
    """Return a list of human-readable violations for one script path.

    The path's *basename* is what we validate; the directory layout is
    enforced separately by the caller via the ``--ignore-dirs`` mechanism.
    ``display`` controls how the path is shown in error messages (defaults
    to ``str(path)``).
    """
    errors: list[str] = []
    name = path.name
    shown = display if display is not None else str(path)

    if not name.startswith("run_"):
        errors.append(f"{shown}: example script must be named 'run_<model>_<train-backend>.sh'")
        return errors

    if not name.endswith(".sh"):
        errors.append(f"{shown}: example script must end with '.sh'")
        return errors

    tokens = _split_tokens(path.stem)

    found_forbidden = [t for t in tokens if t in FORBIDDEN_TOKENS]
    if found_forbidden:
        errors.append(
            f"{shown}: filename contains forbidden token(s) {found_forbidden}. "
            f"Inference-backend ({{vllm,sglang,trtllm}}), platform ({{npu,amd}}), "
            f"machine-type ({{gb200,b200,blackwell}}), and quantization "
            f"({{fp8}}) selections must be exposed as env-var toggles inside "
            f"the script, not embedded in the filename."
        )

    if not tokens or tokens[-1] not in ALLOWED_BACKENDS:
        errors.append(
            f"{shown}: filename must end with '_<train-backend>.sh' where "
            f"train-backend ∈ {list(ALLOWED_BACKENDS)} "
            f"(e.g. 'run_qwen3_8b_fsdp.sh'). Anything after the train-backend "
            f"(e.g. '_fp8', '_gb200', '_multi_rs') must be folded into an "
            f"env-var toggle. If this is an intentional exception, add it to "
            f"DEFAULT_IGNORE_FILES or pass '--ignore-files {shown}'."
        )

    return errors


def collect_scripts(
    root: Path,
    repo_root: Path,
    ignore_dirs: tuple[str, ...],
    ignore_files: tuple[str, ...],
) -> list[Path]:
    return sorted(
        p for p in root.rglob("*.sh") if p.is_file() and not _is_ignored(p, repo_root, ignore_dirs, ignore_files)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("examples"),
        help="Directory to scan (default: examples)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root used for relative-path matching against --ignore-dirs/--ignore-files",
    )
    parser.add_argument(
        "--ignore-dirs",
        nargs="*",
        default=list(DEFAULT_IGNORE_DIRS),
        help="Directories (relative to --repo-root) whose scripts are exempt",
    )
    parser.add_argument(
        "--ignore-files",
        nargs="*",
        default=list(DEFAULT_IGNORE_FILES),
        help="Individual files (relative to --repo-root) that are exempt",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    root = args.root.resolve()
    if not root.is_dir():
        print(f"❌  --root '{args.root}' does not exist or is not a directory.", file=sys.stderr)
        return 2

    scripts = collect_scripts(
        root,
        repo_root,
        tuple(args.ignore_dirs),
        tuple(args.ignore_files),
    )

    all_errors: list[str] = []
    for script in scripts:
        try:
            display = script.relative_to(repo_root).as_posix()
        except ValueError:
            display = str(script)
        all_errors.extend(check_filename(script, display=display))

    if all_errors:
        print("❌  Example script naming violations:\n", file=sys.stderr)
        for err in all_errors:
            print("  - " + err, file=sys.stderr)
        print(
            "\nNaming convention (see examples/README.md):\n"
            "  run_<model>_<train-backend>.sh   (train-backend is the LAST token)\n"
            f"  train-backend ∈ {list(ALLOWED_BACKENDS)}\n"
            f"  must NOT contain any of {list(FORBIDDEN_TOKENS)} as a word.\n",
            file=sys.stderr,
        )
        return 1

    print(f"✅  {len(scripts)} example scripts under '{args.root}' follow the naming convention.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
