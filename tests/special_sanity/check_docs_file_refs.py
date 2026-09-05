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

"""Verify that repo file paths mentioned in the docs actually exist.

Docs drift silently: a workflow gets renamed, a config is deprecated, or an
example script moves, and the prose keeps pointing at the old path. Nothing
fails, so readers hit the dead path instead. This hook fails fast instead.

Two reference styles are checked:

1. **Inline literals** — a repo-relative path inside double backticks, e.g.
   ``` ``verl/trainer/config/ppo_trainer.yaml`` ```. Only literals that look
   repo-relative (they start with a known top-level directory) are checked,
   so bare filenames and prose stay out of scope.

2. **Runnable commands** — a path passed to ``bash``/``python``/``sh`` inside
   a code block, e.g. ``bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh``.
   These are what a reader copy-pastes, so a stale one is the most
   user-visible kind of rot.

Deliberately **not** flagged, since each is a legitimate miss rather than rot:

* Placeholders the reader is meant to substitute (``path/to/model``,
  ``your_script.py``, ``test_xxx.py``).
* Paths inside a submodule (``recipe/``). The working tree is empty until
  ``git submodule update --init``, so checking them would fail spuriously
  depending on checkout state.
* Paths that resolve relative to the *parent* of the repo. The install guides
  say ``git clone ... && bash verl/scripts/install_*.sh``, which is correct
  from the directory above the clone.
* Files a reader is told to create (``config.yaml``, ``ray_start.sh``).

Usage::

    python3 tests/special_sanity/check_docs_file_refs.py
    python3 tests/special_sanity/check_docs_file_refs.py --docs-dir docs

Exits 1 and prints ``file:line -> path`` for each unresolvable reference.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Top-level directories that make a literal look repo-relative. A literal not
# starting with one of these is prose or a bare filename, not a path claim.
REPO_ROOTS = (
    "verl/",
    "examples/",
    "tests/",
    "docs/",
    "scripts/",
    ".github/",
)

# Submodules: tracked as a commit, so the working tree is empty until
# `git submodule update --init --recursive`. Checking these would depend on
# checkout state rather than on whether the docs are correct.
SUBMODULE_PREFIXES = ("recipe/",)

# Substrings marking a path the reader is meant to replace, not open.
PLACEHOLDER_MARKERS = (
    "path/to",
    "your_",
    "xxx",
    "<",  # e.g. run_<model>_<backend>.sh
    "{",  # e.g. {model}.yaml
)

# Specific references that are correct but unresolvable from the repo root,
# or that are known-missing and tracked separately.
# Keep this list short and say why for each entry.
ALLOW_LIST = {
    # The install guides run these from the *parent* of the clone:
    #   git clone ... verl && bash verl/scripts/install_vllm_mcore_npu.sh
    # The real files are scripts/install_*.sh, one level down.
    "verl/scripts/install_vllm_mcore_npu.sh",
    "verl/scripts/install_sglang_mcore_npu.sh",
    # --- Known-missing, left for the owning teams rather than guessed at. ---
    # The Ascend vLLM guide points at a Qwen3-30B vLLM script, but
    # examples/ascend_extras/ only ships an SGLang variant for that model
    # (run_qwen3_30b_a3b_megatron.sh). Substituting it would silently change
    # the documented rollout backend, so this needs an Ascend-team decision:
    # either add the vLLM script or repoint the guide.
    "examples/grpo_trainer/run_qwen3moe-30b_grpo_megatron_vllm_npu.sh",
    # The search-tool and agentic-RL guides reference examples/sglang_multiturn/,
    # which no longer exists anywhere in the repo (nor in the verl-recipe
    # submodule). These walkthroughs need a rewrite against the current
    # agent-loop examples rather than a path substitution.
    "examples/sglang_multiturn/search_r1_like/local_dense_retriever/download.py",
    "examples/sglang_multiturn/search_r1_like/local_dense_retriever/retrieval_server.py",
    "examples/sglang_multiturn/search_r1_like/run_qwen2_5_3b_search_multiturn_fsdp.sh",
    "examples/sglang_multiturn/config/tool_config/search_tool_config.yaml",
    "examples/sglang_multiturn/run_qwen2_5_3b_gsm8k_tool_agent_mlflow_fsdp.sh",
}

# ``some/path.ext`` — inline literal.
INLINE_RE = re.compile(r"``([A-Za-z0-9_.][A-Za-z0-9_./-]*\.(?:py|sh|yaml|yml|rst|md|txt|json))``")

# bash/python/sh <path> — runnable command inside a code block.
_ROOTS_ALT = "|".join(re.escape(root) for root in REPO_ROOTS)
COMMAND_RE = re.compile(r"(?:^|[\s|&;(])(?:bash|sh|python3?)\s+((?:" + _ROOTS_ALT + r")[A-Za-z0-9_./-]*\.(?:sh|py))")


def _is_exempt(ref: str) -> bool:
    """Return whether ``ref`` is a legitimate non-path reference."""
    if ref in ALLOW_LIST:
        return True
    if any(marker in ref for marker in PLACEHOLDER_MARKERS):
        return True
    return any(ref.startswith(prefix) for prefix in SUBMODULE_PREFIXES)


def _looks_repo_relative(ref: str) -> bool:
    return ref.startswith(REPO_ROOTS)


def check_file(doc: Path, repo_root: Path) -> list[str]:
    """Return ``file:line -> ref`` strings for unresolvable references in ``doc``."""
    problems: list[str] = []
    try:
        lines = doc.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError as exc:  # pragma: no cover - unreadable file
        return [f"{doc}: could not read ({exc})"]

    try:
        shown_doc = doc.relative_to(repo_root).as_posix()
    except ValueError:
        shown_doc = str(doc)

    for lineno, line in enumerate(lines, 1):
        refs: list[str] = []
        refs.extend(m.group(1) for m in INLINE_RE.finditer(line) if _looks_repo_relative(m.group(1)))
        refs.extend(m.group(1) for m in COMMAND_RE.finditer(line))

        for ref in refs:
            if _is_exempt(ref):
                continue
            if (repo_root / ref).exists():
                continue
            problems.append(f"{shown_doc}:{lineno} -> {ref}")

    return problems


def collect_docs(docs_dir: Path) -> list[Path]:
    """Return sorted ``.rst``/``.md`` files under ``docs_dir``, skipping build output."""
    return sorted(
        p for p in docs_dir.rglob("*") if p.is_file() and p.suffix in {".rst", ".md"} and "_build" not in p.parts
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"), help="Directory to scan (default: docs)")
    parser.add_argument("--repo-root", type=Path, default=Path("."), help="Repository root (default: .)")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    docs_dir = (repo_root / args.docs_dir) if not args.docs_dir.is_absolute() else args.docs_dir
    if not docs_dir.is_dir():
        print(f"❌  --docs-dir '{args.docs_dir}' does not exist or is not a directory.", file=sys.stderr)
        return 2

    docs = collect_docs(docs_dir)
    problems: list[str] = []
    for doc in docs:
        problems.extend(check_file(doc, repo_root))

    if problems:
        print("❌  Docs reference files that do not exist:\n", file=sys.stderr)
        for problem in problems:
            print("  - " + problem, file=sys.stderr)
        print(
            "\nEach line above is a path mentioned in the docs that is missing from the repo.\n"
            "Fix the path, or - if the reference is intentional (a placeholder, a\n"
            "submodule path, or a file the reader creates) - extend ALLOW_LIST or\n"
            "PLACEHOLDER_MARKERS in tests/special_sanity/check_docs_file_refs.py\n"
            "with a comment explaining why.\n",
            file=sys.stderr,
        )
        return 1

    print(f"✅  All repo file references in {len(docs)} docs under '{args.docs_dir}' resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
