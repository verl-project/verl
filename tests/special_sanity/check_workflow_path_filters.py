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

"""Verify that literal paths in workflow ``paths:`` filters still exist.

A ``paths:`` / ``paths-ignore:`` filter decides whether a workflow runs at all.
When a file is renamed or deleted, a filter entry naming it keeps parsing fine
and GitHub reports nothing -- the entry just stops matching. The failure is
silent and points the wrong way in each direction:

* A **positive** entry (``- "verl/trainer/foo.py"``) stops firing, so edits to
  the thing it was meant to guard no longer trigger the workflow.
* A **negative** entry (``- "!verl/trainer/foo.py"``) stops excluding, so the
  "skip this expensive job for unrelated changes" intent is lost and the job
  runs more often than intended.

Only *literal* entries are checked. Anything containing a glob metacharacter
(``*``, ``?``, ``[``) or a ``${{ }}`` expression is a pattern rather than a
concrete file, and may legitimately match nothing today.

Paths under a submodule are skipped: ``recipe/`` has an empty working tree
until ``git submodule update --init``, so checking it would fail depending on
checkout state rather than on correctness.

Usage::

    python3 tests/special_sanity/check_workflow_path_filters.py
    python3 tests/special_sanity/check_workflow_path_filters.py --workflows-dir .github/workflows

Exits 1 and prints ``workflow:line -> path`` for each entry naming something
that does not exist.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Keys whose list items are path filters.
PATH_FILTER_KEYS = ("paths", "paths-ignore")

# Prefixes backed by a git submodule: empty working tree in a plain clone.
SUBMODULE_PREFIXES = ("recipe/",)

# Entries that are patterns, not concrete paths.
GLOB_CHARS = "*?[]"

# Known-missing entries that are intentionally kept. Each needs a reason.
ALLOW_LIST: dict[str, str] = {}


def _is_pattern(value: str) -> bool:
    """True if the entry is a glob or expression rather than a literal path."""
    return any(c in value for c in GLOB_CHARS) or "${{" in value


def _iter_filter_entries(lines: list[str]):
    """Yield ``(line_number, raw_value)`` for each item under a path-filter key."""
    in_filter = False
    key_indent = 0
    item_indent: int | None = None
    for lineno, raw in enumerate(lines, 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())

        if stripped.endswith(":") and stripped[:-1].strip() in PATH_FILTER_KEYS:
            in_filter = True
            key_indent = indent
            item_indent = None
            continue

        if not in_filter:
            continue

        # Anything that is not a list item ends the block (a sibling mapping key
        # such as `branches:`, or the start of the next section).
        if not stripped.startswith("- "):
            in_filter = False
            continue

        # YAML allows a sequence to sit at the key's own indent or deeper, so the
        # first item fixes the level. A later item that dedents past it belongs to
        # an enclosing sequence -- e.g. the next `- name:` step -- not to us.
        if item_indent is None:
            if indent < key_indent:
                in_filter = False
                continue
            item_indent = indent
        elif indent < item_indent:
            in_filter = False
            continue

        value = stripped[2:].strip()
        if " #" in value:  # trailing comment
            value = value.split(" #", 1)[0].strip()
        value = value.strip("\"'")
        if value:
            yield lineno, value


def check_workflow_path_filters(repo_root: Path, workflows_dir: Path) -> list[tuple[str, int, str]]:
    """Return ``(workflow_name, line_number, path)`` for each stale literal entry."""
    violations: list[tuple[str, int, str]] = []
    for workflow in sorted(workflows_dir.glob("*.yml")) + sorted(workflows_dir.glob("*.yaml")):
        lines = workflow.read_text(encoding="utf-8").splitlines()
        for lineno, value in _iter_filter_entries(lines):
            path = value[1:] if value.startswith("!") else value
            if not path or _is_pattern(path):
                continue
            if path.startswith(SUBMODULE_PREFIXES):
                continue
            if path in ALLOW_LIST:
                continue
            if not (repo_root / path).exists():
                violations.append((workflow.name, lineno, path))
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflows-dir",
        default=".github/workflows",
        help="directory holding the workflow files (default: .github/workflows)",
    )
    args = parser.parse_args(argv)

    workflows_dir = Path(args.workflows_dir)
    if not workflows_dir.is_dir():
        print(f"workflows directory not found: {workflows_dir}", file=sys.stderr)
        return 2

    # Filters are repo-relative, so resolve against the repo root.
    repo_root = Path.cwd()
    violations = check_workflow_path_filters(repo_root, workflows_dir)

    if not violations:
        return 0

    print("Workflow path filters referencing files that do not exist:\n")
    for name, lineno, path in violations:
        print(f"  {name}:{lineno} -> {path}")
    print(
        "\nA filter entry naming a missing file never matches. Point it at the "
        "current path, drop it if the job it guarded is gone, or add it to "
        "ALLOW_LIST in this script with a reason."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
