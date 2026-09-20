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

"""Make a uv-driven workflow trigger on the files that define its environment.

Jobs that run ``uv run --frozen`` install nothing themselves: the committed
``pyproject.toml`` / ``uv.lock`` pair *is* the environment they execute in. A
``paths:`` filter built around ``**/*.py`` therefore skips the whole suite for
exactly the change class that alters what gets installed - a dependency bump, a
new index, a re-resolve - and the first signal that the environment no longer
builds arrives on main.

That is not hypothetical: switching the verl-wheelhouse index to its per-torch
URL (a ``pyproject.toml`` + ``uv.lock`` diff, no ``.py`` touched) ran zero GPU
jobs, because only the Ascend workflows happened to list ``pyproject.toml``.

The rule: if a workflow invokes uv and filters by ``paths``, that filter must
include both dependency files. A workflow with no ``paths`` filter already runs
on every change and is left alone.

Usage::

    python3 tests/special_sanity/check_uv_workflow_paths.py

Exits with status 1 (and prints the offending workflows) on any violation.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml

DEPENDENCY_FILES = ("pyproject.toml", "uv.lock")

# A lock-driven uv invocation - a `run:` step, or the UV_RUN env var the GPU
# jobs build their command from. `uv pip` is deliberately excluded: the Ascend
# workflows use it as a faster pip against a prebuilt image, resolving nothing
# from the lock, so a lock change gives them nothing to re-test.
UV_COMMAND = re.compile(r"""(?:^|[ \t;&|(="'])uv[ \t]+(?:-\S+[ \t]+)*(?:run|sync|lock|export)\b""")


def uses_uv(text: str) -> bool:
    """Whether the workflow actually runs a lock-driven uv command.

    Matched line by line, and never on a whole-line comment: a workflow that
    only *mentions* `uv run` - in a comment saying it launches ambient python
    instead, as the Ascend e2e jobs do, or in a `- name: Install uv` step
    whose own command is plain pip - is not uv-driven.
    """
    return any(UV_COMMAND.search(line) for line in text.splitlines() if not line.lstrip().startswith("#"))


def workflow_triggers(config: dict[Any, Any]) -> dict[str, Any]:
    """The `on:` block. YAML 1.1 parses the bare key `on` as the boolean True."""
    for key in ("on", True):
        value = config.get(key)
        if isinstance(value, dict):
            return value
    return {}


def check_workflow(path: Path, display: str) -> list[str]:
    text = path.read_text()
    if not uses_uv(text):
        return []

    try:
        config = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return [f"{display}: could not be parsed as YAML ({exc})"]
    if not isinstance(config, dict):
        return [f"{display}: is not a YAML mapping"]

    errors = []
    for event, spec in workflow_triggers(config).items():
        if not isinstance(spec, dict):
            continue  # e.g. `workflow_dispatch:` with no config - no filter
        ignored = [f for f in DEPENDENCY_FILES if f in (spec.get("paths-ignore") or [])]
        if ignored:
            errors.append(
                f"{display}: `on.{event}.paths-ignore` excludes {', '.join(ignored)}, "
                "but this workflow runs uv against them"
            )
        paths = spec.get("paths")
        if paths is None:
            continue  # no filter: already runs on every change
        missing = [f for f in DEPENDENCY_FILES if f not in paths]
        if missing:
            errors.append(f"{display}: `on.{event}.paths` is missing {', '.join(missing)}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", type=Path, default=Path("."), help="Repository root (default: .)")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    workflow_dir = repo_root / ".github" / "workflows"
    if not workflow_dir.is_dir():
        print(f"❌  {workflow_dir} does not exist.", file=sys.stderr)
        return 2

    errors: list[str] = []
    uv_workflows = 0
    for path in sorted([*workflow_dir.glob("*.yml"), *workflow_dir.glob("*.yaml")]):
        display = path.relative_to(repo_root).as_posix()
        if not uses_uv(path.read_text()):
            continue
        uv_workflows += 1
        errors.extend(check_workflow(path, display))

    if errors:
        print("❌  uv-driven workflows must trigger on their dependency files:\n", file=sys.stderr)
        for err in errors:
            print("  - " + err, file=sys.stderr)
        print(
            "\nThese jobs run `uv run --frozen`, so pyproject.toml / uv.lock ARE the environment\n"
            "under test. A paths filter that omits them silently skips the whole suite for every\n"
            "dependency change. Add both to each `paths:` list (a workflow with no filter is fine).\n",
            file=sys.stderr,
        )
        return 1

    print(f"✅  All {uv_workflows} uv-driven workflows trigger on pyproject.toml / uv.lock.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
