# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

#!/usr/bin/env python3
"""
Validate that test file subfolders mirror the top-level package layout.

Usage examples
--------------

# Typical run (defaults: impl_root=my_project, tests_root=tests)
python check_tests_structure.py

# Custom layout and extra allowed folders
python check_tests_structure.py \
    --impl-root verl \
    --tests-root tests \
    --allow-dirs special_e2e special_sanity special_standalone special_distributed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def discover_allowed_modules(impl_root: Path, extra: list[str]) -> set[str]:
    """Return the set of first-level directories that tests may live under."""
    allowed = {p.name for p in impl_root.iterdir() if p.is_dir()}
    allowed.update(extra)
    return allowed


def find_violations(tests_root: Path, allowed: set[str], allowed_files: list[str]) -> list[str]:
    """Return a list of error strings for test files in the wrong place."""
    errors: list[str] = []
    for test_file in tests_root.rglob("test*.py"):
        if str(test_file) in allowed_files:
            continue
        rel_parts = test_file.relative_to(tests_root).parts
        if len(rel_parts) < 2:
            errors.append(f"{test_file}: must be inside one of {sorted(allowed)} (not at tests root)")
            continue

        first_folder = rel_parts[0]
        if first_folder not in allowed:
            errors.append(
                f"{test_file}: subfolder '{first_folder}' under tests/ is not an allowed module. "
                f"The valid ones are: {sorted(allowed)}"
            )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Check that test files follow tests/<module>/… layout.")
    parser.add_argument(
        "--impl-root",
        type=Path,
        default="verl",
        help="Implementation root (default: my_project)",
    )
    parser.add_argument(
        "--tests-root",
        type=Path,
        default="tests",
        help="Root of test tree (default: tests)",
    )
    parser.add_argument(
        "--allow-dirs",
        nargs="*",
        default=["special_e2e", "special_sanity", "special_standalone", "special_distributed"],
        help="Extra top-level test folders that are exempt from the rule",
    )
    parser.add_argument(
        "--allow-files",
        nargs="*",
        default=[
            "tests/test_protocol_on_cpu.py",
            "tests/test_base_config_on_cpu.py",
            "tests/test_protocol_v2_on_cpu.py",
        ],
        help="Extra top-level test folders that are exempt from the rule",
    )
    args = parser.parse_args()

    if not args.impl_root.is_dir():
        raise Exception(f"Implementation root '{args.impl_root}' does not exist.")
    if not args.tests_root.is_dir():
        raise Exception(f"Tests root '{args.tests_root}' does not exist.")

    allowed = discover_allowed_modules(args.impl_root, args.allow_dirs)
    violations = find_violations(args.tests_root, allowed, args.allow_files)

    if violations:
        print("❌  Test layout violations found:\n", file=sys.stderr)
        for err in violations:
            print("  -", err, file=sys.stderr)

        print(
            f"\nGuideline:\n  Place each test file under   tests/<module_name>/…\n  where <module_name> is "
            f"one of the top-level packages inside '{args.impl_root}', or is explicitly listed via --allow-dirs.\n",
            file=sys.stderr,
        )
        raise Exception("❌  Test layout violations found.")

    print("✅  Tests folder structure looks good.")


if __name__ == "__main__":
    main()
