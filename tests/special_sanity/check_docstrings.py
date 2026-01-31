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

"""
Python script to check docstrings for functions and classes in specified files.
Checks that every public function and class has proper docstring documentation.
"""

import ast
import os
import sys


class DocstringChecker(ast.NodeVisitor):
    """AST visitor to check for missing docstrings in functions and classes."""

    def __init__(self, filename: str):
        self.filename = filename
        self.missing_docstrings: list[tuple[str, str, int]] = []
        self.current_class = None
        self.function_nesting_level = 0

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definitions and check for docstrings."""
        if not node.name.startswith("_") and self.function_nesting_level == 0:
            if not self._has_docstring(node):
                func_name = f"{self.current_class}.{node.name}" if self.current_class else node.name
                self.missing_docstrings.append((func_name, self.filename, node.lineno))

        self.function_nesting_level += 1
        self.generic_visit(node)
        self.function_nesting_level -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Visit async function definitions and check for docstrings."""
        if not node.name.startswith("_") and self.function_nesting_level == 0:
            if not self._has_docstring(node):
                func_name = f"{self.current_class}.{node.name}" if self.current_class else node.name
                self.missing_docstrings.append((func_name, self.filename, node.lineno))

        self.function_nesting_level += 1
        self.generic_visit(node)
        self.function_nesting_level -= 1

    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definitions and check for docstrings."""
        if not node.name.startswith("_"):
            if not self._has_docstring(node):
                self.missing_docstrings.append((node.name, self.filename, node.lineno))

        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def _has_docstring(self, node) -> bool:
        """Check if a node has a docstring."""
        return ast.get_docstring(node) is not None


def check_file_docstrings(filepath: str) -> list[tuple[str, str, int]]:
    """Check docstrings in a single file."""
    try:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content, filename=filepath)
        checker = DocstringChecker(filepath)
        checker.visit(tree)
        return checker.missing_docstrings

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return []


def main():
    """Main function to check docstrings in specified files."""

    files_to_check = [
        "verl/trainer/ppo/ray_trainer.py",
        "verl/trainer/main_ppo.py",
        "verl/trainer/ppo/reward.py",
        "verl/utils/reward_score/__init__.py",
        "verl/trainer/ppo/core_algos.py",
        "verl/experimental/agent_loop/agent_loop.py",
        "verl/workers/sharding_manager/fsdp_vllm.py",
        "verl/workers/sharding_manager/fsdp_ulysses.py",
    ]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_path = os.path.dirname(os.path.dirname(script_dir))

    if not os.path.exists(repo_path):
        print(f"Repository path {repo_path} does not exist!")
        sys.exit(1)

    os.chdir(repo_path)

    all_missing_docstrings = []

    print("Checking docstrings in specified files...")
    print("=" * 60)

    for file_path in files_to_check:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist!")
            continue

        print(f"Checking {file_path}...")
        missing = check_file_docstrings(file_path)
        all_missing_docstrings.extend(missing)

        if missing:
            print(f"  Found {len(missing)} missing docstrings")
        else:
            print("  All functions and classes have docstrings [OK]")

    print("=" * 60)

    if all_missing_docstrings:
        print(f"\nSUMMARY: Found {len(all_missing_docstrings)} functions/classes missing docstrings:")
        print("-" * 60)

        by_file = {}
        for name, filepath, lineno in all_missing_docstrings:
            if filepath not in by_file:
                by_file[filepath] = []
            by_file[filepath].append((name, lineno))

        for filepath in sorted(by_file.keys()):
            print(f"\n{filepath}:")
            for name, lineno in sorted(by_file[filepath], key=lambda x: x[1]):
                print(f"  - {name} (line {lineno})")

        print(f"\nTotal missing docstrings: {len(all_missing_docstrings)}")

        raise Exception(f"Found {len(all_missing_docstrings)} functions/classes without proper docstrings!")

    else:
        print("\n[OK] All functions and classes have proper docstrings!")


if __name__ == "__main__":
    main()
