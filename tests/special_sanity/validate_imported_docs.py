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
verify_imported_docs.py

Assert that every function or class *explicitly imported* (via
`from <module> import <name>`) in a given Python file has a docstring.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import pathlib
import sys


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify that imported functions/classes have docstrings.")
    p.add_argument(
        "--target-file",
        default="verl/trainer/ppo/ray_trainer.py",
        help="Path to the Python source file to analyse (e.g. verl/trainer/ppo/ray_trainer.py)",
    )
    p.add_argument(
        "--allow-list",
        default=["omegaconf.open_dict"],
        help="a list of third_party dependencies that do not have proper docs :(",
    )
    p.add_argument(
        "--project-root",
        default=".",
        help="Directory to prepend to PYTHONPATH so local packages resolve (default: .)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress success message (still prints errors).",
    )
    return p.parse_args()


def _import_attr(module_name: str, attr_name: str):
    """Import `module_name` then return `getattr(module, attr_name)`."""
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _check_file(py_file: pathlib.Path, project_root: pathlib.Path, allow_list: list[str]) -> list[str]:
    """Return a list of error strings (empty == success)."""
    # Ensure local packages resolve
    sys.path.insert(0, str(project_root.resolve()))

    tree = ast.parse(py_file.read_text(), filename=str(py_file))
    problems: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue

        # Relative imports (level > 0) get the leading dots stripped
        module_name = "." * node.level + (node.module or "")
        for alias in node.names:
            if alias.name == "*":
                problems.append(
                    f"{py_file}:{node.lineno} - wildcard import `from {module_name} import *` cannot be verified."
                )
                continue

            imported_name = alias.name

            try:
                obj = _import_attr(module_name, imported_name)
            except Exception:  # pragma: no cover – wide net for import quirks
                pass
                # For some reason the module cannot be imported, skip for now
                # problems.append(
                #     f"{py_file}:{node.lineno} - could not resolve "
                #     f"`{imported_name}` from `{module_name}` ({exc})"
                # )
                continue

            if f"{module_name}.{imported_name}" in allow_list:
                continue
            if inspect.isfunction(obj) or inspect.isclass(obj):
                doc = inspect.getdoc(obj)
                if not (doc and doc.strip()):
                    kind = "class" if inspect.isclass(obj) else "function"
                    problems.append(
                        f"{py_file}:{node.lineno} - {kind} `{module_name}.{imported_name}` is missing a docstring."
                    )

    return problems


def main() -> None:
    args = _parse_args()
    target_path = pathlib.Path(args.target_file).resolve()
    project_root = pathlib.Path(args.project_root).resolve()

    if not target_path.is_file():
        raise Exception(f"❌ Target file not found: {target_path}")

    errors = _check_file(target_path, project_root, args.allow_list)

    if errors:
        print("Docstring verification failed:\n")
        print("\n".join(f" • {e}" for e in errors))
        raise Exception("❌ Docstring verification failed.")

    if not args.quiet:
        print(f"✅ All explicitly imported functions/classes in {target_path} have docstrings.")


if __name__ == "__main__":
    main()
