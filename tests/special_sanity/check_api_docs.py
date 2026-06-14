# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Fail CI if any function or class that is publicly exported via
``__all__`` lacks a docstring.

Usage
-----
  # Check specific modules or packages
  python check_docstrings.py mypkg.core mypkg.utils

  # Check an entire source tree (all top-level packages under cwd)
  python check_docstrings.py
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import pkgutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable

_ALLOW_LIST = [
    "verl.third_party.vllm.LLM",
    "verl.third_party.vllm.parallel_state",
    "verl.utils.profiler.WorkerProfiler",
    "verl.utils.profiler.WorkerProfilerExtension",
    "verl.utils.profiler.log_gpu_memory_usage",
    "verl.utils.profiler.log_print",
    "verl.utils.profiler.mark_annotate",
    "verl.utils.profiler.mark_end_range",
    "verl.utils.profiler.mark_start_range",
    "verl.models.mcore.qwen2_5_vl.get_vision_model_config",
    "verl.models.mcore.qwen2_5_vl.get_vision_projection_config",
    "verl.models.mcore.mbridge.freeze_moe_router",
    "verl.models.mcore.mbridge.make_value_model",
    "verl.utils.transformers_compat.flash_attn_supports_top_left_mask",
]


def iter_submodules(root: ModuleType) -> Iterable[ModuleType]:
    """Yield *root* and every sub-module inside it."""
    yield root

    def print_pkg_error(pkg_name):
        print(f"[warn] Skipping {pkg_name!r}", file=sys.stderr)

    if getattr(root, "__path__", None):  # only packages have __path__
        for mod_info in pkgutil.walk_packages(root.__path__, prefix=f"{root.__name__}.", onerror=print_pkg_error):
            try:
                yield importlib.import_module(mod_info.name)
            except Exception as exc:
                print(f"[warn] Skipping {mod_info.name!r}: {exc}", file=sys.stderr)


def names_missing_doc(mod: ModuleType) -> list[str]:
    """Return fully-qualified names that need docstrings."""
    missing: list[str] = []
    public = getattr(mod, "__all__", [])
    for name in public:
        obj = getattr(mod, name, None)
        if f"{mod.__name__}.{name}" in _ALLOW_LIST:
            continue
        if obj is None:
            # Exported but not found in the module: flag it anyway.
            missing.append(f"{mod.__name__}.{name}  (not found)")
            continue

        if inspect.isfunction(obj) or inspect.isclass(obj):
            doc = inspect.getdoc(obj)
            if not doc or not doc.strip():
                missing.append(f"{mod.__name__}.{name}")
    return missing


def check_module(qualname: str) -> list[str]:
    """Import *qualname* and check it (and sub-modules)."""
    try:
        module = importlib.import_module(qualname)
    except ModuleNotFoundError as exc:
        print(f"[error] Cannot import '{qualname}': {exc}", file=sys.stderr)
        return [qualname]

    missing: list[str] = []
    for submod in iter_submodules(module):
        missing.extend(names_missing_doc(submod))
    return missing


def autodiscover_packages() -> list[str]:
    """Detect top-level packages under CWD when no argument is given."""
    pkgs: list[str] = []
    for p in Path.cwd().iterdir():
        if p.is_dir() and (p / "__init__.py").exists():
            pkgs.append(p.name)
    return pkgs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "modules",
        nargs="*",
        help="Fully-qualified module or package names (defaults to every top-level package found in CWD).",
    )
    args = parser.parse_args()

    targets = args.modules or autodiscover_packages()
    if not targets:
        raise ValueError("[error] No modules specified and none detected automatically.")

    all_missing: list[str] = []
    for modname in targets:
        all_missing.extend(check_module(modname))

    if all_missing:
        print("\nMissing docstrings:")
        for name in sorted(all_missing):
            print(f"  - {name}")
        raise ValueError("Missing docstrings detected. Please enhance them with docs accordingly.")

    print("✅ All exported functions/classes have docstrings.")


if __name__ == "__main__":
    main()
