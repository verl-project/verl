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

"""Keep the uv launch path GPU-only, so Ascend NPU runs keep using ambient python.

The uv flow (``pyproject.toml`` / ``uv.lock``) resolves only the CUDA backends —
x86_64 Linux, CPython 3.12, cu130 — so ``vllm-ascend`` / ``sglang-ascend`` /
``mindspeed`` cannot come from it. Shell scripts therefore invoke uv only inside
a branch gated on both toggles::

    LAUNCH=(python3)
    RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
    if [ "${VERL_USE_UV:-1}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ]; then
        LAUNCH=(uv run --frozen --all-packages --extra vllm --extra megatron python3)
        RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen ...")
    fi

This check enforces two rules:

1. Every uv command in a shell script sits inside such a gate — including the
   ``py_executable`` string handed to Ray, which would otherwise start worker
   actors under uv on a device the lockfile does not cover.
2. NPU-only trees and scripts (``examples/ascend_extras/``, ``tests/special_npu/``,
   ``*_npu*.sh``, ``*ascend*.sh``, ``*mindspeed*.sh``) invoke uv nowhere at all.

Usage::

    python3 tests/special_sanity/check_uv_gpu_only.py
    python3 tests/special_sanity/check_uv_gpu_only.py --roots examples recipe tests

Exits with status 1 (and prints offending lines) on any violation.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DEFAULT_ROOTS = ("examples", "recipe", "tests")

# A uv invocation: the `uv` word followed by a subcommand, either as a bare
# command or embedded in a string (`py_executable="uv -v run ..."`).
UV_COMMAND = re.compile(r"""(?:^|[\s;&|(="'])uv\s+(?:-\S+\s+)*(?:run|sync|pip|lock|venv|tool|add|export)\b""")

# The gate that makes a branch GPU-only: the opt-out toggle plus the device
# probe. Both clauses must be present; the inference-backend clause that some
# scripts add on top (vllm/sglang only) is optional.
USE_UV_CLAUSE = re.compile(r"\$\{VERL_USE_UV:-1\}")
GPU_CLAUSE = re.compile(r"""\[\s*"\$\{DEVICE:-gpu\}"\s*=\s*gpu\s*\]""")

# Trees and filename markers that only ever run on Ascend NPU.
NPU_DIRS = ("examples/ascend_extras", "tests/special_npu")
NPU_NAME_MARKERS = ("_npu", "ascend", "mindspeed")


def is_npu_only(rel_path: str) -> bool:
    if any(rel_path == d or rel_path.startswith(d + "/") for d in NPU_DIRS):
        return True
    name = Path(rel_path).name.lower()
    return any(marker in name for marker in NPU_NAME_MARKERS)


def strip_comment(line: str) -> str:
    """Drop a whole-line comment; keep everything else verbatim.

    Trailing comments are left in place — quoting makes them ambiguous, and a
    uv command hiding behind one still deserves to be flagged.
    """
    return "" if line.lstrip().startswith("#") else line


def uv_command_lines(lines: list[str]) -> list[int]:
    return [i for i, line in enumerate(lines) if UV_COMMAND.search(strip_comment(line))]


def enclosing_ifs(lines: list[str], idx: int) -> list[int]:
    """Indices of the ``if`` lines whose *then* branch contains ``lines[idx]``, innermost first.

    An ``if`` whose ``else`` / ``elif`` branch holds the line does not count: a
    gate only protects what follows its own ``then``.
    """
    found: list[int] = []
    depth = 0
    in_else = False
    for j in range(idx - 1, -1, -1):
        stripped = strip_comment(lines[j]).strip()
        if stripped == "fi":
            depth += 1
        elif depth == 0 and (stripped == "else" or stripped.startswith("elif ")):
            in_else = True
        elif stripped.startswith("if ") and not stripped.endswith("fi"):
            if depth == 0:
                if not in_else:
                    found.append(j)
                in_else = False
            else:
                depth -= 1
    return found


def is_gpu_gate(line: str) -> bool:
    return bool(USE_UV_CLAUSE.search(line) and GPU_CLAUSE.search(line))


def check_script(lines: list[str], display: str) -> list[str]:
    uv_lines = uv_command_lines(lines)
    if not uv_lines:
        return []

    if is_npu_only(display):
        return [
            f"{display}:{i + 1}: NPU-only script must not invoke uv "
            f"(uv.lock covers CUDA backends only): {lines[i].strip()}"
            for i in uv_lines
        ]

    errors = []
    for i in uv_lines:
        gates = enclosing_ifs(lines, i)
        if any(is_gpu_gate(lines[j]) for j in gates):
            continue
        where = (
            f"gated only by line {gates[0] + 1} (`{lines[gates[0]].strip()}`)"
            if gates
            else "not inside a GPU-gated `then` branch"
        )
        errors.append(
            f"{display}:{i + 1}: uv command {where}; it must run inside "
            f'`if [ "${{VERL_USE_UV:-1}}" != 0 ] && [ "${{DEVICE:-gpu}}" = gpu ]; then`: {lines[i].strip()}'
        )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--roots",
        nargs="*",
        default=list(DEFAULT_ROOTS),
        help=f"Directories to scan for *.sh, relative to --repo-root (default: {' '.join(DEFAULT_ROOTS)})",
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."), help="Repository root (default: .)")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    scripts: list[Path] = []
    for root in args.roots:
        root_path = (repo_root / root).resolve()
        if not root_path.is_dir():
            print(f"❌  --roots entry '{root}' does not exist or is not a directory.", file=sys.stderr)
            return 2
        scripts.extend(root_path.rglob("*.sh"))

    errors: list[str] = []
    uv_scripts = 0
    for script in sorted(set(scripts)):
        lines = script.read_text().split("\n")
        if not uv_command_lines(lines):
            continue
        uv_scripts += 1
        errors.extend(check_script(lines, script.relative_to(repo_root).as_posix()))

    if errors:
        print("❌  uv must only run on the GPU branch:\n", file=sys.stderr)
        for err in errors:
            print("  - " + err, file=sys.stderr)
        print(
            "\nThe uv lockfile resolves CUDA backends only (x86_64 Linux / cp312 / cu130), so every\n"
            "uv command — including the py_executable string passed to Ray — must live inside\n"
            '  if [ "${VERL_USE_UV:-1}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ]; then\n'
            "and NPU-only scripts must not reference uv at all. See docs/start/install.rst.\n",
            file=sys.stderr,
        )
        return 1

    print(f"✅  Every uv command in the {uv_scripts} shell scripts that use uv sits behind the DEVICE=gpu gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
