#!/usr/bin/env python3

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Extract commands from existing Ascend RST documentation.

Test-side selectors keep installation guidance and Quick Start tables readable.
Only local CANN/data paths are adapted for execution; installation commands and
dependency versions remain sourced from the documentation.
"""

import argparse
import json
import os
import re
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]

INSTALL_DOC = "docs/ascend_tutorial/zh/get_start/install_guidance.rst"

QUICK_DOC = "docs/ascend_tutorial/zh/get_start/quick_start.rst"
CONTAINER_DOC = "docs/ascend_tutorial/zh/get_start/container_source_install.rst"

CASES = ("fsdp2_vllm", "megatron_vllm", "fsdp2_sglang", "megatron_sglang")

BLOCKS_BY_FILE = {
    CONTAINER_DOC: (
        "container-image-vllm",
        "container-image-sglang",
        "container-prerequisites-ubuntu",
        "container-conda",
    ),
    INSTALL_DOC: (
        "installation-vllm",
        "installation-sglang",
    ),
    QUICK_DOC: (
        "quickstart-environment",
        "quickstart-data",
        *(f"quickstart-{case}" for case in CASES),
    ),
}

FILE_BY_BLOCK = {name: path for path, names in BLOCKS_BY_FILE.items() for name in names}

# Match executable lines, not comments or prose mentioning the same command.
BLOCK_ANCHORS = {
    "installation-vllm": r"^bash verl/scripts/install_vllm_mcore_npu\.sh(?:\s|$)",
    "installation-sglang": r"^bash verl/scripts/install_sglang_mcore_npu\.sh(?:\s|$)",
    "quickstart-environment": r"^source /usr/local/Ascend/ascend-toolkit/set_env\.sh\s*$",
    "quickstart-data": r"^python3 examples/data_preprocess/gsm8k\.py(?:\s|$)",
}

DIRECTIVE_RE = re.compile(r"^( *)\.\. (?:code|code-block):: (bash|python)[ \t]*$")


class DoctestError(ValueError):
    """Missing, ambiguous or malformed documented commands."""


def extract_rst_blocks(text, source="input", *, name=None):
    """Read existing RST code directives without requiring documentation markers."""
    lines = text.splitlines()
    blocks = []
    index = 0
    while index < len(lines):
        opening = DIRECTIVE_RE.fullmatch(lines[index])
        index += 1
        if not opening:
            continue
        base_indent = len(opening[1])
        block_name = None
        while index < len(lines) and lines[index].lstrip().startswith(":"):
            option = lines[index].strip()
            if option.startswith(":name:"):
                block_name = option.removeprefix(":name:").strip()
            index += 1
        if index >= len(lines) or lines[index].strip():
            raise DoctestError(f"Missing blank line after directive in {source}")
        while index < len(lines) and not lines[index].strip():
            index += 1
        if index >= len(lines):
            raise DoctestError(f"Empty code block in {source}")
        indent = len(lines[index]) - len(lines[index].lstrip(" "))
        if indent <= base_indent:
            raise DoctestError(f"Unindented code block in {source}")
        body = []
        while index < len(lines):
            line = lines[index]
            if line.strip():
                line_indent = len(line) - len(line.lstrip(" "))
                if line_indent <= base_indent:
                    break
                if line_indent < indent:
                    raise DoctestError(f"Inconsistent code indentation in {source}")
            body.append(line[indent:])
            index += 1
        if name is None or name == block_name:
            blocks.append("\n".join(body).rstrip() + "\n")
    return blocks


def extract_doctest_block(text, name, source="input"):
    """Select exactly one existing code block or Quick Start table command."""
    if name in BLOCKS_BY_FILE[CONTAINER_DOC]:
        matches = extract_rst_blocks(text, source, name=name)
    elif name in BLOCK_ANCHORS:
        matches = [
            block for block in extract_rst_blocks(text, source) if re.search(BLOCK_ANCHORS[name], block, re.MULTILINE)
        ]
    elif name.removeprefix("quickstart-") in CASES:
        case = name.removeprefix("quickstart-")
        script = re.escape(f"tests/special_npu/quick_start/run_qwen3_0_6b_{case}_ascend.sh")
        matches = [
            match + "\n"
            for match in re.findall(rf"^[ \t]+- (bash {script}(?:[ \t]+[^\n]*)?)[ \t]*$", text, re.MULTILINE)
        ]
    else:
        raise DoctestError(f"Unknown documentation block: {name}")
    if len(matches) > 1:
        raise DoctestError(f"Ambiguous documentation block {name!r} in {source}")
    return matches[0] if matches else None


def adapt_runtime_block(block, name):
    """Substitute local CANN/data paths only; keep raw extraction unchanged."""
    for path, variable in (
        ("/usr/local/Ascend/ascend-toolkit/set_env.sh", "CANN_ENV_SCRIPT"),
        ("/usr/local/Ascend/nnal/atb/set_env.sh", "ATB_ENV_SCRIPT"),
    ):
        if value := os.environ.get(variable):
            block = re.sub(
                rf"^source {re.escape(path)}[ \t]*$",
                lambda _: f"source {shlex.quote(value)}",
                block,
                flags=re.MULTILINE,
            )
    if name == "quickstart-data":
        lines = block.splitlines()
        for index, line in enumerate(lines):
            if not re.match(BLOCK_ANCHORS[name], line):
                continue
            for option, variable in (
                ("--local_dataset_path", "GSM8K_DATASET_PATH"),
                ("--local_save_dir", "GSM8K_OUTPUT_DIR"),
            ):
                if value := os.environ.get(variable):
                    argument = r"""(?:"[^"\n]*"|'[^'\n]*'|[^\s;]+)"""
                    pattern = rf"({re.escape(option)}[ \t]+){argument}"
                    if re.search(pattern, line):
                        line = re.sub(pattern, lambda match: match[1] + shlex.quote(value), line)
                    elif option in line:
                        raise DoctestError(f"Missing value for {option}")
                    elif option == "--local_save_dir":
                        line += f" {option} {shlex.quote(value)}"
                    else:
                        raise DoctestError(f"Documented preprocessing command is missing {option}")
            lines[index] = line
        block = "\n".join(lines) + "\n"
    return block


def require_doctest_block(name):
    """Resolve a test-side selector and require exactly one documented command."""
    if name not in FILE_BY_BLOCK:
        raise DoctestError(f"Unknown documentation block: {name}")
    path = FILE_BY_BLOCK[name]
    text = (REPO_ROOT / path).read_text(encoding="utf-8")
    block = extract_doctest_block(text, name, path)
    if block is None:
        raise DoctestError(f"Missing documentation block {name!r} in {path}")
    return block


def validate_documents():
    """Require all installation and Quick Start commands without executing them."""
    for name in FILE_BY_BLOCK:
        require_doctest_block(name)
    for backend in ("vllm", "sglang"):
        container_image(backend)


def container_image(backend):
    """Read one literal CANN image from a user-visible command without executing it."""
    block = require_doctest_block(f"container-image-{backend}")
    values = re.findall(r"^export IMAGE=(.+)$", block, re.MULTILINE)
    if len(values) != 1:
        raise DoctestError(f"Expected one export IMAGE assignment for {backend}")
    try:
        words = shlex.split(values[0])
    except ValueError as error:
        raise DoctestError(f"Invalid image assignment for {backend}: {error}") from error
    if len(words) != 1 or not re.fullmatch(
        r"swr\.cn-south-1\.myhuaweicloud\.com/ascendhub/cann:[A-Za-z0-9_.-]+", words[0]
    ):
        raise DoctestError(f"Expected a literal tagged CANN image for {backend}")
    if words[0].endswith(":latest"):
        raise DoctestError(f"Specify a versioned CANN image for {backend}")
    return words[0]


def installation_matrix(case_name="all"):
    """Select runtime cases and obtain each backend's image from the public guide."""
    if case_name != "all" and case_name not in CASES:
        raise DoctestError(f"Unknown installation case: {case_name}")
    cases = CASES if case_name == "all" else (case_name,)
    return {"include": [{"case": case, "image": container_image(case.rsplit("_", 1)[1])} for case in cases]}


def main():
    """Extract a documented command or validate all registered selectors."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract")
    extract.add_argument("name", choices=sorted(FILE_BY_BLOCK))
    extract.add_argument("--runtime", action="store_true", help="Apply test-only local path overrides")
    commands.add_parser("validate")
    matrix = commands.add_parser("matrix")
    matrix.add_argument("--case", choices=("all", *CASES), default="all")
    args = parser.parse_args()
    if args.command == "extract":
        block = require_doctest_block(args.name)
        if args.runtime:
            block = adapt_runtime_block(block, args.name)
        sys.stdout.write(block)
    elif args.command == "matrix":
        print(json.dumps(installation_matrix(args.case)))
    else:
        validate_documents()
        print(f"Validated {len(FILE_BY_BLOCK)} documentation blocks")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DoctestError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
