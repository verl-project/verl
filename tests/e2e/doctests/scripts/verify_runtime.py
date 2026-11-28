#!/usr/bin/env python3
# Runtime checks shared by the installation and Quick Start workers.

#
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


"""Check verl Ascend doctest prerequisites without launching training."""

import argparse
import importlib
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def fail(message: str) -> None:
    """Print one failure message to stderr."""
    print(f"verify_runtime: {message}", file=sys.stderr)


def check_input_file(path_text: str, label: str) -> bool:
    """Require each supplied input path to be a nonempty file."""
    if not path_text:
        return True
    path = Path(path_text)
    if not path.is_file():
        fail(f"{label} not found: {path_text}")
        return False
    if path.stat().st_size == 0:
        fail(f"{label} is empty: {path_text}")
        return False
    return True


def check_model_dir(model_path: str) -> bool:
    """Require a supplied model directory to contain config.json."""
    if not model_path:
        return True
    if not Path(model_path, "config.json").is_file():
        fail(f"model config.json not found under: {model_path}")
        return False
    return True


def npu_device_count(failures: list[str]) -> int:
    """Return the visible NPU count, recording import or device-query failures."""
    try:
        import torch

        importlib.import_module("torch_npu")  # Register the NPU extension before querying devices.
    except Exception as error:
        message = f"cannot import torch/torch_npu for device check: {error}"
        failures.append(message)
        fail(message)
        return -1
    try:
        return torch.npu.device_count()
    except Exception as error:
        message = f"cannot query NPU devices: {error}"
        failures.append(message)
        fail(message)
        return -1


def check_imports(backend: str = "") -> bool:
    """Import core/backend packages and log their installed versions for diagnosis."""
    print(f"Python: {sys.version.split()[0]}")
    ok = True
    modules = ["verl", "torch", "torch_npu"]
    if backend == "vllm":
        modules.extend(("vllm", "vllm_ascend"))
    elif backend == "sglang":
        modules.append("sglang")
    for module in modules:
        try:
            imported = importlib.import_module(module)
            try:
                installed = version(module.replace("_", "-"))
            except PackageNotFoundError:
                installed = getattr(imported, "__version__", "unknown (no package metadata)")
            print(f"{module}: {installed}")
        except Exception as error:
            fail(f"cannot import {module}: {error}")
            ok = False
    return ok


def parse_args() -> argparse.Namespace:
    """Parse the optional runtime checks requested by the caller."""
    parser = argparse.ArgumentParser(description="Verify verl Ascend doctest runtime prerequisites.")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--train-file", default="")
    parser.add_argument("--test-file", default="")
    parser.add_argument("--min-devices", type=int, default=0)
    parser.add_argument("--check-imports", action="store_true")
    parser.add_argument("--backend", choices=("vllm", "sglang"), default="")
    args = parser.parse_args()
    if args.min_devices < 0:
        parser.error("--min-devices must be nonnegative")
    if args.backend and not args.check_imports:
        parser.error("--backend requires --check-imports")
    return args


def main() -> int:
    """Run the requested checks and return a nonzero status on any failure."""
    args = parse_args()
    ok = True
    ok &= check_model_dir(args.model_path)
    ok &= check_input_file(args.train_file, "train file")
    ok &= check_input_file(args.test_file, "test file")
    if args.check_imports:
        ok &= check_imports(args.backend)
    if args.min_devices > 0:
        failures: list[str] = []
        count = npu_device_count(failures)
        if count >= 0 and count < args.min_devices:
            message = f"requires at least {args.min_devices} NPU device(s), found {count}"
            failures.append(message)
            fail(message)
        ok &= not failures
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
