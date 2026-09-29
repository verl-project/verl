#!/usr/bin/env python3
# verify_runtime.py：快速入门/安装测试共用的运行时自检（模型、数据、NPU 数量、关键导入）。

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
from pathlib import Path


def fail(message: str) -> None:
    """Print one failure message to stderr（错误信息统一走 stderr）。"""
    print(f"verify_runtime: {message}", file=sys.stderr)


def check_input_file(path_text: str, label: str) -> bool:
    """检查一个输入文件是否存在且非空（空文件视为无效输入）。"""
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
    """检查模型目录是否包含 config.json（HuggingFace 权重目录的最小要求）。"""
    if not model_path:
        return True
    if not Path(model_path, "config.json").is_file():
        fail(f"model config.json not found under: {model_path}")
        return False
    return True


def npu_device_count(failures: list[str]) -> int:
    """返回可见 NPU 数量；torch/torch_npu 不可用时返回 -1 并记录原因。"""
    try:
        import torch
        import torch_npu
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
    """检查核心包和所选 rollout 后端均可导入。"""
    ok = True
    modules = ["verl", "torch", "torch_npu"]
    if backend == "vllm":
        modules.extend(("vllm", "vllm_ascend"))
    elif backend == "sglang":
        modules.append("sglang")
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as error:
            fail(f"cannot import {module}: {error}")
            ok = False
    return ok


def parse_args() -> argparse.Namespace:
    """解析自检参数；未提供的检查项自动跳过。"""
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
    """执行全部自检并返回退出码：0 通过，1 存在失败项。"""
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
