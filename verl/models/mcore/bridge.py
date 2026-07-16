# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

try:
    from megatron.bridge import AutoBridge
    from megatron.bridge.training.utils.train_utils import LinearForLastLayer, freeze_moe_router, make_value_model
except ImportError as e:
    from importlib.metadata import PackageNotFoundError, version

    _PIN = "0.5.0"
    _CMD = f"`pip install --no-deps megatron-bridge=={_PIN}`"
    try:
        _installed = version("megatron-bridge")
    except PackageNotFoundError:
        _installed = None
    if _installed is None:
        msg = (
            f"Megatron-Bridge is not installed. For stacks matching verl's stable Dockerfiles, "
            f"install with {_CMD} (--no-deps is required so pip does not reinstall "
            "megatron-core/transformer-engine/torch over your container build). "
            "If you are on a published tag predating that pin "
            "(e.g. verlai/verl:sgl0512.dev2 or vllm023.dev1), see docker/README.md for the "
            "CI-proven install command instead of assuming this pin."
        )
    elif _installed == _PIN:
        msg = (
            f"Megatron-Bridge {_installed} (verl's Dockerfile pin) is installed, but the import "
            f"failed: {e}. This usually means megatron-core or transformer-engine is missing or "
            "mismatched - check your Megatron-LM install matches the image/Dockerfile stack."
        )
    else:
        msg = (
            f"Megatron-Bridge {_installed} is installed but the import failed: {e} - {_installed} "
            f"is likely too old or incomplete for verl (e.g. 0.3.1 lacks LinearForLastLayer). "
            f"On a current stable-Dockerfile stack reinstall with {_CMD}; on stale published "
            "tags see docker/README.md."
        )
    raise ImportError(msg) from e

__all__ = [
    "AutoBridge",
    "LinearForLastLayer",
    "freeze_moe_router",
    "make_value_model",
]
