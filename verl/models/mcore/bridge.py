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
except ImportError:
    # `pip install verl[mcore]` or
    print("Megatron-Bridge package not found. Please install Megatron-Bridge with `pip install megatron-bridge`")
    raise

__all__ = [
    "AutoBridge",
    "LinearForLastLayer",
    "freeze_moe_router",
    "make_value_model",
]
