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

from verl.models.mcore.patch import apply_patch_megatron_v012_with_torch_v28_v29, neutralize_broken_flash_attn_cute

# Must precede the `.registry` import below: it reaches megatron's GPT layer specs,
# whose optional FA4 probe crashes on a broken `flash_attn.cute`.
neutralize_broken_flash_attn_cute()

from .registry import (  # noqa: E402
    get_mcore_engine_forward_fn,
    get_mcore_forward_fn,
    get_mcore_forward_fused_fn,
    get_mcore_forward_fused_model_engine_fn,
    get_mcore_weight_converter,
    hf_to_mcore_config,
    init_mcore_model,
)

__all__ = [
    "hf_to_mcore_config",
    "init_mcore_model",
    "get_mcore_forward_fn",
    "get_mcore_weight_converter",
    "get_mcore_forward_fused_fn",
    "get_mcore_engine_forward_fn",
    "get_mcore_forward_fused_model_engine_fn",
]

apply_patch_megatron_v012_with_torch_v28_v29()
