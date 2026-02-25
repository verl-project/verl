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

"""
ModelOpt integration for verl.

Supports NVFP4 quantization with Megatron QAT training + vLLM low-precision inference.

Module Structure:
- qat.py: QAT quantization config, apply_qat, QuantizationMetadata
- weight_processor.py: QATWeightPostProcessor for converting QAT weights to quantized format
- vllm_patch.py: vLLM monkey patches for NVFP4 inference (Linear, MoE, KV Cache)

Usage:
    # Training side
    from verl.utils.modelopt import apply_qat, QATWeightPostProcessor

    # Inference side
    from verl.utils.modelopt import apply_vllm_modelopt_patches
"""

from verl.utils.modelopt.qat import NVFP4_WEIGHT_ONLY_CFG, QuantizationMetadata, apply_qat
from verl.utils.modelopt.vllm_patch import apply_vllm_modelopt_patches
from verl.utils.modelopt.weight_processor import QATWeightPostProcessor

__all__ = [
    "NVFP4_WEIGHT_ONLY_CFG",
    "apply_qat",
    "QuantizationMetadata",
    "QATWeightPostProcessor",
    "apply_vllm_modelopt_patches",
]
