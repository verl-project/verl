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
- quantize.py: Quantization config builder, apply_qat, QuantizationMetadata
- weight_processor.py: QATWeightPostProcessor for converting QAT weights to quantized format
- vllm_modelopt_patch.py: vLLM monkey patches for ModelOpt NVFP4 inference (Linear, MoE, KV Cache)

Usage:
    # Training side
    from verl.utils.modelopt import apply_qat, QATWeightPostProcessor

    # Inference side (dynamic weight reload lifecycle)
    from verl.utils.modelopt import apply_modelopt_nvfp4_patches, prepare_modelopt_for_weight_reload, modelopt_process_weights_after_loading
"""

from verl.utils.modelopt.quantize import (
    # DEFAULT_IGNORE_PATTERNS,
    QuantizationMetadata,
    apply_qat,
    build_quantize_config,
)
from verl.utils.modelopt.vllm_modelopt_patch import (
    apply_modelopt_nvfp4_patches,
    modelopt_process_weights_after_loading,
    prepare_modelopt_for_weight_reload,
)
from verl.utils.modelopt.weight_processor import QATWeightPostProcessor


__all__ = [
    "build_quantize_config",
    "apply_qat",
    "QuantizationMetadata",
    "QATWeightPostProcessor",
    "apply_modelopt_nvfp4_patches",
    "prepare_modelopt_for_weight_reload",
    "modelopt_process_weights_after_loading",
]
