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
QAT (Quantization-Aware Training) module for verl.

Supports NVFP4 (W4A4 and W4A16) quantization modes for FSDP training.

Module Structure:
- core.py: QATConfig, apply_qat, enable_qat_fuse (training setup)
- linear.py: QATLinear layer with Triton kernels for fake quantization
- quantizer.py: QATQuantizer for true quantization + scale computation utilities
- vllm_patch.py: Patches for vLLM dynamic weight loading

Usage:
    from verl.utils.qat import apply_qat, QATConfig

    config = QATConfig(enable=True, mode="w4a16")
    model = apply_qat(model, config)  # Before FSDP wrapping
"""

from verl.utils.qat.core import (
    QATConfig,
    apply_qat,
    enable_qat_fuse,
    invalidate_all_scales,
    load_quantization_config,
)
from verl.utils.qat.vllm_patch import (
    apply_qat_patches,
    manual_process_weights_after_loading,
    prepare_qat_for_load_weights,
)

__all__ = [
    # Core
    "QATConfig",
    "apply_qat",
    "load_quantization_config",
    "enable_qat_fuse",
    "invalidate_all_scales",
    # vLLM Patch
    "apply_qat_patches",
    "manual_process_weights_after_loading",
    "prepare_qat_for_load_weights",
]
