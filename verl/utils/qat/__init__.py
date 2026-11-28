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

Megatron QAT uses ModelOpt fake quantization; the trainer exports weights in the rollout's
compressed-tensors or FP8 format.

Module Structure:
- compressed_tensors.py: compressed-tensors rollout config and trainer-side weight quantizer
- modelopt.py: ModelOpt fake quantization and Megatron QAT workflow helpers
- nvfp4.py: NVFP4 fake quantization and export kernels
"""

from verl.utils.qat.compressed_tensors import (
    CompressedTensorsWeightQuantizer,
    Fp8ConfigWeightQuantizer,
    build_compressed_tensors_config,
    dispatch_rollout_weight_quantizer,
    is_all_mxfp8_config,
    qat_modes_for_config,
)

__all__ = [
    "CompressedTensorsWeightQuantizer",
    "Fp8ConfigWeightQuantizer",
    "build_compressed_tensors_config",
    "dispatch_rollout_weight_quantizer",
    "is_all_mxfp8_config",
    "qat_modes_for_config",
]
