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


from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import _default_disabled_quantizer_cfg

# ---------------------------------------------------------------------------
# NVFP4 quantization config
# ---------------------------------------------------------------------------

NVFP4_WEIGHT_ONLY_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
        "*input_quantizer": {"enable": False},
        **_default_disabled_quantizer_cfg,
    },
    "algorithm": "max",
}

# ---------------------------------------------------------------------------
# QAT application
# ---------------------------------------------------------------------------


def apply_qat(model: nn.Module, qat_mode: str):
    """Apply Quantization-Aware Training to the model.

    Args:
        model: The Megatron model to apply QAT to.
        qat_mode: QAT mode, now only support "w4a16" for weight-only quantization.

    Returns:
        The quantized model.
    """
    if qat_mode != "w4a16":
        raise ValueError(f"Only 'w4a16' is supported, got: {qat_mode}")

    mtq.quantize(model, NVFP4_WEIGHT_ONLY_CFG)
    return model


@dataclass
class QuantizationMetadata:
    """Metadata for a quantized module."""

    qformat: str
    weight_quantizer: Any
    input_quantizer: Any
    module: torch.nn.Module
    vpp_idx: int
    block_size: int = 16  # Default NVFP4 block size
    # Fields for EP synchronization - store amax values for non-local experts
    weight_amax: Optional[torch.Tensor] = None
    input_amax: Optional[torch.Tensor] = None
    is_local: bool = True  # Whether this expert is local to current EP rank
    global_expert_idx: Optional[int] = None  # Global expert index for MoE experts
    local_expert_idx: Optional[int] = None  # Local expert index on this EP rank
