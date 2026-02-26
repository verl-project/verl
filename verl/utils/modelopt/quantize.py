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

"""ModelOpt NVFP4 quantization config and application for Megatron QAT."""

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import _default_disabled_quantizer_cfg


logger = logging.getLogger(__name__)


_NVFP4_W4A16_QUANTIZER_CFG = {
    "*weight_quantizer": {
        "num_bits": (2, 1),
        "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        "axis": None,
        "enable": True,
    },
    "*input_quantizer": {"enable": False},
}


def _ignore_patterns_to_quant_cfg(ignore_patterns: list[str]) -> dict:
    """Convert user-provided ignore patterns to ModelOpt ``quant_cfg`` entries.

    Each pattern is wrapped with ``*`` on both ends (if not already present)
    so that it performs glob-style substring matching against module names.
    For example, ``"lm_head"`` becomes ``"*lm_head*"`` and ``"mlp.gate."``
    becomes ``"*mlp.gate.*"`` (the trailing dot prevents matching
    ``mlp.gate_proj``).
    """
    cfg = {}
    for pattern in ignore_patterns:
        key = pattern
        if not key.startswith("*"):
            key = f"*{key}"
        if not key.endswith("*"):
            key = f"{key}*"
        cfg[key] = {"enable": False}
    return cfg


def build_quantize_config(
    qat_mode: str,
    ignore_patterns: list[str] | None = None,
) -> dict:
    """Build a complete ModelOpt quantization config for ``mtq.quantize``.

    Args:
        qat_mode: Quantization mode. Currently only ``"w4a16"`` is supported.
        ignore_patterns: Layer name patterns to skip quantization for.
            Uses glob-style matching (e.g. ``"lm_head"`` matches ``*lm_head*``).
            If *None*, uses :data:`DEFAULT_IGNORE_PATTERNS`.

    Returns:
        A config dict suitable for ``mtq.quantize()``.
    """
    if qat_mode != "w4a16":
        raise ValueError(f"Only 'w4a16' is supported, got: {qat_mode}")

    if ignore_patterns is None:
        ignore_patterns = []

    ignore_cfg = _ignore_patterns_to_quant_cfg(ignore_patterns)

    quant_cfg = {
        **_NVFP4_W4A16_QUANTIZER_CFG,
        **_default_disabled_quantizer_cfg,
        **ignore_cfg,
    }
    logger.info("Built NVFP4 %s quantize config, ignore_patterns=%s", qat_mode, ignore_patterns)

    return {"quant_cfg": quant_cfg, "algorithm": "max"}


def apply_qat(
    model: nn.Module,
    qat_mode: str,
    ignore_patterns: list[str] | None = None,
) -> nn.Module:
    """Apply Quantization-Aware Training to a Megatron model.

    Args:
        model: The Megatron model to quantize.
        qat_mode: Quantization mode. Currently only ``"w4a16"`` is supported.
        ignore_patterns: Layer name patterns to skip quantization for.
            If *None*, uses :data:`DEFAULT_IGNORE_PATTERNS`.

    Returns:
        The quantized model (modified in-place).
    """
    config = build_quantize_config(qat_mode, ignore_patterns)
    mtq.quantize(model, config)
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
    weight_amax: Optional[torch.Tensor] = None
    input_amax: Optional[torch.Tensor] = None
    is_local: bool = True
    global_expert_idx: Optional[int] = None
    local_expert_idx: Optional[int] = None
