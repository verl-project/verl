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

import copy

import modelopt.torch.quantization as mtq
import torch.nn as nn
from modelopt.torch.quantization.config import _default_disabled_quantizer_cfg

# MXFP4 = E2M1 weights with one E8M0 (power-of-two) scale per 32 contiguous K elements: the on-disk
# format of DeepSeek-V4-Flash routed experts and the format Megatron-Bridge re-quantizes them to on
# every vLLM weight sync, so training fake-quant and rollout weights share one grid.
_MXFP4_WEIGHT_CFG = {
    "num_bits": (2, 1),
    "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
}

# DeepSeek-V4 report: "FP4 quantization-aware training for MoE expert weights" -- routed experts,
# weights only. Megatron paths are decoder.layers.N.mlp.experts.linear_fc{1,2}; the glob does not
# match mlp.shared_experts, the router, attention, or any input/output quantizer. The leading "*"
# disable is load-bearing: an unmatched TensorQuantizer defaults to ENABLED per-tensor INT8.
_MXFP4_EXPERTS_QUANT_CFG = [
    {"quantizer_name": "*", "enable": False},
    {"quantizer_name": "*mlp.experts*weight_quantizer", "cfg": _MXFP4_WEIGHT_CFG, "enable": True},
]

_NVFP4_W4A16_QUANTIZER_CFG = {
    "*weight_quantizer": {
        "num_bits": (2, 1),
        "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        "axis": None,
        "enable": True,
    },
    "*input_quantizer": {"enable": False},
}


def _ignore_patterns_to_quant_cfg(ignore_patterns: list[str]) -> list[dict]:
    cfg = []
    mapping = {
        "lm_head": "*output_layer*",
        "*mlp.gate": "*router*",
        "*self_attn*": "*self_attention*",
    }
    for pattern in ignore_patterns:
        key = pattern
        if key in mapping:
            key = mapping[key]
        cfg.append({"quantizer_name": key, "enable": False})
    return cfg


def build_quantize_config(
    qat_mode: str,
    ignore_patterns: list[str] | None = None,
) -> dict:
    """Build a complete ModelOpt quantization config for ``mtq.quantize``."""
    if qat_mode == "w4a16":
        quant_cfg = mtq.normalize_quant_cfg_list(_NVFP4_W4A16_QUANTIZER_CFG)
        algorithm = "max"
    elif qat_mode == "mxfp4_experts":
        quant_cfg = copy.deepcopy(_MXFP4_EXPERTS_QUANT_CFG)
        # Dynamic block quantization derives its scale per call; modelopt asserts it is never
        # calibrated, so no algorithm / forward_loop.
        algorithm = None
    else:
        raise ValueError(f"Only 'w4a16' and 'mxfp4_experts' are supported, got: {qat_mode}")

    if ignore_patterns is None:
        ignore_patterns = []

    ignore_cfg = _ignore_patterns_to_quant_cfg(ignore_patterns)

    disabled_cfg = copy.deepcopy(_default_disabled_quantizer_cfg)
    if isinstance(disabled_cfg, dict):
        disabled_cfg = mtq.normalize_quant_cfg_list(disabled_cfg)
    quant_cfg.extend(disabled_cfg)
    quant_cfg.extend(ignore_cfg)
    return {"quant_cfg": quant_cfg, "algorithm": algorithm}


def apply_qat(
    model: nn.Module,
    qat_mode: str,
    ignore_patterns: list[str] | None = None,
) -> nn.Module:
    """Apply Quantization-Aware Training to a Megatron model."""
    config = build_quantize_config(qat_mode, ignore_patterns)
    if qat_mode == "mxfp4_experts":
        from verl.utils.modelopt.checkpoint import preserve_mxfp4_checkpoint_methods

        with preserve_mxfp4_checkpoint_methods(model):
            mtq.quantize(model, config)
    else:
        mtq.quantize(model, config)
    return model
