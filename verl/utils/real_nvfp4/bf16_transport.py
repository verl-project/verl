# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Fail-closed attestation for the BF16 actor-to-rollout refit stream."""

import logging
import re
from collections.abc import Iterator
from typing import Any

import torch

logger = logging.getLogger(__name__)

_EXPERT_WEIGHT_RE = re.compile(r"^.*\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)\.weight$")
_EXPERT_KEY_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
_PACKED_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")


def attest_real_nvfp4_bf16_transport(
    weights: Iterator[tuple[str, torch.Tensor]],
    *,
    expected_expert_weights: int,
    location: str = "actor_export",
    hf_config: Any = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Pass plain actor weights through while proving refit is not pre-packed.

    Routed-expert matrices must remain floating point. Quantized weight and
    scale tensors are produced only inside the vLLM worker.
    """

    expected_names = None
    if hf_config is not None:
        from .config import real_nvfp4_moe_layer_indices, validate_real_nvfp4_model_contract

        validate_real_nvfp4_model_contract(hf_config)
        get = hf_config.get if isinstance(hf_config, dict) else lambda key: getattr(hf_config, key, None)
        num_experts = int(get("num_experts") or get("n_routed_experts") or 0)
        expected_names = {
            f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
            for layer in real_nvfp4_moe_layer_indices(hf_config)
            for expert in range(num_experts)
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        if len(expected_names) != expected_expert_weights:
            raise RuntimeError("real NVFP4 configured expert-weight count disagrees with the HF model layout")

    expert_weights = 0
    seen_expert_names = set()
    for name, tensor in weights:
        if name.endswith(_PACKED_SUFFIXES):
            raise RuntimeError(f"real NVFP4 BF16 refit unexpectedly contained packed tensor {name}")
        if _EXPERT_WEIGHT_RE.match(name):
            if name in seen_expert_names:
                raise RuntimeError(f"real NVFP4 refit contains duplicate expert weight {name}")
            if expected_names is not None and (_EXPERT_KEY_RE.match(name) is None or name not in expected_names):
                raise RuntimeError(f"real NVFP4 refit contains unexpected expert weight {name}")
            seen_expert_names.add(name)
            expert_weights += 1
            if tensor.dtype not in {torch.bfloat16, torch.float16, torch.float32}:
                raise RuntimeError(f"real NVFP4 expert refit tensor must be floating point, got {name}: {tensor.dtype}")
        yield name, tensor

    if expected_names is not None and seen_expert_names != expected_names:
        missing = sorted(expected_names - seen_expert_names)
        raise RuntimeError(f"real NVFP4 refit is missing {len(missing)} expert weights: {missing[:8]}")
    if expert_weights != expected_expert_weights:
        raise RuntimeError(
            "real NVFP4 BF16 refit expert-weight count mismatch: "
            f"expected {expected_expert_weights}, got {expert_weights}"
        )
    logger.warning(
        "VERL_REAL_NVFP4_EXPORT PASS location=%s transport=bf16 quantization=vllm_worker expert_weights=%d expected=%d",
        location,
        expert_weights,
        expected_expert_weights,
    )
