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

import re
from collections.abc import Iterator
from typing import Any

import torch

from .config import _hf_get, real_nvfp4_moe_layer_indices, validate_real_nvfp4_model_contract

_EXPERT_WEIGHT_RE = re.compile(r"^.*\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)\.weight$")
_PACKED_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")


def attest_real_nvfp4_bf16_transport(
    weights: Iterator[tuple[str, torch.Tensor]],
    hf_config: Any,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Pass plain actor weights through while proving refit is not pre-packed.

    Every routed-expert projection must arrive exactly once and in floating
    point. Quantized weight and scale tensors are produced only inside the
    vLLM worker.
    """

    validate_real_nvfp4_model_contract(hf_config)
    num_experts = int(_hf_get(hf_config, "num_experts") or _hf_get(hf_config, "n_routed_experts"))
    expected_names = {
        f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
        for layer in real_nvfp4_moe_layer_indices(hf_config)
        for expert in range(num_experts)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }

    seen_expert_names = set()
    for name, tensor in weights:
        if name.endswith(_PACKED_SUFFIXES):
            raise RuntimeError(f"real NVFP4 BF16 refit unexpectedly contained packed tensor {name}")
        if _EXPERT_WEIGHT_RE.match(name):
            if name not in expected_names:
                raise RuntimeError(f"real NVFP4 refit contains unexpected expert weight {name}")
            if name in seen_expert_names:
                raise RuntimeError(f"real NVFP4 refit contains duplicate expert weight {name}")
            if tensor.dtype not in {torch.bfloat16, torch.float16, torch.float32}:
                raise RuntimeError(f"real NVFP4 expert refit tensor must be floating point, got {name}: {tensor.dtype}")
            seen_expert_names.add(name)
        yield name, tensor

    if seen_expert_names != expected_names:
        missing = sorted(expected_names - seen_expert_names)
        raise RuntimeError(f"real NVFP4 refit is missing {len(missing)} expert weights: {missing[:8]}")
