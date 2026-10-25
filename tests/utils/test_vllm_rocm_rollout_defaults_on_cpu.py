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
"""vLLM rollout defaults that differ on ROCm."""

import pytest
import torch

from verl.plugin.platform.platform_rocm import PlatformROCm
from verl.utils.vllm.vllm_quant_utils import set_default_vllm_quantization_backends

FP8_BLOCK = {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [128, 128]}


@pytest.mark.parametrize("hip", [None, "7.2"], ids=["cuda", "rocm"])
def test_deep_gemm_is_only_the_default_backend_off_rocm(monkeypatch, hip):
    monkeypatch.setattr(torch.version, "hip", hip, raising=False)
    monkeypatch.setattr("verl.utils.vllm.vllm_quant_utils._is_hopper", lambda: False)
    engine_kwargs = {}

    set_default_vllm_quantization_backends(FP8_BLOCK, engine_kwargs)

    if hip is None:
        assert engine_kwargs == {"linear_backend": "deep_gemm", "moe_backend": "deep_gemm"}
    else:
        # DeepGEMM has no ROCm kernels; vLLM's automatic selection picks AITER / Triton.
        assert engine_kwargs == {}


def test_rocm_rollout_keeps_deepseek_v4_weights_reloadable(monkeypatch):
    monkeypatch.delenv("VLLM_ROCM_DSV4_BPRESHUFFLE", raising=False)
    assert PlatformROCm().rollout_env_vars()["VLLM_ROCM_DSV4_BPRESHUFFLE"] == "0"

    monkeypatch.setenv("VLLM_ROCM_DSV4_BPRESHUFFLE", "1")
    assert PlatformROCm().rollout_env_vars()["VLLM_ROCM_DSV4_BPRESHUFFLE"] == "1"
