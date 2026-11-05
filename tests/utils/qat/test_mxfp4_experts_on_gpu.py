# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Experts-only MXFP4 QAT (``qat.mode=mxfp4_experts``) for DeepSeek-V4-style checkpoints.

The mode fake-quantizes only the routed-expert weights (E2M1 values, E8M0 scale per 32 elements, the grid
Megatron-Bridge re-quantizes those tensors to on every rollout sync) and leaves every other linear alone, so it
can be stacked on Transformer Engine's real FP8 GEMMs.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from verl.utils.qat import compressed_tensors as ct
from verl.utils.qat.modelopt import _check_mxfp4_experts, apply_qat_to_modules, build_quantize_config
from verl.workers.config import QATEngineConfig

_DSV4_NATIVE_FP8 = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "scale_fmt": "ue8m0",
    "weight_block_size": [128, 128],
}


def _dispatch(model_type: str, qat_mode: str | None, native=_DSV4_NATIVE_FP8):
    hf_config = SimpleNamespace(model_type=model_type, num_hidden_layers=4, quantization_config=native)
    qat_config = None if qat_mode is None else SimpleNamespace(enable=True, mode=qat_mode)
    return ct.dispatch_rollout_weight_quantizer(
        model_config=SimpleNamespace(hf_config=hf_config, local_path=None),
        rollout_config=SimpleNamespace(name="vllm", quantization=None, checkpoint_engine=None),
        qat_config=qat_config,
    )


def test_quantize_config_enables_only_routed_expert_weights():
    cfg = build_quantize_config("mxfp4_experts", ["lm_head"])
    assert cfg["algorithm"] is None
    quant_cfg = cfg["quant_cfg"]
    assert quant_cfg[0] == {"quantizer_name": "*", "enable": False}
    expert = quant_cfg[1]
    assert expert["quantizer_name"] == "*mlp.experts*weight_quantizer" and expert["enable"] is True
    assert expert["cfg"]["num_bits"] == (2, 1)
    assert expert["cfg"]["block_sizes"] == {-1: 32, "type": "dynamic", "scale_bits": (8, 0)}
    # Nothing re-enables the activations or any other linear afterwards.
    assert all(not entry.get("enable", False) for entry in quant_cfg[2:])


def test_dispatch_pairs_mxfp4_experts_with_deepseek_v4_native_fp8():
    quantizer = _dispatch("deepseek_v4", "mxfp4_experts")
    assert isinstance(quantizer, ct.Fp8ConfigWeightQuantizer)
    # Plain FP8 QAT on the same checkpoint keeps working.
    assert isinstance(_dispatch("deepseek_v4", "fp8"), ct.Fp8ConfigWeightQuantizer)
    assert isinstance(_dispatch("deepseek_v4", None), ct.Fp8ConfigWeightQuantizer)


def test_dispatch_rejects_mxfp4_experts_on_other_native_fp8_models():
    with pytest.raises(ValueError, match="does not match rollout quantization"):
        _dispatch("qwen3_moe", "mxfp4_experts")


class _MoELayer(nn.Module):
    def __init__(self, hidden: int = 256, inter: int = 128, n_experts: int = 2):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate = nn.Linear(hidden, n_experts, bias=False)
        self.mlp.experts = nn.ModuleList([nn.Linear(hidden, inter, bias=False) for _ in range(n_experts)])
        self.mlp.shared_experts = nn.Linear(hidden, inter, bias=False)
        self.lm_head = nn.Linear(hidden, 32, bias=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="modelopt fake-quant kernels need a GPU")
def test_apply_qat_fake_quantizes_routed_experts_only():
    mtq = pytest.importorskip("modelopt.torch.quantization")
    torch.manual_seed(0)
    model = _MoELayer().cuda().bfloat16()
    apply_qat_to_modules([model], QATEngineConfig(enable=True, mode="mxfp4_experts"))

    enabled = [n for n, q in model.named_modules() if isinstance(q, mtq.nn.TensorQuantizer) and q.is_enabled]
    assert enabled == ["mlp.experts.0.weight_quantizer", "mlp.experts.1.weight_quantizer"]

    expert = model.mlp.experts[0]
    weight = expert.weight.detach()
    fq = expert.weight_quantizer(weight)
    assert not torch.equal(fq, weight), "random bf16 weights are not on the MXFP4 grid"
    assert torch.equal(expert.weight_quantizer(fq), fq), "fake-quant must be idempotent on-grid"
    # E2M1 keeps at most 1 mantissa bit: the relative step inside a 32-block is coarse but bounded.
    assert (fq.float() - weight.float()).abs().max() < 0.1

    # The untouched linears still run the plain bf16 matmul.
    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(x, model.mlp.shared_experts.weight)
    assert torch.equal(model.mlp.shared_experts(x), ref)
    assert expert(x).shape == (4, 128)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="modelopt fake-quant kernels need a GPU")
def test_check_mxfp4_experts_fails_fast_on_stray_quantizers():
    pytest.importorskip("modelopt.torch.quantization")
    model = _MoELayer().cuda().bfloat16()
    apply_qat_to_modules([model], QATEngineConfig(enable=True, mode="mxfp4_experts"))
    _check_mxfp4_experts([model])  # the sanctioned layout passes

    model.mlp.shared_experts.weight_quantizer.enable()
    with pytest.raises(RuntimeError, match="unexpected quantizers enabled"):
        _check_mxfp4_experts([model])

    for expert in model.mlp.experts:
        expert.weight_quantizer.disable()
    model.mlp.shared_experts.weight_quantizer.disable()
    with pytest.raises(RuntimeError, match="no routed-expert weight_quantizer enabled"):
        _check_mxfp4_experts([model])
