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
"""Check real Transformers checkpoint schemas without requiring NPU operators."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe as hf


@pytest.fixture
def npu_patch(monkeypatch):
    # Load separately so a CPU import stub cannot leak into the production module.
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    path = Path(__file__).resolve().parents[2] / "verl/models/transformers/npu_patch.py"
    spec = importlib.util.spec_from_file_location("qwen3_vl_moe_npu_schema_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(hf, "Qwen3VLMoeTextSparseMoeBlock", hf.Qwen3VLMoeTextSparseMoeBlock)
    monkeypatch.setattr(hf.Qwen3VLMoeTextExperts, "forward", hf.Qwen3VLMoeTextExperts.forward)
    monkeypatch.setattr(hf.Qwen3VLMoeTextRMSNorm, "forward", hf.Qwen3VLMoeTextRMSNorm.forward)
    monkeypatch.setattr(hf, "apply_rotary_pos_emb", hf.apply_rotary_pos_emb)
    return module


def _config():
    return hf.Qwen3VLMoeTextConfig(
        hidden_size=64,
        moe_intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
    )


def test_patch_preserves_native_checkpoint_schema(npu_patch):
    """Native HF checkpoints must load strictly before and after the NPU patch."""
    original = hf.Qwen3VLMoeTextSparseMoeBlock(_config())
    for parameter in original.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    state_dict = original.state_dict()

    npu_patch._patch_qwen3_vl_moe()
    patched = hf.Qwen3VLMoeTextSparseMoeBlock(_config())
    patched.load_state_dict(state_dict, strict=True)
    for name, parameter in patched.state_dict().items():
        torch.testing.assert_close(parameter, state_dict[name], rtol=0, atol=0)


def test_patch_preserves_modern_router(npu_patch):
    """The native router and its output-capture hooks remain usable on NPU."""
    if not hasattr(hf, "Qwen3VLMoeTextTopKRouter"):
        pytest.skip("Requires the Transformers 5 Qwen3-VL-MoE API.")
    original = hf.Qwen3VLMoeTextSparseMoeBlock(_config())
    original_forward = type(original).forward
    npu_patch._patch_qwen3_vl_moe()
    patched = hf.Qwen3VLMoeTextSparseMoeBlock(_config())
    assert type(patched) is type(original)
    assert type(patched).forward is original_forward
    assert type(patched.gate) is type(original.gate)
    for parameter in patched.gate.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    inputs = torch.randn(16, 64)
    logits, weights, indices = patched.gate(inputs)
    assert logits.shape == (16, 4)
    assert weights.shape == indices.shape == (16, 2)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(16))
