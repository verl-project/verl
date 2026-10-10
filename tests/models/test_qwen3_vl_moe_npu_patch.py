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
"""Numerical and checkpoint regressions using real Ascend MoE operators."""

import pytest
import torch
from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe as hf

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Requires an Ascend NPU.")


@pytest.fixture
def npu_patch(monkeypatch):
    from verl.models.transformers import npu_patch

    if not hasattr(hf, "Qwen3VLMoeTextTopKRouter"):
        pytest.skip("Requires the Transformers 5 Qwen3-VL-MoE API.")
    monkeypatch.setattr(hf, "Qwen3VLMoeTextSparseMoeBlock", hf.Qwen3VLMoeTextSparseMoeBlock)
    monkeypatch.setattr(hf.Qwen3VLMoeTextExperts, "forward", hf.Qwen3VLMoeTextExperts.forward)
    monkeypatch.setattr(hf.Qwen3VLMoeTextRMSNorm, "forward", hf.Qwen3VLMoeTextRMSNorm.forward)
    monkeypatch.setattr(hf, "apply_rotary_pos_emb", hf.apply_rotary_pos_emb)
    torch.npu.set_device(0)
    return npu_patch


def _config(top_k=2):
    config = hf.Qwen3VLMoeTextConfig(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=top_k,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [2, 3, 3]},
    )
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    return config


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape,top_k", [((1, 1, 64), 1), ((2, 7, 64), 2)])
def test_moe_forward_backward(npu_patch, dtype, shape, top_k):
    """Compare outputs, input gradients and every parameter gradient to native HF."""
    torch.manual_seed(17)
    config = _config(top_k)
    reference = hf.Qwen3VLMoeTextSparseMoeBlock(config).to(dtype)
    for parameter in reference.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    state_dict = {name: value.clone() for name, value in reference.state_dict().items()}
    inputs = torch.randn(shape, dtype=dtype)
    upstream = torch.randn(shape)
    # Compare on the same device: BF16 CPU/NPU matmul rounding can change
    # nearly tied top-k selections, independently of the expert implementation.
    reference = reference.to("npu:0")
    reference_inputs = inputs.to("npu:0").requires_grad_(True)
    expected = reference(reference_inputs)
    (expected.float() * upstream.to("npu:0")).sum().backward()
    expected_grads = {name: parameter.grad.cpu().clone() for name, parameter in reference.named_parameters()}
    expected_input_grad = reference_inputs.grad.cpu().clone()
    expected = expected.detach().cpu()

    npu_patch._patch_qwen3_vl_moe()
    actual_model = hf.Qwen3VLMoeTextSparseMoeBlock(config).to(device="npu:0", dtype=dtype)
    actual_model.load_state_dict(state_dict, strict=True)
    npu_inputs = inputs.to("npu:0").requires_grad_(True)
    actual = actual_model(npu_inputs)
    (actual.float() * upstream.to("npu:0")).sum().backward()
    torch.npu.synchronize()

    # GMM and per-expert linear reductions round differently in low precision.
    # Also bound relative L2 error so the absolute tolerance at near-zero
    # elements cannot hide a missing gradient or a wrong expert layout.
    rtol, l2_tol = (0.03, 0.02) if dtype == torch.bfloat16 else (0.005, 0.003)
    actual_values = {"output": actual.detach().cpu(), "input_grad": npu_inputs.grad.cpu()}
    expected_values = {"output": expected, "input_grad": expected_input_grad}
    for name, parameter in actual_model.named_parameters():
        actual_values[name] = parameter.grad.cpu()
        expected_values[name] = expected_grads[name]
    errors = {}
    for name, value in actual_values.items():
        expected_value = expected_values[name]
        difference = value.float() - expected_value.float()
        relative_l2 = (difference.norm() / expected_value.float().norm().clamp_min(1e-8)).item()
        if name == "gate.weight" and top_k == 1:
            # A normalized top-1 weight is constant. Transformers versions
            # retaining FP32 routing weights can leave a tiny cancellation
            # residual instead of the mathematically zero router gradient.
            relative_l2 = None
        errors[name] = {"max_abs": difference.abs().max().item(), "relative_l2": relative_l2}
    print(f"{dtype=}, {shape=}, {top_k=}, errors={errors}")
    for name, value in actual_values.items():
        expected_value = expected_values[name]
        if name == "gate.weight" and top_k == 1:
            torch.testing.assert_close(value, torch.zeros_like(value), rtol=0, atol=1e-6)
            torch.testing.assert_close(expected_value, torch.zeros_like(expected_value), rtol=0, atol=1e-6)
            continue
        # Cancellation in router gradients makes local relative error unstable.
        # Allow one dtype epsilon at the tensor's largest reference magnitude,
        # while independently bounding the error across the whole tensor.
        atol = torch.finfo(dtype).eps * expected_value.float().abs().max().item()
        torch.testing.assert_close(value, expected_value, rtol=rtol, atol=atol)
        assert errors[name]["relative_l2"] <= l2_tol, (name, errors[name])


def test_pretrained_text_model_round_trip(npu_patch, tmp_path):
    """Load a native checkpoint, run on NPU, then save and strictly reload it."""
    torch.manual_seed(23)
    config = _config()
    reference = hf.Qwen3VLMoeTextModel(config).bfloat16().eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    with torch.no_grad():
        reference_output = reference(input_ids, use_cache=False, output_router_logits=True)
        expected = reference_output.last_hidden_state
    checkpoint = tmp_path / "native"
    reference.save_pretrained(checkpoint)
    state_dict = {name: value.clone() for name, value in reference.state_dict().items()}

    npu_patch._patch_qwen3_vl_moe()
    patched, loading_info = hf.Qwen3VLMoeTextModel.from_pretrained(
        checkpoint, dtype=torch.bfloat16, output_loading_info=True
    )
    assert not loading_info["missing_keys"]
    assert not loading_info["unexpected_keys"]
    assert not loading_info["mismatched_keys"]
    for name, value in patched.state_dict().items():
        torch.testing.assert_close(value, state_dict[name], rtol=0, atol=0)
    patched = patched.to("npu:0").eval()
    with torch.no_grad():
        patched_output = patched(input_ids.to("npu:0"), use_cache=False, output_router_logits=True)
        actual = patched_output.last_hidden_state.cpu()
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)
    assert len(patched_output.router_logits) == config.num_hidden_layers
    for actual_router, expected_router in zip(
        patched_output.router_logits, reference_output.router_logits, strict=True
    ):
        assert actual_router.shape == (input_ids.numel(), config.num_experts)
        torch.testing.assert_close(actual_router.cpu(), expected_router, rtol=0.03, atol=0.003)

    saved = tmp_path / "patched"
    patched.cpu().save_pretrained(saved)
    reloaded, reloading_info = hf.Qwen3VLMoeTextModel.from_pretrained(
        saved, dtype=torch.bfloat16, output_loading_info=True
    )
    assert not reloading_info["missing_keys"]
    assert not reloading_info["unexpected_keys"]
    assert not reloading_info["mismatched_keys"]
    for name, value in reloaded.state_dict().items():
        torch.testing.assert_close(value, state_dict[name], rtol=0, atol=0)
    reloaded = reloaded.to("npu:0").eval()
    with torch.no_grad():
        reloaded_output = reloaded(input_ids.to("npu:0"), use_cache=False).last_hidden_state.cpu()
    torch.testing.assert_close(reloaded_output, actual, rtol=0, atol=0)
    error = (actual.float() - expected.float()).abs().max().item()
    print(f"native_checkpoint_loaded=True, patched_checkpoint_reloaded=True, max_abs_error={error}")
