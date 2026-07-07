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

"""GPU unit tests for the experimental dense and MoE W4A8 QAT simulation."""

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("compressed_tensors")
pytest.importorskip("vllm")

if not torch.cuda.is_available():
    pytest.skip("W4A8 fake quantization requires a CUDA GPU", allow_module_level=True)

from verl.utils.qat import linear as qat_linear  # noqa: E402
from verl.utils.qat import vllm_patch  # noqa: E402
from verl.utils.qat.linear import QATLinear, QATMode, STEFP8Quant, fp8_e4m3_fake_quant  # noqa: E402
from verl.utils.qat.quantizer import QATQuantizer  # noqa: E402


def test_fp8_fake_quant_handles_noncontiguous_3d_input():
    x = torch.randn(2, 130, 3, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    assert not x.is_contiguous()

    result = fp8_e4m3_fake_quant(x)

    assert result.shape == x.shape
    assert result.dtype == x.dtype
    assert torch.isfinite(result).all()
    assert not torch.equal(result, x)


def test_fp8_fake_quant_supports_rectangular_pytorch_fallback(monkeypatch):
    from verl.utils.kernel import fp8_kernel

    monkeypatch.setattr(fp8_kernel, "_TRITON_AVAILABLE", False)
    x = torch.randn(3, 130, device="cuda", dtype=torch.bfloat16)

    result = fp8_e4m3_fake_quant(x)

    assert result.shape == x.shape
    assert result.dtype == x.dtype
    assert torch.isfinite(result).all()


def test_fp8_fake_quant_matches_independent_blockwise_reference():
    x = torch.tensor(
        [[-9.0, -2.5, 0.75, 8.0, -1.0, 0.5, 3.0, 6.5]],
        device="cuda",
        dtype=torch.float32,
    )
    result = fp8_e4m3_fake_quant(x, block_size=(1, 4))

    blocks = x.reshape(1, 2, 4)
    descale = blocks.abs().amax(dim=-1, keepdim=True) / 448.0
    quantized = torch.clamp(blocks / descale, min=-448.0, max=448.0).to(torch.float8_e4m3fn)
    expected = (quantized.float() * descale).reshape_as(x)

    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_fp8_ste_passes_gradient_through_unchanged():
    x = torch.randn(2, 3, 130, device="cuda", dtype=torch.float32, requires_grad=True)
    upstream = torch.randn_like(x)

    result = STEFP8Quant.apply(x, (1, 128))
    result.backward(upstream)

    torch.testing.assert_close(x.grad, upstream)


def test_qat_linear_uses_fp8_activation_only_in_w4a8(monkeypatch):
    layer = QATLinear(16, 8, mode=QATMode.W4A8, device=torch.device("cuda"), dtype=torch.float32)
    x = torch.randn(2, 16, device="cuda")
    activation_calls = []

    monkeypatch.setattr(layer, "_fake_quantize_weight", lambda weight: weight)

    def fake_activation(value):
        activation_calls.append(value)
        return value + 1

    monkeypatch.setattr(layer, "_fake_quantize_activation_fp8", fake_activation)
    result = layer(x)

    assert len(activation_calls) == 1
    assert activation_calls[0] is x
    expected = torch.nn.functional.linear(x + 1, layer.weight, layer.bias)
    torch.testing.assert_close(result, expected)


def test_w4a8_and_w4a16_export_identical_weight_payloads():
    params = {
        "model.layers.0.mlp.up_proj.weight": torch.randn(16, 16, dtype=torch.bfloat16),
    }

    def export(mode):
        quantizer = QATQuantizer(mode=mode, device=torch.device("cuda"), param_dtype=torch.bfloat16)
        return dict(quantizer.quantize_with_fusion(params, target_device=torch.device("cpu")))

    w4a8 = export("w4a8")
    w4a16 = export("w4a16")

    assert w4a8.keys() == w4a16.keys()
    assert not any("input_global_scale" in name for name in w4a8)
    for name in w4a8:
        torch.testing.assert_close(w4a8[name], w4a16[name], rtol=0, atol=0)


def test_qat_quantizer_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported QAT mode"):
        QATQuantizer(mode="w4a7", device=torch.device("cuda"))


def test_rollout_wrapper_follows_configured_simulation_mode(monkeypatch):
    received = []

    def original(_self, _layer, x, bias=None):
        received.append((x, bias))
        return x

    monkeypatch.setattr(vllm_patch, "_original_w4a16_apply_weights", original)
    monkeypatch.setattr(qat_linear, "fp8_e4m3_fake_quant", lambda x: x + 1)

    x = torch.zeros(2, 16, device="cuda")
    monkeypatch.setenv(vllm_patch.W4A8_SIMULATION_ENV, "1")
    simulated = vllm_patch.patched_w4a16_apply_weights_with_w4a8_simulation(None, None, x)
    torch.testing.assert_close(simulated, x + 1)

    monkeypatch.setenv(vllm_patch.W4A8_SIMULATION_ENV, "0")
    passthrough = vllm_patch.patched_w4a16_apply_weights_with_w4a8_simulation(None, None, x)
    torch.testing.assert_close(passthrough, x)
    assert len(received) == 2


def test_w4a8_mode_selects_marlin_for_moe_workers(monkeypatch):
    monkeypatch.delenv(vllm_patch.W4A8_SIMULATION_ENV, raising=False)
    monkeypatch.setenv(vllm_patch._VLLM_USE_FLASHINFER_MOE_ENV, "1")
    monkeypatch.setenv(vllm_patch._VLLM_FORCE_MARLIN_ENV, "0")

    vllm_patch.set_w4a8_simulation(True)

    assert vllm_patch.is_w4a8_simulation_enabled()
    assert os.environ[vllm_patch._VLLM_USE_FLASHINFER_MOE_ENV] == "0"
    assert os.environ[vllm_patch._VLLM_FORCE_MARLIN_ENV] == "1"

    vllm_patch.set_w4a8_simulation(False)

    assert not vllm_patch.is_w4a8_simulation_enabled()
    assert os.environ[vllm_patch._VLLM_USE_FLASHINFER_MOE_ENV] == "1"
    assert os.environ[vllm_patch._VLLM_FORCE_MARLIN_ENV] == "0"


def test_w4a8_moe_wrapper_quantizes_both_expert_gemm_inputs(monkeypatch):
    from vllm.model_executor.layers.fused_moe.fused_marlin_moe import MarlinExperts

    class RecordingMarlinExperts(MarlinExperts):
        def __init__(self):
            self.gate_up_input = None

        def apply(self, output, hidden_states, *args, **kwargs):
            self.gate_up_input = hidden_states
            return hidden_states

        def activation(self, activation, output, input):
            output.copy_(input * 2)

    quantized_inputs = []

    def fake_quant(value):
        quantized_inputs.append(value.clone())
        return value + 1

    monkeypatch.setattr(qat_linear, "fp8_e4m3_fake_quant", fake_quant)
    wrapped_cls = vllm_patch._make_w4a8_moe_experts_cls(RecordingMarlinExperts)
    experts = wrapped_cls()

    hidden_states = torch.zeros(2, 16, device="cuda")
    gate_up_result = experts.apply(torch.empty_like(hidden_states), hidden_states)
    torch.testing.assert_close(gate_up_result, hidden_states + 1)
    torch.testing.assert_close(experts.gate_up_input, hidden_states + 1)

    activation_input = torch.full((2, 16), 2.0, device="cuda")
    down_input = torch.empty_like(activation_input)
    experts.activation("silu", down_input, activation_input)
    torch.testing.assert_close(down_input, activation_input * 2 + 1)
    assert len(quantized_inputs) == 2


def test_w4a8_moe_wrapper_factory_is_idempotent():
    from vllm.model_executor.layers.fused_moe.fused_marlin_moe import MarlinExperts

    wrapped_cls = vllm_patch._make_w4a8_moe_experts_cls(MarlinExperts)

    assert vllm_patch._make_w4a8_moe_experts_cls(MarlinExperts) is wrapped_cls
    assert vllm_patch._make_w4a8_moe_experts_cls(wrapped_cls) is wrapped_cls


def test_w4a8_moe_rejects_non_marlin_backend(monkeypatch):
    from types import SimpleNamespace

    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend

    monkeypatch.setenv(vllm_patch.W4A8_SIMULATION_ENV, "1")
    method = SimpleNamespace(nvfp4_backend=NvFp4MoeBackend.FLASHINFER_CUTLASS)

    with pytest.raises(NotImplementedError, match="requires the vLLM NVFP4 Marlin backend"):
        vllm_patch.patched_nvfp4_moe_process_weights_after_loading(method, None)


@pytest.mark.parametrize("enabled", [True, False])
def test_w4a8_moe_process_uses_mode_specific_marlin_experts(monkeypatch, enabled):
    from types import SimpleNamespace

    from vllm.model_executor.layers.fused_moe.fused_marlin_moe import MarlinExperts
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend

    processed = []
    method = SimpleNamespace(nvfp4_backend=NvFp4MoeBackend.MARLIN, experts_cls=MarlinExperts)
    layer = SimpleNamespace()

    monkeypatch.setenv(vllm_patch.W4A8_SIMULATION_ENV, "1" if enabled else "0")
    monkeypatch.setattr(vllm_patch, "_check_first_call", lambda _layer: False)
    monkeypatch.setattr(
        vllm_patch,
        "_process_nvfp4_moe_marlin",
        lambda _method, _layer, is_first_call, experts_cls=None: processed.append((is_first_call, experts_cls)),
    )

    vllm_patch.patched_nvfp4_moe_process_weights_after_loading(method, layer)

    expected_cls = vllm_patch._make_w4a8_moe_experts_cls(MarlinExperts) if enabled else MarlinExperts
    assert method.experts_cls is MarlinExperts
    assert processed == [(False, expected_cls)]


def test_apply_qat_patches_installs_real_vllm_w4a8_wrapper(monkeypatch):
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a16_nvfp4 import (
        CompressedTensorsW4A16Fp4,
    )

    monkeypatch.setenv(vllm_patch.W4A8_SIMULATION_ENV, "1")
    vllm_patch.apply_qat_patches()

    assert vllm_patch._w4a8_apply_patch is not None
    assert CompressedTensorsW4A16Fp4.apply_weights is vllm_patch.patched_w4a16_apply_weights_with_w4a8_simulation
