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
"""Real NPU CE and its fallback must respect tensor device and package availability."""

from unittest.mock import Mock

import pytest
import torch

from verl.utils import torch_functional as verl_F

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Requires an Ascend NPU.")


@pytest.mark.parametrize("native_available,flash_available", [(True, False), (True, True), (False, True)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(7, 13), (2, 3, 13)])
def test_npu_logprobs_forward_backward(monkeypatch, native_available, flash_available, dtype, shape):
    """Compare NPU outputs and gradients against log-softmax without stubbing operators."""
    if native_available and not verl_F.NPU_CROSS_ENTROPY_LOSS_AVAILABLE:
        pytest.skip("This torch-npu version does not provide npu_cross_entropy_loss.")
    torch.npu.set_device(0)
    monkeypatch.delenv("VERL_DISABLE_FLASH_ATTN_CE", raising=False)
    monkeypatch.setattr(verl_F, "FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE", flash_available)
    monkeypatch.setattr(verl_F, "NPU_CROSS_ENTROPY_LOSS_AVAILABLE", native_available)
    flash = Mock(side_effect=AssertionError("NPU logits reached Flash-Attention CE"))
    native = Mock(wraps=verl_F.logprobs_from_logits_torch_npu)
    monkeypatch.setattr(verl_F, "logprobs_from_logits_flash_attn", flash)
    monkeypatch.setattr(verl_F, "logprobs_from_logits_torch_npu", native)
    torch.manual_seed(41)
    inputs = torch.randn(shape, dtype=dtype)
    if len(shape) == 3:
        inputs = inputs.transpose(0, 1)
    logits = inputs.to("npu:0").requires_grad_(True)
    labels = torch.randint(shape[-1], logits.shape[:-1], device="npu:0")
    upstream = torch.randn(labels.shape, device="npu:0", dtype=dtype)
    reference_logits = logits.detach().clone().requires_grad_(True)
    expected = reference_logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    (expected * upstream).sum().backward()

    actual = verl_F.logprobs_from_logits(logits, labels)
    (actual * upstream).sum().backward()
    torch.npu.synchronize()
    assert actual.device == logits.device and actual.shape == labels.shape
    tolerance = {torch.float32: (1e-5, 1e-6), torch.float16: (0.005, 0.001), torch.bfloat16: (0.03, 0.01)}
    rtol, atol = tolerance[dtype]
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(logits.grad, reference_logits.grad, rtol=rtol, atol=atol)
    flash.assert_not_called()
    assert native.call_count == int(native_available)
    output_error = (actual.detach().float() - expected.detach().float()).abs().max().item()
    gradient_error = (logits.grad.float() - reference_logits.grad.float()).abs().max().item()
    print(f"{dtype=}, {shape=}, {native_available=}, {flash_available=}, {output_error=}, {gradient_error=}")
