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
"""CPU log-probs remain usable when optional accelerator kernels are installed."""

from unittest.mock import Mock

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from verl.utils import torch_functional as verl_F


@pytest.mark.parametrize("flash_available,npu_available", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("shape", [(7, 13), (2, 3, 13)])
def test_cpu_logprobs_forward_backward(monkeypatch, flash_available, npu_available, dtype, shape):
    """Optional packages cannot redirect CPU logits into device-only operators."""
    monkeypatch.delenv("VERL_DISABLE_FLASH_ATTN_CE", raising=False)
    monkeypatch.setattr(verl_F, "FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE", flash_available)
    monkeypatch.setattr(verl_F, "NPU_CROSS_ENTROPY_LOSS_AVAILABLE", npu_available)
    flash = Mock(side_effect=AssertionError("CPU logits reached Flash-Attention CE"))
    npu = Mock(side_effect=AssertionError("CPU logits reached NPU CE"))
    monkeypatch.setattr(verl_F, "logprobs_from_logits_flash_attn", flash)
    monkeypatch.setattr(verl_F, "logprobs_from_logits_torch_npu", npu)
    torch.manual_seed(41)
    logits = torch.randn(shape, dtype=dtype)
    # Include strided logits rather than covering only contiguous model output.
    if len(shape) == 3:
        logits = logits.transpose(0, 1)
    logits.requires_grad_(True)
    labels = torch.randint(shape[-1], logits.shape[:-1])
    upstream = torch.randn(labels.shape, dtype=dtype)
    expected_logits = logits.detach().clone().requires_grad_(True)
    expected = expected_logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    (expected * upstream).sum().backward()

    actual = verl_F.logprobs_from_logits(logits, labels)
    (actual * upstream).sum().backward()
    assert actual.device.type == "cpu" and actual.shape == labels.shape
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(logits.grad, expected_logits.grad)
    flash.assert_not_called()
    npu.assert_not_called()


@pytest.mark.parametrize("disable_flash", [False, True])
def test_cuda_dispatch_and_disable_switch(monkeypatch, disable_flash):
    """Check CUDA routing metadata on CPU, including an installed NPU package."""
    monkeypatch.setenv("VERL_DISABLE_FLASH_ATTN_CE", "1" if disable_flash else "0")
    monkeypatch.setattr(verl_F, "FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE", True)
    monkeypatch.setattr(verl_F, "NPU_CROSS_ENTROPY_LOSS_AVAILABLE", True)
    npu = Mock(side_effect=AssertionError("CUDA logits reached NPU CE"))
    monkeypatch.setattr(verl_F, "logprobs_from_logits_torch_npu", npu)
    with FakeTensorMode():
        logits = torch.empty(2, 3, 13, device="cuda")
        labels = torch.empty(2, 3, device="cuda", dtype=torch.long)
        flash = Mock(return_value=torch.empty(6, device="cuda"))
        monkeypatch.setattr(verl_F, "logprobs_from_logits_flash_attn", flash)
        actual = verl_F.logprobs_from_logits(logits, labels, inplace_backward=False)
        assert actual.shape == labels.shape and actual.device == logits.device
        assert flash.call_count == int(not disable_flash)
        if not disable_flash:
            flat_logits, flat_labels = flash.call_args.args
            assert flat_logits.shape == (6, 13) and flat_labels.shape == (6,)
            assert flash.call_args.kwargs == {"inplace_backward": False}
    npu.assert_not_called()
