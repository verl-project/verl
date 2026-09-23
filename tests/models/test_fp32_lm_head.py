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

import pytest
import torch

from verl.models.transformers.lm_head import fp32_linear

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="FP32 lm_head requires CUDA")


def test_fp32_lm_head_forward_and_backward():
    torch.manual_seed(42)
    hidden = torch.randn(2, 13, 19, device="cuda").transpose(1, 2)
    hidden = hidden.to(torch.bfloat16).detach().requires_grad_(True)
    weight = torch.randn(29, 13, device="cuda").to(torch.bfloat16).requires_grad_(True)
    bias = torch.randn(29, device="cuda").to(torch.bfloat16).requires_grad_(True)
    assert not hidden.is_contiguous()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = fp32_linear(hidden, weight, bias) / 0.7
        loss = output.log_softmax(dim=-1).square().mean()
    loss.backward()

    expected_hidden = hidden.detach().float().requires_grad_(True)
    expected_weight = weight.detach().float().requires_grad_(True)
    expected_bias = bias.detach().float().requires_grad_(True)
    expected = torch.nn.functional.linear(expected_hidden, expected_weight, expected_bias) / 0.7
    expected.log_softmax(dim=-1).square().mean().backward()

    assert output.dtype == torch.float32
    torch.testing.assert_close(output, expected, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(hidden.grad, expected_hidden.grad.to(torch.bfloat16))
    torch.testing.assert_close(weight.grad, expected_weight.grad.to(torch.bfloat16))
    torch.testing.assert_close(bias.grad, expected_bias.grad.to(torch.bfloat16))

    rounded_after_projection = torch.nn.functional.linear(hidden.detach(), weight.detach(), bias).float()
    assert not torch.equal(output * 0.7, rounded_after_projection)
