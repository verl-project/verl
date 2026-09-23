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
from verl.utils.experimental.torch_functional import FusedLinearForPPO

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="FP32 lm_head requires CUDA")


@pytest.mark.parametrize("hidden_shape", [(19, 13), (2, 19, 13)])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
@pytest.mark.parametrize("with_bias", [False, True])
def test_fp32_lm_head_forward_and_backward(hidden_shape, temperature, with_bias):
    torch.manual_seed(42)
    device = torch.device("cuda")
    vocab_size = 29
    if len(hidden_shape) == 2:
        hidden = torch.randn(hidden_shape[-1], hidden_shape[0], device=device).t()
    else:
        hidden = torch.randn(hidden_shape[0], hidden_shape[-1], hidden_shape[1], device=device).transpose(1, 2)
    hidden = hidden.to(torch.bfloat16).detach().requires_grad_(True)
    weight = torch.randn(vocab_size, hidden_shape[-1], device=device).to(torch.bfloat16).requires_grad_(True)
    bias = None
    if with_bias:
        bias = torch.randn(vocab_size, device=device).to(torch.bfloat16).requires_grad_(True)
    assert not hidden.is_contiguous()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = fp32_linear(hidden, weight, bias) / temperature
        loss = output.log_softmax(dim=-1).square().mean()
    loss.backward()

    expected_hidden = hidden.detach().float().requires_grad_(True)
    expected_weight = weight.detach().float().requires_grad_(True)
    expected_bias = bias.detach().float().requires_grad_(True) if bias is not None else None
    expected = torch.nn.functional.linear(expected_hidden, expected_weight, expected_bias) / temperature
    expected.log_softmax(dim=-1).square().mean().backward()

    assert output.dtype == torch.float32
    torch.testing.assert_close(output, expected, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(hidden.grad, expected_hidden.grad.to(torch.bfloat16))
    torch.testing.assert_close(weight.grad, expected_weight.grad.to(torch.bfloat16))
    if bias is not None:
        torch.testing.assert_close(bias.grad, expected_bias.grad.to(torch.bfloat16))

    rounded_after_projection = torch.nn.functional.linear(hidden.detach(), weight.detach(), bias).float()
    assert not torch.equal(output * temperature, rounded_after_projection)


def test_fp32_fused_chunk_tail_and_entropy_backward():
    torch.manual_seed(101)
    device = torch.device("cuda")
    hidden = torch.randn(2, 19, 13, device=device).to(torch.bfloat16).requires_grad_(True)
    weight = torch.randn(29, 13, device=device).to(torch.bfloat16).requires_grad_(True)
    labels = torch.randint(29, (2, 19), device=device)

    log_probs, entropy = FusedLinearForPPO(chunk_size=7, lm_head_dtype="float32")(
        hidden, weight, labels, temperature=0.7
    )
    (log_probs.mean() + entropy.mean()).backward()

    assert log_probs.dtype == torch.float32
    assert entropy.dtype == torch.float32
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
