# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import torch.nn.functional as F

from verl.utils.torch_functional import logprobs_from_logits_naive, logprobs_from_logits_v2

LOGPROB_FNS = [logprobs_from_logits_naive, logprobs_from_logits_v2]


def _fp64_logprobs_and_grad(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fp64 logprob -softplus(logsumexp_{j != label} - z_label) and its gradient onehot - softmax."""
    logits = logits.double()
    label_logits = logits.gather(-1, labels.unsqueeze(-1))
    others = logits.scatter(-1, labels.unsqueeze(-1), -torch.inf)
    nll = F.softplus(others.logsumexp(-1, keepdim=True) - label_logits)
    grad = -torch.exp(others - label_logits - nll)
    grad.scatter_(-1, labels.unsqueeze(-1), -torch.expm1(-nll))
    return -nll.squeeze(-1), grad


@pytest.mark.parametrize("logprobs_fn", LOGPROB_FNS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_logprobs_match_log_softmax(logprobs_fn, dtype):
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 37, dtype=dtype, requires_grad=True)
    labels = torch.randint(0, 37, (2, 5))
    logprobs = logprobs_fn(logits, labels)
    (grad,) = torch.autograd.grad(logprobs.sum(), logits)

    reference_logits = logits.detach().float().requires_grad_(True)
    reference = F.log_softmax(reference_logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    reference.sum().backward()
    assert logprobs.dtype == dtype and logprobs.shape == labels.shape
    tol = {"rtol": 1.6e-2, "atol": 1e-2} if dtype == torch.bfloat16 else {"rtol": 1e-5, "atol": 1e-6}
    torch.testing.assert_close(logprobs.float(), reference, **tol)
    torch.testing.assert_close(grad.float(), reference_logits.grad, **tol)


@pytest.mark.parametrize("logprobs_fn", LOGPROB_FNS)
@pytest.mark.parametrize(
    "dtype,label_logits",
    [
        # Margins over the other logits of ~23 to ~73: log_softmax rounds the logprob and label gradient to 0
        (torch.float32, [30.0, 50.0, 80.0]),
        # Margins of ~6 to ~23 in bf16
        (torch.bfloat16, [13.0, 17.0, 30.0]),
    ],
)
def test_logprobs_confident_label(logprobs_fn, dtype, label_logits):
    vocab = 1000
    generator = torch.Generator().manual_seed(0)
    logits = (0.1 * torch.randn(1, 3, vocab, generator=generator)).to(dtype)
    labels = torch.tensor([[3, 500, 997]])
    logits[0, torch.arange(3), labels[0]] = torch.tensor(label_logits, dtype=dtype)
    logits.requires_grad_(True)
    logprobs = logprobs_fn(logits, labels)
    (grad,) = torch.autograd.grad(logprobs.sum(), logits)

    expected_logprobs, expected_grad = _fp64_logprobs_and_grad(logits.detach(), labels)
    rtol = 1e-5 if dtype == torch.float32 else 1e-2
    assert torch.all(logprobs < 0)
    torch.testing.assert_close(logprobs.double(), expected_logprobs, rtol=rtol, atol=0)
    torch.testing.assert_close(grad.double(), expected_grad, rtol=10 * rtol, atol=0)
    # The label gradient balances the others
    label_grad = grad.double().gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(grad.double().sum(-1) - label_grad, -label_grad, rtol=10 * rtol, atol=0)


@pytest.mark.parametrize("logprobs_fn", LOGPROB_FNS)
def test_logprobs_label_without_competitors_has_finite_grad(logprobs_fn):
    logits = torch.tensor([[[2.0, -torch.inf, -torch.inf]]], requires_grad=True)
    logprobs = logprobs_fn(logits, torch.tensor([[0]]))
    (grad,) = torch.autograd.grad(logprobs.sum(), logits)
    assert logprobs.item() == 0.0
    assert torch.all(grad == 0)


@pytest.mark.parametrize("logprobs_fn", LOGPROB_FNS)
def test_logprobs_unlikely_label_grad(logprobs_fn):
    # A -inf label and one far below the others: the gradient is still onehot - softmax, and zero when masked out
    logits = torch.tensor([[[-torch.inf, 0.0, 1.0], [-1e8, 0.0, 1.0]]], requires_grad=True)
    labels = torch.tensor([[0, 0]])
    logprobs = logprobs_fn(logits, labels)
    (grad,) = torch.autograd.grad(logprobs.sum(), logits)
    expected_grad = F.one_hot(labels, 3).double() - torch.softmax(logits.detach().double(), dim=-1)
    assert logprobs[0, 0] == -torch.inf
    torch.testing.assert_close(grad.double(), expected_grad, rtol=1e-5, atol=0)

    masked = torch.where(torch.zeros_like(labels, dtype=torch.bool), logprobs_fn(logits, labels), 0.0)
    (grad,) = torch.autograd.grad(masked.sum(), logits)
    assert torch.all(grad == 0)
