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

from verl.models.transformers.fused_lm_head import fused_lm_head_forward
from verl.utils.experimental import torch_functional as experimental_F


@pytest.mark.parametrize("cast_hidden_states", [False, True])
def test_unsharded_fused_head_preserves_logits_and_gradients(cast_hidden_states, monkeypatch):
    # Exercise the CPU fallback even if FlashAttention is installed.
    monkeypatch.setattr(experimental_F, "_FLASH_ATTN_CROSS_ENTROPY_AVAILABLE", False)
    torch.manual_seed(0)
    head = torch.nn.Linear(8, 16, bias=False)
    hidden = torch.randn(2, 7, 8, requires_grad=True)
    labels = torch.randint(16, (2, 7))
    if cast_hidden_states:
        hidden = hidden.detach().double().requires_grad_()
    original_forward = head.forward
    log_probs, entropy = fused_lm_head_forward(
        head, hidden, labels, 0.7, "torch", cast_hidden_states=cast_hidden_states
    )
    logits = head(hidden.to(head.weight.dtype)) / 0.7
    expected_log_probs = logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    expected_entropy = logits.logsumexp(-1) - (logits.softmax(-1) * logits).sum(-1)
    torch.testing.assert_close(log_probs, expected_log_probs)
    torch.testing.assert_close(entropy, expected_entropy)
    actual_grads = torch.autograd.grad(-log_probs.mean() + 0.01 * entropy.mean(), (hidden, head.weight))
    expected_grads = torch.autograd.grad(
        -expected_log_probs.mean() + 0.01 * expected_entropy.mean(), (hidden, head.weight)
    )
    for actual, expected in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, expected)
    assert head.forward == original_forward
    assert not hasattr(head, "_verl_fused_forward")
