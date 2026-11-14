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

from verl.utils.experimental.torch_functional import FusedLinearForPPO


@pytest.mark.parametrize("temperature", [0.7, 1.3])
def test_fused_linear_for_ppo_divides_fp32_logits_by_temperature(temperature):
    """Log-probs and gradients must use logits.float() / T, the order the rollout sampler uses.

    Dividing the bf16 logits by T before the upcast rounds them a second time.
    """
    torch.manual_seed(0)
    num_tokens, hidden_size, vocab_size = 64, 256, 1000
    hidden = torch.randn(num_tokens, hidden_size).to(torch.bfloat16).requires_grad_(True)
    weight = (torch.randn(vocab_size, hidden_size) * 3 / hidden_size**0.5).to(torch.bfloat16).requires_grad_(True)
    labels = torch.randint(0, vocab_size, (num_tokens,))
    grad = torch.randn(num_tokens)

    log_probs, _ = FusedLinearForPPO(chunk_size=16)(hidden, weight, labels, temperature=temperature)
    (log_probs * grad).sum().backward()

    ref_hidden = hidden.detach().clone().requires_grad_(True)
    ref_weight = weight.detach().clone().requires_grad_(True)
    ref_logits = (ref_hidden @ ref_weight.t()).float() / temperature
    ref_log_probs = torch.log_softmax(ref_logits, dim=-1).gather(-1, labels[:, None]).squeeze(-1)
    (ref_log_probs * grad).sum().backward()

    torch.testing.assert_close(log_probs, ref_log_probs, atol=1e-5, rtol=0)
    torch.testing.assert_close(hidden.grad, ref_hidden.grad, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(weight.grad, ref_weight.grad, atol=1e-2, rtol=1e-2)
