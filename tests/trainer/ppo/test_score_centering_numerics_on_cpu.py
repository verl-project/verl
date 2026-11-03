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

"""Finite-gradient acceptance for the inherited near-zero-tail regularization.

The epsilon floor is an additional numerical approximation. The dense-oracle error is
reported as a diagnostic, rather than treating exact gradient parity as this test's gate.
"""

import pytest
import torch

from verl.trainer.ppo.score_centering import score_centering_auto_correction, score_centering_weight_fn


@pytest.mark.parametrize("logit_gap", [16.118095, 35.0])
@pytest.mark.parametrize("rule,threshold", [(None, 2.0), ("token", 2.0), ("token", "0.5_5.0")])
def test_near_zero_trainer_tail_has_finite_correction_and_sampled_loss_gradients(
    logit_gap, rule, threshold, record_property
):
    # Only token 0 is recorded. The actual sampled token 1 remains outside the head;
    # the declared head-tail model assigns it the remaining sampler mass of 0.5.
    logits = torch.tensor([[0.0, -logit_gap]], requires_grad=True)
    log_p = logits.log_softmax(-1)
    q = torch.tensor([[0.5, 0.5]])
    weight = score_centering_weight_fn(rule, threshold)
    correction, shift, sampler_mass, trainer_mass = score_centering_auto_correction(
        log_p[:, :1], q[:, :1].log(), weight, torch.tensor([False])
    )
    correction_grad = torch.autograd.grad(correction.sum(), logits, retain_graph=True)[0]
    sampled_weight = weight((log_p[:, 1].detach() - q[:, 1].log()).exp())
    loss = (correction - sampled_weight * (log_p[:, 1] - shift)).mean()
    loss_grad = torch.autograd.grad(loss, logits)[0]
    for value in (correction, shift, sampler_mass, trainer_mass, correction_grad, loss, loss_grad):
        assert torch.isfinite(value).all()
    assert shift.item() == 0.0

    oracle_logits = torch.tensor([[0.0, -logit_gap]], requires_grad=True)
    oracle_log_p = oracle_logits.log_softmax(-1)
    oracle_weights = weight((oracle_log_p.detach() - q.log()).exp())
    oracle = (q * oracle_weights * oracle_log_p).sum()
    oracle_grad = torch.autograd.grad(oracle, oracle_logits)[0]
    error = (correction_grad - oracle_grad).abs().max().item()
    record_property("exact_tail_oracle_max_abs_gradient_error", error)
