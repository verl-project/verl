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
"""Dense gradient oracles for automatic score-centering alignment and its tail proxy."""

import pytest
import torch

from verl.trainer.ppo.score_centering import (
    score_centering_auto_correction,
    score_centering_correction,
    score_centering_weight_fn,
)


@pytest.mark.parametrize("rule,threshold", [(None, 2.0), ("token", 2.0), ("token", "0.5_5.0")])
def test_mixed_alignment_matches_dense_score_gradient(rule, threshold):
    # Row 0 has complete recorded support. Row 1 records only part of the sampler.
    # Both rows are off-policy so this checks clipping and the normalizer, not a zero gradient.
    initial = torch.tensor([[1.0, 0.2, -0.3, 0.7, -0.4], [0.3, -0.2, 0.8, -0.7, 0.1]])
    ids = torch.tensor([[0, 1, 2], [0, 1, 2]])
    q_head = torch.tensor([[0.65, 0.35, 0.0], [0.10, 0.20, 0.35]])
    complete = torch.tensor([True, False])
    weight = score_centering_weight_fn(rule, threshold)
    logits = initial.clone().requires_grad_()
    lp = logits.log_softmax(-1)
    correction, shift, q_mass, p_mass = score_centering_auto_correction(
        lp.gather(-1, ids), q_head.log(), weight, complete
    )
    actual_grad = torch.autograd.grad(correction.sum(), logits)[0]

    oracle_logits = initial.clone().requires_grad_()
    oracle_lp = oracle_logits.log_softmax(-1)
    # Explicit dense sampler proxy. The complete row uses the actual support;
    # the incomplete row retains head q and distributes residual mass over the trainer tail.
    support_lp = oracle_logits[0, :2].log_softmax(-1)
    with torch.no_grad():
        aligned_weights = weight((support_lp - q_head[0, :2].log()).exp())
        p = oracle_lp[1].exp()
        q_proxy = torch.cat([q_head[1], (1 - q_head[1].sum()) * p[3:] / p[3:].sum()])
        proxy_weights = weight((oracle_lp[1] - q_proxy.log()).exp())
    oracle = (q_head[0, :2] * aligned_weights * support_lp).sum()
    oracle += (q_proxy * proxy_weights * oracle_lp[1]).sum()
    oracle_grad = torch.autograd.grad(oracle, oracle_logits)[0]
    torch.testing.assert_close(actual_grad, oracle_grad, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(shift[0], lp[0, :2].logsumexp(-1))
    assert shift[1] == 0
    torch.testing.assert_close(q_mass, q_head.sum(-1))
    torch.testing.assert_close(p_mass, torch.stack([lp[0, :2].exp().sum(), lp[1, :3].exp().sum()]))


def test_unit_head_mass_without_receipt_does_not_align():
    logits = torch.tensor([[2.0, 0.8, 0.2, -1.0]], requires_grad=True)
    head = logits.log_softmax(-1)[:, :2]
    q = torch.tensor([[0.7, 0.3]]).log()
    weight = score_centering_weight_fn(None, 2.0)
    correction, shift, _, _ = score_centering_auto_correction(head, q, weight, torch.tensor([False]))
    reference, _, _ = score_centering_correction(head, q, weight)
    torch.testing.assert_close(correction, reference)
    assert shift.item() == 0
    assert torch.isfinite(torch.autograd.grad(correction.sum(), logits)[0]).all()


def test_equal_configured_k_with_cutoff_ties_uses_partial_tail_proxy():
    logits = torch.tensor([[2.0, 1.0, 1.0, 0.0]], requires_grad=True)
    # Sampling top-k=2 retains the tied third token; the recording budget is still two.
    q = torch.tensor([[2.0, 1.0, 1.0, -torch.inf]]).log_softmax(-1)
    head = logits.log_softmax(-1)[:, :2]
    correction, shift, q_mass, _ = score_centering_auto_correction(
        head, q[:, :2], score_centering_weight_fn("token", 2.0), torch.tensor([False])
    )
    assert q_mass.item() == pytest.approx(0.7880584)
    assert shift.item() == 0
    assert torch.isfinite(torch.autograd.grad(correction.sum(), logits)[0]).all()


@pytest.mark.parametrize("rule,threshold", [(None, 2.0), ("token", 2.0), ("token", "0.5_5.0")])
def test_sampled_token_outside_head_keeps_actual_probability(rule, threshold):
    logits = torch.tensor([[1.2, 0.4, -0.1, 0.3]], requires_grad=True)
    lp = logits.log_softmax(-1)
    q_head = torch.tensor([[0.45, 0.25]])
    q_sample = torch.tensor([0.20])
    weight = score_centering_weight_fn(rule, threshold)
    correction, shift, _, _ = score_centering_auto_correction(lp[:, :2], q_head.log(), weight, torch.tensor([False]))
    sampled = lp[:, 2] - shift
    actual_sampled_weight = weight((sampled.detach() - q_sample.log()).exp())
    loss = -(actual_sampled_weight * sampled - correction).mean()
    assert shift.item() == 0
    assert torch.equal(q_sample, torch.tensor([0.20]))
    assert torch.isfinite(torch.autograd.grad(loss, logits)[0]).all()
