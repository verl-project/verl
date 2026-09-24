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

"""CPU coverage for the score centering math module."""

import math

import pytest
import torch

from verl.trainer.ppo.score_centering import (
    dummy_rollout_topk,
    pad_rollout_topk,
    score_centering_correction,
    score_centering_weight_fn,
    topk_log_probs_from_logits,
)


def _full_vocab_centering(logits, sampler_log_probs):
    # Exact centering term over the whole vocabulary: sum_v sg[q_v] log p_v.
    log_p = torch.log_softmax(logits.float(), dim=-1)
    return (sampler_log_probs.exp().detach() * log_p).sum(-1)


def test_topk_log_probs_match_log_softmax_gather_with_gradient():
    torch.manual_seed(0)
    logits = torch.randn(5, 37, dtype=torch.bfloat16, requires_grad=True)
    ids = torch.stack([torch.randperm(37)[:4] for _ in range(5)])
    out = topk_log_probs_from_logits(logits, ids, chunk_size=2)
    ref = torch.log_softmax(logits.float(), dim=-1).gather(-1, ids)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    weight = torch.randn_like(out)
    (out * weight).sum().backward()
    grad_chunked = logits.grad.clone()
    logits.grad = None
    (ref * weight).sum().backward()
    torch.testing.assert_close(grad_chunked.float(), logits.grad.float(), atol=1e-2, rtol=1e-2)


def test_correction_is_zero_on_policy():
    torch.manual_seed(1)
    logits = torch.randn(3, 11)
    log_p = torch.log_softmax(logits, dim=-1)
    ids = torch.topk(log_p, k=4, dim=-1).indices
    head = log_p.gather(-1, ids)
    correction, q_mass, p_mass = score_centering_correction(head, head, score_centering_weight_fn(None, 2.0))
    torch.testing.assert_close(correction, torch.zeros(3), atol=1e-6, rtol=0)
    torch.testing.assert_close(q_mass, p_mass)


def test_topk_correction_gradient_equals_full_vocab_when_k_is_vocab():
    # With k = V the head/tail split is degenerate (rho = 0/0), so only the gradient of the
    # correction is meaningful; it must match the exact full-vocab centering term's gradient.
    torch.manual_seed(2)
    vocab = 9
    trainer_logits = torch.randn(4, vocab, requires_grad=True)
    sampler_log_probs = torch.log_softmax(torch.randn(4, vocab), dim=-1)
    ids = torch.arange(vocab).expand(4, vocab)
    head = topk_log_probs_from_logits(trainer_logits, ids)
    correction, _, _ = score_centering_correction(head, sampler_log_probs, score_centering_weight_fn(None, 2.0))
    ref = _full_vocab_centering(trainer_logits, sampler_log_probs)
    grad_a = torch.autograd.grad(correction.sum(), trainer_logits)[0]
    grad_b = torch.autograd.grad(ref.sum(), trainer_logits)[0]
    torch.testing.assert_close(grad_a, grad_b, atol=1e-5, rtol=1e-5)


def test_topk_correction_gradient_matches_tail_model():
    # With k < V the tail is modeled as rho * p; the expected gradient must equal
    # the exact gradient of sum_v sg[q_hat_v] log p_v with q_hat from the paper.
    torch.manual_seed(3)
    vocab, k = 13, 5
    trainer_logits = torch.randn(2, vocab, requires_grad=True)
    sampler_log_probs = torch.log_softmax(torch.randn(2, vocab), dim=-1)
    ids = torch.topk(sampler_log_probs, k=k, dim=-1).indices
    head = topk_log_probs_from_logits(trainer_logits, ids)
    correction, q_mass, p_mass = score_centering_correction(
        head, sampler_log_probs.gather(-1, ids), score_centering_weight_fn(None, 2.0)
    )
    log_p = torch.log_softmax(trainer_logits, dim=-1)
    p = log_p.exp().detach()
    q_hat = sampler_log_probs.exp().clone()
    in_head = torch.zeros_like(q_hat, dtype=torch.bool).scatter_(1, ids, True)
    rho = (1 - q_mass) / (1 - p_mass)
    q_hat = torch.where(in_head, q_hat, rho.unsqueeze(-1) * p)
    ref = (q_hat.detach() * log_p).sum(-1)
    grad_a = torch.autograd.grad(correction.sum(), trainer_logits)[0]
    grad_b = torch.autograd.grad(ref.sum(), trainer_logits)[0]
    torch.testing.assert_close(grad_a, grad_b, atol=1e-5, rtol=1e-5)


def test_drift_cancels_under_constant_reward():
    # Constant reward, sampler != trainer: vanilla REINFORCE has a nonzero expected gradient,
    # score centering makes it vanish (up to the tail model, exact here with k = V).
    torch.manual_seed(4)
    vocab = 7
    trainer_logits = torch.randn(1, vocab, requires_grad=True)
    sampler_log_probs = torch.log_softmax(torch.randn(1, vocab), dim=-1)
    ids = torch.arange(vocab).expand(1, vocab)
    head = topk_log_probs_from_logits(trainer_logits, ids)
    correction, _, _ = score_centering_correction(head, sampler_log_probs, score_centering_weight_fn(None, 2.0))
    log_p = torch.log_softmax(trainer_logits, dim=-1)
    q = sampler_log_probs.exp()
    expected_vanilla = torch.autograd.grad((q * log_p).sum(), trainer_logits, retain_graph=True)[0]
    expected_centered = torch.autograd.grad((q.detach() * log_p).sum() - correction.sum(), trainer_logits)[0]
    assert expected_vanilla.abs().max() > 1e-3
    torch.testing.assert_close(expected_centered, torch.zeros_like(expected_centered), atol=1e-6, rtol=0)


def test_weight_fn_matches_tis_and_icepop_rules():
    ratio = torch.tensor([0.1, 0.7, 1.0, 3.0, 9.0])
    torch.testing.assert_close(score_centering_weight_fn(None, 2.0)(ratio), torch.ones(5))
    torch.testing.assert_close(score_centering_weight_fn("token", 2.0)(ratio), ratio.clamp(max=2.0))
    torch.testing.assert_close(
        score_centering_weight_fn("token", "0.5_5.0")(ratio), torch.tensor([0.0, 0.7, 1.0, 3.0, 0.0])
    )
    with pytest.raises(ValueError):
        score_centering_weight_fn("sequence", 2.0)


def test_composed_correction_uses_alpha_rho_f_one_over_rho():
    torch.manual_seed(5)
    head_p = torch.log_softmax(torch.randn(2, 6), dim=-1)[:, :3]
    head_q = torch.log_softmax(torch.randn(2, 6), dim=-1)[:, :3]
    tis = score_centering_weight_fn("token", 2.0)
    correction, q_mass, p_mass = score_centering_correction(head_p, head_q, tis)
    rho = (1 - q_mass) / (1 - p_mass)
    alpha = rho * tis(1 / rho)
    residual = head_q.exp() * tis((head_p - head_q).exp()) - alpha.unsqueeze(-1) * head_p.exp()
    torch.testing.assert_close(correction, (residual * head_p).sum(-1))


def test_dummy_rows_are_finite_distributions():
    ids, log_probs = dummy_rollout_topk(3, 4)
    assert ids.dtype == torch.int32 and log_probs.dtype == torch.float32
    assert ids.tolist() == [[0, 1, 2, 3]] * 3
    torch.testing.assert_close(log_probs.exp().sum(-1), torch.ones(3))


def test_pad_rollout_topk_places_heads_after_last_prompt_token():
    k = 2
    response_ids = [[10, 11], [12, 13], [14, 15]]
    response_log_probs = [[-0.1, -2.0]] * 3
    ids, log_probs = pad_rollout_topk(
        response_ids, response_log_probs, k=k, prompt_width=5, response_width=4, response_length=3
    )
    assert ids.shape == (1, 9, k) and log_probs.shape == (1, 9, k)
    assert ids[0, 4:7].tolist() == response_ids
    assert ids[0, 3].tolist() == [0, 1] and ids[0, 7].tolist() == [0, 1]
    assert math.isclose(log_probs[0, 0, 0].item(), -math.log(k), rel_tol=1e-6)
    with pytest.raises(ValueError, match="top-2"):
        pad_rollout_topk(
            response_ids[:2], response_log_probs[:2], k=k, prompt_width=5, response_width=4, response_length=3
        )
