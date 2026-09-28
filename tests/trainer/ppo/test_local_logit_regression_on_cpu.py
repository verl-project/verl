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
"""Tensor-level tests for the LOCAL (Local-Curvature Advantage Logit Regression) policy loss."""

import pytest
import torch

from verl.trainer.ppo.local_logit_regression import (
    compute_old_topk_features,
    compute_policy_loss_local,
    compute_topk_centered_logits,
    is_local_loss,
)
from verl.workers.config.actor import ActorConfig, PolicyLossConfig


def _random_logits(num_rows=7, vocab=11, seed=0):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(num_rows, vocab, generator=generator) * 3
    labels = torch.randint(0, vocab, (num_rows,), generator=generator)
    return logits, labels


def test_old_features_match_brute_force():
    logits, labels = _random_logits()
    ids, log_probs, centered = compute_old_topk_features(logits, labels, topk=4)

    top_logits, top_ids = torch.topk(logits, k=4, dim=-1)
    assert ids.dtype == torch.int32
    assert torch.equal(ids.long(), top_ids)
    torch.testing.assert_close(log_probs, torch.log_softmax(logits, dim=-1).gather(-1, top_ids))
    pbar = torch.softmax(logits, dim=-1).gather(-1, top_ids)
    pbar = pbar / pbar.sum(dim=-1, keepdim=True)
    expected = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1) - (pbar * top_logits).sum(dim=-1)
    torch.testing.assert_close(centered, expected)


def test_chunking_does_not_change_features():
    logits, labels = _random_logits(num_rows=9)
    reference = compute_old_topk_features(logits, labels, topk=3, chunk_size=512)
    chunked = compute_old_topk_features(logits, labels, topk=3, chunk_size=2)
    for a, b in zip(reference, chunked, strict=True):
        torch.testing.assert_close(a, b)


def test_full_support_gives_the_exact_optimal_offset():
    """With K = |vocab| the centre is E_pi[z], the optimal state-dependent offset."""
    logits, labels = _random_logits()
    _, _, centered = compute_old_topk_features(logits, labels, topk=100)  # clipped to the vocabulary
    expected = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1) - (torch.softmax(logits, -1) * logits).sum(-1)
    torch.testing.assert_close(centered, expected)


def test_current_features_equal_old_features_and_ignore_per_state_offsets():
    logits, labels = _random_logits()
    ids, log_probs, old_centered = compute_old_topk_features(logits, labels, topk=4)
    shifted = logits + torch.randn(logits.shape[0], 1) * 10  # one constant per state
    current = compute_topk_centered_logits(shifted, labels, ids, log_probs)
    torch.testing.assert_close(current, old_centered, rtol=1e-5, atol=1e-4)


def test_logit_gradient_is_residual_times_centred_indicator():
    """d/dz 1/2 delta^2 = delta * (e_a - pbar), with pbar zero outside the Top-K support."""
    logits, labels = _random_logits(num_rows=1, vocab=9)
    ids, log_probs, old_centered = compute_old_topk_features(logits, labels, topk=3)
    current = (logits + 0.1 * torch.randn_like(logits)).requires_grad_()
    eta, advantage = 2.0, torch.tensor([0.7])
    delta = compute_topk_centered_logits(current, labels, ids, log_probs) - old_centered - eta * advantage
    (0.5 * delta.square()).sum().backward()

    pbar = torch.zeros(9)
    pbar[ids[0].long()] = torch.softmax(log_probs[0], dim=-1)
    indicator = torch.zeros(9)
    indicator[labels[0]] = 1.0
    torch.testing.assert_close(current.grad[0], delta.detach()[0] * (indicator - pbar))


def test_loss_at_the_old_policy_is_half_eta_squared_mean_advantage_squared():
    advantages = torch.tensor([[1.0, -2.0, 0.5], [3.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 1], [1, 0, 0]])
    zeros = torch.zeros_like(advantages)
    loss, metrics = compute_policy_loss_local(
        centered_logits=zeros, old_centered_logits=zeros, advantages=advantages, response_mask=mask, eta=0.5
    )
    mean_sq = (1.0 + 4.0 + 0.25 + 9.0) / 4
    torch.testing.assert_close(loss, torch.tensor(0.5 * 0.25 * mean_sq))
    assert metrics["actor/local_residual_mse"] == pytest.approx(0.25 * mean_sq)
    assert metrics["actor/local_feature_delta_abs"] == pytest.approx(0.0)


def test_rollout_weights_and_metrics():
    centered = torch.tensor([[0.2, 0.4]])
    old = torch.zeros(1, 2)
    advantages = torch.tensor([[1.0, 1.0]])
    mask = torch.ones(1, 2)
    weights = torch.tensor([[2.0, 0.0]])
    topk_log_probs = torch.log(torch.tensor([[[0.5, 0.3], [0.9, 0.05]]]))
    loss, metrics = compute_policy_loss_local(
        centered_logits=centered,
        old_centered_logits=old,
        advantages=advantages,
        response_mask=mask,
        eta=1.0,
        rollout_is_weights=weights,
        old_topk_log_probs=topk_log_probs,
        log_prob=torch.tensor([[-1.0, -2.0]]),
        old_log_prob=torch.tensor([[-1.5, -2.0]]),
    )
    torch.testing.assert_close(loss, torch.tensor((2.0 * 0.5 * 0.8**2) / 2))
    assert metrics["actor/local_topk_mass"] == pytest.approx((0.8 + 0.95) / 2)
    assert metrics["actor/ppo_kl"] == pytest.approx(-0.25)


def test_is_local_loss():
    assert is_local_loss({"policy_loss": {"loss_mode": "local"}})
    assert not is_local_loss({"policy_loss": {"loss_mode": "vanilla"}})
    assert not is_local_loss({})


def _actor_config(**kwargs):
    policy_loss = PolicyLossConfig(loss_mode="local", **kwargs.pop("policy_loss", {}))
    return ActorConfig(rollout_n=1, ppo_micro_batch_size=2, policy_loss=policy_loss, **kwargs)


def test_actor_config_accepts_local_on_fsdp():
    config = _actor_config(strategy="fsdp2", policy_loss={"local_eta": 5.0, "local_topk": 32})
    assert config.policy_loss.local_eta == 5.0
    assert config.policy_loss.local_topk == 32


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strategy": "megatron"},
        {"strategy": "fsdp", "use_fused_kernels": True},
        {"strategy": "fsdp", "policy_loss": {"local_topk": 0}},
        {"strategy": "fsdp", "policy_loss": {"local_eta": 0.0}},
    ],
)
def test_actor_config_rejects_unsupported_local_settings(kwargs):
    with pytest.raises(ValueError):
        _actor_config(**kwargs)
