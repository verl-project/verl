# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""CPU tests for the TOPR (Tapered Off-Policy REINFORCE) policy loss.

TOPR (https://arxiv.org/abs/2503.14286) reinforces positive-advantage sequences with unit weight
and weights negative-advantage sequences by the sequence-level importance ratio tapered to
``[0, 1]``, keeping their gradient bounded but non-zero where PPO-style clipping zeroes it. These
tests pin the defining properties of the objective on synthetic tensors. No GPU/model needed.
"""

import pytest
import torch

from verl.trainer.ppo.core_algos import compute_policy_loss_topr, get_policy_loss_fn
from verl.workers.config import ActorConfig, PolicyLossConfig


def _make_config(**policy_loss_overrides) -> ActorConfig:
    return ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        use_dynamic_bsz=True,
        policy_loss=PolicyLossConfig(**policy_loss_overrides),
    )


def _make_batch(seq_advantages, log_ratio_per_token, batch_size=None, resp_len=8, seed=0):
    """Build a synthetic batch with a constant advantage per sequence (outcome/group style) and a
    controlled per-token log importance ratio ``log_prob - old_log_prob``."""
    batch_size = batch_size or len(seq_advantages)
    g = torch.Generator().manual_seed(seed)
    old_log_prob = -torch.rand(batch_size, resp_len, generator=g) - 0.5
    log_prob = (old_log_prob + torch.tensor(log_ratio_per_token).unsqueeze(-1)).detach()
    log_prob.requires_grad_(True)
    advantages = torch.tensor(seq_advantages).unsqueeze(-1).expand(batch_size, resp_len).contiguous()
    response_mask = torch.ones(batch_size, resp_len)
    return old_log_prob, log_prob, advantages, response_mask


def _reference_reinforce_loss(log_prob, advantages, response_mask, seq_weights):
    """Hand-computed seq-mean-token-mean REINFORCE surrogate with per-sequence weights."""
    per_token = -advantages * log_prob * seq_weights.unsqueeze(-1)
    seq_token_counts = response_mask.sum(dim=-1)
    seq_means = (per_token * response_mask).sum(dim=-1) / (seq_token_counts + 1e-8)
    seq_valid = (seq_token_counts > 0).float()
    return (seq_means * seq_valid).sum() / seq_valid.sum()


def test_topr_is_registered():
    assert get_policy_loss_fn("topr") is compute_policy_loss_topr


def test_engine_call_convention_pins_sequence_aggregation():
    """The engine's ``ppo_loss`` always passes ``loss_agg_mode=config.loss_agg_mode`` (default
    "token-mean") explicitly. Like gspo/sapo, TOPR must pin "seq-mean-token-mean" regardless, so
    the paper's per-sequence length normalization survives the engine call convention."""
    config = _make_config()
    old_log_prob, log_prob, advantages, mask = _make_batch(
        seq_advantages=[1.0, -1.0, 0.5, -0.5], log_ratio_per_token=[0.0, -1.0, 0.5, -0.5], resp_len=8
    )
    # make sequence lengths ragged so token-mean and seq-mean-token-mean genuinely differ
    mask[0, 4:] = 0.0
    mask[1, 2:] = 0.0

    loss_default, _ = compute_policy_loss_topr(old_log_prob, log_prob, advantages, mask, config=config)
    loss_engine, _ = compute_policy_loss_topr(
        old_log_prob, log_prob, advantages, mask, loss_agg_mode="token-mean", config=config
    )
    torch.testing.assert_close(loss_engine, loss_default)


def test_on_policy_reduces_to_reinforce():
    """With log_prob == old_log_prob every taper weight is 1 and TOPR equals plain REINFORCE, in
    both loss value and gradient."""
    config = _make_config()
    old_log_prob, log_prob, advantages, mask = _make_batch(
        seq_advantages=[1.0, -1.0, 0.5, -0.25], log_ratio_per_token=[0.0, 0.0, 0.0, 0.0]
    )

    loss, metrics = compute_policy_loss_topr(old_log_prob, log_prob, advantages, mask, config=config)
    grad = torch.autograd.grad(loss, log_prob, retain_graph=True)[0]

    ref = _reference_reinforce_loss(log_prob, advantages, mask, torch.ones(4))
    ref_grad = torch.autograd.grad(ref, log_prob)[0]

    torch.testing.assert_close(loss, ref)
    torch.testing.assert_close(grad, ref_grad)
    assert metrics["actor/topr_taper_weight_mean"] == pytest.approx(1.0)


def test_positive_advantage_ignores_importance_ratio():
    """Positive-advantage sequences get unit weight regardless of how off-policy they are: even
    with a strongly shifted importance ratio, the loss equals the plain weight-1 REINFORCE
    surrogate on the same tensors."""
    config = _make_config()
    old_log_prob, log_prob, advantages, mask = _make_batch(
        seq_advantages=[1.0, 2.0, 0.5, 1.5], log_ratio_per_token=[-3.0, 2.0, -1.0, 0.5]
    )
    loss, metrics = compute_policy_loss_topr(old_log_prob, log_prob, advantages, mask, config=config)

    ref = _reference_reinforce_loss(log_prob, advantages, mask, torch.ones(4))
    torch.testing.assert_close(loss, ref)
    assert metrics["actor/topr_taper_weight_mean"] == pytest.approx(1.0)


def test_negative_advantage_gradient_scales_with_tapered_ratio():
    """For a negative-advantage sequence the gradient equals ``clip(ratio, 0, 1)`` times the
    REINFORCE gradient: down-weighted when the sequence became unlikely (ratio < 1) and capped at
    the on-policy magnitude when it became more likely (ratio > 1)."""
    config = _make_config()
    for log_ratio, expected_weight in ((-1.0, torch.exp(torch.tensor(-1.0)).item()), (2.0, 1.0)):
        old_log_prob, log_prob, advantages, mask = _make_batch(seq_advantages=[-1.0], log_ratio_per_token=[log_ratio])
        loss, metrics = compute_policy_loss_topr(old_log_prob, log_prob, advantages, mask, config=config)
        grad = torch.autograd.grad(loss, log_prob, retain_graph=True)[0]

        ref = _reference_reinforce_loss(log_prob, advantages, mask, torch.ones(1))
        ref_grad = torch.autograd.grad(ref, log_prob)[0]

        torch.testing.assert_close(grad, expected_weight * ref_grad)
        assert metrics["actor/topr_taper_weight_mean"] == pytest.approx(expected_weight, rel=1e-5)


def test_negative_advantage_gradient_survives_where_ppo_clips_to_zero():
    """The defining TOPR property: for an off-policy negative-advantage sequence whose ratio fell
    below PPO's clip window, vanilla PPO's gradient is exactly zero (the clipped branch dominates),
    while TOPR keeps a non-zero, ratio-tapered gradient."""
    config = _make_config()
    vanilla_fn = get_policy_loss_fn("vanilla")

    # every sequence is negative-advantage and strongly off-policy: ratio = e^-5 << 1 - clip_ratio
    old_log_prob, log_prob, advantages, mask = _make_batch(
        seq_advantages=[-1.0, -0.5], log_ratio_per_token=[-5.0, -5.0]
    )

    vanilla_loss, _ = vanilla_fn(old_log_prob, log_prob, advantages, mask, config=config)
    vanilla_grad = torch.autograd.grad(vanilla_loss, log_prob, retain_graph=True)[0]
    assert torch.all(vanilla_grad == 0.0), "PPO clipping should zero the gradient in this regime"

    topr_loss, _ = compute_policy_loss_topr(old_log_prob, log_prob, advantages, mask, config=config)
    topr_grad = torch.autograd.grad(topr_loss, log_prob)[0]
    assert torch.all(topr_grad != 0.0)

    # and the surviving gradient is exactly the tapered-ratio-scaled REINFORCE gradient
    ref = _reference_reinforce_loss(log_prob, advantages, mask, torch.ones(2))
    ref_grad = torch.autograd.grad(ref, log_prob)[0]
    torch.testing.assert_close(topr_grad, torch.exp(torch.tensor(-5.0)) * ref_grad)


def test_taper_weight_is_stop_gradient():
    """No gradient flows through the taper weight itself: the TOPR gradient matches the gradient of
    a reference loss built with the weights as constants."""
    config = _make_config()
    old_log_prob, log_prob, advantages, mask = _make_batch(
        seq_advantages=[-1.0, 1.0, -2.0], log_ratio_per_token=[-0.5, 1.0, 0.3]
    )
    loss, _ = compute_policy_loss_topr(old_log_prob, log_prob, advantages, mask, config=config)
    grad = torch.autograd.grad(loss, log_prob, retain_graph=True)[0]

    with torch.no_grad():
        seq_ratio = torch.exp(((log_prob - old_log_prob) * mask).sum(-1) / mask.sum(-1))
        seq_adv = (advantages * mask).sum(-1) / mask.sum(-1)
        weights = torch.where(seq_adv >= 0, torch.ones_like(seq_ratio), seq_ratio.clamp(0.0, 1.0))
    ref = _reference_reinforce_loss(log_prob, advantages, mask, weights)
    ref_grad = torch.autograd.grad(ref, log_prob)[0]

    torch.testing.assert_close(grad, ref_grad)


def test_taper_bounds_are_configurable():
    """The negative-branch bounds come from PolicyLossConfig: raising the upper bound above 1 lets
    a ratio > 1 negative sequence be weighted above the canonical cap."""
    old_log_prob, log_prob, advantages, mask = _make_batch(seq_advantages=[-1.0], log_ratio_per_token=[0.5])

    _, metrics_canonical = compute_policy_loss_topr(old_log_prob, log_prob, advantages, mask, config=_make_config())
    _, metrics_raised = compute_policy_loss_topr(
        old_log_prob, log_prob, advantages, mask, config=_make_config(topr_negative_ratio_upper=2.0)
    )

    assert metrics_canonical["actor/topr_taper_weight_mean"] == pytest.approx(1.0)
    assert metrics_raised["actor/topr_taper_weight_mean"] == pytest.approx(
        torch.exp(torch.tensor(0.5)).item(), rel=1e-5
    )


def test_padded_tokens_do_not_contribute():
    """Tokens outside the response mask affect neither the taper weight nor the loss."""
    config = _make_config()
    g = torch.Generator().manual_seed(1)
    old_log_prob = -torch.rand(2, 8, generator=g)
    log_prob_base = (old_log_prob - 0.5).detach()
    advantages = torch.tensor([[-1.0], [1.0]]).expand(2, 8).contiguous()
    mask = torch.ones(2, 8)
    mask[:, 5:] = 0.0

    log_prob_a = log_prob_base.clone().requires_grad_(True)
    loss_a, _ = compute_policy_loss_topr(old_log_prob, log_prob_a, advantages, mask, config=config)

    corrupted = log_prob_base.clone()
    corrupted[:, 5:] += 100.0  # garbage in the padded region
    log_prob_b = corrupted.requires_grad_(True)
    loss_b, _ = compute_policy_loss_topr(old_log_prob, log_prob_b, advantages, mask, config=config)

    torch.testing.assert_close(loss_a, loss_b)
    grad_b = torch.autograd.grad(loss_b, log_prob_b)[0]
    assert torch.all(grad_b[:, 5:] == 0.0)


def test_rollout_is_weights_compose():
    """Rollout-correction weights multiply the loss exactly like in the other policy losses."""
    config = _make_config()
    old_log_prob, log_prob, advantages, mask = _make_batch(seq_advantages=[1.0, -1.0], log_ratio_per_token=[0.0, 0.0])
    loss_plain, _ = compute_policy_loss_topr(old_log_prob, log_prob, advantages, mask, config=config)
    loss_scaled, _ = compute_policy_loss_topr(
        old_log_prob, log_prob, advantages, mask, config=config, rollout_is_weights=torch.full((2, 8), 0.5)
    )
    torch.testing.assert_close(loss_scaled, 0.5 * loss_plain)


@pytest.mark.parametrize("num_micro_batches", [2, 4])
def test_microbatch_invariance_with_global_batch_info(num_micro_batches):
    """With the engine's global batch info threaded through ``config.global_batch_info``, summing
    the per-micro-batch TOPR losses equals the whole-mini-batch loss (the same normalization
    contract the actor's other losses follow)."""
    batch_size = 8
    config = _make_config()
    config.global_batch_info["dp_size"] = 1
    config.global_batch_info["batch_num_tokens"] = None
    config.global_batch_info["global_batch_size"] = batch_size
    config.global_batch_info["loss_scale_factor"] = None

    old_log_prob, log_prob, advantages, mask = _make_batch(
        seq_advantages=[1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.25, -0.25],
        log_ratio_per_token=[0.0, -1.0, 0.5, -0.5, 1.0, -2.0, 0.0, 0.3],
    )

    whole, _ = compute_policy_loss_topr(old_log_prob, log_prob, advantages, mask, config=config)

    step = batch_size // num_micro_batches
    accum = sum(
        compute_policy_loss_topr(
            old_log_prob[i : i + step],
            log_prob[i : i + step],
            advantages[i : i + step],
            mask[i : i + step],
            config=config,
        )[0]
        for i in range(0, batch_size, step)
    )
    torch.testing.assert_close(accum, whole)
