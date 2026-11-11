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

"""Invariance coverage for registered policy losses on CPU.

Two invariants, following the precedent of #7829 (which established that
splitting an optimizer minibatch into microbatches must not change the
objective):

1. Minibatch invariance: evaluating a loss once on the full minibatch must
   equal recombining per-microbatch losses weighted by valid-token count
   (what token-mean aggregation accumulates to). Losses that compute
   minibatch-level statistics — token selection quotas, covariance ranking,
   batch normalization factors — over whatever slice they receive violate
   this: clip_cov (#8634), kl_cov (#7829), and bypass_mode with
   rollout_is_batch_normalize=True (same class, see #8634 discussion).

2. Rejected-token isolation: a token whose rollout IS weight is zero must
   not influence the loss. Corrupting the log-prob/advantage at those
   positions must leave the scalar loss unchanged. Sequence-level losses
   that fold every response_mask=1 token into a shared importance ratio
   violate this (#8636).
"""

import pytest
import torch

from verl.trainer.config.algorithm import RolloutCorrectionConfig
from verl.trainer.ppo.core_algos import get_policy_loss_fn
from verl.workers.config.actor import ActorConfig, PolicyLossConfig

B, T = 8, 12
RTOL = 1e-5
ATOL = 1e-6


def _tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    old_log_prob = torch.randn(B, T, generator=g) * 0.5 - 1.0
    log_prob = old_log_prob + torch.randn(B, T, generator=g) * 0.1
    advantages = torch.randn(B, T, generator=g)
    response_mask = torch.ones(B, T)
    response_mask[5, 9:] = 0.0
    response_mask[6, 6:] = 0.0
    response_mask[7, 3:] = 0.0
    return old_log_prob, log_prob, advantages, response_mask


def _config(loss_mode: str, rollout_correction: RolloutCorrectionConfig | None = None) -> ActorConfig:
    kwargs = {"loss_mode": loss_mode, "dro_beta": 0.05}
    if rollout_correction is not None:
        kwargs["rollout_correction"] = rollout_correction
    return ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        loss_agg_mode="token-mean",
        policy_loss=PolicyLossConfig(**kwargs),
    )


def _call(loss_mode: str, sl: slice, tensors, weights, rollout_correction=None) -> torch.Tensor:
    old_log_prob, log_prob, advantages, response_mask = tensors
    fn = get_policy_loss_fn(loss_mode)
    cfg = _config(loss_mode, rollout_correction)
    w = weights[sl] if weights is not None else None
    loss, _ = fn(
        old_log_prob[sl],
        log_prob[sl],
        advantages[sl],
        response_mask[sl],
        "token-mean",
        cfg,
        rollout_is_weights=w,
    )
    return loss


# Effective aggregation per loss under our "token-mean" request. SAPO hardcodes
# seq-mean-token-mean internally; geo_mean does torch.mean over batch rows.
_SEQ_AGGREGATED = {"sapo"}
_ROW_AGGREGATED = {"geo_mean"}


def _recombine_microbatches(loss_mode, tensors, weights, n_mb, rollout_correction=None) -> torch.Tensor:
    """Recombine per-microbatch losses the way the effective agg mode accumulates."""
    _, _, _, response_mask = tensors

    def agg_weight(sl: slice) -> torch.Tensor:
        if loss_mode in _ROW_AGGREGATED:
            return torch.tensor(float(sl.stop - sl.start))
        if loss_mode in _SEQ_AGGREGATED:
            return (response_mask[sl].sum(dim=-1) > 0).float().sum()
        return response_mask[sl].sum()

    full = slice(0, B)
    numer = torch.zeros(())
    for i in range(n_mb):
        sl = slice(i * B // n_mb, (i + 1) * B // n_mb)
        torch.manual_seed(0)  # isolate structural (not RNG) divergence for stochastic losses
        numer = numer + _call(loss_mode, sl, tensors, weights, rollout_correction) * agg_weight(sl)
    torch.manual_seed(0)
    loss_full = _call(loss_mode, full, tensors, weights, rollout_correction)
    return loss_full, numer / agg_weight(full)


MICROBATCH_MODES = [
    "vanilla",
    "dppo_tv",
    "dppo_kl",
    "gspo",
    "sapo",
    "gpg",
    "geo_mean",
    "dro",
    "cispo",
    pytest.param(
        "clip_cov",
        marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason="per-microbatch clip token selection, #8634"),
    ),
    pytest.param(
        "kl_cov",
        marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason="per-microbatch cov-token selection, #7829"),
    ),
]


@pytest.mark.parametrize("n_mb", [2, 4])
@pytest.mark.parametrize("loss_mode", MICROBATCH_MODES)
def test_policy_loss_microbatch_invariance(loss_mode, n_mb):
    tensors = _tensors()
    loss_full, loss_recombined = _recombine_microbatches(loss_mode, tensors, weights=None, n_mb=n_mb)
    torch.testing.assert_close(loss_recombined, loss_full, rtol=RTOL, atol=ATOL)


# reinforce loss_type is used so the computed IS weights actually multiply into the
# loss (ppo_clip ignores them, which would hide normalization differences).
BYPASS_RC_BASE = dict(loss_type="reinforce", rollout_is="token", rollout_is_threshold=2.0, bypass_mode=True)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="batch_normalize is computed per microbatch in bypass_mode, see #8634")
def test_bypass_mode_batch_normalize_microbatch_invariant():
    """In bypass_mode the rollout-correction prepass runs inside the per-microbatch
    loss call, so batch_normalize divides by the microbatch mean rather than the
    minibatch mean. Same bug class as #8634; marked xfail pending the fix."""
    # Large log-ratio spread so the normalize factor differs meaningfully across slices.
    g = torch.Generator().manual_seed(0)
    old_log_prob = torch.randn(B, T, generator=g) * 0.5 - 1.0
    log_prob = old_log_prob + torch.randn(B, T, generator=g) * 0.8
    advantages = torch.randn(B, T, generator=g)
    response_mask = torch.ones(B, T)
    tensors = (old_log_prob, log_prob, advantages, response_mask)
    rc = RolloutCorrectionConfig(**BYPASS_RC_BASE, rollout_is_batch_normalize=True)
    loss_full, loss_recombined = _recombine_microbatches("bypass_mode", tensors, None, 2, rc)
    torch.testing.assert_close(loss_recombined, loss_full, rtol=RTOL, atol=ATOL)


def test_bypass_mode_without_batch_normalize_is_microbatch_invariant():
    tensors = _tensors()
    rc = RolloutCorrectionConfig(**BYPASS_RC_BASE, rollout_is_batch_normalize=False)
    loss_full, loss_recombined = _recombine_microbatches("bypass_mode", tensors, None, 2, rc)
    torch.testing.assert_close(loss_recombined, loss_full, rtol=RTOL, atol=ATOL)


REJECT_ISOLATION_MODES = [
    "vanilla",
    "dppo_tv",
    "dppo_kl",
    "gpg",
    "dro",
    "cispo",
    pytest.param(
        "gspo",
        marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason="rejected token pollutes sequence IS ratio, #8636"),
    ),
    "sapo",
    pytest.param(
        "geo_mean",
        marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason="rejected token pollutes sequence IS ratio, #8636"),
    ),
    pytest.param(
        "clip_cov",
        marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason="minibatch statistics include rejected tokens, #8636"),
    ),
    pytest.param(
        "kl_cov",
        marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason="minibatch statistics include rejected tokens, #8636"),
    ),
]


@pytest.mark.parametrize("loss_mode", REJECT_ISOLATION_MODES)
def test_rejected_tokens_do_not_influence_loss(loss_mode):
    """Zero-weight tokens must be fully inert: corrupting their inputs must not change the loss."""
    old_log_prob, log_prob, advantages, response_mask = _tensors()

    weights = torch.ones(B, T)
    rejected = torch.zeros(B, T, dtype=torch.bool)
    rejected[0, 0] = True
    rejected[1, 3] = True
    rejected[3, 5] = True
    weights[rejected] = 0.0

    log_prob_corrupt = log_prob + torch.where(rejected, torch.full((B, T), 8.0), torch.zeros(B, T))
    adv_corrupt = advantages + torch.where(rejected, torch.full((B, T), 50.0), torch.zeros(B, T))

    base = (old_log_prob, log_prob, advantages, response_mask)
    corrupt = (old_log_prob, log_prob_corrupt, adv_corrupt, response_mask)

    torch.manual_seed(0)
    loss_clean = _call(loss_mode, slice(0, B), base, weights)
    torch.manual_seed(0)
    loss_corrupt = _call(loss_mode, slice(0, B), corrupt, weights)
    torch.testing.assert_close(loss_corrupt, loss_clean, rtol=RTOL, atol=ATOL)
