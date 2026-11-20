# Copyright 2026 Individual Contributor: LOGO127
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

"""Check Clip-Cov quotas, valid-token statistics, and the direct-call fallback."""

import pytest
import torch

from verl.trainer.ppo.core_algos import compute_clip_cov_mask, compute_policy_loss_clip_cov
from verl.workers.config import ActorConfig
from verl.workers.config.actor import PolicyLossConfig

ARGS = dict(clip_cov_lb=1.0, clip_cov_ub=5.0, cliprange_low=0.2, cliprange_high=0.2)


def test_quota_is_one_for_8192_tokens_not_one_per_1024_token_microbatch():
    # adv = +-8, log_prob = +-0.15 positively correlated -> covariance = +1.2
    # everywhere (all eligible); ratio = e^0.15 stays inside the clip range so
    # nothing is origin-clipped either.
    old = torch.zeros(64, 128)
    adv = torch.where(torch.arange(8192).reshape(64, 128) % 2 == 0, 8.0, -8.0)
    log_prob = torch.sign(adv) * 0.15
    valid = torch.ones_like(adv, dtype=torch.bool)
    assert compute_clip_cov_mask(old, log_prob, adv, valid, 0.0002, **ARGS).sum() == 1
    assert (
        sum(
            compute_clip_cov_mask(o, p, a, m, 0.0002, **ARGS).sum()
            for o, p, a, m in zip(old.chunk(8), log_prob.chunk(8), adv.chunk(8), valid.chunk(8), strict=True)
        )
        == 8
    )


@pytest.mark.parametrize("ratio", [0.0002, 0.5, 1.0])
def test_mask_excludes_padding_and_origin_clipped(ratio):
    # All six valid tokens are covariance-eligible (cov = +1.2): advantages and
    # log-probs share the same +-pattern with zero means. Token (1,0) is clipped
    # by the vanilla PPO term (adv > 0, ratio = e^5.15 above clip_ratio_high) and
    # so must be excluded even though its covariance qualifies; (0,2) and (0,3)
    # are padding and must never be selected.
    old = torch.zeros(2, 4)
    old[1, 0] = -5.0
    log_prob = torch.tensor([[0.15, -0.15, 1e6, 1e6], [0.15, -0.15, 0.15, -0.15]])
    adv = torch.tensor([[8.0, -8.0, 1e6, 1e6], [8.0, -8.0, 8.0, -8.0]])
    valid = torch.tensor([[True, True, False, False], [True, True, True, True]])
    clipped = torch.zeros_like(valid)
    clipped[1, 0] = True

    for o, p, a, v, c in [
        (old, log_prob, adv, valid, clipped),
        (old.T, log_prob.T, adv.T, valid.T, clipped.T),
    ]:
        actual = compute_clip_cov_mask(o, p, a, v, ratio, **ARGS)
        assert not actual[~v].any()
        assert not actual[c].any()
        if ratio == 1.0:
            assert actual[v & ~c].all()

    # values in padding positions must not change which valid tokens are selected
    junk_adv = adv.clone()
    junk_adv[~valid] = -1e9
    junk_lp = log_prob.clone()
    junk_lp[~valid] = 1e9
    assert torch.equal(
        compute_clip_cov_mask(old, junk_lp, junk_adv, valid, ratio, **ARGS),
        compute_clip_cov_mask(old, log_prob, adv, valid, ratio, **ARGS),
    )
    assert not compute_clip_cov_mask(old, log_prob, adv, torch.zeros_like(valid), ratio, **ARGS).any()


def test_quota_counts_only_eligible_tokens():
    # 6 positive-advantage tokens (cov = +1.2, eligible) and 2 negative ones
    # (cov = +10.8 > ub, ineligible). old_log_prob keeps every ratio at e^0.15
    # so eligibility is decided by the covariance range alone.
    adv = torch.tensor([[8.0, 8.0, 8.0, -8.0], [-8.0, 8.0, 8.0, 8.0]])
    log_prob = torch.tensor([[1.2, 1.2, 1.2, 0.0], [0.0, 1.2, 1.2, 1.2]])
    old = log_prob - 0.15
    valid = torch.ones_like(adv, dtype=torch.bool)
    mask = compute_clip_cov_mask(old, log_prob, adv, valid, 0.5, **ARGS)
    assert mask.sum() == 4  # quota = int(8 * 0.5), below the 6 eligible tokens
    assert mask.dtype == torch.bool
    assert mask[~(adv > 0)].sum() == 0  # never selects ineligible tokens
    assert compute_clip_cov_mask(old, log_prob, adv, valid, 1.0, **ARGS).sum() == 6


def test_selection_is_deterministic_across_calls():
    generator = torch.Generator().manual_seed(3)
    old = torch.randn(4, 16, generator=generator)
    log_prob = torch.randn(4, 16, generator=generator)
    adv = torch.randn(4, 16, generator=generator)
    valid = torch.ones_like(adv, dtype=torch.bool)
    m1 = compute_clip_cov_mask(old, log_prob, adv, valid, 0.1, **ARGS)
    m2 = compute_clip_cov_mask(old, log_prob, adv, valid, 0.1, **ARGS)
    torch.testing.assert_close(m1, m2)


def test_precomputed_and_direct_selection_match_for_one_full_batch():
    generator = torch.Generator().manual_seed(17)
    old = torch.randn(3, 7, generator=generator)
    log_prob = torch.randn(3, 7, generator=generator, requires_grad=True)
    advantages = torch.randn(3, 7, generator=generator)
    valid = torch.ones_like(advantages, dtype=torch.bool)
    config = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        policy_loss=PolicyLossConfig(loss_mode="clip_cov", clip_cov_ratio=0.2),
    )
    direct, _ = compute_policy_loss_clip_cov(old, log_prob, advantages, valid, config=config)
    selected = compute_clip_cov_mask(
        old, log_prob, advantages, valid, 0.2, clip_cov_lb=1.0, clip_cov_ub=5.0, cliprange_low=0.2, cliprange_high=0.2
    )
    prepared, _ = compute_policy_loss_clip_cov(old, log_prob, advantages, valid, config=config, clip_cov_mask=selected)
    torch.testing.assert_close(direct, prepared)
    torch.testing.assert_close(torch.autograd.grad(direct, log_prob)[0], torch.autograd.grad(prepared, log_prob)[0])


def test_per_microbatch_selection_diverges_from_minibatch_selection():
    # The regression being fixed: selecting independently on each microbatch
    # picks a different set (and a different count) than one minibatch-level
    # selection sliced per microbatch.
    # Deterministic data: all tokens eligible -> both selections hit the quota,
    # but the picked sets differ (minibatch picks 12 of 256, each microbatch
    # picks 3 of its own 64).
    old = torch.zeros(8, 32)
    adv = torch.where(torch.arange(256).reshape(8, 32) % 2 == 0, 8.0, -8.0)
    log_prob = torch.sign(adv) * 0.15
    valid = torch.ones_like(adv, dtype=torch.bool)
    full = compute_clip_cov_mask(old, log_prob, adv, valid, 0.05, **ARGS)
    micro = torch.cat(
        [
            compute_clip_cov_mask(o, p, a, m, 0.05, **ARGS)
            for o, p, a, m in zip(old.chunk(4), log_prob.chunk(4), adv.chunk(4), valid.chunk(4), strict=True)
        ]
    )
    # per-microbatch quotas: max(1, int(0.05 * 64)) * 4 = 12 vs int(0.05 * 256) = 12
    # -> the count matches here by accident, but eligibility statistics and the
    # fixed-seed picks differ, so the selected sets must differ.
    assert full.sum() == micro.sum()
    assert not torch.equal(full, micro)
