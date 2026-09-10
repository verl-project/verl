# Copyright 2026 Individual Contributor: gss10282025
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

"""Check KL-Cov quotas, valid-token statistics, and the direct-call fallback."""

import pytest
import torch

from verl.trainer.ppo.core_algos import compute_kl_cov_mask, compute_policy_loss_kl_cov
from verl.workers.config import ActorConfig
from verl.workers.config.actor import PolicyLossConfig


def test_quota_is_one_for_8192_tokens_not_one_per_1024_token_microbatch():
    adv = torch.linspace(-2, 3, 8192).reshape(64, 128)
    log_prob = adv.square() * 0.1 - 4
    valid = torch.ones_like(adv, dtype=torch.bool)
    assert compute_kl_cov_mask(adv, log_prob, valid, 0.0002).sum() == 1
    assert (
        sum(
            compute_kl_cov_mask(a, p, m, 0.0002).sum()
            for a, p, m in zip(adv.chunk(8), log_prob.chunk(8), valid.chunk(8), strict=True)
        )
        == 8
    )


@pytest.mark.parametrize("ratio", [0.0002, 0.5, 1.0])
def test_mask_excludes_padding_and_uses_all_valid_statistics(ratio):
    adv = torch.tensor([[0.0, 2.0, 1e6], [-1.0, 3.0, -1e6]])
    log_prob = torch.tensor([[0.3, 1.2, 1e6], [-0.8, 2.6, -1e6]])
    valid = torch.tensor([[True, True, False], [True, True, False]])
    # Transposed tensors also exercise non-contiguous masks.
    for a, p, v in [(adv, log_prob, valid), (adv.T, log_prob.T, valid.T)]:
        actual = compute_kl_cov_mask(a, p, v, ratio)
        score = (a[v] - a[v].mean()) * (p[v] - p[v].mean())
        expected = torch.zeros_like(score, dtype=torch.bool)
        expected[score.argsort(descending=True)[: max(1, int(v.sum() * ratio))]] = True
        torch.testing.assert_close(actual[v], expected)
        assert not actual[~v].any()
    assert not compute_kl_cov_mask(adv, log_prob, torch.zeros_like(valid), ratio).any()


def test_precomputed_and_direct_selection_match_for_one_full_batch():
    generator = torch.Generator().manual_seed(17)
    advantages = torch.randn(3, 7, generator=generator)
    log_prob = torch.randn(3, 7, generator=generator, requires_grad=True)
    old = torch.randn(3, 7, generator=generator)
    valid = torch.ones_like(advantages, dtype=torch.bool)
    config = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        policy_loss=PolicyLossConfig(loss_mode="kl_cov", kl_cov_ratio=0.2),
    )
    direct, _ = compute_policy_loss_kl_cov(old, log_prob, advantages, valid, config=config)
    selected = compute_kl_cov_mask(advantages, log_prob, valid, 0.2)
    prepared, _ = compute_policy_loss_kl_cov(old, log_prob, advantages, valid, config=config, kl_cov_mask=selected)
    torch.testing.assert_close(direct, prepared)
    torch.testing.assert_close(torch.autograd.grad(direct, log_prob)[0], torch.autograd.grad(prepared, log_prob)[0])
