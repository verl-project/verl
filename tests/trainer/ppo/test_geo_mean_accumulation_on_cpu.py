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

"""The optimizer-minibatch GMPO objective must survive microbatch splitting."""

from types import SimpleNamespace

import pytest
import torch

from verl.trainer.ppo.core_algos import compute_policy_loss_geo_mean


def _inputs():
    generator = torch.Generator().manual_seed(41)
    old_logp = torch.randn(8, 5, generator=generator, dtype=torch.float64) * 0.1
    logp = old_logp + torch.randn(8, 5, generator=generator, dtype=torch.float64) * 0.2
    advantages = torch.tensor([1, -1, 0.5, -0.5, 2, -2, 0.25, -0.25], dtype=torch.float64)
    lengths = torch.tensor([5, 3, 4, 2, 1, 5, 0, 4])
    mask = (torch.arange(5)[None, :] < lengths[:, None]).double()
    return old_logp, logp, advantages[:, None].expand_as(logp), mask


def _config(global_batch_size=8, dp_size=1):
    return SimpleNamespace(
        clip_ratio=0.4,
        clip_ratio_low=None,
        clip_ratio_high=None,
        global_batch_info={"global_batch_size": global_batch_size, "dp_size": dp_size},
    )


@pytest.mark.parametrize("sizes", [(8,), (1,) * 8, (3, 1, 4), (7, 1)])
@pytest.mark.parametrize("use_rollout_weights", [False, True])
def test_geo_mean_microbatch_loss_and_gradient_match_full_batch(sizes, use_rollout_weights):
    """Compare full and uneven accumulated calls, including a fully masked row."""
    old, initial, advantages, mask = _inputs()
    weights = torch.linspace(0.8, 1.2, initial.numel()).reshape_as(initial) if use_rollout_weights else None
    full = initial.clone().requires_grad_()
    full_loss, _ = compute_policy_loss_geo_mean(
        old, full, advantages, mask, config=_config(), rollout_is_weights=weights
    )
    full_gradient = torch.autograd.grad(full_loss, full)[0]

    split = initial.clone().requires_grad_()
    losses = []
    start = 0
    for size in sizes:
        part = slice(start, start + size)
        loss, _ = compute_policy_loss_geo_mean(
            old[part],
            split[part],
            advantages[part],
            mask[part],
            config=_config(),
            rollout_is_weights=None if weights is None else weights[part],
        )
        losses.append(loss)
        start += size
    split_loss = torch.stack(losses).sum()
    split_gradient = torch.autograd.grad(split_loss, split)[0]
    torch.testing.assert_close(split_loss, full_loss, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(split_gradient, full_gradient, rtol=1e-12, atol=1e-12)


def test_geo_mean_dp_average_matches_full_batch_with_uneven_rank_batches():
    """Model DP's gradient average after differently sized rank-local batches."""
    old, initial, advantages, mask = _inputs()
    full = initial.clone().requires_grad_()
    expected, _ = compute_policy_loss_geo_mean(old, full, advantages, mask, config=_config())
    expected_gradient = torch.autograd.grad(expected, full)[0]
    split = initial.clone().requires_grad_()
    losses = []
    for part in [slice(0, 2), slice(2, 3), slice(3, 5), slice(5, 8)]:
        loss, _ = compute_policy_loss_geo_mean(
            old[part], split[part], advantages[part], mask[part], config=_config(dp_size=2)
        )
        losses.append(loss)
    averaged = torch.stack(losses).sum() / 2
    actual_gradient = torch.autograd.grad(averaged, split)[0]
    torch.testing.assert_close(averaged, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("global_batch_size,dp_size", [(None, 1), (0, 1), (8, 0)])
def test_geo_mean_rejects_missing_or_invalid_minibatch_metadata(global_batch_size, dp_size):
    """Do not silently restore the local-mean bug when metadata is missing."""
    old, logp, advantages, mask = _inputs()
    with pytest.raises(ValueError, match="geo_mean requires"):
        compute_policy_loss_geo_mean(old, logp, advantages, mask, config=_config(global_batch_size, dp_size))
