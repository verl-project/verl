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

"""CPU coverage for loss aggregation modes."""

import pytest
import torch

from verl.trainer.ppo.core_algos import agg_loss


@pytest.mark.parametrize("dp_size", [1, 2, 4])
@pytest.mark.parametrize("micro_batch_size", [1, 2])
def test_token_mean_ppo_tis_gradient_is_partition_invariant(dp_size, micro_batch_size):
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
    from verl.workers.config.actor import ActorConfig

    # Unequal lengths, mixed advantages and non-unit IS weights expose both
    # local-token normalization and accidentally dropped correction weights.
    mask = torch.arange(8)[None, :] < torch.tensor([1, 8, 2, 7, 3, 6, 4, 5])[:, None]
    features = torch.linspace(-2.0, 2.0, 64).reshape(8, 8)
    advantages = torch.tensor([-1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0])[:, None].expand(8, 8)
    weights = torch.linspace(0.3, 2.0, 64).reshape(8, 8)

    def gradient(partitioned):
        theta = torch.tensor(0.17, requires_grad=True)
        config = ActorConfig(
            strategy="megatron",
            rollout_n=1,
            ppo_micro_batch_size_per_gpu=1,
            clip_ratio_low=0.2,
            clip_ratio_high=0.28,
            clip_ratio_c=10.0,
            global_batch_info={"dp_size": dp_size if partitioned else 1, "batch_num_tokens": int(mask.sum())},
        )
        rank_size = 8 // dp_size
        partitions = (
            [slice(0, 8)]
            if not partitioned
            else [
                slice(start, min(start + micro_batch_size, rank_start + rank_size))
                for rank_start in range(0, 8, rank_size)
                for start in range(rank_start, rank_start + rank_size, micro_batch_size)
            ]
        )
        loss = sum(
            compute_policy_loss_vanilla(
                torch.zeros_like(features[part]),
                theta * features[part],
                advantages[part],
                mask[part],
                "token-mean",
                config,
                rollout_is_weights=weights[part],
            )[0]
            for part in partitions
        )
        # Simulate the DP mean reduction after accumulating local microbatches.
        if partitioned:
            loss = loss / dp_size
        return torch.autograd.grad(loss, theta)[0]

    torch.testing.assert_close(gradient(True), gradient(False))


def test_token_sum_masks_tokens_and_scales_for_dp():
    loss_mat = torch.tensor([[1.0, 2.0, 30.0], [4.0, 50.0, 6.0]])
    loss_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])

    loss = agg_loss(loss_mat, loss_mask, loss_agg_mode="token-sum", dp_size=4)

    assert loss.item() == pytest.approx((1.0 + 2.0 + 4.0 + 6.0) * 4)


@pytest.mark.parametrize("num_micro_batches", [1, 2, 4])
def test_token_sum_is_microbatch_invariant(num_micro_batches):
    loss_mat = torch.arange(1, 25, dtype=torch.float32).reshape(8, 3)
    loss_mask = torch.tensor([[1.0, 1.0, 0.0]] * 8)
    step = loss_mat.shape[0] // num_micro_batches

    accumulated = sum(
        agg_loss(loss_mat[i : i + step], loss_mask[i : i + step], loss_agg_mode="token-sum")
        for i in range(0, loss_mat.shape[0], step)
    )
    whole = agg_loss(loss_mat, loss_mask, loss_agg_mode="token-sum")

    torch.testing.assert_close(accumulated, whole)


@pytest.mark.parametrize("dp_size", [2, 4])
def test_token_sum_matches_global_sum_after_fsdp_mean_reduction(dp_size):
    loss_mat = torch.arange(1, 25, dtype=torch.float32).reshape(8, 3)
    loss_mask = torch.tensor([[1.0, 1.0, 0.0]] * 8)
    rank_step = loss_mat.shape[0] // dp_size

    rank_losses = [
        agg_loss(
            loss_mat[i : i + rank_step],
            loss_mask[i : i + rank_step],
            loss_agg_mode="token-sum",
            dp_size=dp_size,
        )
        for i in range(0, loss_mat.shape[0], rank_step)
    ]
    fsdp_reduced = torch.stack(rank_losses).mean()
    global_sum = agg_loss(loss_mat, loss_mask, loss_agg_mode="token-sum")

    torch.testing.assert_close(fsdp_reduced, global_sum)
