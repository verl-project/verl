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

import pytest
import torch

from verl.trainer.ppo.core_algos import agg_loss


def _drgrpo_oracle(loss_mat: torch.Tensor, loss_mask: torch.Tensor, normalizer: int, batch_size: int):
    seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
    seq_mask = torch.sum(loss_mask, dim=-1) > 0
    return torch.sum(seq_losses[seq_mask]) / (batch_size * normalizer)


def test_seq_mean_token_sum_norm_matches_single_dp_oracle():
    loss_mat = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ]
    )
    loss_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )

    loss = agg_loss(loss_mat, loss_mask, loss_agg_mode="seq-mean-token-sum-norm")
    expected = _drgrpo_oracle(loss_mat, loss_mask, normalizer=loss_mask.shape[-1], batch_size=3)

    torch.testing.assert_close(loss, expected)


def test_seq_mean_token_sum_norm_respects_loss_scale_factor():
    loss_mat = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ]
    )
    loss_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    loss_scale_factor = 10

    loss = agg_loss(
        loss_mat,
        loss_mask,
        loss_agg_mode="seq-mean-token-sum-norm",
        loss_scale_factor=loss_scale_factor,
    )
    expected = _drgrpo_oracle(loss_mat, loss_mask, normalizer=loss_scale_factor, batch_size=3)

    torch.testing.assert_close(loss, expected)


def test_seq_mean_token_sum_norm_gradient_scale():
    loss_mat = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ],
        requires_grad=True,
    )
    loss_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    loss_scale_factor = 8

    loss = agg_loss(
        loss_mat,
        loss_mask,
        loss_agg_mode="seq-mean-token-sum-norm",
        loss_scale_factor=loss_scale_factor,
    )
    loss.backward()

    expected_grad = loss_mask / (loss_mask.shape[0] * loss_scale_factor)
    torch.testing.assert_close(loss_mat.grad, expected_grad)


def test_seq_mean_token_sum_norm_dp_average_matches_full_batch():
    full_loss_mat = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
        requires_grad=True,
    )
    full_loss_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ]
    )
    loss_scale_factor = 6

    full_loss = agg_loss(
        full_loss_mat,
        full_loss_mask,
        loss_agg_mode="seq-mean-token-sum-norm",
        loss_scale_factor=loss_scale_factor,
    )
    full_loss.backward()

    rank0_loss_mat = full_loss_mat[:2].detach().clone().requires_grad_(True)
    rank1_loss_mat = full_loss_mat[2:].detach().clone().requires_grad_(True)
    rank0_loss_mask = full_loss_mask[:2]
    rank1_loss_mask = full_loss_mask[2:]

    rank0_loss = agg_loss(
        rank0_loss_mat,
        rank0_loss_mask,
        loss_agg_mode="seq-mean-token-sum-norm",
        dp_size=2,
        global_batch_size=full_loss_mask.shape[0],
        loss_scale_factor=loss_scale_factor,
    )
    rank1_loss = agg_loss(
        rank1_loss_mat,
        rank1_loss_mask,
        loss_agg_mode="seq-mean-token-sum-norm",
        dp_size=2,
        global_batch_size=full_loss_mask.shape[0],
        loss_scale_factor=loss_scale_factor,
    )
    averaged_rank_loss = (rank0_loss + rank1_loss) / 2
    averaged_rank_loss.backward()

    torch.testing.assert_close(averaged_rank_loss, full_loss.detach())
    rank_grad = torch.cat([rank0_loss_mat.grad, rank1_loss_mat.grad], dim=0)
    torch.testing.assert_close(rank_grad, full_loss_mat.grad)


def test_seq_mean_token_sum_norm_rejects_missing_global_batch_for_dp():
    loss_mat = torch.ones(2, 4)
    loss_mask = torch.ones(2, 4)

    with pytest.raises(ValueError, match="global_batch_size is required"):
        agg_loss(loss_mat, loss_mask, loss_agg_mode="seq-mean-token-sum-norm", dp_size=2)


def test_seq_mean_token_sum_norm_empty_rows_are_explicit():
    loss_mat = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    loss_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    loss_scale_factor = 3

    fallback_loss = agg_loss(
        loss_mat,
        loss_mask,
        loss_agg_mode="seq-mean-token-sum-norm",
        loss_scale_factor=loss_scale_factor,
    )
    explicit_global_loss = agg_loss(
        loss_mat,
        loss_mask,
        loss_agg_mode="seq-mean-token-sum-norm",
        global_batch_size=loss_mask.shape[0],
        loss_scale_factor=loss_scale_factor,
    )

    non_empty_row_count = 2
    fallback_expected = _drgrpo_oracle(
        loss_mat,
        loss_mask,
        normalizer=loss_scale_factor,
        batch_size=non_empty_row_count,
    )
    explicit_expected = _drgrpo_oracle(
        loss_mat,
        loss_mask,
        normalizer=loss_scale_factor,
        batch_size=loss_mask.shape[0],
    )

    torch.testing.assert_close(fallback_loss, fallback_expected)
    torch.testing.assert_close(explicit_global_loss, explicit_expected)
    assert not torch.isclose(fallback_loss, explicit_global_loss)
