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
"""
Test that agg_loss seq-mean modes produce gradients consistent with token-mean
when using the new engine path (global_batch_size is provided).

The core invariant: the effective loss (after FSDP gradient averaging) should
equal the global mean, regardless of dp_size.

FSDP gradient semantics:
- Each rank computes loss_i.backward() → local gradient g_i
- FSDP all-reduces gradients with AVERAGE: g_avg = sum(g_i) / dp_size
- Effective loss: L_eff = sum(loss_i) / dp_size

For this to yield the correct global mean:
- token-mean: loss_i = local_token_sum / global_num_tokens * dp_size
- seq-mean:   loss_i = local_seq_sum / global_batch_size * dp_size
              = local_seq_sum / (global_batch_size / dp_size)

The bug (before fix): seq-mean used seq_denominator = global_batch_size * dp_size,
yielding loss_i = local_seq_sum / (global_batch_size * dp_size), which is
dp_size^2 times too small.
"""

import torch

from verl.trainer.ppo.core_algos import agg_loss


def simulate_fsdp_effective_loss(per_rank_losses: list[float], dp_size: int) -> float:
    """Simulate the effective loss after FSDP gradient averaging.

    FSDP averages gradients across DP ranks:
        g_avg = sum(g_i) / dp_size

    Since g_i = grad(loss_i), the effective loss is:
        L_eff = sum(loss_i) / dp_size
    """
    return sum(per_rank_losses) / dp_size


class TestAggLossSeqMeanDpScaling:
    """Test that seq-mean modes are invariant to dp_size on the new engine path."""

    def _make_uniform_data(self, batch_size: int, seq_len: int, loss_value: float = 1.0):
        """Create uniform loss matrix and mask for testing."""
        loss_mat = torch.full((batch_size, seq_len), loss_value)
        loss_mask = torch.ones(batch_size, seq_len)
        return loss_mat, loss_mask

    def test_token_mean_dp_invariant(self):
        """Verify token-mean gives consistent effective loss across dp_size values."""
        global_batch_size = 32
        seq_len = 10

        for dp_size in [1, 2, 4, 8]:
            local_bsz = global_batch_size // dp_size
            global_tokens = global_batch_size * seq_len  # all tokens valid

            loss_mat, loss_mask = self._make_uniform_data(local_bsz, seq_len)

            loss = agg_loss(
                loss_mat=loss_mat,
                loss_mask=loss_mask,
                loss_agg_mode="token-mean",
                dp_size=dp_size,
                batch_num_tokens=global_tokens,
            )

            # Simulate FSDP: all dp_size ranks have the same loss value
            per_rank_losses = [loss.item()] * dp_size
            effective = simulate_fsdp_effective_loss(per_rank_losses, dp_size)

            assert abs(effective - 1.0) < 1e-6, (
                f"token-mean effective loss should be 1.0 with dp_size={dp_size}, got {effective}"
            )

    def test_seq_mean_token_mean_dp_invariant(self):
        """Verify seq-mean-token-mean gives consistent effective loss across dp_size values."""
        global_batch_size = 32
        seq_len = 10

        for dp_size in [1, 2, 4, 8]:
            local_bsz = global_batch_size // dp_size

            loss_mat, loss_mask = self._make_uniform_data(local_bsz, seq_len)

            loss = agg_loss(
                loss_mat=loss_mat,
                loss_mask=loss_mask,
                loss_agg_mode="seq-mean-token-mean",
                dp_size=dp_size,
                global_batch_size=global_batch_size,
            )

            per_rank_losses = [loss.item()] * dp_size
            effective = simulate_fsdp_effective_loss(per_rank_losses, dp_size)

            assert abs(effective - 1.0) < 1e-6, (
                f"seq-mean-token-mean effective loss should be 1.0 with dp_size={dp_size}, got {effective}"
            )

    def test_seq_mean_token_sum_dp_invariant(self):
        """Verify seq-mean-token-sum gives consistent effective loss across dp_size values."""
        global_batch_size = 32
        seq_len = 10

        for dp_size in [1, 2, 4, 8]:
            local_bsz = global_batch_size // dp_size

            loss_mat, loss_mask = self._make_uniform_data(local_bsz, seq_len)

            loss = agg_loss(
                loss_mat=loss_mat,
                loss_mask=loss_mask,
                loss_agg_mode="seq-mean-token-sum",
                dp_size=dp_size,
                global_batch_size=global_batch_size,
            )

            # With token-sum, each seq_loss = sum of 10 tokens × 1.0 = 10.0
            # Mean over batch = 10.0
            per_rank_losses = [loss.item()] * dp_size
            effective = simulate_fsdp_effective_loss(per_rank_losses, dp_size)

            assert abs(effective - 10.0) < 1e-5, (
                f"seq-mean-token-sum effective loss should be 10.0 with dp_size={dp_size}, got {effective}"
            )

    def test_seq_mean_token_sum_norm_dp_invariant(self):
        """Verify seq-mean-token-sum-norm gives consistent effective loss across dp_size values."""
        global_batch_size = 32
        seq_len = 10

        for dp_size in [1, 2, 4, 8]:
            local_bsz = global_batch_size // dp_size

            loss_mat, loss_mask = self._make_uniform_data(local_bsz, seq_len)

            loss = agg_loss(
                loss_mat=loss_mat,
                loss_mask=loss_mask,
                loss_agg_mode="seq-mean-token-sum-norm",
                dp_size=dp_size,
                global_batch_size=global_batch_size,
                loss_scale_factor=seq_len,  # normalize by seq_len
            )

            # token-sum-norm: each seq_loss = sum(10) / 10 = 1.0, mean = 1.0
            per_rank_losses = [loss.item()] * dp_size
            effective = simulate_fsdp_effective_loss(per_rank_losses, dp_size)

            assert abs(effective - 1.0) < 1e-6, (
                f"seq-mean-token-sum-norm effective loss should be 1.0 with dp_size={dp_size}, got {effective}"
            )

    def test_seq_mean_matches_token_mean_with_equal_lengths(self):
        """When all sequences have equal lengths, seq-mean-token-mean should equal token-mean."""
        global_batch_size = 32
        seq_len = 10
        dp_size = 4
        local_bsz = global_batch_size // dp_size
        global_tokens = global_batch_size * seq_len

        loss_mat, loss_mask = self._make_uniform_data(local_bsz, seq_len, loss_value=2.5)

        token_mean_loss = agg_loss(
            loss_mat=loss_mat,
            loss_mask=loss_mask,
            loss_agg_mode="token-mean",
            dp_size=dp_size,
            batch_num_tokens=global_tokens,
        )

        seq_mean_loss = agg_loss(
            loss_mat=loss_mat,
            loss_mask=loss_mask,
            loss_agg_mode="seq-mean-token-mean",
            dp_size=dp_size,
            global_batch_size=global_batch_size,
        )

        # Both should give the same effective loss after FSDP averaging
        token_effective = simulate_fsdp_effective_loss([token_mean_loss.item()] * dp_size, dp_size)
        seq_effective = simulate_fsdp_effective_loss([seq_mean_loss.item()] * dp_size, dp_size)

        assert abs(token_effective - seq_effective) < 1e-6, (
            f"token-mean effective={token_effective} should match seq-mean-token-mean effective={seq_effective}"
        )

    def test_legacy_path_still_correct(self):
        """Verify legacy path (global_batch_size=None) still works correctly."""
        local_bsz = 8
        seq_len = 10

        loss_mat, loss_mask = self._make_uniform_data(local_bsz, seq_len)

        # Legacy path: global_batch_size is None, uses local_bsz as denominator
        loss = agg_loss(
            loss_mat=loss_mat,
            loss_mask=loss_mask,
            loss_agg_mode="seq-mean-token-mean",
            # dp_size, global_batch_size not provided → use defaults
        )

        # Each seq has token-mean = 1.0, mean over 8 seqs = 1.0
        assert abs(loss.item() - 1.0) < 1e-6, (
            f"Legacy path seq-mean-token-mean should give 1.0, got {loss.item()}"
        )

    def test_uneven_sequence_lengths(self):
        """Test with varying sequence lengths to verify masking works correctly."""
        dp_size = 4
        global_batch_size = 8  # 2 per GPU
        seq_len = 10
        local_bsz = global_batch_size // dp_size  # 2

        loss_mat = torch.ones(local_bsz, seq_len) * 3.0
        loss_mask = torch.ones(local_bsz, seq_len)
        # Second sequence is shorter: only 5 valid tokens
        loss_mask[1, 5:] = 0

        loss = agg_loss(
            loss_mat=loss_mat,
            loss_mask=loss_mask,
            loss_agg_mode="seq-mean-token-mean",
            dp_size=dp_size,
            global_batch_size=global_batch_size,
        )

        # seq0: token-mean = 3.0 (10 tokens)
        # seq1: token-mean = 3.0 (5 tokens, all 3.0)
        # mean over 2 seqs = 3.0, but denominator is global_batch_size/dp_size = 2
        # loss_on_this_gpu = (3.0 + 3.0) / 2 = 3.0
        per_rank_losses = [loss.item()] * dp_size
        effective = simulate_fsdp_effective_loss(per_rank_losses, dp_size)

        assert abs(effective - 3.0) < 1e-6, (
            f"seq-mean-token-mean with uneven lengths should give 3.0, got {effective}"
        )
