# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
CPU unit tests for agg_loss mathematical correctness (token-mean mode).

No GPU or distributed setup required -- runs on CPU.

Covers:
  - policy-loss-style aggregation with uniform and variable-length masks
  - entropy/KL-style aggregation (same agg_loss path, different toy inputs)
  - the micro-batch splitting invariant: sum of micro-batch agg_loss values
    must equal the full-batch agg_loss when global_batch_info is correct
  - demonstration that without global_batch_info the sum is wrong (#5625)

Usage:
    pytest tests/workers/actor/test_agg_loss_math.py -v
"""

import torch

from verl.trainer.ppo.core_algos import agg_loss


class TestAggLossTokenMean:
    """Verify that agg_loss(token-mean) with global_batch_info produces
    the correct scaled loss for DDP/FSDP gradient reduction."""

    def _random_batch(self, bs, seq_len, seed=0):
        gen = torch.Generator().manual_seed(seed)
        loss_mat = torch.randn(bs, seq_len, generator=gen).abs()
        loss_mask = torch.ones(bs, seq_len)
        return loss_mat, loss_mask

    def test_single_rank_no_splitting(self):
        """With dp_size=1, batch_num_tokens=mask.sum(), should equal plain mean."""
        loss_mat, mask = self._random_batch(4, 8)
        expected = (loss_mat * mask).sum() / mask.sum()
        actual = agg_loss(loss_mat, mask, "token-mean", dp_size=1, batch_num_tokens=int(mask.sum().item()))
        torch.testing.assert_close(actual, expected)

    def test_dp_size_scales_linearly(self):
        """loss should scale linearly with dp_size (pre-compensates DDP averaging)."""
        loss_mat, mask = self._random_batch(4, 8)
        T = int(mask.sum().item())

        loss_dp1 = agg_loss(loss_mat, mask, "token-mean", dp_size=1, batch_num_tokens=T)
        loss_dp4 = agg_loss(loss_mat, mask, "token-mean", dp_size=4, batch_num_tokens=T)
        torch.testing.assert_close(loss_dp4, loss_dp1 * 4)

    def test_microbatch_sum_equals_full_batch(self):
        bs, seq_len, dp_size = 8, 16, 2
        loss_mat, mask = self._random_batch(bs, seq_len)
        T_global = int(mask.sum().item()) * dp_size

        full_loss = agg_loss(loss_mat, mask, "token-mean", dp_size=dp_size, batch_num_tokens=T_global)

        N = 4
        micro_bs = bs // N
        micro_loss_sum = torch.tensor(0.0)
        for i in range(N):
            sl = slice(i * micro_bs, (i + 1) * micro_bs)
            ml = agg_loss(loss_mat[sl], mask[sl], "token-mean", dp_size=dp_size, batch_num_tokens=T_global)
            micro_loss_sum += ml

        torch.testing.assert_close(micro_loss_sum, full_loss, rtol=1e-5, atol=1e-6)

    def test_without_global_info_microbatch_sum_differs(self):
        bs, seq_len = 8, 16
        loss_mat, mask = self._random_batch(bs, seq_len)

        full_loss = agg_loss(loss_mat, mask, "token-mean")

        N = 4
        micro_bs = bs // N
        micro_loss_sum = torch.tensor(0.0)
        for i in range(N):
            sl = slice(i * micro_bs, (i + 1) * micro_bs)
            ml = agg_loss(loss_mat[sl], mask[sl], "token-mean")
            micro_loss_sum += ml

        ratio = (micro_loss_sum / full_loss).item()
        assert abs(ratio - N) < 0.01, f"Expected micro_loss_sum / full_loss ~ {N} (the bug), got {ratio:.4f}"

    def test_uneven_tokens_without_global_info_is_mean_of_means(self):
        bs, seq_len = 8, 16
        gen = torch.Generator().manual_seed(7)
        loss_mat = torch.randn(bs, seq_len, generator=gen).abs()

        mask = torch.ones(bs, seq_len)
        mask[0, 8:] = 0
        mask[1, 12:] = 0

        full_loss = agg_loss(loss_mat, mask, "token-mean")

        N = 4
        micro_bs = bs // N
        micro_loss_sum = torch.tensor(0.0)
        for i in range(N):
            sl = slice(i * micro_bs, (i + 1) * micro_bs)
            ml = agg_loss(loss_mat[sl], mask[sl], "token-mean")
            micro_loss_sum += ml

        ratio = (micro_loss_sum / full_loss).item()
        assert abs(ratio - N) > 0.05, (
            f"With uneven masks the ratio should NOT be exactly {N}, "
            f"got {ratio:.4f} -- mean-of-means distortion expected"
        )

    def test_uneven_tokens_with_global_info_still_exact(self):
        bs, seq_len, dp_size = 8, 16, 2
        gen = torch.Generator().manual_seed(7)
        loss_mat = torch.randn(bs, seq_len, generator=gen).abs()

        mask = torch.ones(bs, seq_len)
        mask[0, 8:] = 0
        mask[1, 12:] = 0
        T_global = int(mask.sum().item()) * dp_size

        full_loss = agg_loss(loss_mat, mask, "token-mean", dp_size=dp_size, batch_num_tokens=T_global)

        N = 4
        micro_bs = bs // N
        micro_loss_sum = torch.tensor(0.0)
        for i in range(N):
            sl = slice(i * micro_bs, (i + 1) * micro_bs)
            ml = agg_loss(loss_mat[sl], mask[sl], "token-mean", dp_size=dp_size, batch_num_tokens=T_global)
            micro_loss_sum += ml

        torch.testing.assert_close(micro_loss_sum, full_loss, rtol=1e-5, atol=1e-6)


class TestAggLossEntropyKL:

    def test_entropy_like_microbatch_invariant(self):
        bs, seq_len, dp_size = 8, 16, 2
        gen = torch.Generator().manual_seed(99)
        entropy_mat = torch.randn(bs, seq_len, generator=gen).abs()

        mask = torch.ones(bs, seq_len)
        mask[0, 10:] = 0
        mask[3, 6:] = 0
        T_global = int(mask.sum().item()) * dp_size

        full = agg_loss(entropy_mat, mask, "token-mean", dp_size=dp_size, batch_num_tokens=T_global)

        N = 4
        micro_bs = bs // N
        partial = torch.tensor(0.0)
        for i in range(N):
            sl = slice(i * micro_bs, (i + 1) * micro_bs)
            partial += agg_loss(entropy_mat[sl], mask[sl], "token-mean", dp_size=dp_size, batch_num_tokens=T_global)

        torch.testing.assert_close(partial, full, rtol=1e-5, atol=1e-6)

    def test_kl_like_microbatch_invariant(self):
        bs, seq_len, dp_size = 8, 16, 2
        gen = torch.Generator().manual_seed(77)
        kl_mat = torch.randn(bs, seq_len, generator=gen) * 0.5

        mask = torch.ones(bs, seq_len)
        mask[1, 12:] = 0
        mask[2, 4:] = 0
        T_global = int(mask.sum().item()) * dp_size

        full = agg_loss(kl_mat, mask, "token-mean", dp_size=dp_size, batch_num_tokens=T_global)

        N = 4
        micro_bs = bs // N
        partial = torch.tensor(0.0)
        for i in range(N):
            sl = slice(i * micro_bs, (i + 1) * micro_bs)
            partial += agg_loss(kl_mat[sl], mask[sl], "token-mean", dp_size=dp_size, batch_num_tokens=T_global)

        torch.testing.assert_close(partial, full, rtol=1e-5, atol=1e-6)
