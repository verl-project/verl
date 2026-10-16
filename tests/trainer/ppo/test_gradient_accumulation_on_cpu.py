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
Tests for gradient accumulation correctness across different loss aggregation modes.

This module tests that gradient accumulation produces mathematically equivalent results
to processing the entire mini-batch at once, for all supported loss aggregation modes.

Reference: Issue #907 - Tests for gradient accumulation
"""

import pytest
import torch

from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss_vanilla
from verl.utils.torch_functional import masked_mean, masked_sum


# All supported loss aggregation modes
LOSS_AGG_MODES = [
    "token-mean",
    "seq-mean-token-sum",
    "seq-mean-token-mean",
    "seq-mean-token-sum-norm",
]


def create_mock_batch(batch_size: int, seq_len: int, seed: int = 42) -> dict:
    """Create a mock batch of data for testing.

    Args:
        batch_size: Number of sequences in the batch
        seq_len: Sequence length (response length)
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing mock batch data
    """
    torch.manual_seed(seed)

    # Create response mask with at least one valid token per sequence
    response_mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.float32)
    # Ensure at least one valid token per sequence
    rows_without_one = (response_mask.sum(dim=-1) == 0).nonzero(as_tuple=True)[0]
    if len(rows_without_one) > 0:
        response_mask[rows_without_one, -1] = 1.0

    # Create log probabilities and advantages
    log_prob = torch.randn(batch_size, seq_len)
    old_log_prob = torch.randn(batch_size, seq_len)
    advantages = torch.randn(batch_size, seq_len)

    return {
        "log_prob": log_prob,
        "old_log_prob": old_log_prob,
        "advantages": advantages,
        "response_mask": response_mask,
    }


def compute_loss_without_accumulation(
    batch: dict, loss_agg_mode: str, clip_ratio: float = 0.2
) -> torch.Tensor:
    """Compute policy loss for the entire batch without gradient accumulation.

    Args:
        batch: Dictionary containing batch data
        loss_agg_mode: Loss aggregation mode
        clip_ratio: PPO clipping ratio

    Returns:
        Scalar loss tensor
    """
    log_prob = batch["log_prob"]
    old_log_prob = batch["old_log_prob"]
    advantages = batch["advantages"]
    response_mask = batch["response_mask"]

    # Compute policy loss using vanilla PPO
    ratio = torch.exp(log_prob - old_log_prob)
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # Aggregate loss
    loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
    )

    return loss


def compute_loss_with_accumulation(
    batch: dict, loss_agg_mode: str, num_micro_batches: int, clip_ratio: float = 0.2
) -> torch.Tensor:
    """Compute policy loss with gradient accumulation over micro-batches.

    This simulates the gradient accumulation logic in dp_actor.py where:
    - loss_scale_factor = 1 / gradient_accumulation (for non-dynamic batch)
    - Each micro-batch loss is scaled by this factor before accumulation

    Args:
        batch: Dictionary containing batch data
        loss_agg_mode: Loss aggregation mode
        num_micro_batches: Number of micro-batches to split into
        clip_ratio: PPO clipping ratio

    Returns:
        Accumulated scalar loss tensor
    """
    batch_size = batch["log_prob"].shape[0]
    micro_batch_size = batch_size // num_micro_batches

    accumulated_loss = torch.tensor(0.0)
    loss_scale_factor = 1.0 / num_micro_batches

    for i in range(num_micro_batches):
        start_idx = i * micro_batch_size
        end_idx = (i + 1) * micro_batch_size

        # Slice micro-batch
        micro_log_prob = batch["log_prob"][start_idx:end_idx]
        micro_old_log_prob = batch["old_log_prob"][start_idx:end_idx]
        micro_advantages = batch["advantages"][start_idx:end_idx]
        micro_response_mask = batch["response_mask"][start_idx:end_idx]

        # Compute policy loss for micro-batch
        ratio = torch.exp(micro_log_prob - micro_old_log_prob)
        pg_losses1 = -micro_advantages * ratio
        pg_losses2 = -micro_advantages * torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        pg_losses = torch.maximum(pg_losses1, pg_losses2)

        # Aggregate loss for micro-batch
        micro_loss = agg_loss(
            loss_mat=pg_losses,
            loss_mask=micro_response_mask,
            loss_agg_mode=loss_agg_mode,
        )

        # Scale and accumulate
        accumulated_loss = accumulated_loss + micro_loss * loss_scale_factor

    return accumulated_loss


def compute_loss_with_token_weighted_accumulation(
    batch: dict, loss_agg_mode: str, num_micro_batches: int, clip_ratio: float = 0.2
) -> torch.Tensor:
    """Compute policy loss with token-weighted gradient accumulation.

    For token-mean mode, the correct accumulation requires weighting each
    micro-batch loss by the proportion of valid tokens it contains.

    Args:
        batch: Dictionary containing batch data
        loss_agg_mode: Loss aggregation mode
        num_micro_batches: Number of micro-batches to split into
        clip_ratio: PPO clipping ratio

    Returns:
        Accumulated scalar loss tensor
    """
    batch_size = batch["log_prob"].shape[0]
    micro_batch_size = batch_size // num_micro_batches
    total_tokens = batch["response_mask"].sum().item()

    accumulated_loss = torch.tensor(0.0)

    for i in range(num_micro_batches):
        start_idx = i * micro_batch_size
        end_idx = (i + 1) * micro_batch_size

        # Slice micro-batch
        micro_log_prob = batch["log_prob"][start_idx:end_idx]
        micro_old_log_prob = batch["old_log_prob"][start_idx:end_idx]
        micro_advantages = batch["advantages"][start_idx:end_idx]
        micro_response_mask = batch["response_mask"][start_idx:end_idx]

        # Compute policy loss for micro-batch
        ratio = torch.exp(micro_log_prob - micro_old_log_prob)
        pg_losses1 = -micro_advantages * ratio
        pg_losses2 = -micro_advantages * torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        pg_losses = torch.maximum(pg_losses1, pg_losses2)

        micro_tokens = micro_response_mask.sum().item()

        if loss_agg_mode == "token-mean":
            # For token-mean, weight by token proportion
            token_weight = micro_tokens / total_tokens
            micro_loss = agg_loss(
                loss_mat=pg_losses,
                loss_mask=micro_response_mask,
                loss_agg_mode=loss_agg_mode,
            )
            accumulated_loss = accumulated_loss + micro_loss * token_weight
        else:
            # For sequence-based modes, weight by batch size proportion
            batch_weight = micro_batch_size / batch_size
            micro_loss = agg_loss(
                loss_mat=pg_losses,
                loss_mask=micro_response_mask,
                loss_agg_mode=loss_agg_mode,
            )
            accumulated_loss = accumulated_loss + micro_loss * batch_weight

    return accumulated_loss


class TestGradientAccumulationLossEquivalence:
    """Test that gradient accumulation produces equivalent losses."""

    @pytest.mark.parametrize("loss_agg_mode", LOSS_AGG_MODES)
    @pytest.mark.parametrize("batch_size", [8, 16, 32])
    @pytest.mark.parametrize("num_micro_batches", [2, 4])
    def test_token_weighted_accumulation_equivalence(
        self, loss_agg_mode: str, batch_size: int, num_micro_batches: int
    ):
        """Test that token-weighted accumulation matches full batch loss.

        This tests the corrected gradient accumulation logic that properly
        weights micro-batch losses based on token counts (for token-mean)
        or batch sizes (for sequence-based modes).
        """
        if batch_size % num_micro_batches != 0:
            pytest.skip("batch_size must be divisible by num_micro_batches")

        batch = create_mock_batch(batch_size=batch_size, seq_len=64, seed=42)

        # Compute loss without accumulation (full batch)
        loss_full = compute_loss_without_accumulation(batch, loss_agg_mode)

        # Compute loss with token-weighted accumulation
        loss_accum = compute_loss_with_token_weighted_accumulation(
            batch, loss_agg_mode, num_micro_batches
        )

        # Assert losses are equal within tolerance
        torch.testing.assert_close(
            loss_accum,
            loss_full,
            rtol=1e-5,
            atol=1e-6,
            msg=f"Loss mismatch for {loss_agg_mode} with {num_micro_batches} micro-batches",
        )

    @pytest.mark.parametrize("loss_agg_mode", ["seq-mean-token-sum", "seq-mean-token-mean"])
    @pytest.mark.parametrize("batch_size", [8, 16])
    @pytest.mark.parametrize("num_micro_batches", [2, 4])
    def test_sequence_based_simple_accumulation(
        self, loss_agg_mode: str, batch_size: int, num_micro_batches: int
    ):
        """Test that sequence-based modes work with simple batch-weighted accumulation.

        For sequence-mean based modes, simple averaging of micro-batch losses
        (weighted by batch size) should produce correct results.
        """
        if batch_size % num_micro_batches != 0:
            pytest.skip("batch_size must be divisible by num_micro_batches")

        batch = create_mock_batch(batch_size=batch_size, seq_len=64, seed=42)

        # Compute loss without accumulation
        loss_full = compute_loss_without_accumulation(batch, loss_agg_mode)

        # Compute loss with simple accumulation
        loss_accum = compute_loss_with_accumulation(batch, loss_agg_mode, num_micro_batches)

        # Assert losses are equal within tolerance
        torch.testing.assert_close(
            loss_accum,
            loss_full,
            rtol=1e-5,
            atol=1e-6,
            msg=f"Loss mismatch for {loss_agg_mode} with simple accumulation",
        )


class TestAggLossConsistency:
    """Test agg_loss function consistency properties."""

    @pytest.mark.parametrize("loss_agg_mode", LOSS_AGG_MODES)
    def test_agg_loss_deterministic(self, loss_agg_mode: str):
        """Test that agg_loss produces deterministic results."""
        torch.manual_seed(42)
        loss_mat = torch.randn(8, 64)
        loss_mask = torch.randint(0, 2, (8, 64), dtype=torch.float32)
        loss_mask[loss_mask.sum(dim=-1) == 0, -1] = 1.0

        loss1 = agg_loss(loss_mat, loss_mask, loss_agg_mode)
        loss2 = agg_loss(loss_mat, loss_mask, loss_agg_mode)

        assert torch.equal(loss1, loss2), "agg_loss should be deterministic"

    @pytest.mark.parametrize("loss_agg_mode", LOSS_AGG_MODES)
    def test_agg_loss_mask_zero_contribution(self, loss_agg_mode: str):
        """Test that masked positions don't contribute to loss."""
        torch.manual_seed(42)
        loss_mat = torch.randn(8, 64)
        loss_mask = torch.zeros(8, 64)
        loss_mask[:, -1] = 1.0  # Only last token is valid

        # Modify masked positions - should not affect loss
        loss_mat_modified = loss_mat.clone()
        loss_mat_modified[:, :-1] = loss_mat[:, :-1] * 100

        loss1 = agg_loss(loss_mat, loss_mask, loss_agg_mode)
        loss2 = agg_loss(loss_mat_modified, loss_mask, loss_agg_mode)

        torch.testing.assert_close(
            loss1,
            loss2,
            rtol=1e-5,
            atol=1e-6,
            msg="Masked positions should not affect loss",
        )


class TestGradientAccumulationGradients:
    """Test that gradients are correctly accumulated."""

    @pytest.mark.parametrize("loss_agg_mode", LOSS_AGG_MODES)
    @pytest.mark.parametrize("num_micro_batches", [2, 4])
    def test_gradient_accumulation_equivalence(self, loss_agg_mode: str, num_micro_batches: int):
        """Test that accumulated gradients match full batch gradients.

        This test creates a simple linear model and verifies that gradients
        computed with accumulation match those from processing the full batch.
        """
        batch_size = 16
        seq_len = 64
        hidden_dim = 32

        torch.manual_seed(42)

        # Create a simple model
        model = torch.nn.Linear(hidden_dim, 1, bias=False)

        # Create input data
        x = torch.randn(batch_size, seq_len, hidden_dim)
        response_mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.float32)
        response_mask[response_mask.sum(dim=-1) == 0, -1] = 1.0
        total_tokens = response_mask.sum().item()

        # Compute gradients without accumulation
        model.zero_grad()
        output_full = model(x).squeeze(-1)  # (batch_size, seq_len)
        loss_full = agg_loss(output_full, response_mask, loss_agg_mode)
        loss_full.backward()
        grad_full = model.weight.grad.clone()

        # Compute gradients with accumulation
        model.zero_grad()
        micro_batch_size = batch_size // num_micro_batches

        for i in range(num_micro_batches):
            start_idx = i * micro_batch_size
            end_idx = (i + 1) * micro_batch_size

            micro_x = x[start_idx:end_idx]
            micro_mask = response_mask[start_idx:end_idx]
            micro_tokens = micro_mask.sum().item()

            output_micro = model(micro_x).squeeze(-1)
            loss_micro = agg_loss(output_micro, micro_mask, loss_agg_mode)

            # Apply correct scaling based on loss mode
            if loss_agg_mode == "token-mean":
                scale_factor = micro_tokens / total_tokens
            else:
                scale_factor = micro_batch_size / batch_size

            scaled_loss = loss_micro * scale_factor
            scaled_loss.backward()

        grad_accum = model.weight.grad.clone()

        # Assert gradients match
        torch.testing.assert_close(
            grad_accum,
            grad_full,
            rtol=1e-4,
            atol=1e-5,
            msg=f"Gradient mismatch for {loss_agg_mode} with {num_micro_batches} micro-batches",
        )


class TestEdgeCases:
    """Test edge cases for gradient accumulation."""

    @pytest.mark.parametrize("loss_agg_mode", LOSS_AGG_MODES)
    def test_single_micro_batch_equals_full_batch(self, loss_agg_mode: str):
        """Test that using 1 micro-batch equals processing full batch."""
        batch = create_mock_batch(batch_size=16, seq_len=64, seed=42)

        loss_full = compute_loss_without_accumulation(batch, loss_agg_mode)
        loss_accum = compute_loss_with_token_weighted_accumulation(batch, loss_agg_mode, 1)

        torch.testing.assert_close(
            loss_accum, loss_full, rtol=1e-5, atol=1e-6, msg="Single micro-batch should equal full batch"
        )

    @pytest.mark.parametrize("loss_agg_mode", LOSS_AGG_MODES)
    def test_uneven_token_distribution(self, loss_agg_mode: str):
        """Test gradient accumulation with uneven token distribution across micro-batches."""
        batch_size = 8
        seq_len = 64

        torch.manual_seed(42)

        # Create response mask with uneven token distribution
        response_mask = torch.zeros(batch_size, seq_len)
        # First half has many tokens, second half has few
        response_mask[:4, :32] = 1.0
        response_mask[4:, -4:] = 1.0

        batch = {
            "log_prob": torch.randn(batch_size, seq_len),
            "old_log_prob": torch.randn(batch_size, seq_len),
            "advantages": torch.randn(batch_size, seq_len),
            "response_mask": response_mask,
        }

        loss_full = compute_loss_without_accumulation(batch, loss_agg_mode)
        loss_accum = compute_loss_with_token_weighted_accumulation(batch, loss_agg_mode, 2)

        torch.testing.assert_close(
            loss_accum,
            loss_full,
            rtol=1e-5,
            atol=1e-6,
            msg=f"Uneven token distribution failed for {loss_agg_mode}",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
