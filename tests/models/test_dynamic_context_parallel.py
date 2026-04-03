"""
Tests for Dynamic Context Parallel (DCP) strategy components in verl.

Tests cover:
  1. Core strategy: local_cp_size computation via dynamic_cp_split_batch
  2. Batch splitting: sequence distribution across DP-CP ranks
  3. Loss normalization: DCP vs non-DCP produce aligned loss values
  4. Gradient norm: DCP vs non-DCP produce aligned gradient norms
  5. Output merging via dynamic_cp_merge_output
  6. Consistency across different local_cp_size for same data
  7. Edge cases

Usage:
    torchrun --nproc_per_node=8 tests/models/test_dynamic_context_parallel.py
    torchrun --nproc_per_node=4 tests/models/test_dynamic_context_parallel.py
"""

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import numpy as np
import pytest
import torch
import torch.distributed as dist
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.megatron_utils import dynamic_cp_split_batch
from verl.utils.torch_functional import masked_sum
from verl.workers.config.engine import McoreEngineConfig

# ============================================================================
# Helpers
# ============================================================================


def make_engine_config(max_seqlen_per_dp_cp_rank: int) -> McoreEngineConfig:
    """Create a minimal McoreEngineConfig for DCP testing."""
    return McoreEngineConfig(
        dynamic_context_parallel=True,
        max_seqlen_per_dp_cp_rank=max_seqlen_per_dp_cp_rank,
        context_parallel_size=1,
    )


def make_batch(seq_lens: list[int], vocab_size: int = 100, device="cuda") -> TensorDict:
    """Create a TensorDict batch with nested tensors, matching the real engine data format.

    Returns a TensorDict with:
      - input_ids: nested tensor [bsz, j1]
      - loss_mask: nested tensor [bsz, j1]  (first 20% prompt, rest response)
    """
    input_ids_list = []
    loss_mask_list = []
    for sl in seq_lens:
        ids = torch.randint(0, vocab_size, (sl,), device=device)
        prompt_len = max(1, sl // 5)
        mask = torch.zeros(sl, device=device, dtype=torch.float32)
        mask[prompt_len:] = 1.0
        input_ids_list.append(ids)
        loss_mask_list.append(mask)

    input_ids_nested = torch.nested.nested_tensor(input_ids_list, layout=torch.jagged)
    loss_mask_nested = torch.nested.nested_tensor(loss_mask_list, layout=torch.jagged)

    batch = TensorDict(
        {"input_ids": input_ids_nested, "loss_mask": loss_mask_nested},
        batch_size=len(seq_lens),
    )
    return batch


def compute_sft_loss(log_probs: torch.Tensor, loss_mask: torch.Tensor, batch_num_tokens: float, dp_size: int):
    """SFT loss: -sum(log_prob * shifted_mask) / batch_num_tokens * dp_size"""
    loss_mask_shifted = torch.roll(loss_mask, shifts=-1, dims=0)
    loss = -masked_sum(log_probs, loss_mask_shifted) / batch_num_tokens * dp_size
    return loss


class SimpleModel(torch.nn.Module):
    """Minimal model for testing loss/grad computation."""

    def __init__(self, vocab_size=100, hidden_dim=64):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_dim)
        self.proj = torch.nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids):
        h = self.embed(input_ids)
        logits = self.proj(h)
        return torch.nn.functional.log_softmax(logits, dim=-1)


# ============================================================================
# Distributed initialization
# ============================================================================


def init_dist():
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(seconds=300))
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)


# ============================================================================
# Test 1: Core strategy - local_cp_size via dynamic_cp_split_batch
# ============================================================================


class TestLocalCPSizeComputation:
    """Test that dynamic_cp_split_batch computes correct local_cp_size."""

    def _get_cp_size(self, seq_lens, dp_size, dp_rank, max_seqlen_per_dp_cp_rank):
        torch.manual_seed(0)
        batch = make_batch(seq_lens, device="cpu")
        config = make_engine_config(max_seqlen_per_dp_cp_rank)
        result = dynamic_cp_split_batch(batch, config, dp_size, dp_rank)
        return tu.get_non_tensor_data(result, "local_cp_size", None)

    def test_exact_fit(self):
        """max_seq_len == max_seqlen_per_dp_cp_rank => cp=1"""
        cp = self._get_cp_size([1024] * 8, dp_size=8, dp_rank=0, max_seqlen_per_dp_cp_rank=1024)
        assert cp == 1

    def test_double_length(self):
        """max_seq_len == 2x threshold => cp=2"""
        cp = self._get_cp_size([2048] * 8, dp_size=8, dp_rank=0, max_seqlen_per_dp_cp_rank=1024)
        assert cp == 2

    def test_round_up_to_power_of_2(self):
        """ratio=3 should round up to cp=4"""
        cp = self._get_cp_size([3072] * 8, dp_size=8, dp_rank=0, max_seqlen_per_dp_cp_rank=1024)
        assert cp == 4

    def test_clamp_to_dp_size(self):
        """cp should never exceed dp_size"""
        cp = self._get_cp_size([100000], dp_size=8, dp_rank=0, max_seqlen_per_dp_cp_rank=1024)
        assert cp == 8

    def test_min_coverage_constraint(self):
        """2 seqs, 8 dp ranks => cp>=4 so each sub-group gets data"""
        cp = self._get_cp_size([512] * 2, dp_size=8, dp_rank=0, max_seqlen_per_dp_cp_rank=1024)
        assert cp >= 4
        assert 8 // cp <= 2

    def test_single_sequence(self):
        """1 seq => full CP"""
        cp = self._get_cp_size([512], dp_size=8, dp_rank=0, max_seqlen_per_dp_cp_rank=1024)
        assert cp == 8

    def test_always_power_of_2(self):
        """cp should always be a power of 2"""
        for max_sl in [100, 500, 1000, 1500, 2000, 3000, 5000, 10000]:
            for num_seqs in [1, 2, 3, 4, 7, 8, 16]:
                for dp in [2, 4, 8]:
                    cp = self._get_cp_size(
                        [max_sl] * num_seqs, dp_size=dp, dp_rank=0, max_seqlen_per_dp_cp_rank=1024
                    )
                    assert cp & (cp - 1) == 0, f"cp={cp} not power-of-2 for max_sl={max_sl}, n={num_seqs}, dp={dp}"
                    assert 1 <= cp <= dp

    @pytest.mark.parametrize(
        "max_seq_len,num_seqs,dp_size,max_per_rank,expected_cp",
        [
            (1024, 8, 8, 1024, 1),
            (2048, 8, 8, 1024, 2),
            (4096, 8, 8, 1024, 4),
            (8192, 8, 8, 1024, 8),
            (512, 4, 4, 1024, 1),
            (512, 1, 4, 1024, 4),   # coverage constraint
            (2048, 2, 8, 1024, 4),  # max(2, 4) = 4 due to coverage
        ],
    )
    def test_parametrized(self, max_seq_len, num_seqs, dp_size, max_per_rank, expected_cp):
        cp = self._get_cp_size([max_seq_len] * num_seqs, dp_size, 0, max_per_rank)
        assert cp == expected_cp, f"Expected cp={expected_cp}, got {cp}"


# ============================================================================
# Test 2: Batch splitting via dynamic_cp_split_batch
# ============================================================================


class TestBatchSplitting:
    """Test that dynamic_cp_split_batch distributes sequences correctly."""

    def test_even_split(self):
        """8 seqs, dp=8, short seqs => cp=1, each rank gets 1 seq"""
        torch.manual_seed(0)
        batch = make_batch([512] * 8, device="cpu")
        config = make_engine_config(1024)
        for rank in range(8):
            result = dynamic_cp_split_batch(batch.clone(), config, dp_size=8, dp_rank=rank)
            cp = tu.get_non_tensor_data(result, "local_cp_size", None)
            dp = result["dp_size"]
            assert cp == 1
            assert dp == 8
            assert result.batch_size[0] == 1

    def test_full_cp(self):
        """1 seq => cp=8, all ranks get same seq"""
        torch.manual_seed(0)
        batch = make_batch([2048], device="cpu")
        config = make_engine_config(1024)
        for rank in range(8):
            result = dynamic_cp_split_batch(batch.clone(), config, dp_size=8, dp_rank=rank)
            cp = tu.get_non_tensor_data(result, "local_cp_size", None)
            dp = result["dp_size"]
            assert cp == 8
            assert dp == 1
            assert result.batch_size[0] == 1

    def test_partial_cp(self):
        """4 long seqs, dp=8 => cp=2, local_dp=4, each gets 1 seq"""
        torch.manual_seed(0)
        batch = make_batch([2048] * 4, device="cpu")
        config = make_engine_config(1024)
        for rank in range(8):
            result = dynamic_cp_split_batch(batch.clone(), config, dp_size=8, dp_rank=rank)
            assert tu.get_non_tensor_data(result, "local_cp_size", None) == 2
            assert result["dp_size"] == 4
            assert result.batch_size[0] == 1

    def test_cp_ranks_share_data(self):
        """Ranks in the same CP sub-group get same local_dp_rank."""
        torch.manual_seed(0)
        batch = make_batch([2048] * 4, device="cpu")
        config = make_engine_config(1024)
        results = {}
        for rank in range(8):
            result = dynamic_cp_split_batch(batch.clone(), config, dp_size=8, dp_rank=rank)
            cp = tu.get_non_tensor_data(result, "local_cp_size", None)
            local_dp_rank = rank // cp
            results[rank] = (local_dp_rank, result.batch_size[0])
        # Ranks 0,1 in same CP group; ranks 2,3 in same CP group
        assert results[0][0] == results[1][0]
        assert results[2][0] == results[3][0]

    def test_all_sequences_covered(self):
        """All sequences are assigned to exactly one DP sub-group."""
        torch.manual_seed(0)
        batch = make_batch([512] * 7, device="cpu")
        config = make_engine_config(1024)
        # Get cp size
        probe = dynamic_cp_split_batch(batch.clone(), config, dp_size=8, dp_rank=0)
        cp = tu.get_non_tensor_data(probe, "local_cp_size", None)

        total = 0
        for dp_rank in range(0, 8, cp):
            result = dynamic_cp_split_batch(batch.clone(), config, dp_size=8, dp_rank=dp_rank)
            total += result.batch_size[0]
        assert total == 7


# ============================================================================
# Test 3 & 4: Loss and gradient norm alignment (distributed)
# ============================================================================


@dataclass
class DCPTestConfig:
    seq_lens: list[int]
    max_seqlen_per_dp_cp_rank: int
    expected_cp: Optional[int] = None
    description: str = ""


def dcp_test_configs():
    return [
        DCPTestConfig(seq_lens=[512] * 8, max_seqlen_per_dp_cp_rank=1024, expected_cp=1, description="cp=1, short"),
        DCPTestConfig(seq_lens=[2048] * 8, max_seqlen_per_dp_cp_rank=1024, expected_cp=2, description="cp=2, medium"),
        DCPTestConfig(seq_lens=[4096] * 8, max_seqlen_per_dp_cp_rank=1024, expected_cp=4, description="cp=4, long"),
        DCPTestConfig(seq_lens=[8192], max_seqlen_per_dp_cp_rank=1024, expected_cp=8, description="cp=8, single"),
    ]


def _run_dcp_forward(model, batch, engine_config, dp_size, dp_rank):
    """Run model forward through DCP split, return (loss, log_probs, local_cp_size).

    Follows the real engine flow:
      1. batch_num_tokens computed on full batch BEFORE split
      2. dynamic_cp_split_batch splits and sets dp_size=local_dp_size
      3. loss = -masked_sum / batch_num_tokens * local_dp_size
    """
    # Step 1: batch_num_tokens on full batch (before split)
    full_loss_mask = batch["loss_mask"]
    batch_num_tokens = full_loss_mask.values().sum().item()

    # Step 2: split
    split_batch = dynamic_cp_split_batch(batch.clone(), engine_config, dp_size, dp_rank)
    local_cp_size = tu.get_non_tensor_data(split_batch, "local_cp_size", None)
    local_dp_size = split_batch["dp_size"]

    # Step 3: forward on local data
    local_ids = split_batch["input_ids"].values()
    local_masks = split_batch["loss_mask"].values()

    log_probs = model(local_ids)
    log_probs_selected = log_probs.gather(1, local_ids.unsqueeze(1)).squeeze(1)

    loss = compute_sft_loss(log_probs_selected, local_masks, batch_num_tokens, local_dp_size)
    return loss, local_cp_size


@pytest.mark.parametrize("test_config", dcp_test_configs(), ids=lambda c: c.description)
def test_loss_alignment(test_config):
    """DCP loss (all-reduced) should match single-rank reference loss."""
    init_dist()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if test_config.expected_cp is not None and test_config.expected_cp > world_size:
        pytest.skip(f"Need {test_config.expected_cp} GPUs, have {world_size}")

    torch.manual_seed(42)
    vocab_size, hidden_dim = 100, 64
    model = SimpleModel(vocab_size, hidden_dim).cuda()
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    # Same data on all ranks
    torch.manual_seed(42)
    batch = make_batch(test_config.seq_lens, vocab_size=vocab_size)
    config = make_engine_config(test_config.max_seqlen_per_dp_cp_rank)

    # --- Reference: all data, no splitting ---
    all_ids = batch["input_ids"].values()
    all_masks = batch["loss_mask"].values()
    global_num_tokens = all_masks.sum()
    ref_log_probs = model(all_ids).gather(1, all_ids.unsqueeze(1)).squeeze(1)
    ref_loss = -masked_sum(ref_log_probs, torch.roll(all_masks, -1, 0)) / global_num_tokens
    ref_loss_val = ref_loss.detach()

    # --- DCP path ---
    model.zero_grad()
    dcp_loss, local_cp_size = _run_dcp_forward(model, batch, config, world_size, rank)

    if test_config.expected_cp is not None:
        assert local_cp_size == test_config.expected_cp

    dcp_loss_reduced = dcp_loss.detach().clone()
    dist.all_reduce(dcp_loss_reduced, op=dist.ReduceOp.SUM)
    dcp_loss_avg = dcp_loss_reduced / world_size

    torch.testing.assert_close(
        dcp_loss_avg, ref_loss_val, rtol=1e-4, atol=1e-4,
        msg=f"Loss mismatch [{test_config.description}]: dcp={dcp_loss_avg.item():.6f}, ref={ref_loss_val.item():.6f}",
    )
    if rank == 0:
        print(f"  PASS [{test_config.description}] cp={local_cp_size}, loss={ref_loss_val.item():.6f}")


@pytest.mark.parametrize("test_config", dcp_test_configs(), ids=lambda c: c.description)
def test_grad_norm_alignment(test_config):
    """DCP gradient norms (after all-reduce) should match single-rank reference."""
    init_dist()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if test_config.expected_cp is not None and test_config.expected_cp > world_size:
        pytest.skip(f"Need {test_config.expected_cp} GPUs, have {world_size}")

    torch.manual_seed(42)
    vocab_size, hidden_dim = 100, 64

    # --- Reference ---
    ref_model = SimpleModel(vocab_size, hidden_dim).cuda()
    for p in ref_model.parameters():
        dist.broadcast(p.data, src=0)

    torch.manual_seed(42)
    batch = make_batch(test_config.seq_lens, vocab_size=vocab_size)
    config = make_engine_config(test_config.max_seqlen_per_dp_cp_rank)

    all_ids = batch["input_ids"].values()
    all_masks = batch["loss_mask"].values()
    global_num_tokens = all_masks.sum()
    ref_log_probs = ref_model(all_ids).gather(1, all_ids.unsqueeze(1)).squeeze(1)
    ref_loss = -masked_sum(ref_log_probs, torch.roll(all_masks, -1, 0)) / global_num_tokens
    ref_loss.backward()
    ref_grad_norm = torch.nn.utils.clip_grad_norm_(ref_model.parameters(), float("inf"))

    # --- DCP ---
    dcp_model = SimpleModel(vocab_size, hidden_dim).cuda()
    for p in dcp_model.parameters():
        dist.broadcast(p.data, src=0)

    dcp_loss, local_cp_size = _run_dcp_forward(dcp_model, batch, config, world_size, rank)
    dcp_loss.backward()

    for p in dcp_model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world_size

    dcp_grad_norm = torch.nn.utils.clip_grad_norm_(dcp_model.parameters(), float("inf"))

    torch.testing.assert_close(
        dcp_grad_norm, ref_grad_norm, rtol=1e-2, atol=1e-4,
        msg=f"Grad norm mismatch [{test_config.description}]: dcp={dcp_grad_norm.item():.6f}, ref={ref_grad_norm.item():.6f}",
    )
    if rank == 0:
        print(f"  PASS [{test_config.description}] cp={local_cp_size}, grad_norm={ref_grad_norm.item():.6f}")


# ============================================================================
# Test 5: Output merging (requires megatron mpu, tested structurally)
# ============================================================================
# NOTE: dynamic_cp_merge_output uses mpu.get_data_parallel_group() which
# requires full Megatron initialization. We test its structural properties
# without calling it directly.


def test_merge_output_identity():
    """When local_cp_size == dp_size, merge should return input unchanged."""
    from verl.utils.megatron_utils import dynamic_cp_merge_output

    outputs = {"log_probs": torch.randn(3, 10)}
    result = dynamic_cp_merge_output(outputs, dp_size=8, dp_rank=0, local_cp_size=8)
    assert result is outputs  # identity - no merging needed


# ============================================================================
# Test 6: Consistency across different local_cp_size for same data
# ============================================================================


def test_loss_consistent_across_cp_sizes():
    """Loss should be the same regardless of local_cp_size for the same data."""
    init_dist()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size < 4:
        pytest.skip("Need at least 4 GPUs")

    vocab_size, hidden_dim = 100, 64
    torch.manual_seed(42)
    batch = make_batch([1024] * world_size, vocab_size=vocab_size)

    losses = []
    for max_per_rank in [2048, 1024, 512, 256]:
        torch.manual_seed(42)
        model = SimpleModel(vocab_size, hidden_dim).cuda()
        for p in model.parameters():
            dist.broadcast(p.data, src=0)

        config = make_engine_config(max_per_rank)
        dcp_loss, cp = _run_dcp_forward(model, batch, config, world_size, rank)

        loss_reduced = dcp_loss.detach().clone()
        dist.all_reduce(loss_reduced, op=dist.ReduceOp.SUM)
        loss_avg = loss_reduced / world_size
        losses.append((cp, loss_avg.item()))

    ref_loss = losses[0][1]
    for cp, loss_val in losses[1:]:
        assert abs(loss_val - ref_loss) < 1e-4, f"Loss mismatch: cp={cp} loss={loss_val:.6f} vs ref={ref_loss:.6f}"

    if rank == 0:
        for cp, loss_val in losses:
            print(f"  cp={cp}, loss={loss_val:.6f}")
        print("  PASS [loss_consistent_across_cp_sizes]")


# ============================================================================
# Test 7: Edge cases
# ============================================================================


class TestEdgeCases:

    def _get_cp_dp(self, seq_lens, dp_size, dp_rank, max_per_rank):
        torch.manual_seed(0)
        batch = make_batch(seq_lens, device="cpu")
        config = make_engine_config(max_per_rank)
        result = dynamic_cp_split_batch(batch, config, dp_size, dp_rank)
        cp = tu.get_non_tensor_data(result, "local_cp_size", None)
        dp = result["dp_size"]
        return cp, dp

    def test_num_seqs_less_than_dp(self):
        """3 seqs, 8 ranks => cp>=4"""
        cp, dp = self._get_cp_dp([1024] * 3, dp_size=8, dp_rank=0, max_per_rank=2048)
        assert cp >= 4
        assert dp <= 2

    def test_uneven_sequence_distribution(self):
        """All sequences should be covered even with uneven counts."""
        for num_seqs in [1, 2, 3, 5, 7, 9, 15]:
            torch.manual_seed(0)
            batch = make_batch([512] * num_seqs, device="cpu")
            config = make_engine_config(1024)

            probe = dynamic_cp_split_batch(batch.clone(), config, dp_size=8, dp_rank=0)
            cp = tu.get_non_tensor_data(probe, "local_cp_size", None)

            total = 0
            for dp_rank in range(0, 8, cp):
                result = dynamic_cp_split_batch(batch.clone(), config, dp_size=8, dp_rank=dp_rank)
                total += result.batch_size[0]
            assert total == num_seqs, f"num_seqs={num_seqs}: assigned {total}"

    def test_dp_size_1(self):
        """Single GPU: cp=1"""
        cp, dp = self._get_cp_dp([4096], dp_size=1, dp_rank=0, max_per_rank=1024)
        assert cp == 1
        assert dp == 1

    def test_very_short_sequences(self):
        """Very short seqs => cp=1"""
        cp, _ = self._get_cp_dp([10] * 8, dp_size=8, dp_rank=0, max_per_rank=1024)
        assert cp == 1

    def test_dp_size_metadata_overwritten(self):
        """dynamic_cp_split_batch should overwrite dp_size to local_dp_size."""
        torch.manual_seed(0)
        batch = make_batch([2048] * 4, device="cpu")
        tu.assign_non_tensor(batch, dp_size=999)  # pre-set a wrong value
        config = make_engine_config(1024)
        result = dynamic_cp_split_batch(batch, config, dp_size=8, dp_rank=0)
        # Should be overwritten to local_dp_size, not 999
        assert result["dp_size"] == 4

    def test_local_cp_size_attached_as_metadata(self):
        """local_cp_size should be attached to the batch as non-tensor data."""
        torch.manual_seed(0)
        batch = make_batch([4096] * 8, device="cpu")
        config = make_engine_config(1024)
        result = dynamic_cp_split_batch(batch, config, dp_size=8, dp_rank=0)
        cp = tu.get_non_tensor_data(result, "local_cp_size", None)
        assert cp is not None
        assert isinstance(cp, int)
        assert cp == 4


if __name__ == "__main__":
    pytest.main([__file__, "-svv"])
