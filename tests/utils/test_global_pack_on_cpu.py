# Copyright (c) 2026 Google LLC. All rights reserved.
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

"""CPU tests for the global FFD packing planner (``verl.utils.global_pack``, ``VERL_TPU_GLOBAL_PACK=1``).

Engine-side tests (agreement with the TPU engine's per-rank packing and loss equivalence of packed rows) live in
verl-hardware-plugin next to the engine.
"""

from __future__ import annotations

import math
import random

import pytest
import torch

from verl.utils.global_pack import (
    bin_lower_bound,
    ffd_pack_bins,
    get_global_pack_partitions,
    global_pack_partitions_from_env,
    segment_slot_len,
)

BIN = 65536


def _bench397_lengths(seed: int, num_trajs: int = 1024, max_response_len: int = 57344) -> list[int]:
    """Synthetic 397B DeepSWE 64K length mix (prompt + response): ~70% of trajectories are longer than 40K."""
    g = torch.Generator().manual_seed(seed)
    prompt_lens = torch.randint(2160, 2400 + 1, (num_trajs,), generator=g).tolist()
    out = []
    for i, p_len in enumerate(prompt_lens):
        cap = min(max_response_len, BIN - p_len)
        rem = i % 20
        if rem < 8:
            lo, hi = min(50000, cap), min(62000, cap)
        elif rem < 14:
            lo, hi = min(40000, cap), min(50000, cap)
        elif rem < 18:
            lo, hi = min(28000, cap), min(40000, cap)
        else:
            lo, hi = min(12000, cap), min(28000, cap)
        r_len = int(torch.randint(lo, max(lo, hi) + 1, (1,), generator=g).item())
        out.append(p_len + min(r_len, cap))
    return out


def _gamma_lengths(seed: int, n: int = 1024, mean: float = 33000.0) -> list[int]:
    torch.manual_seed(seed)
    dist = torch.distributions.Gamma(torch.tensor(2.0), torch.tensor(2.0 / mean))
    return [int(x) for x in dist.sample((n,)).clamp(1500, BIN).tolist()]


def _per_rank_ffd_max(lengths: list[int], dp: int, bin_len: int) -> int:
    k = len(lengths) // dp
    return max(len(ffd_pack_bins(lengths[r * k : (r + 1) * k], bin_len)) for r in range(dp))


def _check_plan(lengths, partitions, dp, bin_len, align, max_segments=0):
    n = len(lengths)
    assert len(partitions) == dp
    flat = [i for p in partitions for i in p]
    assert sorted(flat) == list(range(n)), "every sequence must be assigned exactly once"
    assert {len(p) for p in partitions} == {n // dp}, "equal-size dispatch chunks"
    costs = []
    for p in partitions:
        bins = ffd_pack_bins([lengths[i] for i in p], bin_len, align=align, max_segments=max_segments)
        for b in bins:
            assert sum(segment_slot_len(lengths[p[i]], bin_len, align) for i in b) <= bin_len
            if max_segments:
                assert len(b) <= max_segments
        costs.append(len(bins))
    return costs


# --------------------------------------------------------------------------------------------- planner


def _legacy_ffd_count(lengths: list[int], bin_len: int) -> int:
    """Reference: plain first-fit decreasing on raw lengths (the TPU engine's historical per-rank packing)."""
    used: list[int] = []
    for ln in sorted(lengths, reverse=True):
        for b in range(len(used)):
            if used[b] + ln <= bin_len:
                used[b] += ln
                break
        else:
            used.append(ln)
    return len(used)


def test_ffd_pack_bins_matches_plain_ffd():
    """align=1 / no segment cap is exactly plain first-fit decreasing."""
    rng = random.Random(0)
    for _ in range(50):
        lengths = [rng.randint(1, 900) for _ in range(rng.randint(1, 30))]
        bins = ffd_pack_bins(lengths, 1024)
        assert len(bins) == _legacy_ffd_count(lengths, 1024)
        assert sorted(i for b in bins for i in b) == list(range(len(lengths)))
        assert all(sum(lengths[i] for i in b) <= 1024 for b in bins)


def test_ffd_alignment_and_segment_cap():
    assert segment_slot_len(1, 64, 64) == 64
    assert segment_slot_len(65, 256, 64) == 128
    assert segment_slot_len(250, 256, 64) == 256
    with pytest.raises(ValueError):
        segment_slot_len(257, 256, 64)
    # 4 x 60 fits one 256 bin unaligned, but 64-aligned (4 x 64) still fits; 5 x 50 needs 2 bins aligned.
    assert len(ffd_pack_bins([60] * 4, 256, align=64)) == 1
    assert len(ffd_pack_bins([50] * 5, 256, align=1)) == 1
    assert len(ffd_pack_bins([50] * 5, 256, align=64)) == 2
    assert len(ffd_pack_bins([10] * 8, 256, max_segments=3)) == 3


@pytest.mark.parametrize("seed", [1001, 1002, 1003])
def test_bench397_distribution_reaches_lower_bound(seed):
    """BENCH_DIST=397b, 1024 trajectories, CP=2 on 256 devices -> 128 DP ranks, 65536-token bins."""
    dp = 128
    lengths = _bench397_lengths(seed)
    partitions, stats = get_global_pack_partitions(lengths, dp, BIN, align=64)
    costs = _check_plan(lengths, partitions, dp, BIN, align=64)
    per_rank = _per_rank_ffd_max(lengths, dp, BIN)
    lb = sum(stats.lower_bound_per_rank)
    token_bound = math.ceil(sum(lengths) / (dp * BIN))
    print(
        f"seed={seed} tokens={sum(lengths)} per-rank FFD={per_rank} global={max(costs)} lower_bound={lb} "
        f"token_bound={token_bound} fill={stats.fill_ratio:.3f}"
    )
    assert max(costs) == stats.micro_batches
    assert stats.micro_batches == lb  # optimal: the planner reaches its lower bound on this mix
    assert stats.micro_batches < per_rank
    # Per-rank FFD of each reordered (contiguous) dispatch chunk reproduces the plan.
    reordered = [lengths[j] for p in partitions for j in p]
    assert _per_rank_ffd_max(reordered, dp, BIN) == max(_check_plan(lengths, partitions, dp, BIN, align=1))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_33k_mean_distribution_near_token_bound(seed):
    """A ~33K-mean mix (~33M tokens/step): global packing reaches ceil(tokens / (dp * 64K))."""
    dp = 128
    lengths = _gamma_lengths(seed)
    partitions, stats = get_global_pack_partitions(lengths, dp, BIN, align=64)
    costs = _check_plan(lengths, partitions, dp, BIN, align=64)
    token_bound = math.ceil(sum(lengths) / (dp * BIN))
    print(
        f"seed={seed} tokens={sum(lengths)} per-rank FFD={_per_rank_ffd_max(lengths, dp, BIN)} global={max(costs)} "
        f"token_bound={token_bound} fill={stats.fill_ratio:.3f}"
    )
    assert max(costs) <= token_bound + 1
    assert max(costs) <= _per_rank_ffd_max(lengths, dp, BIN)


@pytest.mark.parametrize("dp,n,bin_len,max_segments", [(4, 32, 1024, 0), (8, 64, 2048, 3), (3, 27, 512, 0)])
def test_random_small_plans_are_valid(dp, n, bin_len, max_segments):
    rng = random.Random(dp * 1000 + n)
    for _ in range(20):
        lengths = [rng.randint(1, bin_len) for _ in range(n)]
        partitions, stats = get_global_pack_partitions(lengths, dp, bin_len, align=8, max_segments=max_segments)
        costs = _check_plan(lengths, partitions, dp, bin_len, align=8, max_segments=max_segments)
        slots = [segment_slot_len(x, bin_len, 8) for x in lengths]
        assert max(costs) >= -(-bin_lower_bound(slots, bin_len, max_segments) // dp)
        assert max(costs) == stats.micro_batches


def test_mini_batches_are_planned_separately():
    """Each rank's m-th mini-batch (contiguous slice of its chunk, no shuffle) is its share of mini-batch m."""
    dp, m = 4, 2
    rng = random.Random(5)
    lengths = [rng.randint(100, 4000) for _ in range(64)]
    partitions, stats = get_global_pack_partitions(lengths, dp, 4096, align=64, num_mini_batches=m)
    per_rank = len(lengths) // dp
    per_mb = per_rank // m
    for p in partitions:
        assert len(p) == per_rank
        for mb in range(m):
            chunk = p[mb * per_mb : (mb + 1) * per_mb]
            assert all(mb * 32 <= i < (mb + 1) * 32 for i in chunk)
    assert len(stats.max_bins_per_rank) == m


def test_plan_rejects_uneven_batches_and_oversized_sequences():
    with pytest.raises(ValueError):
        get_global_pack_partitions([10] * 10, 4, 64)
    with pytest.raises(ValueError):
        get_global_pack_partitions([100] * 4, 4, 64)


def test_env_settings(monkeypatch):
    monkeypatch.delenv("VERL_TPU_GLOBAL_PACK_BIN_LEN", raising=False)
    monkeypatch.delenv("VERL_TPU_BIN_SLOT_LEN", raising=False)
    assert global_pack_partitions_from_env([10] * 8, 4) is None
    monkeypatch.setenv("VERL_TPU_BIN_SLOT_LEN", "256")
    monkeypatch.setenv("VERL_TPU_GLOBAL_PACK_ALIGN", "64")
    partitions, stats = global_pack_partitions_from_env([100] * 8, 4)
    assert stats.bin_len == 256 and all(len(p) == 2 for p in partitions)
    assert stats.micro_batches == 1  # two 128-slot segments per 256 bin
