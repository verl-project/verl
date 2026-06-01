# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
Dynamic Context Parallelism Scheduler for verl.

Adapted from Megatron-LM's data_schedule.py and data_schedule_utils.py.
Provides a scheduler that dynamically assigns CP sizes to micro-batches based
on sequence lengths, and redistributes data across DP ranks via all-to-all.
"""

import logging
import os
from collections import deque
from functools import lru_cache
from math import ceil, log2
from typing import Callable

import torch
import torch.distributed
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_id

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

MAX_SHAPE_DIMS = 8


# =============================================================================
# Workload estimation helpers
# =============================================================================


@lru_cache(maxsize=128)
def dcp_gpus_needed(seq_len: int, max_seq_len_per_rank: int, min_cp_size: int = 1) -> int:
    """Number of GPUs needed for a sequence, rounded up to next power of 2.

    Args:
        seq_len: The sequence length.
        max_seq_len_per_rank: Max sequence length each rank can handle.
        min_cp_size: Minimum CP group size.

    Returns:
        Number of GPUs needed (power of 2, >= min_cp_size).
    """
    if seq_len <= max_seq_len_per_rank:
        return max(1, min_cp_size)
    raw = max(1, 2 ** ceil(log2(seq_len / max_seq_len_per_rank)))
    return max(min_cp_size, raw)


@lru_cache(maxsize=128)
def dcp_get_total_workload(
    seq_length: int, max_seq_len_per_rank: int, cp_size: int | None = None, min_cp_size: int = 1
) -> float:
    """Estimate workload of a sample for scheduling balance.

    Workload is proportional to seq_len^2 / cp_size (self-attention complexity).

    Args:
        seq_length: The sequence length.
        max_seq_len_per_rank: Max sequence length each rank can handle.
        cp_size: Override CP size. If None, computed from seq_length.
        min_cp_size: Minimum CP group size.

    Returns:
        Estimated workload as a float.
    """
    if cp_size is None:
        cp_size = dcp_gpus_needed(seq_length, max_seq_len_per_rank, min_cp_size)
    return (seq_length * seq_length) / cp_size


def dcp_make_buckets_equal(
    sample_seqlens: list[tuple[int, int]],
    compute_estimator: Callable,
    max_seq_len_per_rank: int,
    min_cp_size: int = 1,
) -> list[deque]:
    """Split samples into buckets of roughly equal work, one per unique CP size.

    Args:
        sample_seqlens: List of (sample_id, seq_len) tuples, sorted by seq_len desc.
        compute_estimator: Callable(seq_len, cp_size=None) -> workload.
        max_seq_len_per_rank: Max sequence length per rank.
        min_cp_size: Minimum CP group size.

    Returns:
        List of deques, each containing (sample_id, seq_len) tuples.
    """
    seqlens = [seq_len for _, seq_len in sample_seqlens]
    k = len({dcp_gpus_needed(L, max_seq_len_per_rank, min_cp_size) for L in seqlens})

    work = []
    for _, s in sample_seqlens:
        cp_size = dcp_gpus_needed(s, max_seq_len_per_rank, min_cp_size)
        work.append(compute_estimator(s, cp_size))
    total_work = sum(work)
    target = total_work / k
    buckets, cur, cur_work = [], [], 0.0
    remaining_k = k

    for i, (sample_id, seq_len) in enumerate(sample_seqlens):
        w = compute_estimator(seq_len)
        projected = cur_work + w
        if cur and (projected > target * 1.1 or len(sample_seqlens) - i <= remaining_k - len(buckets)):
            buckets.append(deque(cur))
            cur, cur_work = [], 0.0
            remaining_k -= 1
        cur.append((sample_id, seq_len))
        cur_work += w

    if cur:
        buckets.append(deque(cur))
    return buckets


# =============================================================================
# Core scheduling algorithm
# =============================================================================


def next_hdp_group(
    sample_seqlens: list[tuple[int, int]],
    compute_estimator: Callable[[int], float],
    total_gpus: int,
    gpus_needed_fn: Callable[[int], int],
    make_buckets_equal_fn: Callable,
    max_seq_len_per_rank: float,
    get_total_workload_fn: Callable,
    delta: float = 0.05,
    strategy: str = "dp",
    eps_bucket: float = 0.10,
) -> tuple[list[list[int]], list[tuple[int, int]], list[float], list[list[int]]]:
    """Form one balanced micro-batch group across DPxCP ranks.

    This is the core greedy scheduling algorithm. It assigns samples to GPU groups
    such that the workload is balanced across all GPUs.

    Args:
        sample_seqlens: List of (sample_id, seq_len) tuples.
        compute_estimator: Callable(seq_len) -> per-GPU workload estimate.
        total_gpus: Total number of DPxCP GPUs.
        gpus_needed_fn: Callable(seq_len) -> number of GPUs needed.
        make_buckets_equal_fn: Callable to create balanced buckets.
        max_seq_len_per_rank: Max tokens per rank for packing.
        get_total_workload_fn: Callable(seq_len, cp_size) -> workload.
        delta: Balance threshold (default 5%).
        strategy: "dp" or "pp" scan strategy.
        eps_bucket: Bucket balance tolerance (default 10%).

    Returns:
        Tuple of:
            - micro_batches: Per-GPU list of sequence lengths.
            - leftovers: Unassigned (sample_id, seq_len) tuples.
            - exec_times: Per-GPU estimated execution times.
            - sample_ids_per_gpu: Per-GPU list of sample IDs.
    """
    if not sample_seqlens:
        return (
            [[] for _ in range(total_gpus)],
            [],
            [0.0 for _ in range(total_gpus)],
            [[] for _ in range(total_gpus)],
        )

    buckets = make_buckets_equal_fn(sample_seqlens, compute_estimator)

    micro_batches = [[] for _ in range(total_gpus)]
    exec_times = [0.0 for _ in range(total_gpus)]
    sample_ids_per_gpu = [[] for _ in range(total_gpus)]
    packing_sequence_len = {}

    gpu_group_id = [None] * total_gpus
    group_members = {}
    group_size = {}
    next_gid = 0

    pp_cursor = 0
    prev_needed = None
    check_balance = False

    while buckets:
        sample_seq_tuple = bucket_idx = None
        needed = None

        scan_order = (
            range(len(buckets)) if strategy == "dp" else [(pp_cursor + i) % len(buckets) for i in range(len(buckets))]
        )

        for idx in scan_order:
            if not buckets[idx]:
                continue
            cand_tuple = buckets[idx][0]
            cand_seq_len = cand_tuple[1]
            needed = gpus_needed_fn(cand_seq_len)

            candidate_gids = [gid for gid, sz in group_size.items() if sz == needed]
            free_ranks = [r for r, gid in enumerate(gpu_group_id) if gid is None]
            if candidate_gids or len(free_ranks) >= needed:
                sample_seq_tuple, bucket_idx = cand_tuple, idx
                break

        if sample_seq_tuple is None:
            break

        if strategy == "pp":
            pp_cursor = (bucket_idx + 1) % len(buckets)

        sample_id, seq_len = sample_seq_tuple
        needed = gpus_needed_fn(seq_len)
        if prev_needed is None:
            prev_needed = needed

        # Find best existing group or create new one
        candidate_gids = [
            gid
            for gid, sz in group_size.items()
            if sz == needed and packing_sequence_len[gid] + seq_len / needed <= max_seq_len_per_rank
        ]
        if candidate_gids:
            best_gid, best_load = min(
                ((gid, max(exec_times[r] for r in group_members[gid])) for gid in candidate_gids),
                key=lambda t: t[1],
            )
        else:
            best_gid, best_load = None, float("inf")

        free_ranks = [r for r, gid in enumerate(gpu_group_id) if gid is None]
        if len(free_ranks) >= needed:
            free_sorted = sorted(free_ranks, key=lambda r: exec_times[r])
            new_members = free_sorted[:needed]
            new_load = exec_times[new_members[-1]]

            if new_load < best_load:
                best_gid = None
                chosen_members = new_members
            else:
                chosen_members = group_members[best_gid]
        else:
            if best_gid is None:
                break
            chosen_members = group_members[best_gid]

        if best_gid is None:
            best_gid = next_gid
            next_gid += 1
            group_members[best_gid] = chosen_members
            group_size[best_gid] = needed
            for r in chosen_members:
                gpu_group_id[r] = best_gid

        per_gpu_cost = compute_estimator(seq_len)

        packing_sequence_len[best_gid] = packing_sequence_len.get(best_gid, 0) + seq_len / needed
        for r in chosen_members:
            micro_batches[r].append(seq_len)
            exec_times[r] += per_gpu_cost
            sample_ids_per_gpu[r].append(sample_id)

        buckets[bucket_idx].popleft()

        while buckets and not buckets[0]:
            buckets.pop(0)
            pp_cursor %= max(1, len(buckets))

        if needed < prev_needed:
            check_balance = True

        if check_balance and buckets and max(exec_times) - min(exec_times) <= delta * max(exec_times):
            break

    leftovers = []
    for b in buckets:
        for sample_seq_tuple in b:
            leftovers.append(sample_seq_tuple)

    # Fill empty GPUs by expanding smallest groups
    total_work_before = sum(len(mb) for mb in micro_batches)

    def fill_empty_gpus(micro_batches, exec_times, sample_ids_per_gpu, group_members, group_size):
        empty_gpus = [i for i in range(total_gpus) if not micro_batches[i]]
        if not empty_gpus:
            return (micro_batches, exec_times, sample_ids_per_gpu, group_members, group_size)

        existing_group_sizes = set(group_size.values())
        assert existing_group_sizes, (
            "No existing groups found, cannot redistribute. Try increasing 'max_seqlen_per_dp_cp_rank'."
        )

        min_group_size = min(existing_group_sizes)
        next_power = min(min_group_size * 2, total_gpus)

        for gid, size in group_size.items():
            if size == min_group_size:
                members = group_members[gid]
                needed_count = next_power - min_group_size
                group_start_gpu = members[0]
                group_end_gpu = members[-1]
                empty_gpu = [idx for idx, work in enumerate(micro_batches) if not work][0]
                assert not all(work for work in micro_batches[empty_gpu : empty_gpu + needed_count]), (
                    "Empty GPUs were detected but not enough to expand."
                )
                work_to_push = micro_batches[group_end_gpu + 1 : empty_gpu]
                exec_times_to_push = exec_times[group_end_gpu + 1 : empty_gpu]
                sample_ids_to_push = sample_ids_per_gpu[group_end_gpu + 1 : empty_gpu]

                new_micro_batches = [[] for _ in micro_batches]
                new_exec_times = [0.0] * len(exec_times)
                new_sample_ids_per_gpu = [[] for _ in sample_ids_per_gpu]

                for i in range(group_start_gpu):
                    new_micro_batches[i] = micro_batches[i]
                    new_exec_times[i] = exec_times[i]
                    new_sample_ids_per_gpu[i] = sample_ids_per_gpu[i]

                for i in range(group_start_gpu, group_end_gpu + needed_count + 1):
                    new_micro_batches[i] = micro_batches[group_end_gpu]
                    new_exec_times[i] = get_total_workload_fn(micro_batches[group_end_gpu][0], next_power)
                    new_sample_ids_per_gpu[i] = sample_ids_per_gpu[group_end_gpu]

                for i, work in enumerate(work_to_push):
                    new_micro_batches[group_end_gpu + needed_count + 1 + i] = work
                    new_exec_times[group_end_gpu + needed_count + 1 + i] = exec_times_to_push[i]
                    new_sample_ids_per_gpu[group_end_gpu + needed_count + 1 + i] = sample_ids_to_push[i]

                group_size[gid] = next_power
                group_members[gid] = list(range(members[0], members[-1] + needed_count + 1))
                for pushed_gid in group_size.keys():
                    if pushed_gid > gid:
                        group_members[pushed_gid] = [x + needed_count for x in group_members[pushed_gid]]

                return (
                    new_micro_batches,
                    new_exec_times,
                    new_sample_ids_per_gpu,
                    group_members,
                    group_size,
                )

    empty_gpus = any(not micro_batches[i] for i in range(total_gpus))
    while empty_gpus:
        micro_batches, exec_times, sample_ids_per_gpu, group_members, group_size = fill_empty_gpus(
            micro_batches, exec_times, sample_ids_per_gpu, group_members, group_size
        )
        empty_gpus = any(not micro_batches[i] for i in range(total_gpus))

    total_work_after = sum(len(mb) for mb in micro_batches)
    assert total_work_after >= total_work_before, f"Samples were removed: {total_work_before} -> {total_work_after}"

    return micro_batches, leftovers, exec_times, sample_ids_per_gpu


def align_sample_id_groups(sample_id_groups: list, microbatch_group_size_per_vp_stage: int) -> list:
    """Align len(sample_id_groups) to microbatch_group_size_per_vp_stage for VPP.

    This preserves the dynamic-CP layout inside each microbatch: ranks that share
    a sample ID form one CP group, and CP group sizes must be non-increasing as
    we move across the DPxCP ranks. When a tail group is split, empty rank slots
    are filled by expanding existing CP groups rather than by slicing each
    rank's sample list independently.
    """
    multiple = int(microbatch_group_size_per_vp_stage)
    remainder = (-len(sample_id_groups)) % multiple
    i = len(sample_id_groups) - 1

    def fill_empty_by_expanding_cp(sample_id_group):
        def fill_empty(group):
            empty_size = sum(1 for ids in group if len(ids) == 0)
            i = len(group) - 1 - empty_size
            prev_cp_size = 0
            while i >= 0:
                sid0 = group[i][0]
                cp_size = 0
                while i >= 0 and sid0 in group[i]:
                    cp_size += 1
                    i -= 1
                if cp_size > prev_cp_size and prev_cp_size != 0:
                    start_idx = i + 1 + cp_size
                    end_idx = -empty_size + prev_cp_size if -empty_size + prev_cp_size < 0 else None
                    group[start_idx + 2 * prev_cp_size : end_idx] = group[start_idx + prev_cp_size : -empty_size]
                    group[start_idx + prev_cp_size : start_idx + 2 * prev_cp_size] = group[
                        start_idx : start_idx + prev_cp_size
                    ]
                    break
                if cp_size <= empty_size and i == -1:
                    end_idx = -empty_size + cp_size if -empty_size + cp_size < 0 else None
                    group[2 * cp_size : end_idx] = group[cp_size:-empty_size]
                    group[cp_size : 2 * cp_size] = group[0:cp_size]
                    break
                prev_cp_size = cp_size
            return group

        while sample_id_group and len(sample_id_group[-1]) == 0:
            sample_id_group = fill_empty(sample_id_group)
        return sample_id_group

    def split_group(sample_id_group):
        total_dcp_ranks = len(sample_id_group)
        cu_ranks = [0]
        prev_cp_size = 0

        while cu_ranks[-1] != total_dcp_ranks:
            start_rank = cu_ranks[-1]
            if not sample_id_group[start_rank]:
                return None, None
            sid0 = sample_id_group[start_rank][0]
            cp_size = 0
            for rank in range(start_rank, total_dcp_ranks):
                if sid0 in sample_id_group[rank]:
                    cp_size += 1
                else:
                    break
            if prev_cp_size != 0 and cp_size > prev_cp_size:
                raise AssertionError(f"split_group: CP size is not decreasing: prev={prev_cp_size}, cur={cp_size}")
            cu_ranks.append(start_rank + cp_size)
            prev_cp_size = cp_size

        if len(cu_ranks) == 2:
            return None, None

        split_idx = 0
        while cu_ranks[split_idx] < total_dcp_ranks // 2:
            split_idx += 1

        old_group = [ids[:] for ids in sample_id_group[: cu_ranks[split_idx]]] + [
            [] for _ in range(total_dcp_ranks - cu_ranks[split_idx])
        ]
        new_group = [ids[:] for ids in sample_id_group[cu_ranks[split_idx] :]] + [
            [] for _ in range(cu_ranks[split_idx])
        ]
        old_group = fill_empty_by_expanding_cp(old_group)
        new_group = fill_empty_by_expanding_cp(new_group)
        return new_group, old_group

    attempts_since_split = 0
    while remainder > 0:
        if i < 0:
            if attempts_since_split >= len(sample_id_groups):
                raise AssertionError("align_sample_id_groups: no tail microbatch has enough ids to split")
            i = len(sample_id_groups) - 1

        group1, group2 = split_group(sample_id_groups[i])
        if group1 is not None and group2 is not None:
            sample_id_groups[i] = group1
            sample_id_groups.append(group2)
            remainder -= 1
            attempts_since_split = 0
        else:
            attempts_since_split += 1
        i -= 1

    return sample_id_groups


# =============================================================================
# Data collection: gather global sequence lengths
# =============================================================================


def _get_global_seqlens_and_ids(
    local_seqlens: torch.Tensor, dp_group
) -> tuple[list[tuple[int, int]], torch.Tensor, torch.Tensor]:
    """Gather sequence lengths from all DP ranks and assign global IDs.

    Args:
        local_seqlens: 1D int tensor of sequence lengths on this rank.
        dp_group: Data parallel process group.

    Returns:
        global_id_seqlens: List of (global_id, seq_len) for ALL samples across all ranks.
        global_ids_this_rank: 1D tensor of global IDs present on this rank.
        offsets: Cumulative offset tensor for each DP rank.
    """
    num_local = local_seqlens.shape[0]
    local_len = torch.tensor([num_local], dtype=torch.int32, device=local_seqlens.device)
    dp_counts = [torch.zeros_like(local_len) for _ in range(dp_group.size())]
    torch.distributed.all_gather(dp_counts, local_len, group=dp_group)

    dp_counts_cpu = torch.stack(dp_counts, dim=0).cpu().view(-1)
    max_count = int(dp_counts_cpu.max().item())

    # Pad local seqlens to max_count for all_gather
    if num_local < max_count:
        padded = torch.cat(
            [local_seqlens, torch.zeros(max_count - num_local, dtype=torch.int32, device=local_seqlens.device)],
            dim=0,
        )
    else:
        padded = local_seqlens

    gathered = [torch.empty_like(padded) for _ in range(dp_group.size())]
    torch.distributed.all_gather(gathered, padded, group=dp_group)

    # Trim each rank's data to actual length
    for rank_idx, seqlen_tensor in enumerate(gathered):
        gathered[rank_idx] = seqlen_tensor[: dp_counts_cpu[rank_idx]]

    all_seqlens = torch.cat(gathered, dim=0).cpu().tolist()

    # Compute offsets
    csum = torch.cumsum(dp_counts_cpu, dim=0, dtype=torch.int32)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int32), csum], dim=0)

    # Global IDs
    dp_rank = dp_group.rank()
    global_id_seqlens = [(i, all_seqlens[i]) for i in range(len(all_seqlens))]
    start_idx = offsets[dp_rank].item()
    end_idx = offsets[dp_rank + 1].item()
    global_ids_this_rank = torch.arange(start_idx, end_idx, dtype=torch.int32, device=local_seqlens.device)

    return global_id_seqlens, global_ids_this_rank, offsets


# =============================================================================
# All-to-all routing
# =============================================================================


def _unique_keys(keys: list[str]) -> list[str]:
    seen = set()
    unique = []
    for key in keys:
        if key not in seen:
            unique.append(key)
            seen.add(key)
    return unique


def _group_rank_indices(group, within_group) -> list[int]:
    """Return ranks from `group` expressed as rank indices in `within_group`."""
    within_ranks = torch.distributed.get_process_group_ranks(within_group)
    rank_to_idx = {rank: idx for idx, rank in enumerate(within_ranks)}
    return [rank_to_idx[rank] for rank in torch.distributed.get_process_group_ranks(group)]


def _infer_same_cp_dcp_rank(
    dp_rank: int,
    peer_dcp_rank: int,
    dp_size: int,
    cp_size: int,
    current_dp_rank_to_dcp_rank: list[int],
) -> int:
    """Map a DP rank to the DPxCP rank that shares `peer_dcp_rank`'s CP column."""
    if cp_size == 1:
        return current_dp_rank_to_dcp_rank[dp_rank]
    if dp_size == 1:
        return peer_dcp_rank

    dp_stride = current_dp_rank_to_dcp_rank[1] - current_dp_rank_to_dcp_rank[0]
    if abs(dp_stride) == cp_size:
        cp_rank = peer_dcp_rank % cp_size
        return dp_rank * cp_size + cp_rank
    if abs(dp_stride) == 1:
        cp_rank = peer_dcp_rank // dp_size
        return cp_rank * dp_size + dp_rank

    raise ValueError(
        "Cannot infer DPxCP rank layout from current DP group mapping "
        f"{current_dp_rank_to_dcp_rank} with dp_size={dp_size}, cp_size={cp_size}"
    )


def _dp_src_rank_for_gid(gid: int, offsets: torch.Tensor) -> int:
    return int(torch.bucketize(torch.tensor(gid, dtype=offsets.dtype), offsets[1:] - 1).item())


def _shape_row(tensor: torch.Tensor, device) -> list[int]:
    shape = list(tensor.shape)
    if len(shape) > MAX_SHAPE_DIMS:
        raise ValueError(f"DCP routing supports tensors with at most {MAX_SHAPE_DIMS} dims, got shape={shape}")
    return [len(shape), *shape, *([0] * (MAX_SHAPE_DIMS - len(shape)))]


def _shape_from_row(row: torch.Tensor) -> tuple[int, ...]:
    ndim = int(row[0].item())
    if ndim == 0:
        return ()
    return tuple(int(x.item()) for x in row[1 : 1 + ndim])


def _stack_or_pad_samples(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Stack per-sample dense tensors, right-padding dimensions when DP ranks used different max lengths."""
    if not tensors:
        return torch.empty(0)
    shapes = [tuple(t.shape) for t in tensors]
    if len(set(shapes)) == 1:
        return torch.stack(tensors, dim=0)

    ndim = max(len(shape) for shape in shapes)
    if any(len(shape) != ndim for shape in shapes):
        raise ValueError(f"Cannot pad tensors with different ranks: {shapes}")

    max_shape = tuple(max(shape[dim] for shape in shapes) for dim in range(ndim))
    output = tensors[0].new_zeros((len(tensors), *max_shape))
    for i, tensor in enumerate(tensors):
        slices = (i, *[slice(0, size) for size in tensor.shape])
        output[slices] = tensor
    return output


def _reroute_samples(
    local_samples: list[dict[str, torch.Tensor]],
    global_ids_this_rank: torch.Tensor,
    global_id_seqlens: list[tuple[int, int]],
    sample_id_groups: list[list[list[int]]],
    offsets: torch.Tensor,
    dp_group,
    dcp_group,
    tensor_keys: list[str],
    scalar_keys: list[str],
    padded_keys: list[str] | None = None,
) -> dict[int, dict[str, torch.Tensor]]:
    """Reroute samples to correct DCP ranks via all-to-all.

    Args:
        local_samples: List of per-sample dicts on this rank.
        global_ids_this_rank: Global IDs of samples on this rank.
        global_id_seqlens: Full list of (global_id, seq_len) tuples.
        sample_id_groups: Per-microbatch, per-rank sample ID assignment.
            sample_id_groups[mb_idx][rank] = [global_id, ...]
        offsets: Cumulative sample count offsets per DP rank.
        dp_group: Data parallel process group that owns distinct data.
        dcp_group: DPxCP process group used for dynamic CP scheduling/routing.
        tensor_keys: Keys in sample dicts that are variable-length tensors.
        scalar_keys: Keys in sample dicts that are per-sample scalars.

    Returns:
        Dict mapping global_id -> sample dict for samples assigned to this rank.
    """
    total_dcp_gpus = dcp_group.size()
    dcp_rank = dcp_group.rank()
    dp_rank_to_dcp_rank = _group_rank_indices(dp_group, dcp_group)
    gid2local = {int(gid): i for i, gid in enumerate(global_ids_this_rank)}
    global_ids_set = set(gid2local.keys())

    def _gid_to_src_dcp_rank(gid: int) -> int:
        dp_src_rank = _dp_src_rank_for_gid(gid, offsets)
        return _infer_same_cp_dcp_rank(
            dp_src_rank,
            dcp_rank,
            dp_group.size(),
            dcp_group.size() // dp_group.size(),
            dp_rank_to_dcp_rank,
        )

    # Build combined sample ID groups per rank
    combined = [[] for _ in range(total_dcp_gpus)]
    for sample_id_group in sample_id_groups:
        for d in range(total_dcp_gpus):
            combined[d].extend(sample_id_group[d])
    for d in range(total_dcp_gpus):
        combined[d].sort()

    # Each CP replica owns the same local samples for its DP rank. To avoid duplicate
    # sends, a rank only serves destinations in its DP group expressed inside the
    # DPxCP group, matching Megatron-LM's hybrid DPxCP routing.
    send_groups = [[] for _ in range(total_dcp_gpus)]
    for dest_rank in dp_rank_to_dcp_rank:
        for gid in combined[dest_rank]:
            if gid in global_ids_set:
                send_groups[dest_rank].append(gid)
    send_ids_sorted = [gid for group in send_groups for gid in group]
    send_num_per_rank = [len(group) for group in send_groups]

    # Build recv plan
    recv_groups = [[] for _ in range(total_dcp_gpus)]
    for gid in combined[dcp_rank]:
        src_rank = _gid_to_src_dcp_rank(gid)
        recv_groups[src_rank].append(gid)

    recv_ids_sorted = [gid for d in range(total_dcp_gpus) for gid in recv_groups[d]]
    recv_counts = [len(recv_groups[d]) for d in range(total_dcp_gpus)]

    if padded_keys is None:
        padded_keys = []
    all_keys = _unique_keys(tensor_keys + scalar_keys + padded_keys)
    recv_samples = [{k: None for k in all_keys} for _ in range(sum(recv_counts))]
    dev = torch.cuda.current_device()

    def _pack_key_payload(key: str):
        parts = []
        numels = []
        shapes = []
        for gid in send_ids_sorted:
            sample = local_samples[gid2local[gid]]
            if key not in sample:
                numels.append(0)
                shapes.append([0, *([0] * MAX_SHAPE_DIMS)])
                continue
            t = sample[key].to(dev, non_blocking=True)
            numels.append(t.numel())
            shapes.append(_shape_row(t, dev))
            parts.append(t.reshape(-1))

        dtype = torch.float32
        for s in local_samples:
            if key in s:
                dtype = s[key].dtype
                break

        send_tensor = torch.cat(parts, dim=0) if parts else torch.empty(0, device=dev, dtype=dtype)
        send_numels = torch.tensor(numels, dtype=torch.int64, device=dev)
        send_shapes = torch.tensor(shapes, dtype=torch.int64, device=dev).reshape(-1)
        return send_tensor, send_numels, send_shapes

    shape_width = MAX_SHAPE_DIMS + 1
    for key in all_keys:
        send_tensor, send_numels, send_shapes = _pack_key_payload(key)

        recv_numels = torch.empty(sum(recv_counts), dtype=torch.int64, device=dev)
        torch.distributed.all_to_all_single(
            output=recv_numels,
            input=send_numels,
            output_split_sizes=recv_counts,
            input_split_sizes=send_num_per_rank,
            group=dcp_group,
        )

        recv_shapes = torch.empty(sum(recv_counts) * shape_width, dtype=torch.int64, device=dev)
        torch.distributed.all_to_all_single(
            output=recv_shapes,
            input=send_shapes,
            output_split_sizes=[count * shape_width for count in recv_counts],
            input_split_sizes=[count * shape_width for count in send_num_per_rank],
            group=dcp_group,
        )
        recv_shapes = recv_shapes.reshape(-1, shape_width)

        input_elem_splits = [0] * total_dcp_gpus
        cursor = 0
        for rank, count in enumerate(send_num_per_rank):
            for _ in range(count):
                input_elem_splits[rank] += int(send_numels[cursor].item())
                cursor += 1

        output_elem_splits = [0] * total_dcp_gpus
        cursor = 0
        for rank, count in enumerate(recv_counts):
            for _ in range(count):
                output_elem_splits[rank] += int(recv_numels[cursor].item())
                cursor += 1

        recv_size = sum(output_elem_splits)
        recv_tensor = torch.empty(recv_size, device=dev, dtype=send_tensor.dtype)

        torch.distributed.all_to_all_single(
            output=recv_tensor,
            input=send_tensor,
            output_split_sizes=output_elem_splits,
            input_split_sizes=input_elem_splits,
            group=dcp_group,
        )

        cursor = 0
        for i, _gid in enumerate(recv_ids_sorted):
            numel = int(recv_numels[i].item())
            if numel == 0:
                recv_samples[i].pop(key, None)
                continue
            shape = _shape_from_row(recv_shapes[i])
            recv_samples[i][key] = recv_tensor[cursor : cursor + numel].reshape(shape)
            cursor += numel

    return {recv_id: recv_samples[i] for i, recv_id in enumerate(recv_ids_sorted)}


# =============================================================================
# Micro-batch building
# =============================================================================


def _build_micro_batches_from_samples(
    samples_with_id: dict[int, dict[str, torch.Tensor]],
    sample_id_groups: list[list[list[int]]],
    dcp_rank: int,
    tensor_keys: list[str],
    scalar_keys: list[str],
    padded_keys: list[str] | None = None,
) -> tuple[list[dict[str, torch.Tensor]], list[int]]:
    """Build packed micro-batches from scheduled samples.

    Args:
        samples_with_id: Mapping from global_id -> sample dict.
        sample_id_groups: Per-microbatch, per-rank sample ID assignment.
        dcp_rank: This rank's index in the DCP group.
        tensor_keys: Variable-length tensor keys.
        scalar_keys: Per-sample scalar keys.

    Returns:
        Tuple of:
            - List of packed sample dicts (one per micro-batch).
            - List of local_cp_size per micro-batch.
    """
    num_micro_batches = len(sample_id_groups)
    local_cp_sizes = []

    for i in range(num_micro_batches):
        my_ids = sample_id_groups[i][dcp_rank]
        # local_cp_size = number of ranks that share the same first sample
        first_id = my_ids[0] if my_ids else -1
        cp_size = sum(1 for rank_ids in sample_id_groups[i] if first_id in rank_ids)
        local_cp_sizes.append(cp_size)

    packed_batches = []
    for i in range(num_micro_batches):
        my_ids = sample_id_groups[i][dcp_rank]
        samples = [samples_with_id[gid] for gid in my_ids]

        # Compute cu_seqlens (cumulative sequence lengths) for packed sequences
        seq_lens = torch.tensor(
            [samples_with_id[gid]["_seq_len"] for gid in my_ids],
            dtype=torch.int32,
            device=torch.cuda.current_device(),
        )
        cu_seqlens = torch.zeros(len(my_ids) + 1, dtype=torch.int32, device=torch.cuda.current_device())
        cu_seqlens[1:] = torch.cumsum(seq_lens, dim=0)

        packed = {
            "_samples": samples,
            "cu_seqlens": cu_seqlens,
            "max_seqlen": seq_lens.max() if len(seq_lens) > 0 else torch.tensor(0, dtype=torch.int32),
        }

        packed_batches.append(packed)

    return packed_batches, local_cp_sizes


# =============================================================================
# NestedTensor <-> per-sample conversion utilities
# =============================================================================


def _nested_tensor_to_samples(
    batch: TensorDict,
    tensor_keys: list[str],
    scalar_keys: list[str],
    padded_keys: list[str] | None = None,
) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor]:
    """Convert a TensorDict with NestedTensors into per-sample dicts.

    Args:
        batch: TensorDict containing NestedTensor fields.
        tensor_keys: Keys that are variable-length (NestedTensor).
        scalar_keys: Keys that are per-sample scalars.
        padded_keys: Keys that are padded tensors (dim 0 = batch size).

    Returns:
        Tuple of:
            - List of per-sample dicts.
            - 1D tensor of sequence lengths.
    """
    if padded_keys is None:
        padded_keys = []

    input_ids = batch["input_ids"]
    assert input_ids.is_nested, "input_ids must be a NestedTensor"
    seq_lens = input_ids.offsets().diff()
    num_samples = len(seq_lens)

    # Unbind NestedTensors
    unbound = {}
    for key in tensor_keys:
        if key in batch.keys():
            val = batch[key]
            if isinstance(val, torch.Tensor) and val.is_nested:
                unbound[key] = val.unbind()
            else:
                unbound[key] = None
        else:
            unbound[key] = None

    samples = []
    for i in range(num_samples):
        sample = {}
        for key in tensor_keys:
            if unbound[key] is not None:
                sample[key] = unbound[key][i]
            else:
                # If key doesn't exist or is not nested, skip
                pass
        for key in scalar_keys:
            if key in batch.keys():
                val = batch[key]
                if isinstance(val, torch.Tensor):
                    sample[key] = val[i : i + 1] if val.dim() > 0 else val.unsqueeze(0)
        for key in padded_keys:
            if key in batch.keys():
                val = batch[key]
                if isinstance(val, torch.Tensor) and val.dim() >= 1 and val.shape[0] == num_samples:
                    sample[key] = val[i]
        # Store seq_len as metadata for routing
        sample["_seq_len"] = int(seq_lens[i].item())
        samples.append(sample)

    return samples, seq_lens


def _samples_to_nested_tensor_batch(
    packed_batch: dict[str, torch.Tensor],
    cu_seqlens: torch.Tensor,
    tensor_keys: list[str],
    scalar_keys: list[str],
    padded_keys: list[str],
    local_cp_size: int,
    non_tensor_data: dict,
) -> TensorDict:
    """Convert a packed sample dict back to a TensorDict with NestedTensors.

    Args:
        packed_batch: Packed sample dict with concatenated tensors.
        cu_seqlens: Cumulative sequence lengths.
        tensor_keys: Variable-length tensor keys.
        scalar_keys: Per-sample scalar keys.
        local_cp_size: CP size for this micro-batch.
        non_tensor_data: Non-tensor metadata to preserve.

    Returns:
        TensorDict suitable for verl's forward pass.
    """
    samples = packed_batch["_samples"]
    num_seqs = len(samples)
    data_dict = {}

    for key in tensor_keys:
        per_seq = [sample[key] for sample in samples if key in sample and sample[key] is not None]
        if not per_seq:
            continue
        if len(per_seq) != num_seqs:
            raise ValueError(f"Key {key!r} is missing from some DCP-routed samples")
        data_dict[key] = torch.nested.as_nested_tensor(per_seq, layout=torch.jagged)

    for key in scalar_keys:
        tensors = [sample[key].reshape(-1) for sample in samples if key in sample and sample[key] is not None]
        if tensors:
            if len(tensors) != num_seqs:
                raise ValueError(f"Key {key!r} is missing from some DCP-routed samples")
            data_dict[key] = torch.cat(tensors, dim=0)

    for key in padded_keys:
        tensors = [sample[key] for sample in samples if key in sample and sample[key] is not None]
        if tensors:
            if len(tensors) != num_seqs:
                raise ValueError(f"Key {key!r} is missing from some DCP-routed samples")
            data_dict[key] = _stack_or_pad_samples(tensors)

    td = TensorDict(data_dict, batch_size=[num_seqs])

    # Assign non-tensor metadata
    for k, v in non_tensor_data.items():
        tu.assign_non_tensor(td, **{k: v})
    tu.assign_non_tensor_data(td, "local_cp_size", local_cp_size)

    return td


# =============================================================================
# Main scheduler class
# =============================================================================


def _build_reverse_routing_plans(routing_info: dict, dp_group, dcp_group):
    """Pre-compute send/recv plans for reverse routing (reused across keys).

    Returns:
        send_by_dest: send_by_dest[dest_rank] = [gid, ...] — ordered by dest rank
        recv_by_src:  recv_by_src[src_rank]  = [gid, ...] — ordered by src rank
        my_output_gids: global IDs whose outputs sit on this rank (scheduler order)
        send_ids_flat: flattened gid list in send order
        recv_ids_flat: flattened gid list in recv order
    """
    sample_id_groups = routing_info["sample_id_groups"]
    offsets = routing_info["offsets"]
    global_ids_this_rank = routing_info["global_ids_this_rank"]
    total = dcp_group.size()
    dcp_rank = dcp_group.rank()
    dp_rank_to_dcp_rank = _group_rank_indices(dp_group, dcp_group)
    dp_size = dp_group.size()
    cp_size = dcp_group.size() // dp_size

    def _gid_to_orig_rank_for_peer(gid: int, peer_dcp_rank: int) -> int:
        dp_src_rank = _dp_src_rank_for_gid(gid, offsets)
        return _infer_same_cp_dcp_rank(dp_src_rank, peer_dcp_rank, dp_size, cp_size, dp_rank_to_dcp_rank)

    # Combined sample assignment per rank (same as forward routing)
    combined = [[] for _ in range(total)]
    for grp in sample_id_groups:
        for d in range(total):
            combined[d].extend(grp[d])

    # my_output_gids preserves micro-batch concatenation order (matches postprocess output)
    my_output_gids = list(combined[dcp_rank])

    # Sort combined for consistent all-to-all communication plans
    for d in range(total):
        combined[d].sort()

    original_gids = set(int(g) for g in global_ids_this_rank)

    # Reverse: outputs go from scheduled rank → original rank
    send_by_dest = [[] for _ in range(total)]
    for gid in my_output_gids:
        send_by_dest[_gid_to_orig_rank_for_peer(gid, dcp_rank)].append(gid)
    for dest_rank in range(total):
        send_by_dest[dest_rank].sort()

    recv_by_src = [[] for _ in range(total)]
    for src_rank in range(total):
        for gid in combined[src_rank]:
            if gid in original_gids and _gid_to_orig_rank_for_peer(gid, src_rank) == dcp_rank:
                recv_by_src[src_rank].append(gid)

    send_ids_flat = [gid for dest in range(total) for gid in send_by_dest[dest]]
    recv_ids_flat = [gid for src in range(total) for gid in recv_by_src[src]]

    return send_by_dest, recv_by_src, my_output_gids, send_ids_flat, recv_ids_flat


def reverse_route_outputs(
    model_output: dict[str, torch.Tensor],
    routing_info: dict,
    dp_group,
    dcp_group=None,
) -> dict[str, torch.Tensor]:
    """Reverse-route outputs back to original DP ranks via all-to-all.

    The forward scheduling sent samples from their original ranks to scheduled
    ranks. After the forward pass, outputs must be returned to the original
    ranks so that the caller sees them in the original sample order.

    Args:
        model_output: Dict of NestedTensors (keys like 'log_probs', 'entropy').
        routing_info: Dict returned by DynamicCPScheduler.schedule().
        dp_group: Data parallel process group.

    Returns:
        Dict of NestedTensors in the original sample order for this rank.
    """
    if dcp_group is None:
        dcp_group = dp_group
    total_dcp_gpus = dcp_group.size()
    dev = torch.cuda.current_device()

    send_by_dest, recv_by_src, my_output_gids, send_ids_flat, recv_ids_flat = _build_reverse_routing_plans(
        routing_info, dp_group, dcp_group
    )

    send_counts = [len(send_by_dest[d]) for d in range(total_dcp_gpus)]
    recv_counts = [len(recv_by_src[d]) for d in range(total_dcp_gpus)]

    # All-to-all for each output key
    reversed_output = {}
    shape_width = MAX_SHAPE_DIMS + 1

    for key, val in model_output.items():
        if not isinstance(val, torch.Tensor):
            reversed_output[key] = val
            continue

        per_sample = list(val.unbind()) if val.is_nested else [val[i] for i in range(val.shape[0])]
        gid_to_out = {}
        for i, gid in enumerate(my_output_gids):
            if i < len(per_sample):
                gid_to_out[gid] = per_sample[i]

        parts = []
        send_numel_values = []
        send_shape_values = []
        for gid in send_ids_flat:
            if gid in gid_to_out:
                out = gid_to_out[gid].to(dev)
                parts.append(out.reshape(-1))
                send_numel_values.append(out.numel())
                send_shape_values.append(_shape_row(out, dev))
            else:
                send_numel_values.append(0)
                send_shape_values.append([0, *([0] * MAX_SHAPE_DIMS)])
        send_tensor = torch.cat(parts, dim=0) if parts else torch.empty(0, device=dev, dtype=val.dtype)
        send_numels = torch.tensor(send_numel_values, dtype=torch.int64, device=dev)
        recv_numels = torch.empty(len(recv_ids_flat), dtype=torch.int64, device=dev)
        torch.distributed.all_to_all_single(
            output=recv_numels,
            input=send_numels,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            group=dcp_group,
        )

        send_shapes = torch.tensor(send_shape_values, dtype=torch.int64, device=dev).reshape(-1)
        recv_shapes = torch.empty(len(recv_ids_flat) * shape_width, dtype=torch.int64, device=dev)
        torch.distributed.all_to_all_single(
            output=recv_shapes,
            input=send_shapes,
            output_split_sizes=[count * shape_width for count in recv_counts],
            input_split_sizes=[count * shape_width for count in send_counts],
            group=dcp_group,
        )
        recv_shapes = recv_shapes.reshape(-1, shape_width)

        send_elem_splits = [0] * total_dcp_gpus
        cursor = 0
        for rank, count in enumerate(send_counts):
            for _ in range(count):
                send_elem_splits[rank] += int(send_numels[cursor].item())
                cursor += 1

        recv_elem_splits = [0] * total_dcp_gpus
        cursor = 0
        for rank, count in enumerate(recv_counts):
            for _ in range(count):
                recv_elem_splits[rank] += int(recv_numels[cursor].item())
                cursor += 1

        recv_tensor = torch.empty(sum(recv_elem_splits), device=dev, dtype=send_tensor.dtype)

        torch.distributed.all_to_all_single(
            output=recv_tensor,
            input=send_tensor,
            output_split_sizes=recv_elem_splits,
            input_split_sizes=send_elem_splits,
            group=dcp_group,
        )

        gid_to_result = {}
        cursor = 0
        for i, gid in enumerate(recv_ids_flat):
            n = int(recv_numels[i].item())
            shape = _shape_from_row(recv_shapes[i])
            gid_to_result[gid] = recv_tensor[cursor : cursor + n].reshape(shape)
            cursor += n

        # Rebuild in original order (sorted by global_id within this rank)
        original_gids_sorted = sorted(int(g) for g in routing_info["global_ids_this_rank"])
        result_tensors = [gid_to_result[gid] for gid in original_gids_sorted if gid in gid_to_result]

        if result_tensors:
            reversed_output[key] = torch.nested.as_nested_tensor(result_tensors, layout=torch.jagged)
        else:
            reversed_output[key] = val

    return reversed_output


# =============================================================================
# PP broadcast for dynamic CP metadata
# =============================================================================


def broadcast_dcp_metadata_to_pp(
    micro_batches: list[TensorDict],
    pp_group,
) -> list[TensorDict]:
    """Broadcast dynamic CP metadata from first/last PP stage to middle stages.

    Middle PP stages need local_cp_size and cu_seqlens for attention computation
    but do not receive full batch data. This broadcasts the metadata from the
    first PP stage (rank 0 in pp_group).

    Reference: megatron-lm broadcast_to_pp_group() in data_schedule_utils.py.

    Args:
        micro_batches: List of TensorDict micro-batches (populated on first/last PP).
        pp_group: Pipeline parallel process group.

    Returns:
        micro_batches with metadata populated on all PP stages.
    """
    if pp_group.size() <= 2:
        # Only first and last stage — no middle stages to broadcast to
        return micro_batches

    pp_src_rank = torch.distributed.get_process_group_ranks(pp_group)[0]
    dev = torch.cuda.current_device()

    if pp_group.rank() == 0:
        # First PP stage: pack metadata and broadcast
        num_mb = len(micro_batches)
        cp_sizes = []
        cu_seqlens_list = []
        for mb in micro_batches:
            cp_size = tu.get_non_tensor_data(mb, key="local_cp_size", default=1)
            cp_sizes.append(int(cp_size))
            # Extract cu_seqlens from NestedTensor offsets
            input_ids = mb["input_ids"]
            if input_ids.is_nested:
                offsets = input_ids.offsets().to(dtype=torch.int32, device=dev)
                cu_seqlens_list.append(offsets)
            else:
                n = input_ids.shape[0]
                cu_seqlens_list.append(torch.arange(n + 1, dtype=torch.int32, device=dev))

        header = torch.tensor(
            [num_mb, *cp_sizes, *[int(cu.numel()) for cu in cu_seqlens_list]],
            dtype=torch.int32,
            device=dev,
        )
        info = torch.cat([header, *cu_seqlens_list], dim=0)

        info_len = torch.tensor(info.shape[0], dtype=torch.int32, device=dev)
        torch.distributed.broadcast(info_len, pp_src_rank, group=pp_group)
        torch.distributed.broadcast(info, pp_src_rank, group=pp_group)
    else:
        info_len = torch.tensor(0, dtype=torch.int32, device=dev)
        torch.distributed.broadcast(info_len, pp_src_rank, group=pp_group)
        info = torch.empty(int(info_len.item()), dtype=torch.int32, device=dev)
        torch.distributed.broadcast(info, pp_src_rank, group=pp_group)

        if pp_group.rank() != pp_group.size() - 1:
            # Middle PP stage: unpack metadata
            num_mb = int(info[0].item())
            cp_sizes = info[1 : 1 + num_mb].tolist()
            cu_lens = info[1 + num_mb : 1 + 2 * num_mb].tolist()
            cu_start = 1 + 2 * num_mb

            micro_batches = []
            for i in range(num_mb):
                cu_len = int(cu_lens[i])
                cu_seqlens = info[cu_start : cu_start + cu_len].to(device=dev, dtype=torch.int64)
                cu_start += cu_len
                seq_lens = cu_seqlens.diff().tolist()
                dummy = [torch.zeros(int(seq_len), dtype=torch.long, device=dev) for seq_len in seq_lens]
                input_ids = torch.nested.as_nested_tensor(dummy, layout=torch.jagged)
                td = TensorDict({"input_ids": input_ids}, batch_size=[len(dummy)])
                tu.assign_non_tensor_data(td, "local_cp_size", int(cp_sizes[i]))
                tu.assign_non_tensor_data(td, "_dcp_scheduled", True)
                micro_batches.append(td)

    return micro_batches


# =============================================================================
# Main scheduler class
# =============================================================================


class DynamicCPScheduler:
    """Dynamic Context Parallelism Scheduler.

    Collects sequence lengths from all DP ranks, schedules samples into
    micro-batches with optimal CP sizes, routes samples via all-to-all,
    and builds packed micro-batches.

    Args:
        max_seqlen_per_dp_cp_rank: Max sequence length per DPxCP rank.
        dp_size: Data parallel world size.
        min_cp_size: Minimum CP group size (default 1).
        microbatch_group_size_per_vp_stage: For VPP alignment (default None).
    """

    # verl data keys
    # Core keys that are always routed if present.
    # Additional NestedTensor keys in the batch are auto-detected in schedule().
    TENSOR_KEYS = ["input_ids", "position_ids", "loss_mask", "prompts", "responses", "attention_mask"]
    SCALAR_KEYS = ["temperature"]

    def __init__(
        self,
        max_seqlen_per_dp_cp_rank: int,
        dp_size: int,
        cp_size: int = 1,
        min_cp_size: int = 1,
        microbatch_group_size_per_vp_stage: int | None = None,
    ):
        self.max_seqlen_per_dp_cp_rank = max_seqlen_per_dp_cp_rank
        self.dp_size = dp_size
        self.cp_size = cp_size
        self.total_dcp_gpus = dp_size * cp_size
        self.min_cp_size = min_cp_size
        self.microbatch_group_size_per_vp_stage = microbatch_group_size_per_vp_stage

    def _get_groups_and_subsamples(self, sample_id_seqlens: list[tuple[int, int]]) -> list[list[list[int]]]:
        """Schedule samples into micro-batch groups with dynamic CP sizes.

        Args:
            sample_id_seqlens: List of (global_id, seq_len) tuples.

        Returns:
            sample_id_groups: Per-microbatch, per-rank assignment of sample IDs.
                sample_id_groups[mb_idx][rank] = [global_id, ...]
        """
        mslpr = self.max_seqlen_per_dp_cp_rank
        min_cp = self.min_cp_size

        workload_fn = lambda seq_len, cp_size=None: dcp_get_total_workload(seq_len, mslpr, cp_size, min_cp)
        gpus_fn = lambda seq_len: dcp_gpus_needed(seq_len, mslpr, min_cp)
        buckets_fn = lambda sample_seqlens, compute_est: dcp_make_buckets_equal(
            sample_seqlens, compute_est, mslpr, min_cp
        )

        sample_id_groups = []
        sample_id_seqlens = sorted(sample_id_seqlens, key=lambda x: x[1], reverse=True)
        oversized = [(sample_id, seq_len, gpus_fn(seq_len)) for sample_id, seq_len in sample_id_seqlens]
        oversized = [
            (sample_id, seq_len, needed) for sample_id, seq_len, needed in oversized if needed > self.total_dcp_gpus
        ]
        if oversized:
            sample_id, seq_len, needed = oversized[0]
            raise ValueError(
                "Dynamic CP cannot schedule sample "
                f"{sample_id} with seq_len={seq_len}: needs {needed} ranks but DPxCP group has "
                f"{self.total_dcp_gpus}. Increase max_seqlen_per_dp_cp_rank or the DPxCP group size."
            )

        while sample_id_seqlens:
            num_left_before = len(sample_id_seqlens)
            mb, sample_id_seqlens, exec_times, sample_ids = next_hdp_group(
                sample_id_seqlens,
                workload_fn,
                self.total_dcp_gpus,
                gpus_needed_fn=gpus_fn,
                make_buckets_equal_fn=buckets_fn,
                max_seq_len_per_rank=mslpr,
                get_total_workload_fn=workload_fn,
            )
            if len(sample_id_seqlens) == num_left_before:
                raise RuntimeError(
                    "Dynamic CP scheduler made no progress. "
                    "Check max_seqlen_per_dp_cp_rank, min_cp_size, and DPxCP group size."
                )
            sample_id_groups.append(sample_ids)

        if self.microbatch_group_size_per_vp_stage is not None and self.microbatch_group_size_per_vp_stage > 1:
            sample_id_groups = align_sample_id_groups(sample_id_groups, self.microbatch_group_size_per_vp_stage)

        return sample_id_groups

    def schedule(
        self,
        batch: TensorDict,
        dp_group,
        dcp_group=None,
        non_tensor_data: dict | None = None,
    ) -> tuple[list[TensorDict], dict]:
        """Run the full dynamic CP scheduling pipeline.

        Args:
            batch: Local TensorDict with NestedTensor data for this DP rank.
            dp_group: Data parallel process group with distinct data.
            dcp_group: DPxCP process group used by dynamic context parallelism.
            non_tensor_data: Non-tensor metadata to propagate to micro-batches.

        Returns:
            Tuple of:
                - micro_batches: List of TensorDict, each with local_cp_size
                  and _dcp_scheduled=True set as non-tensor data.
                - routing_info: Dict with routing metadata for reverse_route_outputs().
        """
        if non_tensor_data is None:
            non_tensor_data = {}
        if dcp_group is None:
            dcp_group = dp_group
        assert dcp_group.size() == self.total_dcp_gpus, (
            f"DCP group size {dcp_group.size()} does not match dp_size * cp_size "
            f"({self.dp_size} * {self.cp_size} = {self.total_dcp_gpus})"
        )

        # Determine which keys exist in this batch. Known verl fields keep a
        # stable order, but their actual tensor layout still decides routing:
        # jagged NestedTensors are sent as variable-length tensors; dense tensors
        # with batch dimension are routed as per-sample padded tensors.
        ordered_keys = _unique_keys([k for k in self.TENSOR_KEYS if k in batch.keys()] + list(batch.keys()))
        tensor_keys = []
        padded_keys = []
        scalar_keys = []
        for k in ordered_keys:
            val = batch[k]
            if not isinstance(val, torch.Tensor):
                continue
            if k in self.SCALAR_KEYS:
                scalar_keys.append(k)
            elif val.is_nested:
                tensor_keys.append(k)
            elif val.dim() >= 1:
                padded_keys.append(k)

        # Step 1: Convert NestedTensor batch to per-sample format
        local_samples, local_seqlens = _nested_tensor_to_samples(batch, tensor_keys, scalar_keys, padded_keys)

        # Step 2: Gather global sequence lengths from all DP ranks
        local_seqlens_gpu = local_seqlens.to(dtype=torch.int32, device=get_device_id())
        global_id_seqlens, global_ids_this_rank, offsets = _get_global_seqlens_and_ids(local_seqlens_gpu, dp_group)

        # Step 3: Schedule samples into groups
        sample_id_groups = self._get_groups_and_subsamples(global_id_seqlens)

        # Validate: all samples are assigned
        assigned = set()
        for group in sample_id_groups:
            for rank_ids in group:
                assigned.update(rank_ids)
        assert len(assigned) == len(global_id_seqlens), (
            f"Scheduling assigned {len(assigned)} samples but expected {len(global_id_seqlens)}"
        )

        # Step 4: Reroute samples via all-to-all
        samples_this_rank = _reroute_samples(
            local_samples,
            global_ids_this_rank,
            global_id_seqlens,
            sample_id_groups,
            offsets,
            dp_group,
            dcp_group,
            tensor_keys,
            scalar_keys,
            padded_keys,
        )

        # Add _seq_len to received samples
        for gid, sample in samples_this_rank.items():
            sample["_seq_len"] = global_id_seqlens[gid][1]

        # Step 5: Build packed micro-batches
        dcp_rank = dcp_group.rank()
        packed_batches, local_cp_sizes = _build_micro_batches_from_samples(
            samples_this_rank,
            sample_id_groups,
            dcp_rank,
            tensor_keys,
            scalar_keys,
            padded_keys,
        )

        # Mark micro-batches as scheduler-managed (skip legacy dynamic_cp_merge_output)
        non_tensor_data["_dcp_scheduled"] = True

        # Step 6: Convert packed batches to TensorDict with NestedTensor
        micro_batches = []
        for i, (packed, cp_size) in enumerate(zip(packed_batches, local_cp_sizes, strict=True)):
            td = _samples_to_nested_tensor_batch(
                packed,
                packed_batches[i]["cu_seqlens"],
                tensor_keys,
                scalar_keys,
                padded_keys,
                local_cp_size=cp_size,
                non_tensor_data=non_tensor_data,
            )
            micro_batches.append(td)

        # Build routing info for reverse_route_outputs()
        routing_info = {
            "global_id_seqlens": global_id_seqlens,
            "global_ids_this_rank": global_ids_this_rank,
            "sample_id_groups": sample_id_groups,
            "offsets": offsets,
        }

        logger.info(
            f"DynamicCPScheduler: {len(global_id_seqlens)} samples -> "
            f"{len(micro_batches)} micro-batches, "
            f"local_cp_sizes={local_cp_sizes}"
        )

        return micro_batches, routing_info
