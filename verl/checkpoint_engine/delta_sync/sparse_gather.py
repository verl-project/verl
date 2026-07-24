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
"""Sharded delta: diff each rank's local FSDP shard, gather only the changes to rank 0.

The default delta path all-gathers the full parameter (``DTensor.full_tensor()``) and
byte-diffs it against a full-model pinned-CPU snapshot on rank 0. This module instead lets
each rank keep a pinned snapshot of only *its* shard, byte-diff the shard locally, and
gather just the changed ``(within-parameter position, value)`` pairs to rank 0 -- so the
all-gather volume drops to the sparsity ratio (~1-3%) and rank 0 no longer needs a
full-model snapshot. The gathered result is bit-identical to the full-tensor diff, so the
downstream encode + broadcast and the receiver are unchanged.

Scope: FSDP2 ``Shard(0)`` DTensors (the common case) + replicated / non-DTensor params.
Other shard dims are strided in the flattened layout and raise NotImplementedError.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    pass

_DTYPE_INT = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def shard_delta_indices(
    local_new: torch.Tensor,
    local_snap: torch.Tensor,
    offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Byte-diff a local shard against its snapshot; return (global_positions, values).

    Positions are int64 indices into the *full flattened parameter* (offset + local index).
    Dtype-agnostic, bytewise (view-as-int), no arithmetic -- matches ``bytewise_diff_mask``.
    """
    es = local_new.element_size()
    int_dtype = _DTYPE_INT.get(es)
    if int_dtype is None:
        raise ValueError(f"unsupported element size {es}")
    mask = local_new.view(int_dtype) != local_snap.view(int_dtype)
    local_idx = mask.nonzero(as_tuple=False).view(-1)
    values = local_new[local_idx]
    global_idx = local_idx.to(torch.int64) + offset
    return global_idx, values


def gather_slot_entries_to_rank0(
    idx_concat: torch.Tensor,
    val_concat: torch.Tensor,
    counts: torch.Tensor,
    group: dist.ProcessGroup | None = None,
    max_round_bytes: int | None = None,
) -> list | None:
    """Variable-length sparse gather, batched: one collective round for K parameters.

    Each rank passes its K per-parameter deltas concatenated (``idx_concat``,
    ``val_concat``) plus the per-parameter length vector ``counts`` ([K] int64).
    One all_gather exchanges the K x world count matrix; two padded gathers move
    the blobs. Rank 0 slices per (rank, param) and returns K ``(idx, val)`` pairs
    (None elsewhere) -- bit-identical to K individual gathers, ~K x fewer
    collectives and host syncs.
    """
    rank = dist.get_rank(group)
    world = dist.get_world_size(group)
    dst = dist.get_global_rank(group, 0) if group is not None else 0
    dev = idx_concat.device
    k = int(counts.numel())

    counts_all = [torch.zeros_like(counts) for _ in range(world)]
    dist.all_gather(counts_all, counts.to(dev), group=group)
    counts_cpu = torch.stack(counts_all).cpu().tolist()  # one D2H sync instead of `world`

    if max_round_bytes is not None and k > 1:
        # Deterministic sub-rounds: every rank sees the same counts matrix, so all
        # ranks derive the SAME slot partition (no per-rank trigger asymmetry) and
        # each padded round's largest blob stays within the byte budget.
        per_elem = idx_concat.element_size() + val_concat.element_size()
        budget = max(int(max_round_bytes) // per_elem, 1)  # elements per rank per round
        cuts = [0]
        run = [0] * world
        for i in range(k):
            run = [run[r] + counts_cpu[r][i] for r in range(world)]
            if max(run) > budget and cuts[-1] != i:
                cuts.append(i)
                run = [counts_cpu[r][i] for r in range(world)]
        cuts.append(k)
        if len(cuts) > 2:
            out_all: list = []
            my_off = [0]
            for i in range(k):
                my_off.append(my_off[-1] + counts_cpu[rank][i])
            for lo, hi in zip(cuts[:-1], cuts[1:], strict=False):
                sub_counts = torch.tensor(counts_cpu[rank][lo:hi], dtype=torch.int64, device=dev)
                sub = gather_slot_entries_to_rank0(
                    idx_concat[my_off[lo] : my_off[hi]],
                    val_concat[my_off[lo] : my_off[hi]],
                    sub_counts,
                    group=group,
                )
                if rank == 0:
                    out_all.extend(sub)
            return out_all if rank == 0 else None

    totals = [sum(c) for c in counts_cpu]
    max_n = max(totals) if totals else 0
    if max_n == 0:
        if rank != 0:
            return None
        empty_i = torch.empty(0, dtype=idx_concat.dtype, device=dev)
        empty_v = torch.empty(0, dtype=val_concat.dtype, device=dev)
        return [(empty_i, empty_v) for _ in range(k)]

    idx_pad = torch.zeros(max_n, dtype=idx_concat.dtype, device=dev)
    val_pad = torch.zeros(max_n, dtype=val_concat.dtype, device=dev)
    n = int(idx_concat.numel())
    idx_pad[:n] = idx_concat
    val_pad[:n] = val_concat

    idx_list = [torch.zeros(max_n, dtype=idx_pad.dtype, device=dev) for _ in range(world)] if rank == 0 else None
    val_list = [torch.zeros(max_n, dtype=val_concat.dtype, device=dev) for _ in range(world)] if rank == 0 else None
    dist.gather(idx_pad, idx_list, dst=dst, group=group)
    dist.gather(val_pad, val_list, dst=dst, group=group)
    if rank != 0:
        return None

    # per-rank cumulative offsets into each blob, sliced per param then stitched across ranks
    offs = [[0] * (k + 1) for _ in range(world)]
    for r in range(world):
        for i in range(k):
            offs[r][i + 1] = offs[r][i] + counts_cpu[r][i]
    out = []
    for i in range(k):
        idx_pieces = [idx_list[r][offs[r][i] : offs[r][i + 1]] for r in range(world) if counts_cpu[r][i]]
        val_pieces = [val_list[r][offs[r][i] : offs[r][i + 1]] for r in range(world) if counts_cpu[r][i]]
        if idx_pieces:
            out.append((torch.cat(idx_pieces), torch.cat(val_pieces)))
        else:
            out.append(
                (torch.empty(0, dtype=idx_concat.dtype, device=dev), torch.empty(0, dtype=val_concat.dtype, device=dev))
            )
    return out
