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

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

_DTYPE_INT = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _prod(xs) -> int:
    n = 1
    for x in xs:
        n *= int(x)
    return n


def local_shard_view(param: torch.Tensor, mesh_rank0_only: bool = True):
    """Return (local_flat_shard, within_param_flat_offset, contributes).

    * For a ``Shard(0)`` DTensor: the rank's local rows, flattened, and their flat offset
      into the full (flattened) parameter, computed purely locally from the DTensor spec
      (``compute_local_shape_and_global_offset`` does no collective).
    * For a replicated / plain tensor: the whole tensor is present on every rank, so only
      one rank should contribute it -- ``contributes`` is False on the others to avoid
      double-counting (gated on the mesh coordinate, else global rank 0).
    """
    if not isinstance(param, DTensor):
        return param.reshape(-1), 0, (dist.get_rank() == 0 if dist.is_initialized() else True)

    placements = param.placements
    for p in placements:
        if p.is_shard() and p.dim != 0:
            raise NotImplementedError(
                f"sharded delta only supports Shard(0) (FSDP2 default); got placements={placements}"
            )

    # A parameter is replicated along any Replicate mesh dim (e.g. the ulysses/SP dim of a
    # 2D FSDP mesh). Every rank on that dim holds the *same* shard, so only the coord-0 rank
    # should contribute -- otherwise the gather double-counts it.
    coord = param.device_mesh.get_coordinate()
    contributes = True
    if coord is not None:
        for mesh_dim, p in enumerate(placements):
            if p.is_replicate() and coord[mesh_dim] != 0:
                contributes = False
                break

    if all(p.is_replicate() for p in placements):
        return param.to_local().reshape(-1), 0, contributes

    _, global_offset = compute_local_shape_and_global_offset(
        param.shape, param.device_mesh, param.placements
    )
    inner = _prod(param.shape[1:])
    offset = int(global_offset[0]) * inner
    return param.to_local().reshape(-1), offset, contributes


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


def gather_v_to_rank0(
    local_idx: torch.Tensor,
    local_val: torch.Tensor,
    group=None,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """Gather variable-length (idx, val) from every rank to rank 0 (pad-to-max ``dist.gather``).

    Returns (idx, val) on rank 0 (concatenated, padding stripped) and (None, None) elsewhere.
    The counts are exchanged with a fixed-size all_gather; the payloads then ride a single
    padded gather to rank 0 -- so nothing all-gathers the full tensor.
    """
    rank = dist.get_rank(group)
    world = dist.get_world_size(group)
    dev = local_idx.device

    n = int(local_idx.numel())
    cnt = torch.tensor([n], dtype=torch.long, device=dev)
    counts = [torch.zeros(1, dtype=torch.long, device=dev) for _ in range(world)]
    dist.all_gather(counts, cnt, group=group)
    counts = [int(c.item()) for c in counts]
    max_n = max(counts) if counts else 0
    if max_n == 0:
        return (torch.empty(0, dtype=torch.int64, device=dev),
                torch.empty(0, dtype=local_val.dtype, device=dev)) if rank == 0 else (None, None)

    idx_pad = torch.zeros(max_n, dtype=torch.int64, device=dev)
    val_pad = torch.zeros(max_n, dtype=local_val.dtype, device=dev)
    idx_pad[:n] = local_idx
    val_pad[:n] = local_val

    idx_list = [torch.zeros(max_n, dtype=torch.int64, device=dev) for _ in range(world)] if rank == 0 else None
    val_list = [torch.zeros(max_n, dtype=local_val.dtype, device=dev) for _ in range(world)] if rank == 0 else None
    dist.gather(idx_pad, idx_list, dst=0, group=group)
    dist.gather(val_pad, val_list, dst=0, group=group)

    if rank != 0:
        return None, None
    idx = torch.cat([idx_list[r][: counts[r]] for r in range(world)])
    val = torch.cat([val_list[r][: counts[r]] for r in range(world)])
    return idx, val
