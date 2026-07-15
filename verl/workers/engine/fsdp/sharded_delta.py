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
"""FSDP shard-view helper for the sharded delta export: reads the DTensor spec
to expose this rank's ``Shard(0)`` slice and its flat offset in the full
parameter -- the FSDP trainer backend's half of the shard-export contract."""

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

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
