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
"""The shard-export contract between training engines and the sharded delta engine.

``BaseEngine.get_per_tensor_param_shard`` yields ``(name, local_shard, ShardSpec)``
per local parameter, in an order identical on every rank. The spec describes the
parameter's distribution declaratively with torch's own vocabulary -- a
:class:`~torch.distributed.device_mesh.DeviceMesh` plus
:class:`~torch.distributed.tensor.placement_types.Placement` per mesh dim -- and
the engine derives everything else (this rank's flat offset, the gather group,
whether this rank contributes) via ``compute_local_shape_and_global_offset``.
DTensor-based trainers (FSDP, veomni, ...) pass ``param.device_mesh`` /
``param.placements`` verbatim; ``mesh=None`` means the local tensor already is
the whole parameter (replicated / unsharded).

``to_hf`` is reserved for trainers whose logical parameter differs from the HF
tensor(s) (e.g. Megatron fused qkv): a pure-permutation callable mapping the
gather group's dense shards to ``[(hf_name, hf_tensor)]``. It is None for
DTensor trainers and lands with the Megatron follow-up PR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

__all__ = ["ShardSpec", "derive_placement"]


@dataclass
class ShardSpec:
    """Declarative placement descriptor for one exported local parameter shard."""

    # Logical (full) tensor shape; the distribution facts below refer to it.
    full_shape: tuple
    # Distribution: torch DeviceMesh + per-mesh-dim Placement. None = unsharded.
    mesh: Optional[object] = None
    placements: Optional[tuple] = None
    # Reserved (Megatron follow-up): pure-permutation shards -> [(hf_name, hf_tensor)].
    to_hf: Optional[Callable[[list[torch.Tensor]], list[tuple[str, torch.Tensor]]]] = None

    @classmethod
    def from_param(cls, param: torch.Tensor) -> ShardSpec:
        if isinstance(param, DTensor):
            return cls(full_shape=tuple(param.shape), mesh=param.device_mesh, placements=tuple(param.placements))
        return cls(full_shape=tuple(param.shape))


def _prod(xs) -> int:
    n = 1
    for x in xs:
        n *= int(x)
    return n


def derive_placement(spec: ShardSpec):
    """Derive ``(flat_offset, contributes, gather_group)`` for THIS rank from the spec.

    * unsharded (``mesh is None``): offset 0; only global rank 0 contributes; no group
      (the local tensor is already the full parameter).
    * ``Shard(0)`` over one mesh dim (+ any number of ``Replicate`` dims): the flat
      offset comes from ``compute_local_shape_and_global_offset`` (pure math, no
      collective); only ranks at coordinate 0 of every Replicate dim contribute; the
      gather group is the Shard dim's subgroup.

    Other shard dims raise -- same scope as the export that produces the spec.
    """
    import torch.distributed as dist

    if spec.mesh is None:
        return 0, (dist.get_rank() == 0 if dist.is_initialized() else True), None

    placements = spec.placements
    shard_dims = [d for d, p in enumerate(placements) if p.is_shard()]
    for d in shard_dims:
        if placements[d].dim != 0:
            raise NotImplementedError(
                f"sharded delta only supports Shard(0) (FSDP2 default); got placements={placements}"
            )
    assert len(shard_dims) <= 1, f"at most one Shard dim is supported; got placements={placements}"

    coord = spec.mesh.get_coordinate()
    contributes = True
    if coord is not None:
        for d, p in enumerate(placements):
            if p.is_replicate() and coord[d] != 0:
                contributes = False
                break

    if not shard_dims:
        # replicated across every mesh dim: full tensor on each rank, no gather
        return 0, contributes, None

    _, global_offset = compute_local_shape_and_global_offset(spec.full_shape, spec.mesh, list(placements))
    inner = _prod(spec.full_shape[1:])
    offset = int(global_offset[0]) * inner
    group = spec.mesh.get_group(mesh_dim=shard_dims[0])
    return offset, contributes, group
