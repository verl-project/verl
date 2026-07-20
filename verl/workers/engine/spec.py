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

__all__ = ["BlockPlacement", "ShardSpec", "derive_placement", "translate_flat_indices"]


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
    # Explicit placement override for trainers whose sharding is not fully captured
    # by DTensor placements (e.g. veomni's manual expert-dim split): ``place`` is an
    # int flat offset or a BlockPlacement in a *virtual* full tensor, and
    # ``gather_group`` is the ProcessGroup covering every rank that holds a block
    # (pass a real group object -- the engine treats ``None`` as "unsharded").
    # Every rank is assumed to contribute.
    place: Optional[object] = None
    gather_group: Optional[object] = None

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


def _row_major_strides(shape) -> tuple:
    strides, acc = [], 1
    for d in reversed([int(x) for x in shape]):
        strides.append(acc)
        acc *= d
    return tuple(reversed(strides))


@dataclass(frozen=True)
class BlockPlacement:
    """This rank's local shard is one hyper-rectangular block of the full tensor:
    ``full[o0:o0+l0, o1:o1+l1, ...]`` with ``local_shape=(l0, l1, ...)`` and
    ``global_offset=(o0, o1, ...)``. Produced by :func:`derive_placement` for every
    sharded geometry, including the dim-0 cut (FSDP2 ``Shard(0)``): that block is
    flat-contiguous, and :func:`translate_flat_indices` detects it via
    ``is_flat_contiguous`` and keeps the single-add fast path."""

    local_shape: tuple
    global_offset: tuple
    full_shape: tuple

    @property
    def local_strides(self) -> tuple:
        return _row_major_strides(self.local_shape)

    @property
    def full_strides(self) -> tuple:
        return _row_major_strides(self.full_shape)

    @property
    def is_flat_contiguous(self) -> bool:
        """True when the block is one contiguous flat range (only dim 0 is cut).

        Then every trailing dim is whole, so the trailing offsets are all zero
        and the block occupies flat positions ``[flat_offset, flat_offset+numel)``.
        """
        return all(int(lo) == int(fu) for lo, fu in zip(self.local_shape[1:], self.full_shape[1:], strict=False))

    @property
    def flat_offset(self) -> int:
        """Flat start of the block; only meaningful when ``is_flat_contiguous``."""
        return int(self.global_offset[0]) * _prod(self.full_shape[1:]) if self.full_shape else 0


def translate_flat_indices(lidx: torch.Tensor, place) -> torch.Tensor:
    """Map shard-local flat positions to full-tensor flat positions.

    ``place`` is what :func:`derive_placement` returned: an ``int`` for the
    identity cases (unsharded / replicated / explicit exporter overrides, translate
    = add), or a :class:`BlockPlacement`. A flat-contiguous block (only dim 0 cut,
    e.g. FSDP2 ``Shard(0)``) keeps the single-add fast path; any other block does a
    mixed-radix decompose by the local shape, adds the per-dim offset, and recomposes
    with the full-tensor strides -- a few divmods on the nnz tensor, no collectives.
    """
    if isinstance(place, int):
        return lidx + place if place else lidx
    if place.is_flat_contiguous:
        off = place.flat_offset
        return lidx + off if off else lidx
    out = torch.zeros_like(lidx)
    rem = lidx
    for lstride, off, fstride in zip(place.local_strides, place.global_offset, place.full_strides, strict=False):
        coord = torch.div(rem, lstride, rounding_mode="floor")
        rem = rem - coord * lstride
        out = out + (coord + int(off)) * fstride
    return out


def derive_placement(spec: ShardSpec):
    """Derive ``(place, contributes, gather_group)`` for THIS rank from the spec.

    ``place`` feeds :func:`translate_flat_indices`:

    * unsharded (``mesh is None``) or fully replicated: ``0``; no group (the local
      tensor is already the full parameter). Explicit exporter overrides
      (``spec.place``) may also be plain ``int`` offsets.
    * any sharded geometry: a :class:`BlockPlacement` computed from
      ``compute_local_shape_and_global_offset`` (pure math, no collective). For a
      single Shard dim, only ranks at coordinate 0 of every Replicate dim
      contribute and the gather group is the Shard dim's subgroup; the FSDP2
      default ``Shard(0)`` yields a flat-contiguous block, which keeps the add
      fast path in :func:`translate_flat_indices`. With several Shard dims (e.g.
      automodel's EP x FSDP ``(Shard(0), Shard(1))`` expert mesh) every rank holds
      a distinct block, the gather group spans the whole mesh (created once and
      cached), and Replicate dims are not supported alongside them.

    ``_StridedShard`` placements (interleaved local tensors, from some HSDP/TP
    orderings) are rejected: the local tensor is not a single block.
    """
    import torch.distributed as dist

    if spec.place is not None:
        return spec.place, True, spec.gather_group

    if spec.mesh is None:
        return 0, (dist.get_rank() == 0 if dist.is_initialized() else True), None

    placements = spec.placements
    for p in placements:
        if type(p).__name__ == "_StridedShard":
            raise NotImplementedError(
                f"sharded delta does not support _StridedShard (local tensor is not one block); "
                f"got placements={placements}"
            )
    shard_dims = [d for d, p in enumerate(placements) if p.is_shard()]

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

    local_shape, global_offset = compute_local_shape_and_global_offset(spec.full_shape, spec.mesh, list(placements))

    if len(shard_dims) == 1:
        group = spec.mesh.get_group(mesh_dim=shard_dims[0])
        return BlockPlacement(tuple(local_shape), tuple(global_offset), tuple(spec.full_shape)), contributes, group

    # several Shard dims: every rank owns a distinct block; gather spans the whole mesh.
    assert all(p.is_shard() for p in placements), (
        f"Replicate dims are not supported alongside multiple Shard dims; got placements={placements}"
    )
    group = _flattened_mesh_group(spec.mesh)
    return BlockPlacement(tuple(local_shape), tuple(global_offset), tuple(spec.full_shape)), contributes, group


_FLAT_GROUPS: dict = {}


def _flattened_mesh_group(mesh):
    """One process group spanning every rank of ``mesh``, created once per mesh.

    ``dist.new_group`` must be called by all processes in the same order; every
    trainer rank walks the export in an identical order, so the cache stays in
    lockstep. The group's rank 0 is the mesh's smallest global rank, which keeps
    the gathered result on the engine master whenever it is part of the mesh.
    """
    import torch.distributed as dist

    ranks = tuple(sorted(int(r) for r in mesh.mesh.flatten().tolist()))
    got = _FLAT_GROUPS.get(ranks)
    if got is None:
        got = dist.new_group(list(ranks))
        _FLAT_GROUPS[ranks] = got
    return got
