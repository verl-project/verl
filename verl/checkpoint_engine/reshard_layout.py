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

"""Layout helpers shared by NCCL M2N Reshard checkpoint backends."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import torch

from verl.workers.engine.spec import ShardSpec

__all__ = [
    "LocalWeightDesc",
    "ReshardLayout",
    "build_reshard_layouts",
    "local_shape",
    "local_weight_desc_from_shard_api",
]


@dataclass(frozen=True)
class LocalWeightDesc:
    """Describe one rank-local tensor and how it is partitioned.

    "source_shard_dim" and "destination_shard_dim" are tensor dimensions,
    not device-mesh dimensions. "None" means that the tensor is replicated on
    that side of the transfer.

    For a sharded source, "source_shard_size" is the size of the device-mesh
    dimension that shards the tensor. It is ignored for a replicated source.
    "source_mesh_dims" records the source mesh as the logical
    ``(replica_size, shard_size)`` expected by M2N. It is ``None`` when the
    shard export has no mesh and therefore cannot describe its replica count.
    """

    # The producer has already converted this to the consumer's parameter name.
    name: str
    tensor: torch.Tensor
    # Shape before either the source or destination partition is applied.
    global_shape: torch.Size
    destination_shard_dim: int | None
    source_shard_dim: int | None
    source_shard_size: int
    source_mesh_dims: tuple[int, int] | None = None


@dataclass(frozen=True)
class ReshardLayout:
    """Describe one side of a transfer as a logical two-dimensional mesh.

    "mesh_dims" gives the number of ranks along each mesh dimension.
    "placements" has one entry for each mesh dimension: an integer means that
    mesh dimension shards the corresponding tensor dimension, while "None"
    means replication. For example, placements=(None, 0) replicates the tensor
    along mesh dimension 0 and shards tensor dimension 0 along mesh dimension 1.

    "start_rank" is the first rank occupied by this mesh in the dedicated M2N
    communicator, not a rank in the trainer's global process group. The backend
    maps participating processes into contiguous M2N source and destination
    ranges before building this layout. "local_shape" is the tensor shape held
    by one rank after applying the placements.
    """

    mesh_dims: tuple[int, int]
    start_rank: int
    placements: tuple[int | None, int | None]
    local_shape: torch.Size


def local_shape(global_shape: Sequence[int], shard_dim: int | None, shard_size: int) -> torch.Size:
    """Return one rank's shape after an even partition on "shard_dim".

    A "None" shard dimension represents replication, so it leaves the global
    shape unchanged.
    """

    if shard_size <= 0:
        raise ValueError(f"shard_size must be positive, got {shard_size}")
    shape = list(global_shape)
    if not 1 <= len(shape) <= 3:
        raise ValueError(f"NCCL M2N supports tensor ranks 1 through 3, got {len(shape)} for shape {tuple(shape)}")
    if shard_dim is not None:
        if not 0 <= shard_dim < len(shape) or shape[shard_dim] % shard_size:
            raise ValueError(f"shape {tuple(shape)} cannot be sharded on dim {shard_dim} by {shard_size}")
        shape[shard_dim] //= shard_size
    return torch.Size(shape)


def local_weight_desc_from_shard_api(
    exported: tuple[str, torch.Tensor, ShardSpec],
    *,
    destination_shard_dim: int | None | Literal["source"] = "source",
) -> LocalWeightDesc:
    """Convert one engine shard export into a transport-independent descriptor.

    "ShardSpec.placements" describes the trainer tensor over its device mesh.
    Destination placement is supplied by the producer because it is a
    model/consumer policy, not a property of the generic layout library. Passing
    "source" preserves the source tensor placement.
    """

    if not isinstance(exported, tuple) or len(exported) != 3:
        raise TypeError("get_per_tensor_param_shard() must yield (name, tensor, ShardSpec)")
    name, tensor, spec = exported
    if not isinstance(name, str) or not isinstance(tensor, torch.Tensor) or not isinstance(spec, ShardSpec):
        raise TypeError("invalid get_per_tensor_param_shard() item")
    if spec.to_hf_chunk is not None or spec.hf_slots is not None:
        # A deferred format conversion may need multiple source shards. The
        # no-gather path requires conversion before constructing this descriptor.
        raise NotImplementedError(
            f"Reshard requires {name!r} in final HF-local form; ShardSpec converter metadata is unsupported"
        )
    if spec.place is not None or spec.gather_group is not None or not spec.contributes:
        # These fields describe sharding that is not necessarily represented by
        # mesh/placements. Treating mesh=None as replication would silently send
        # an incomplete tensor for Megatron and veomni explicit-placement exports.
        raise NotImplementedError(f"Reshard does not support explicit ShardSpec placement metadata for {name!r}")
    if spec.mesh is None and spec.placements is not None:
        raise ValueError(f"source placements for {name!r} require a source mesh")

    source_dim, source_size, source_mesh_dims = None, 1, None
    if spec.mesh is not None:
        mesh_ndim = int(spec.mesh.ndim)
        if mesh_ndim not in (1, 2):
            raise NotImplementedError(f"Reshard supports one- or two-dimensional source meshes for {name!r}")
        if not isinstance(spec.placements, tuple) or len(spec.placements) != mesh_ndim:
            raise ValueError(f"invalid source placements for {name!r}")

        mesh_shape = tuple(int(spec.mesh.size(mesh_dim=axis)) for axis in range(mesh_ndim))
        source_mesh_dims = (1, mesh_shape[0]) if mesh_ndim == 1 else mesh_shape

        # Each placement belongs to a device-mesh dimension; Shard.dim is the
        # tensor dimension split along that mesh dimension. For example,
        # (Replicate(), Shard(0)) on a (DP, FSDP) mesh splits tensor dimension 0
        # across FSDP while DP replicas hold identical shards. The current M2N
        # descriptor supports at most one such sharded mesh dimension.
        sharded = []
        unsupported = []
        for axis, placement in enumerate(spec.placements):
            if placement.is_shard():
                sharded.append((axis, placement))
            elif not placement.is_replicate():
                unsupported.append(placement)
        if unsupported or len(sharded) > 1:
            raise NotImplementedError(f"Reshard supports one source Shard placement for {name!r}")
        if sharded:
            mesh_dim, placement = sharded[0]
            if mesh_dim != mesh_ndim - 1:
                raise NotImplementedError(
                    f"Reshard requires the sharded source mesh dimension to be innermost for {name!r}"
                )
            source_dim = int(placement.dim)
            source_size = source_mesh_dims[1]
        else:
            source_size = source_mesh_dims[1]

    full_shape = torch.Size(spec.full_shape)
    destination_dim = source_dim if destination_shard_dim == "source" else destination_shard_dim
    expected = local_shape(full_shape, source_dim, source_size)
    if tensor.numel() != math.prod(expected):
        raise ValueError(f"local shard for {name!r} has {tensor.numel()} elements, expected {math.prod(expected)}")

    # FSDP may expose a flattened local shard. The element-count check above
    # makes this reshape safe and restores the expected rank-local tensor shape.
    return LocalWeightDesc(
        name=name,
        tensor=tensor.reshape(expected),
        global_shape=full_shape,
        destination_shard_dim=destination_dim,
        source_shard_dim=source_dim,
        source_shard_size=source_size,
        source_mesh_dims=source_mesh_dims,
    )


def build_reshard_layouts(
    weight: LocalWeightDesc,
    *,
    source_replica_size: int,
    source_shard_size: int,
    destination_replica_size: int,
    destination_shard_size: int,
) -> tuple[ReshardLayout, ReshardLayout]:
    """Build source and destination layouts in one combined communicator.

    A sharded side is modeled as (replica_size, shard_size). For a replicated
    weight, all ranks are flattened onto the first mesh dimension and the
    second dimension is one. This lets every weight use the same two-placement
    representation without assigning parallelism roles to the mesh dimensions.

    Source ranks occupy the first contiguous communicator range. Destination
    ranks immediately follow them, so the destination "start_rank" equals the
    source world size. Selecting participants, including any pipeline-stage
    filtering, is the caller's responsibility and happens before this function.
    """

    if min(source_replica_size, source_shard_size, destination_replica_size, destination_shard_size) <= 0:
        raise ValueError("Reshard topology sizes must be positive")
    if weight.source_mesh_dims is not None:
        exported_replica_size, exported_shard_size = weight.source_mesh_dims
        if exported_replica_size != source_replica_size:
            raise ValueError(
                f"source replica size {exported_replica_size} does not match configured source replica size "
                f"{source_replica_size}"
            )
        if exported_shard_size != source_shard_size:
            raise ValueError(
                f"source mesh shard size {exported_shard_size} does not match configured {source_shard_size}"
            )
    if weight.source_shard_dim is not None and weight.source_shard_size != source_shard_size:
        raise ValueError(f"source shard size {weight.source_shard_size} does not match configured {source_shard_size}")

    # With no tensor shard dimension, placements=(None, None) replicates the
    # full tensor across the flattened rank dimension and a singleton dimension.
    source_world_size = source_replica_size * source_shard_size
    source_dims = (
        (source_replica_size, source_shard_size) if weight.source_shard_dim is not None else (source_world_size, 1)
    )
    destination_dims = (
        (destination_replica_size, destination_shard_size)
        if weight.destination_shard_dim is not None
        else (destination_replica_size * destination_shard_size, 1)
    )
    return (
        ReshardLayout(
            mesh_dims=source_dims,
            start_rank=0,
            placements=(None, weight.source_shard_dim),
            local_shape=local_shape(weight.global_shape, weight.source_shard_dim, weight.source_shard_size),
        ),
        ReshardLayout(
            mesh_dims=destination_dims,
            start_rank=source_world_size,
            placements=(None, weight.destination_shard_dim),
            local_shape=local_shape(weight.global_shape, weight.destination_shard_dim, destination_shard_size),
        ),
    )
