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

"""CPU-only tests for NCCL M2N Reshard layout construction."""

from __future__ import annotations

import pytest
import torch
from torch.distributed.tensor import Replicate, Shard

from verl.checkpoint_engine.reshard_layout import (
    build_reshard_layouts,
    local_shape,
    local_weight_desc_from_shard_api,
)
from verl.workers.engine.spec import ShardSpec


class _Mesh:
    ndim = 2

    @staticmethod
    def size(mesh_dim=None):
        return 32 if mesh_dim is None else (2, 16)[mesh_dim]


class _OneDimensionalMesh:
    ndim = 1

    @staticmethod
    def size(mesh_dim=None):
        return 16


@pytest.mark.parametrize("shape", [(), (2, 2, 2, 2)])
def test_local_shape_rejects_tensor_ranks_unsupported_by_m2n(shape):
    with pytest.raises(ValueError, match="supports tensor ranks 1 through 3"):
        local_shape(shape, shard_dim=None, shard_size=1)


def test_shard_api_maps_sharded_source_to_sharded_destination():
    exported = (
        "model.layers.0.mlp.down_proj.weight",
        torch.empty(16, device="meta"),
        ShardSpec(full_shape=(32, 8), mesh=_Mesh(), placements=(Replicate(), Shard(0))),
    )

    weight = local_weight_desc_from_shard_api(exported, destination_shard_dim=1)
    source, destination = build_reshard_layouts(
        weight,
        source_replica_size=2,
        source_shard_size=16,
        destination_replica_size=8,
        destination_shard_size=4,
    )

    assert weight.tensor.shape == (2, 8)
    assert weight.source_shard_size == 16
    assert weight.source_mesh_dims == (2, 16)
    assert (source.mesh_dims, source.placements) == ((2, 16), (None, 0))
    assert (destination.mesh_dims, destination.placements) == ((8, 4), (None, 1))
    assert destination.local_shape == (32, 2)


def test_shard_api_preserves_replicated_layouts():
    exported = (
        "model.layers.0.input_layernorm.weight",
        torch.empty(32, device="meta"),
        ShardSpec(full_shape=(32,), mesh=_Mesh(), placements=(Replicate(), Replicate())),
    )

    weight = local_weight_desc_from_shard_api(exported, destination_shard_dim=None)
    source, destination = build_reshard_layouts(
        weight,
        source_replica_size=2,
        source_shard_size=16,
        destination_replica_size=8,
        destination_shard_size=4,
    )

    assert weight.source_shard_dim is None
    assert weight.source_mesh_dims == (2, 16)
    assert source.mesh_dims == destination.mesh_dims == (32, 1)
    assert source.placements == destination.placements == (None, None)
    assert source.local_shape == destination.local_shape == (32,)


def test_shard_api_accepts_one_dimensional_source_mesh():
    exported = (
        "model.layers.0.mlp.down_proj.weight",
        torch.empty(16, device="meta"),
        ShardSpec(full_shape=(32, 8), mesh=_OneDimensionalMesh(), placements=(Shard(0),)),
    )

    weight = local_weight_desc_from_shard_api(exported, destination_shard_dim=1)
    source, _ = build_reshard_layouts(
        weight,
        source_replica_size=1,
        source_shard_size=16,
        destination_replica_size=8,
        destination_shard_size=4,
    )

    assert weight.source_mesh_dims == (1, 16)
    assert (source.mesh_dims, source.placements) == ((1, 16), (None, 0))


def test_shard_api_rejects_deferred_hf_conversion():
    exported = (
        "decoder.layers.0.self_attention.linear_qkv.weight",
        torch.empty(1, device="meta"),
        ShardSpec(
            full_shape=(16,),
            mesh=_Mesh(),
            placements=(Replicate(), Shard(0)),
            to_hf_chunk=lambda start, tensor: [],
            hf_slots=[("model.layers.0.self_attn.q_proj.weight", (16,))],
        ),
    )

    with pytest.raises(NotImplementedError, match="converter metadata"):
        local_weight_desc_from_shard_api(exported)


def test_shard_api_rejects_explicit_placement_metadata():
    exported = (
        "decoder.layers.0.self_attention.linear_qkv.weight",
        torch.empty(16, device="meta"),
        ShardSpec(full_shape=(16,), place=0, gather_group=object()),
    )

    with pytest.raises(NotImplementedError, match="explicit ShardSpec placement metadata"):
        local_weight_desc_from_shard_api(exported)


def test_shard_api_rejects_multiple_sharded_mesh_dimensions():
    exported = (
        "model.layers.0.mlp.down_proj.weight",
        torch.empty(8, device="meta"),
        ShardSpec(full_shape=(32, 8), mesh=_Mesh(), placements=(Shard(0), Shard(1))),
    )

    with pytest.raises(NotImplementedError, match="one source Shard placement"):
        local_weight_desc_from_shard_api(exported)


def test_shard_api_rejects_outer_sharded_mesh_dimension():
    exported = (
        "model.layers.0.mlp.down_proj.weight",
        torch.empty(128, device="meta"),
        ShardSpec(full_shape=(32, 8), mesh=_Mesh(), placements=(Shard(0), Replicate())),
    )

    with pytest.raises(NotImplementedError, match="sharded source mesh dimension to be innermost"):
        local_weight_desc_from_shard_api(exported)


def test_shard_api_rejects_wrong_local_tensor_size():
    exported = (
        "model.layers.0.mlp.down_proj.weight",
        torch.empty(15, device="meta"),
        ShardSpec(full_shape=(32, 8), mesh=_Mesh(), placements=(Replicate(), Shard(0))),
    )

    with pytest.raises(ValueError, match="has 15 elements, expected 16"):
        local_weight_desc_from_shard_api(exported)


def test_layout_builder_rejects_source_shard_size_mismatch():
    exported = (
        "model.layers.0.mlp.down_proj.weight",
        torch.empty(16, device="meta"),
        ShardSpec(full_shape=(32, 8), mesh=_Mesh(), placements=(Replicate(), Shard(0))),
    )
    weight = local_weight_desc_from_shard_api(exported, destination_shard_dim=1)

    with pytest.raises(ValueError, match="source mesh shard size 16 does not match configured 8"):
        build_reshard_layouts(
            weight,
            source_replica_size=2,
            source_shard_size=8,
            destination_replica_size=8,
            destination_shard_size=4,
        )


def test_layout_builder_rejects_source_replica_size_mismatch():
    exported = (
        "model.layers.0.mlp.down_proj.weight",
        torch.empty(16, device="meta"),
        ShardSpec(full_shape=(32, 8), mesh=_Mesh(), placements=(Replicate(), Shard(0))),
    )
    weight = local_weight_desc_from_shard_api(exported, destination_shard_dim=1)

    with pytest.raises(ValueError, match="source replica size 2 does not match configured source replica size 4"):
        build_reshard_layouts(
            weight,
            source_replica_size=4,
            source_shard_size=16,
            destination_replica_size=8,
            destination_shard_size=4,
        )
