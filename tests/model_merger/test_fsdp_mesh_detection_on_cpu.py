# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from verl.model_merger.fsdp_model_merger import FSDPModelMerger


def _write_shards(rank, directory, mesh_dim_names):
    directory = Path(directory)
    dist.init_process_group(
        "gloo",
        init_method=(directory / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        mesh_shape = (1, 2) if len(mesh_dim_names) == 2 else (2,)
        mesh = init_device_mesh("cpu", mesh_shape, mesh_dim_names=mesh_dim_names)
        placements = [Replicate(), Shard(0)] if len(mesh_dim_names) == 2 else [Shard(0)]
        local = torch.arange(rank * 4, rank * 4 + 4, dtype=torch.bfloat16).reshape(2, 2)
        weight = DTensor.from_local(local, mesh, placements, shape=torch.Size([4, 2]), stride=(2, 1))
        # FSDP2 shards parameters, not persistent buffers. The buffer sorts first.
        torch.save(
            {"a_buffer": torch.tensor(7.0), "weight": weight},
            directory / f"model_world_size_2_rank_{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope="module", params=[("fsdp",), ("dp_shard",), ("ddp", "fsdp")])
def checkpoint(tmp_path_factory, request):
    tmp_path = tmp_path_factory.mktemp("fsdp_mesh")
    mp.spawn(_write_shards, args=(str(tmp_path), request.param), nprocs=2, join=True)
    merger = FSDPModelMerger.__new__(FSDPModelMerger)
    merger.config = SimpleNamespace(local_dir=tmp_path)
    return merger, request.param


def test_replicated_buffer_does_not_hide_sharded_weights(checkpoint):
    merger, expected_names = checkpoint
    state = merger._load_rank_zero_state_dict(2)
    mesh, names = merger._extract_device_mesh_info(state, 2)
    count, shape = merger._calculate_shard_configuration(mesh, names)
    assert count == 2
    assert names == expected_names
    merged = merger._load_and_merge_state_dicts(2, count, shape, names)
    torch.testing.assert_close(merged["weight"], torch.arange(8, dtype=torch.bfloat16).reshape(4, 2))
    assert merged["a_buffer"].item() == 7


def test_dtensor_first_mesh_is_unchanged(checkpoint):
    merger, expected_names = checkpoint
    state = merger._load_rank_zero_state_dict(2)
    state = {"a_weight": state["weight"], "z_buffer": state["a_buffer"]}
    mesh, names = merger._extract_device_mesh_info(state, 2)
    count, _ = merger._calculate_shard_configuration(mesh, names)
    assert count == 2
    assert names == expected_names
