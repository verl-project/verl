# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

_checkpoint = pytest.importorskip("fsdp_turbo.models.kimi.kimi_k3.checkpoint")
iter_checkpoint_weights = _checkpoint.iter_checkpoint_weights
map_checkpoint_indices = _checkpoint.map_checkpoint_indices


def _export_local(param, layout):
    return next(iter_checkpoint_weights({"experts.gate_up_proj": param}, device="cpu", rollout_layout=layout))


@pytest.mark.parametrize("projection,shape", [("gate_up_proj", (4, 3, 8)), ("down_proj", (4, 4, 3))])
def test_full_export_and_sparse_delta_share_hf_coordinates(projection, shape):
    name = f"model.layers.7.block_sparse_moe.experts.{projection}"
    initial = torch.arange(torch.tensor(shape).prod()).reshape(shape).float()
    changed = initial + 2
    seed = dict(iter_checkpoint_weights({name: initial}, device="cpu"))
    indices = torch.arange(initial.numel())
    slots, _, counts, hf_indices, values = map_checkpoint_indices(name, shape, indices, changed.flatten())
    offset = 0
    for (key, output_shape), count in zip(slots, counts.tolist(), strict=True):
        assert tuple(seed[key].shape) == output_shape
        seed[key].flatten()[hf_indices[offset : offset + count]] = values[offset : offset + count]
        offset += count
    expected = dict(iter_checkpoint_weights({name: changed}, device="cpu"))
    assert set(seed) == set(expected)
    for key in expected:
        torch.testing.assert_close(seed[key], expected[key], rtol=0, atol=0)
    if projection == "gate_up_proj":
        torch.testing.assert_close(expected[name.rsplit(".", 1)[0] + ".2.w1.weight"], changed[2, :, :4].T)
        torch.testing.assert_close(expected[name.rsplit(".", 1)[0] + ".2.w3.weight"], changed[2, :, 4:].T)
    else:
        torch.testing.assert_close(expected[name.rsplit(".", 1)[0] + ".2.w2.weight"], changed[2].T)


def _check_colocated_export(rank, rendezvous):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=4)
    try:
        mesh = DeviceMesh("cpu", torch.arange(4).reshape(2, 2))
        original = torch.arange(4 * 6 * 8).reshape(4, 6, 8).float()
        layout = dict(
            expert_parallel_size=2, tensor_parallel_size=2, data_parallel_size=1, pipeline_model_parallel_size=1
        )
        # Two eFSDP matrix shards x two expert shards. Check initial weights
        # and a changed version, including delivery to both rollout replicas.
        for version in (0, 1):
            weights = original + version
            param = distribute_tensor(weights, mesh, [Shard(1), Shard(0)])
            name, local = _export_local(param, layout)
            with patch.object(DTensor, "full_tensor", side_effect=AssertionError("must not gather packed tensor")):
                exported = dict(iter_checkpoint_weights({"experts.gate_up_proj": param}, device="cpu"))
            for expert in range(4):
                torch.testing.assert_close(exported[f"experts.{expert}.w1.weight"], weights[expert, :, :4].T)
                torch.testing.assert_close(exported[f"experts.{expert}.w3.weight"], weights[expert, :, 4:].T)
            start = (rank % 2) * 2
            assert name == f"experts.gate_up_proj.__verl_packed_local__.{start}"
            torch.testing.assert_close(local, weights[start : start + 2], rtol=0, atol=0)
            with patch.object(DTensor, "redistribute", side_effect=AssertionError("must send matrix shards directly")):
                name, local = _export_local(
                    param,
                    {**layout, "expert_parallel_size": 4, "data_parallel_size": 2},
                )
            assert name.endswith(f".{rank}")
            torch.testing.assert_close(local, weights[rank : rank + 1], rtol=0, atol=0)
        reordered = DeviceMesh("cpu", torch.tensor([[0, 2], [1, 3]]))
        param = distribute_tensor(original, reordered, [Shard(1), Shard(0)])
        _, local = _export_local(param, layout)
        torch.testing.assert_close(local, original[rank % 2 * 2 : rank % 2 * 2 + 2], rtol=0, atol=0)

        # 896 IDs, uneven matrix-row shards, both growing and shrinking EP,
        # source replicas, and buffer chunks that cross expert boundaries.
        original = torch.arange(896 * 7 * 5).reshape(896, 7, 5).float()
        configurations = [
            (torch.arange(4).reshape(2, 2), [Shard(1), Shard(0)], 4),
            (torch.arange(4).reshape(2, 2), [Shard(1), Shard(0)], 1),
            (torch.arange(4), [Shard(0)], 2),
            (torch.arange(4).reshape(2, 2), [Replicate(), Shard(0)], 4),
        ]
        for ranks, placements, target_ep in configurations:
            mesh = DeviceMesh("cpu", ranks)
            for version in (0, 1):
                weights = original + version
                param = distribute_tensor(weights, mesh, placements)
                with (
                    patch("fsdp_turbo.models.kimi.kimi_k3.checkpoint._RESHARD_BUFFER_BYTES", 512),
                    patch.object(DTensor, "redistribute", side_effect=AssertionError("must not gather")),
                ):
                    name, local = _export_local(
                        param,
                        dict(
                            expert_parallel_size=target_ep,
                            tensor_parallel_size=1,
                            data_parallel_size=target_ep,
                            pipeline_model_parallel_size=1,
                        ),
                    )
                count = 896 // target_ep
                start = rank % target_ep * count
                assert name.endswith(f".{start}")
                torch.testing.assert_close(local, weights[start : start + count], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_colocated_export_gathers_matrix_shards_and_keeps_expert_ownership(tmp_path):
    mp.spawn(_check_colocated_export, args=(str(tmp_path / "gloo_rendezvous"),), nprocs=4, join=True)


def _check_column_shard_export(rank, rendezvous):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=6)
    try:
        original = torch.arange(896 * 7 * 12).reshape(896, 7, 12).float()
        configurations = [
            (torch.arange(6).reshape(3, 2), [Shard(2), Shard(0)], 1),
            (torch.arange(6).reshape(2, 3, 1), [Replicate(), Shard(2), Shard(0)], 2),
            (torch.tensor([[0, 2], [1, 4], [3, 5]]), [Shard(2), Shard(0)], 2),
            (torch.arange(6).reshape(1, 2, 3), [Shard(0), Shard(1), Shard(2)], 2),
        ]
        for ranks, placements, target_ep in configurations:
            mesh = DeviceMesh("cpu", ranks)
            for version in (0, 1):
                weights = original + version
                param = distribute_tensor(weights, mesh, placements)
                shape, offset = compute_local_shape_and_global_offset(weights.shape, mesh, placements)
                expected_local = weights[
                    tuple(slice(begin, begin + size) for begin, size in zip(offset, shape, strict=True))
                ]
                torch.testing.assert_close(
                    param.to_local(),
                    expected_local,
                    rtol=0,
                    atol=0,
                    msg=f"source shard mismatch: ranks={ranks.tolist()} placements={placements}",
                )
                with (
                    patch("fsdp_turbo.models.kimi.kimi_k3.checkpoint._RESHARD_BUFFER_BYTES", 512),
                    patch.object(DTensor, "redistribute", side_effect=AssertionError("must send tiles directly")),
                ):
                    name, local = _export_local(
                        param,
                        dict(
                            expert_parallel_size=target_ep,
                            tensor_parallel_size=1,
                            data_parallel_size=target_ep,
                            pipeline_model_parallel_size=1,
                        ),
                    )
                count = 896 // target_ep
                start = rank % target_ep * count
                assert name.endswith(f".{start}")
                torch.testing.assert_close(
                    local,
                    weights[start : start + count],
                    rtol=0,
                    atol=0,
                    msg=f"ranks={ranks.tolist()} placements={placements} target_ep={target_ep} version={version}",
                )

        mesh = DeviceMesh("cpu", torch.arange(6).reshape(3, 2))
        param = distribute_tensor(original, mesh, [Shard(2), Shard(0)])
        _, local = _export_local(
            param,
            dict(expert_parallel_size=2, tensor_parallel_size=2, data_parallel_size=1, pipeline_model_parallel_size=1),
        )
        torch.testing.assert_close(local, original[rank % 2 * 448 : rank % 2 * 448 + 448], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_column_shards_support_factor_three_and_updated_weights(tmp_path):
    mp.spawn(_check_column_shard_export, args=(str(tmp_path / "gloo_columns"),), nprocs=6, join=True)
