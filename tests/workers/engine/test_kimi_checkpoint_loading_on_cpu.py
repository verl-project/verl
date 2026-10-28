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

from collections import defaultdict

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard, distribute_tensor

_checkpoint = pytest.importorskip("fsdp_turbo.models.kimi.kimi_k3.checkpoint")
_add_kimi_packed_tasks = _checkpoint._add_kimi_packed_tasks
_copy_task = _checkpoint._copy_task
_TensorSpec = _checkpoint._TensorSpec
load_kimi_k3_checkpoint_to_local_shards = _checkpoint.load_hf_checkpoint_shards


@pytest.mark.parametrize("projection", ["gate_up_proj", "down_proj"])
def test_packed_local_slice_matches_external_experts(tmp_path, projection):
    raw = {}
    for expert in range(4):
        for index, (name, shape) in enumerate((("w1", (4, 6)), ("w2", (6, 4)), ("w3", (4, 6)))):
            raw[f"experts.{expert}.{name}.weight"] = (
                torch.arange(24).reshape(shape).float() + 100 * expert + 1000 * index
            )
    packed_gate = torch.stack(
        [torch.cat((raw[f"experts.{e}.w1.weight"].T, raw[f"experts.{e}.w3.weight"].T), dim=1) for e in range(4)]
    )
    packed_down = torch.stack([raw[f"experts.{e}.w2.weight"].T for e in range(4)])
    packed = packed_gate if projection == "gate_up_proj" else packed_down
    # Cross the gate/up boundary and retain only part of the eFSDP matrix.
    offset = (1, 1, 3)
    shape = (2, 2, 3)
    destination = torch.full(shape, float("nan"))
    spec = _TensorSpec(f"experts.{projection}", tuple(packed.shape), shape, offset, destination)
    shard = tmp_path / "experts.safetensors"
    save_file(raw, shard)
    tasks = defaultdict(list)
    _add_kimi_packed_tasks(spec, {name: shard.name for name in raw}, tasks)
    with safe_open(shard, framework="pt", device="cpu") as handle:
        for task in tasks[shard.name]:
            _copy_task(task, handle.get_slice(task.source_key))
    torch.testing.assert_close(destination, packed[1:3, 1:3, 3:6], rtol=0, atol=0)


def test_streaming_load_preserves_buffers_and_rejects_incomplete_weights(tmp_path):
    model = torch.nn.Linear(3, 2, bias=False, device="meta")
    model.register_buffer("rotary", torch.arange(3).float(), persistent=False)
    weight = torch.arange(6).reshape(2, 3).float()
    save_file({"weight": weight}, tmp_path / "model.safetensors")
    load_kimi_k3_checkpoint_to_local_shards(model, str(tmp_path), materialize_device="cpu")
    torch.testing.assert_close(model.weight, weight, rtol=0, atol=0)
    torch.testing.assert_close(model.rotary, torch.arange(3).float(), rtol=0, atol=0)
    save_file({"unrelated": weight}, tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError, match="missing target parameters"):
        load_kimi_k3_checkpoint_to_local_shards(model, str(tmp_path), materialize_device="cpu")


@pytest.mark.parametrize("padding", ["zero", "nonzero", "too_short"])
def test_streaming_load_validates_kda_head_padding(tmp_path, padding):
    model = torch.nn.Module()
    model.layer = torch.nn.Module()
    model.layer.self_attn = torch.nn.Module()
    model.layer.self_attn.A_log = torch.nn.Parameter(torch.empty(96, device="meta"))
    weight = torch.cat((torch.arange(96).float(), torch.zeros(32)))
    if padding == "nonzero":
        weight[-1] = 1
    elif padding == "too_short":
        weight = weight[:95].clone()
    save_file({"layer.self_attn.A_log": weight}, tmp_path / "model.safetensors")
    if padding == "zero":
        load_kimi_k3_checkpoint_to_local_shards(model, str(tmp_path), materialize_device="cpu")
        torch.testing.assert_close(model.layer.self_attn.A_log, weight[:96], rtol=0, atol=0)
    else:
        error = ValueError if padding == "nonzero" else RuntimeError
        match = "Nonzero inactive" if padding == "nonzero" else "shape mismatch"
        with pytest.raises(error, match=match):
            load_kimi_k3_checkpoint_to_local_shards(model, str(tmp_path), materialize_device="cpu")


def _check_sharded_meta_load(rank, directory):
    dist.init_process_group("gloo", init_method=f"file://{directory}/rendezvous", rank=rank, world_size=2)
    try:
        mesh = DeviceMesh("cpu", [0, 1])
        model = torch.nn.Linear(6, 4, bias=False, device="meta")
        model.weight = torch.nn.Parameter(distribute_tensor(model.weight, mesh, [Shard(0)]))
        original_parameter = model.weight
        model.register_buffer("rotary", torch.arange(3).float(), persistent=False)
        load_kimi_k3_checkpoint_to_local_shards(model, directory, materialize_device="cpu")
        assert model.weight is original_parameter
        assert model.weight.placements == (Shard(0),)
        assert model.weight.device_mesh == mesh
        assert model.weight.shape == (4, 6)
        expected = torch.arange(24).reshape(4, 6).float()[rank * 2 : rank * 2 + 2]
        torch.testing.assert_close(model.weight.to_local(), expected, rtol=0, atol=0)
        torch.testing.assert_close(model.rotary, torch.arange(3).float(), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_streaming_load_materializes_dtensor_with_its_original_layout(tmp_path):
    save_file({"weight": torch.arange(24).reshape(4, 6).float()}, tmp_path / "model.safetensors")
    mp.spawn(_check_sharded_meta_load, args=(str(tmp_path),), nprocs=2, join=True)


def test_native_stream_loader_does_not_silently_skip_new_adapter_parameters(tmp_path):
    model = torch.nn.Linear(3, 2, bias=False, device="meta")
    model.register_parameter("lora_A", torch.nn.Parameter(torch.empty(1, 3, device="meta")))
    save_file({"weight": torch.arange(6).reshape(2, 3).float()}, tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError, match="missing target parameters.*lora_A"):
        load_kimi_k3_checkpoint_to_local_shards(model, str(tmp_path), materialize_device="cpu")


def test_native_loader_preserves_tied_meta_parameters(tmp_path):
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.empty(2, 3, device="meta"))
    model.alias = model.weight
    save_file({"weight": torch.arange(6).reshape(2, 3).float()}, tmp_path / "model.safetensors")
    load_kimi_k3_checkpoint_to_local_shards(model, str(tmp_path), materialize_device="cpu")
    assert model.weight is model.alias
    torch.testing.assert_close(model.weight, torch.arange(6).reshape(2, 3).float())
