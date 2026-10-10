# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from unittest.mock import Mock, call

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from verl.utils import fsdp_utils


@pytest.mark.parametrize("device_name, non_blocking", [("cuda", True), ("npu", False)])
@pytest.mark.parametrize("empty_cache", [False, True])
def test_offload_fsdp2_model_to_cpu_copy_policy(monkeypatch, device_name, non_blocking, empty_cache):
    model = Mock()
    device_module = Mock()
    calls = Mock()
    calls.attach_mock(model.to, "to")
    calls.attach_mock(device_module.empty_cache, "empty_cache")
    monkeypatch.setattr(fsdp_utils, "get_device_name", lambda: device_name)
    monkeypatch.setattr(fsdp_utils, "get_torch_device", lambda: device_module)

    fsdp_utils.offload_fsdp2_model_to_cpu(model, empty_cache=empty_cache)

    expected = [call.to("cpu", non_blocking=non_blocking)]
    if empty_cache:
        expected.append(call.empty_cache())
    assert calls.mock_calls == expected


@pytest.mark.parametrize("device_name, non_blocking", [("cuda", True), ("npu", False)])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("cpu_offload", [None, fsdp_utils.CPUOffloadPolicy()])
def test_load_full_state_dict_cpu_transfer(monkeypatch, device_name, non_blocking, rank, cpu_offload):
    if fsdp_utils.version.parse(torch.__version__) >= fsdp_utils.version.parse("2.7.0"):
        from torch.distributed.checkpoint import state_dict
    else:
        from verl.third_party.torch.distributed.checkpoint import state_dict

    model = Mock()
    model.to.return_value = model
    model.to_empty.return_value = model
    buffer = Mock()
    buffer_data = buffer.data
    model.named_buffers.return_value = [("rotary_emb", buffer)]
    model.buffers.return_value = [buffer]
    device = object()
    device_module = Mock()
    load_state = Mock()
    broadcast = Mock()
    monkeypatch.setattr(fsdp_utils, "get_device_name", lambda: device_name)
    monkeypatch.setattr(fsdp_utils, "get_device_id", lambda: device)
    monkeypatch.setattr(fsdp_utils, "get_torch_device", lambda: device_module)
    monkeypatch.setattr(fsdp_utils.dist, "get_rank", lambda: rank)
    monkeypatch.setattr(fsdp_utils.dist, "broadcast", broadcast)
    monkeypatch.setattr(state_dict, "set_model_state_dict", load_state)
    full_state = {"weight": torch.ones(4)} if rank == 0 else {}

    fsdp_utils.fsdp2_load_full_state_dict(model, full_state, cpu_offload=cpu_offload)

    assert load_state.call_args.args == (model, full_state)
    options = load_state.call_args.kwargs["options"]
    assert options.full_state_dict and options.broadcast_from_rank0
    assert options.cpu_offload == (cpu_offload is not None)
    broadcast.assert_called_once_with(buffer, src=0)
    expected = [call(device=device, non_blocking=True)] if rank == 0 else []
    if rank != 0:
        model.to_empty.assert_called_once_with(device=device)
    else:
        model.to_empty.assert_not_called()
    if cpu_offload is not None:
        expected.append(call("cpu", non_blocking=non_blocking))
        buffer_data.to.assert_called_once_with(device)
        assert buffer.data is buffer_data.to.return_value
    else:
        buffer_data.to.assert_not_called()
    assert model.to.call_args_list == expected
    device_module.empty_cache.assert_not_called()


def test_load_fsdp2_model_to_gpu_uses_non_blocking_copy(monkeypatch):
    model = Mock()
    device = object()
    monkeypatch.setattr(fsdp_utils, "get_device_id", lambda: device)

    fsdp_utils.load_fsdp2_model_to_gpu(model)

    model.to.assert_called_once_with(device, non_blocking=True)


def _sharded_snapshot_worker(rank, world_size, rendezvous_file):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        # On a CPU mesh the shards already live on CPU, as they do on GPU workers with param_offload=True.
        mesh = init_device_mesh("cpu", (world_size,))
        model = torch.nn.Linear(4, 4, bias=False)
        fully_shard(model, mesh=mesh)
        torch.nn.init.constant_(model.weight, 1.0)

        cpu_sharded_state, _ = fsdp_utils.fsdp2_sharded_save_to_cpu(model)
        torch.nn.init.constant_(model.weight, 0.0)

        saved_weight, _ = cpu_sharded_state["weight"]
        torch.testing.assert_close(saved_weight, torch.ones_like(saved_weight))
    finally:
        dist.destroy_process_group()


def test_fsdp2_sharded_save_to_cpu_copies_cpu_shards(tmp_path):
    world_size = 2
    rendezvous_file = str(tmp_path / "fsdp2_rdzv")
    mp.spawn(
        _sharded_snapshot_worker,
        args=(world_size, rendezvous_file),
        nprocs=world_size,
        join=True,
    )
