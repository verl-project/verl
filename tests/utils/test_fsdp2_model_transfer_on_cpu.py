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

from unittest.mock import Mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from verl.utils import fsdp_utils


def test_offload_fsdp2_model_to_cpu_uses_non_blocking_copy():
    model = Mock()

    fsdp_utils.offload_fsdp2_model_to_cpu(model, empty_cache=False)

    model.to.assert_called_once_with("cpu", non_blocking=True)


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
