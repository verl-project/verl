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

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from verl.utils.fsdp_utils import collect_lora_params

peft = pytest.importorskip("peft")


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 6, bias=False)
        self.head = torch.nn.Linear(6, 4)


def _base_export_worker(rank, world_size, rendezvous_file):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(0)
        model = peft.get_peft_model(_TinyModel(), peft.LoraConfig(r=2, target_modules=["proj"]))
        expected = {
            "proj.weight": model.base_model.model.proj.base_layer.weight.detach().clone(),
            "head.weight": model.base_model.model.head.weight.detach().clone(),
            "head.bias": model.base_model.model.head.bias.detach().clone(),
        }
        fully_shard(model, mesh=init_device_mesh("cpu", (world_size,)))

        params = collect_lora_params(model, layered_summon=False, base_sync_done=False)

        # Base names only, with the LoRA wrapper stripped and no adapter tensors.
        assert sorted(params) == sorted(expected)
        # Still sharded: each rank holds its slice until the caller gathers one tensor at a time.
        for name, param in params.items():
            assert isinstance(param, DTensor), name
            assert param.to_local().numel() < param.numel(), name
            torch.testing.assert_close(param.full_tensor(), expected[name])

        with pytest.raises(ValueError, match="rollout.layered_summon=True cannot export the LoRA base weights"):
            collect_lora_params(model, layered_summon=True, base_sync_done=False)
    finally:
        dist.destroy_process_group()


def test_fsdp2_lora_base_export_stays_sharded(tmp_path):
    world_size = 2
    mp.spawn(
        _base_export_worker,
        args=(world_size, str(tmp_path / "fsdp2_rdzv")),
        nprocs=world_size,
        join=True,
    )
