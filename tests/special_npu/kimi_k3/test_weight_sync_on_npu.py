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
# ruff: noqa: E402

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard, distribute_tensor

pytest.importorskip("torch_npu")

from fsdp_turbo.models.kimi.kimi_k3.checkpoint import iter_checkpoint_weights

from verl.utils.device import get_torch_device
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


def _engine(param):
    engine = object.__new__(FSDPEngine)
    engine.module = torch.nn.Module()
    engine.module.experts = torch.nn.Module()
    engine.module.experts.gate_up_proj = torch.nn.Parameter(param)
    engine.module.iter_checkpoint_weights = iter_checkpoint_weights
    engine.model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="kimi_k3"), lora={})
    engine._uses_fsdp2_cpu_offload_policy = engine._is_offload_param = engine._qat_enabled = engine._is_lora = False
    return engine


def _check_weight_sync(rank, rendezvous):
    get_torch_device().set_device(rank)
    dist.init_process_group("cpu:gloo,npu:hccl", init_method=f"file://{rendezvous}", rank=rank, world_size=4)
    try:
        original = torch.arange(8 * 7 * 12, device=f"npu:{rank}").reshape(8, 7, 12).float()
        for ranks in (torch.arange(4).reshape(2, 2), torch.tensor([[0, 2], [1, 3]])):
            mesh = DeviceMesh("npu", ranks)
            for matrix_axis in (1, 2):
                for version in (0, 1):
                    weights = original + version
                    param = distribute_tensor(weights, mesh, [Shard(matrix_axis), Shard(0)])
                    engine = _engine(param)
                    # Force chunks to cross matrix and expert boundaries.
                    with patch("fsdp_turbo.models.kimi.kimi_k3.checkpoint._RESHARD_BUFFER_BYTES", 512):
                        global_weights = dict(engine.get_per_tensor_param()[0])
                        for expert in range(8):
                            torch.testing.assert_close(
                                global_weights[f"experts.{expert}.w1.weight"], weights[expert, :, :6].T, rtol=0, atol=0
                            )
                            torch.testing.assert_close(
                                global_weights[f"experts.{expert}.w3.weight"], weights[expert, :, 6:].T, rtol=0, atol=0
                            )
                        for ep, tp, dp in ((1, 1, 1), (2, 2, 1), (4, 1, 4)):
                            layout = dict(
                                expert_parallel_size=ep,
                                tensor_parallel_size=tp,
                                data_parallel_size=dp,
                                pipeline_model_parallel_size=1,
                            )
                            name, local = next(engine.get_per_tensor_param(rollout_layout=layout)[0])
                            start = rank % ep * (8 // ep)
                            assert name == f"experts.gate_up_proj.__verl_packed_local__.{start}"
                            assert local.device.type == "npu"
                            torch.testing.assert_close(local, weights[start : start + 8 // ep], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_global_and_colocated_weight_sync(tmp_path):
    mp.spawn(_check_weight_sync, args=(str(tmp_path / "weight_sync"),), nprocs=4, join=True)
