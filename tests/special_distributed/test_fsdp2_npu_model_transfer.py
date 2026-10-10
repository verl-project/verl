# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright 2026 Individual Contributor: leovzhang
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

"""Check complete FSDP2 parameter values across NPU/CPU transfers, without model weights.

Launch from the repository root on two or more NPUs:
    PYTHONPATH=. torchrun --standalone --nproc-per-node=2 \
        tests/special_distributed/test_fsdp2_npu_model_transfer.py

The uneven vectors require CPU repadding even with two ranks. With 32 ranks,
their length is 4304, reproducing the 135-element shards and 119-element tail
from the Qwen3-Omni vision bias failure. Evenly sharded vectors are controls.
"""

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401 -- register the NPU backend before importing verl
from torch.distributed import init_device_mesh

from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    fsdp2_load_full_state_dict,
    fully_shard,
    load_fsdp_model_to_gpu,
    offload_fsdp_model_to_cpu,
)


def _build_model(mesh):
    world_size = mesh.size()
    lengths = {"uneven": world_size * 135 - world_size // 2, "even": world_size * 135}
    model = torch.nn.Sequential()
    expected = {}
    for index in range(27):
        layer = torch.nn.Module()
        for name, length in lengths.items():
            # Exactly representable BF16 values, distinct across layers and positions.
            value = ((torch.arange(length) + index) % 127 + 1).to(torch.bfloat16) / 128
            expected[f"{index}.{name}"] = value
            layer.register_parameter(name, torch.nn.Parameter(value.to(get_device_id())))
        fully_shard(layer, mesh=mesh)
        model.append(layer)
    fully_shard(model, mesh=mesh)
    return model, expected


def _check_parameters(model, expected, phase, *, full=False):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    failures = []
    for name, param in model.named_parameters():
        if full:
            # Every rank must finish the collectives, even if an earlier parameter differs.
            actual = param.full_tensor().cpu()
            reference = expected[name]
        else:
            actual = param.to_local()
            reference = expected[name].chunk(world_size)[rank]
            assert actual.device.type == "cpu", f"{phase}: {name} was not offloaded"
        try:
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        except AssertionError as error:
            failures.append(f"{name}: {error}")

    failed_ranks = torch.tensor(int(bool(failures)), device=get_device_id())
    dist.all_reduce(failed_ranks)
    assert failed_ranks.item() == 0, f"{phase}, rank {rank}: " + (
        "\n".join(failures) if failures else "parameter mismatch on another rank"
    )


@torch.no_grad()
def main():
    assert get_device_name() == "npu", "this regression requires Ascend NPUs"
    _, rank, world_size = initialize_global_process_group()
    try:
        assert world_size >= 2, "launch with at least two ranks to exercise uneven shards"
        mesh = init_device_mesh("npu", (world_size,))
        model, expected = _build_model(mesh)
        _check_parameters(model, expected, "initial sharding", full=True)

        # Cover the initialization path as well as explicit runtime offloading.
        full_state = {name: value.clone() for name, value in expected.items()} if rank == 0 else {}
        fsdp2_load_full_state_dict(model, full_state, device_mesh=mesh, cpu_offload=CPUOffloadPolicy())
        _check_parameters(model, expected, "initial load/offload")
        load_fsdp_model_to_gpu(model)
        _check_parameters(model, expected, "initial full export", full=True)

        for iteration in range(5):
            # Inspect the CPU shards immediately; no synchronize() may hide an unfinished copy.
            offload_fsdp_model_to_cpu(model, empty_cache=False)
            _check_parameters(model, expected, f"CPU shards, iteration {iteration}")
            load_fsdp_model_to_gpu(model)
            _check_parameters(model, expected, f"full export, iteration {iteration}", full=True)

            # Also test the production round trip without intervening host reads or synchronization.
            offload_fsdp_model_to_cpu(model, empty_cache=False)
            load_fsdp_model_to_gpu(model)
            get_torch_device().synchronize()
            _check_parameters(model, expected, f"unobserved round trip, iteration {iteration}", full=True)

        if rank == 0:
            print(f"FSDP2 NPU model transfer passed: {world_size} ranks, 27 uneven/even pairs, 5 round trips")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
