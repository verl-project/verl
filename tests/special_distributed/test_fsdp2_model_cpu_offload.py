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

"""Check FSDP2 CPU offloading on CUDA and NPU, without external model weights.

Launch from the repository root on two or more accelerators:
    PYTHONPATH=. torchrun --standalone --nproc-per-node=2 \
        tests/special_distributed/test_fsdp2_model_cpu_offload.py

The uneven vectors require CPU repadding even with two ranks. With 32 ranks,
their length is 4304, reproducing the 135-element shards and 119-element tail
from the Qwen3-Omni vision bias failure. Evenly sharded vectors are controls.
"""

import torch
import torch.distributed as dist
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
    # Put large shards first so small preceding transfers cannot drain the queued
    # work before the uneven D2H copy. On two ranks these have 4+3 and 4+4 rows.
    for index, rows in enumerate((world_size * 4 - 1, world_size * 4)):
        layer = torch.nn.Linear(262144, rows, bias=False, device=get_device_id())
        value = torch.full((rows, 262144), (index + 1) / 8, dtype=layer.weight.dtype)
        with torch.no_grad():
            layer.weight.copy_(value)
        expected[f"{len(model)}.weight"] = value
        fully_shard(layer, mesh=mesh)
        model.append(layer)
    for index in range(27):
        layer = torch.nn.Module()
        for name, length in lengths.items():
            # Exactly representable BF16 values, distinct across layers and positions.
            value = ((torch.arange(length) + index) % 127 + 1).to(torch.bfloat16) / 128
            expected[f"{len(model)}.{name}"] = value
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

    # Report each rank separately, after every local comparison and collective
    # needed to reconstruct parameters has finished.
    failed_rank = torch.tensor([int(bool(failures))], device=get_device_id())
    failed_ranks = [torch.empty_like(failed_rank) for _ in range(world_size)]
    dist.all_gather(failed_ranks, failed_rank)
    counts = [item.item() for item in failed_ranks]
    assert not any(counts), f"{phase}, failed ranks={counts}, rank {rank}: " + "\n".join(failures)


def _queue_work(matrices):
    # Reuse warmed-up buffers on the actual conversion stream. No host sleep,
    # tensor reads, synchronization, or collectives before the offload call.
    left, right, out = matrices
    for _ in range(64):
        torch.mm(left, right, out=out)


def _update_parameters(model, expected):
    # Change values between trials so a reused allocation containing an earlier
    # result cannot hide an incomplete copy. References own independent storage.
    rank, world_size = dist.get_rank(), dist.get_world_size()
    for name, param in model.named_parameters():
        expected[name].add_(1 / 128)
        param.to_local().copy_(expected[name].chunk(world_size)[rank])
    get_torch_device().synchronize()


@torch.no_grad()
def main():
    device_name = get_device_name()
    assert device_name in ("cuda", "npu"), "this regression requires CUDA or NPU"
    _, rank, world_size = initialize_global_process_group()
    try:
        assert world_size >= 2, "launch with at least two ranks to exercise uneven shards"
        mesh = init_device_mesh(device_name, (world_size,))
        model, expected = _build_model(mesh)
        _check_parameters(model, expected, "initial sharding", full=True)

        # Cover the initialization path as well as explicit runtime offloading.
        full_state = {name: value.clone() for name, value in expected.items()} if rank == 0 else {}
        fsdp2_load_full_state_dict(model, full_state, device_mesh=mesh, cpu_offload=CPUOffloadPolicy())
        _check_parameters(model, expected, "initial load/offload")
        load_fsdp_model_to_gpu(model)
        _check_parameters(model, expected, "initial full export", full=True)

        accelerator = get_torch_device()
        matrices = tuple(torch.ones((2048, 2048), device=get_device_id(), dtype=torch.float16) for _ in range(3))
        _queue_work(matrices)
        accelerator.synchronize()
        for stream_index, stream in enumerate((accelerator.current_stream(), accelerator.Stream())):
            # Setup and previous round trips are complete before changing streams.
            with accelerator.stream(stream):
                for iteration in range(5):
                    phase = f"stream {stream_index}, iteration {iteration}"
                    _update_parameters(model, expected)
                    _queue_work(matrices)
                    offload_fsdp_model_to_cpu(model, empty_cache=False)
                    accelerator.synchronize()
                    # Check after synchronization: failures here are persistent
                    # corruption, not just observing an unfinished DMA transfer.
                    _check_parameters(model, expected, f"CPU shards, {phase}")
                    load_fsdp_model_to_gpu(model)
                    _check_parameters(model, expected, f"full export, {phase}", full=True)

                    # Production round trip without intervening host reads or synchronization.
                    _update_parameters(model, expected)
                    _queue_work(matrices)
                    offload_fsdp_model_to_cpu(model, empty_cache=False)
                    load_fsdp_model_to_gpu(model)
                    accelerator.synchronize()
                    _check_parameters(model, expected, f"unobserved round trip, {phase}", full=True)
            accelerator.synchronize()

        if rank == 0:
            print(f"FSDP2 {device_name} CPU offload passed: {world_size} ranks, default and non-default streams")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
