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

"""Regression test for FSDP1 parameter offload under an unsharded strategy.

``fsdp_size=1`` builds a one-rank shard group, which degrades FSDP1 to
``NO_SHARD``. Combined with verl's default ``use_orig_params=False``, the
per-parameter views registered on the wrapped modules keep pointing at the
pre-move flat-parameter storage: ``FlatParamHandle.flat_param_to()`` only
refreshes those views when ``use_orig_params=True``. The stale views hold a
reference to the old device allocation, so ``offload_fsdp_model_to_cpu`` frees
nothing and the following ``load_fsdp_model_to_gpu`` allocates a second copy.

The offload must be exercised **before any forward pass**: FSDP's pre-forward
unshard re-points the views itself, which hides the leak.

Launch:
    torchrun --nproc-per-node=2 --standalone \
        tests/special_distributed/test_fsdp1_no_shard_offload.py
"""

import torch
import torch.distributed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from transformers import AutoModelForCausalLM, Qwen2Config

from verl.utils.device import get_device_name, get_torch_device
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.workers.engine.fsdp.utils import create_device_mesh, get_sharding_strategy


def _build_no_shard_fsdp1_model(world_size):
    config = Qwen2Config(
        num_hidden_layers=4,
        hidden_size=1024,
        intermediate_size=2048,
        num_attention_heads=8,
        num_key_value_heads=8,
        vocab_size=2048,
    )
    model = AutoModelForCausalLM.from_config(config=config, torch_dtype=torch.float32)
    model = model.to(get_device_name())
    device_mesh = create_device_mesh(world_size=world_size, fsdp_size=1)
    return FSDP(
        model,
        device_mesh=device_mesh,
        sharding_strategy=get_sharding_strategy(device_mesh),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        ),
        use_orig_params=False,
        device_id=get_torch_device().current_device(),
    )


def main():
    if not get_torch_device().is_available():
        print("skipped: accelerator unavailable")
        return
    assert get_torch_device().device_count() >= 2, "test requires at least 2 devices"

    _local_rank, rank, world_size = initialize_global_process_group()
    device_type = get_device_name()

    model = _build_no_shard_fsdp1_model(world_size)

    # Keep the reference copy on the host: a device-side clone would itself be counted
    # by memory_allocated() and mask the very allocation this test measures.
    num_params_before = sum(1 for _ in model.parameters())
    expected = [param.detach().to("cpu", copy=True) for param in model.parameters()]

    get_torch_device().synchronize()
    alloc_before = get_torch_device().memory_allocated()

    # Before any forward: FSDP's pre-forward unshard would re-point the views itself.
    offload_fsdp_model_to_cpu(model)
    get_torch_device().synchronize()
    alloc_offloaded = get_torch_device().memory_allocated()

    handles = model._all_handles
    assert handles, "expected at least one FlatParamHandle"
    for handle in handles:
        assert not handle.uses_sharded_strategy, (
            f"expected an unsharded strategy from fsdp_size=1, got {handle._sharding_strategy}"
        )

    assert alloc_offloaded < 0.1 * alloc_before, (
        f"offload_fsdp_model_to_cpu freed almost nothing: {alloc_before} -> {alloc_offloaded} "
        f"bytes allocated ({alloc_offloaded / max(alloc_before, 1):.1%} still resident on device)"
    )
    for name, param in model.named_parameters():
        assert param.device.type == "cpu", f"{name} still on {param.device} after offload"

    load_fsdp_model_to_gpu(model)
    get_torch_device().synchronize()
    alloc_reloaded = get_torch_device().memory_allocated()

    assert alloc_reloaded <= 1.1 * alloc_before, (
        f"load_fsdp_model_to_gpu double-allocated: {alloc_before} -> {alloc_reloaded} bytes "
        f"allocated ({alloc_reloaded / max(alloc_before, 1):.2f}x the pre-offload footprint)"
    )

    params = list(model.parameters())
    assert len(params) == num_params_before, (
        f"parameter count changed across the round trip: {num_params_before} -> {len(params)}"
    )
    for param, want in zip(params, expected, strict=True):
        assert param.device.type == device_type, f"expected {device_type}, got {param.device}"
        torch.testing.assert_close(param.detach().cpu(), want, atol=0.0, rtol=0.0)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    if rank == 0:
        print("test_fsdp1_no_shard_offload passed")


if __name__ == "__main__":
    main()
