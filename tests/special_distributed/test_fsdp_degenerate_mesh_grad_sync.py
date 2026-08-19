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

"""Regression: fsdp_size=1 on multiple GPUs must still synchronize gradients.

create_device_mesh(world_size, fsdp_size=1) builds a (world_size, 1) mesh, whose
shard dim is degenerate. Selecting HYBRID_SHARD for it makes FSDP1 clamp to
NO_SHARD (the shard group holds a single rank) while still reducing gradients
over that size-1 shard group, so every rank trains its own copy of the model --
with no error, no warning beyond FSDP's strategy-switch notice, and plausible
metrics. See pytorch/pytorch#154888 (acknowledged upstream, closed as
not_planned because FSDP1 is in maintenance mode).

get_sharding_strategy therefore returns NO_SHARD for a degenerate shard dim,
which routes FSDP's non-hybrid path to mesh_dim=0 -- the replicate dim.

This test feeds each rank different data and asserts that post-backward
gradients are identical across ranks (all-reduce averages them). Pre-fix the
gradients differ; post-fix they match bitwise.

Launch:
    torchrun --nproc-per-node=2 --standalone \\
        tests/special_distributed/test_fsdp_degenerate_mesh_grad_sync.py
"""

import torch
import torch.distributed
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

from verl.utils.device import get_device_name
from verl.workers.engine.fsdp.utils import create_device_mesh, get_sharding_strategy


def main():
    torch.distributed.init_process_group()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    assert world_size >= 2, "this regression needs at least 2 ranks"
    device_name = get_device_name()
    torch.get_device_module(device_name).set_device(rank)

    device_mesh = create_device_mesh(world_size=world_size, fsdp_size=1)
    assert device_mesh.ndim == 2, f"fsdp_size=1 is expected to build a 2D mesh, got {device_mesh.shape}"
    assert device_mesh.size(1) == 1, f"shard dim should be degenerate, got {device_mesh.size(1)}"

    sharding_strategy = get_sharding_strategy(device_mesh)

    torch.manual_seed(0)
    module = nn.Linear(64, 64, bias=False).to(device_name)
    model = FSDP(
        module,
        device_mesh=device_mesh,
        sharding_strategy=sharding_strategy,
        use_orig_params=True,
        sync_module_states=True,
        device_id=rank,
    )

    # Different data per rank: without gradient synchronization the resulting
    # gradients differ, which is exactly the failure this test guards against.
    inputs = torch.randn(4, 64, device=device_name) * (rank + 1)
    model(inputs).square().mean().backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients were produced"
    local_norm = torch.stack([g.detach().float().norm() for g in grads]).norm()

    gathered = [torch.zeros_like(local_norm) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, local_norm)
    gathered = torch.stack(gathered)
    assert torch.equal(gathered, gathered[0].expand_as(gathered)), (
        f"gradients are not synchronized across ranks: {gathered.tolist()} (sharding_strategy={sharding_strategy})"
    )

    # Pin the mechanism as well, so a future strategy change that happens to
    # keep gradients in sync still surfaces here rather than silently drifting.
    assert sharding_strategy == ShardingStrategy.NO_SHARD, (
        f"a degenerate shard dim must not select a hybrid strategy, got {sharding_strategy}"
    )

    if rank == 0:
        print(f"[fsdp_size=1] gradient norms across {world_size} ranks: {gathered.tolist()} -- synchronized")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
