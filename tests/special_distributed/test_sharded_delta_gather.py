# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Validate verl.checkpoint_engine.delta_sync.sparse_gather against the full-gather diff.

torchrun --nproc_per_node=4 tests/special_distributed/test_sharded_delta_gather.py
"""

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from verl.checkpoint_engine.delta_sync.sparse_gather import (
    gather_v_batched_to_rank0,
    shard_delta_indices,
)
from verl.workers.engine.spec import ShardSpec, derive_placement


def _run_case(shape, placements, mesh, dev, rank, si):
    """Diff a distributed param via the real sharded module vs a full-gather baseline."""
    torch.manual_seed(si)
    full_old = torch.randn(*shape, dtype=torch.bfloat16, device=dev)
    full_new = full_old.clone()
    g = torch.Generator(device=dev).manual_seed(100 + si)
    numel = full_old.numel()
    k = max(1, numel // 100)
    pert = torch.randint(0, numel, (k,), device=dev, generator=g)
    full_new.view(-1)[pert] += torch.randn(k, dtype=torch.bfloat16, device=dev, generator=g) * 0.1

    dt_old = distribute_tensor(full_old, mesh, placements)
    dt_new = distribute_tensor(full_new, mesh, placements)

    # --- sharded path (real module) ---
    off, contributes, _pg = derive_placement(ShardSpec.from_param(dt_new))
    loc_new = dt_new.to_local().reshape(-1)
    loc_old = dt_old.to_local().reshape(-1)
    off2, _, _ = derive_placement(ShardSpec.from_param(dt_old))
    assert off == off2
    if contributes:
        gidx, gval = shard_delta_indices(loc_new, loc_old, off)
    else:
        gidx = torch.empty(0, dtype=torch.int64, device=dev)
        gval = torch.empty(0, dtype=loc_new.dtype, device=dev)
    counts = torch.tensor([int(gidx.numel())], dtype=torch.int64, device=gidx.device)
    ((sh_idx, sh_val),) = gather_v_batched_to_rank0(gidx, gval, counts)

    # --- baseline: full gather + diff ---
    fo = dt_old.full_tensor().reshape(-1)
    fn = dt_new.full_tensor().reshape(-1)
    bmask = fn.view(torch.int16) != fo.view(torch.int16)
    b_idx = bmask.nonzero(as_tuple=False).view(-1).to(torch.int64)
    b_val = fn[b_idx]

    if rank != 0:
        return True
    so = torch.argsort(sh_idx)
    bo = torch.argsort(b_idx)
    idx_ok = torch.equal(sh_idx[so], b_idx[bo])
    val_ok = torch.equal(sh_val[so].view(torch.int16), b_val[bo].view(torch.int16))
    ok = idx_ok and val_ok and (sh_idx.numel() == b_idx.numel())
    tag = "x".join(p.__class__.__name__[0] for p in placements)
    print(
        f"[shape={shape} mesh={tuple(mesh.shape)} {tag}] nnz sharded={sh_idx.numel()} "
        f"full={b_idx.numel()} idx={idx_ok} val={val_ok} -> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    all_ok = True

    # 1D FSDP mesh, Shard(0) -- uneven shapes stress the offset math.
    mesh1d = init_device_mesh("cuda", (world,))
    for si, shape in enumerate([(4096, 1024), (7, 3), (30001, 8), (128,)]):
        all_ok = _run_case(shape, [Shard(0)], mesh1d, dev, rank, si) and all_ok

    # 2D FSDP x SP(replicate) mesh -- the ulysses case: weights are Shard(0) on the FSDP dim
    # and Replicate on the SP dim, so only SP-coord-0 ranks may contribute (no double-count).
    if world % 2 == 0:
        mesh2d = init_device_mesh("cuda", (world // 2, 2), mesh_dim_names=("fsdp", "sp"))
        for si, shape in enumerate([(4096, 1024), (7, 3), (128,)]):
            all_ok = _run_case(shape, [Shard(0), Replicate()], mesh2d, dev, rank, 50 + si) and all_ok

    if rank == 0:
        print("=" * 50)
        print(f"OVERALL: {'ALL PASS ✅' if all_ok else 'FAIL ❌'}")
        print("=" * 50)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
