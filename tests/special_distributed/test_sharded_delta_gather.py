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

torchrun --nproc_per_node=8 tests/special_distributed/test_sharded_delta_gather.py
"""

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor
from torch.distributed.tensor.placement_types import _StridedShard

from verl.checkpoint_engine.delta_sync.sparse_gather import (
    gather_slot_entries_to_rank0,
    shard_delta_indices,
)
from verl.workers.engine.spec import ShardSpec, derive_dtensor_placement, translate_flat_indices


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
    place, contributes, pg = derive_dtensor_placement(ShardSpec.from_param(dt_new))
    loc_new = dt_new.to_local().reshape(-1)
    loc_old = dt_old.to_local().reshape(-1)
    place2, _, _ = derive_dtensor_placement(ShardSpec.from_param(dt_old))
    assert place == place2

    # One copy of every block and no replicas: the product of the Shard dims' sizes.
    n_shard = 1
    for d, p in enumerate(placements):
        if p.is_shard() or type(p).__name__ == "_StridedShard":
            n_shard *= mesh.size(d)
    assert dist.get_world_size(pg) == n_shard, f"gather group has {dist.get_world_size(pg)} ranks, want {n_shard}"

    if contributes:
        # engine convention: diff in shard-local coordinates, then translate
        lidx, gval = shard_delta_indices(loc_new, loc_old, 0)
        gidx = translate_flat_indices(lidx, place)
    else:
        gidx = torch.empty(0, dtype=torch.int64, device=dev)
        gval = torch.empty(0, dtype=loc_new.dtype, device=dev)
    counts = torch.tensor([int(gidx.numel())], dtype=torch.int64, device=gidx.device)
    gathered = gather_slot_entries_to_rank0(gidx, gval, counts, group=pg)  # None off the group's rank 0

    # --- baseline: full gather + diff ---
    fo = dt_old.full_tensor().reshape(-1)
    fn = dt_new.full_tensor().reshape(-1)
    bmask = fn.view(torch.int16) != fo.view(torch.int16)
    b_idx = bmask.nonzero(as_tuple=False).view(-1).to(torch.int64)
    b_val = fn[b_idx]

    if rank != 0:
        return True
    ((sh_idx, sh_val),) = gathered
    so = torch.argsort(sh_idx)
    bo = torch.argsort(b_idx)
    idx_ok = torch.equal(sh_idx[so], b_idx[bo])
    val_ok = torch.equal(sh_val[so].view(torch.int16), b_val[bo].view(torch.int16))
    ok = idx_ok and val_ok and (sh_idx.numel() == b_idx.numel())
    tag = "x".join("Strided" if type(p).__name__ == "_StridedShard" else type(p).__name__[0] for p in placements)
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

    # 2D FSDP x TP mesh: column-parallel cuts the same dim twice (_StridedShard), row-parallel a second one.
    if world % 4 == 0:
        meshtp = init_device_mesh("cuda", (world // 2, 2), mesh_dim_names=("fsdp", "tp"))
        for si, shape in enumerate([(4096, 1024), (2048, 128), (32, 6)]):
            pl = [_StridedShard(0, split_factor=2), Shard(0)]
            all_ok = _run_case(shape, pl, meshtp, dev, rank, 100 + si) and all_ok
        for si, shape in enumerate([(4096, 1024), (32, 6)]):
            all_ok = _run_case(shape, [Shard(0), Shard(1)], meshtp, dev, rank, 150 + si) and all_ok

    # 2D EFSDP x EP mesh: column-parallel TP's geometry one rank higher, since both dims cut the expert dim.
    if world % 4 == 0:
        meshep = init_device_mesh("cuda", (world // 4, 4), mesh_dim_names=("efsdp", "ep"))
        for si, shape in enumerate([(64, 768, 256), (64, 256, 768), (8, 3, 5)]):
            pl = [_StridedShard(0, split_factor=4), Shard(0)]
            all_ok = _run_case(shape, pl, meshep, dev, rank, 200 + si) and all_ok

    # 3D HSDP x TP and HSDP x EP: same blocks, so what is under test is that the group stops at the Shard dims.
    if world % 8 == 0:
        meshhtp = init_device_mesh("cuda", (2, world // 4, 2), mesh_dim_names=("dp_replicate", "fsdp", "tp"))
        for si, shape in enumerate([(4096, 1024), (32, 6)]):
            pl = [Replicate(), _StridedShard(0, split_factor=2), Shard(0)]
            all_ok = _run_case(shape, pl, meshhtp, dev, rank, 300 + si) and all_ok
        for si, shape in enumerate([(4096, 1024), (32, 6)]):
            all_ok = _run_case(shape, [Replicate(), Shard(0), Shard(1)], meshhtp, dev, rank, 350 + si) and all_ok

        meshhep = init_device_mesh("cuda", (2, world // 4, 2), mesh_dim_names=("dp_replicate", "efsdp", "ep"))
        for si, shape in enumerate([(64, 768, 256), (8, 3, 5)]):
            pl = [Replicate(), _StridedShard(0, split_factor=2), Shard(0)]
            all_ok = _run_case(shape, pl, meshhep, dev, rank, 400 + si) and all_ok

        # replicate degree 4 leaves rank 0's group a quarter of the mesh: the sharpest "group is not the mesh".
        meshwide = init_device_mesh("cuda", (4, world // 8, 2), mesh_dim_names=("dp_replicate", "efsdp", "ep"))
        for si, shape in enumerate([(64, 768, 256), (8, 3, 5)]):
            pl = [Replicate(), _StridedShard(0, split_factor=2), Shard(0)]
            all_ok = _run_case(shape, pl, meshwide, dev, rank, 450 + si) and all_ok

    if rank == 0:
        print("=" * 50)
        print(f"OVERALL: {'ALL PASS ✅' if all_ok else 'FAIL ❌'}")
        print("=" * 50)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
