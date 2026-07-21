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
"""CPU unit tests for the sharded-delta primitives.

The full sharded path (DTensor shards + gather-v across ranks vs the full-gather diff) is
validated bit-identically in a multi-GPU check; see
``tests/special_distributed/test_sharded_delta_gather.py`` (run with torchrun). These
tests cover the process-local pieces that CI can run without a process group.
"""

from __future__ import annotations

import pytest
import torch

from verl.checkpoint_engine.delta_sync.sparse_gather import shard_delta_indices


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_shard_delta_indices_matches_bytewise_diff(dtype):
    torch.manual_seed(0)
    # A "shard" of some parameter, whose flat start in the full param is `offset`.
    shard = torch.randn(1000, dtype=dtype)
    new = shard.clone()
    changed = torch.tensor([3, 17, 500, 999], dtype=torch.int64)
    new[changed] = new[changed] + 0.5
    offset = 4096  # this shard begins at flat position 4096 within the full param

    gidx, gval = shard_delta_indices(new, shard, offset)

    # positions are (offset + local changed index), bytewise-exact values
    assert torch.equal(gidx.sort().values, (changed + offset).sort().values)
    order = torch.argsort(gidx)
    got_pos = (gidx[order] - offset).to(torch.int64)
    assert torch.equal(
        gval[order].view(torch.int16 if dtype == torch.bfloat16 else torch.int32),
        new[got_pos].view(torch.int16 if dtype == torch.bfloat16 else torch.int32),
    )


def test_shard_delta_indices_no_change_is_empty():
    shard = torch.randn(256, dtype=torch.bfloat16)
    gidx, gval = shard_delta_indices(shard.clone(), shard, offset=0)
    assert gidx.numel() == 0
    assert gval.numel() == 0


def test_derive_placement_unsharded():
    # A non-DTensor (replicated / unsharded) param: offset 0, no gather group,
    # and outside a process group rank 0 is assumed -> contributes.
    from verl.workers.engine.spec import ShardSpec, derive_placement

    t = torch.randn(64, 8, dtype=torch.bfloat16)
    spec = ShardSpec.from_param(t)
    assert spec.mesh is None and spec.full_shape == (64, 8)
    offset, contributes, group = derive_placement(spec)
    assert offset == 0 and contributes is True and group is None


def test_spec_to_hf_pure_permutation():
    """A converter spec (Megatron-style) must preserve NaN sentinel positions --
    the property the engine's sparse rebuild relies on."""
    from verl.workers.engine.spec import ShardSpec

    full = torch.arange(24, dtype=torch.float32).view(6, 4)
    shards = [sh.reshape(-1) for sh in full.chunk(3, dim=0)]

    def to_hf(shard_list):
        return [("w", torch.cat(shard_list).view(6, 4))]

    spec = ShardSpec(full_shape=(6, 4), to_hf=to_hf)
    nan_shards = [torch.full_like(sh, float("nan")) for sh in shards]
    nan_shards[1][3] = 42.0
    ((_, rebuilt),) = spec.to_hf(nan_shards)
    fl = rebuilt.reshape(-1)
    pos = (~torch.isnan(fl)).nonzero(as_tuple=False).view(-1)
    assert pos.tolist() == [8 + 3] and fl[pos[0]] == 42.0


def test_gather_v_batched_sub_rounds_world1():
    """max_round_bytes splits the slot list into deterministic sub-rounds; the
    reassembled output must equal the single-round result (world=1 mechanics)."""
    import os

    import torch.distributed as dist

    from verl.checkpoint_engine.delta_sync.sparse_gather import gather_v_batched_to_rank0

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29512")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    torch.manual_seed(7)
    k = 9
    counts = torch.tensor([5, 0, 3, 7, 1, 0, 4, 2, 6], dtype=torch.int64)
    n = int(counts.sum())
    idx = torch.randint(0, 1000, (n,), dtype=torch.int32)
    val = torch.randn(n, dtype=torch.bfloat16)

    ref = gather_v_batched_to_rank0(idx, val, counts)
    per_elem = idx.element_size() + val.element_size()
    for budget_elems in (1, 4, 8):
        got = gather_v_batched_to_rank0(idx, val, counts, max_round_bytes=budget_elems * per_elem)
        assert len(got) == k
        for (ri, rv), (gi, gv) in zip(ref, got, strict=True):
            assert torch.equal(ri, gi)
            assert torch.equal(rv, gv)
