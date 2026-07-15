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
from verl.workers.engine.fsdp.sharded_delta import local_shard_view


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
    assert torch.equal(gval[order].view(torch.int16 if dtype == torch.bfloat16 else torch.int32),
                       new[got_pos].view(torch.int16 if dtype == torch.bfloat16 else torch.int32))


def test_shard_delta_indices_no_change_is_empty():
    shard = torch.randn(256, dtype=torch.bfloat16)
    gidx, gval = shard_delta_indices(shard.clone(), shard, offset=0)
    assert gidx.numel() == 0
    assert gval.numel() == 0


def test_local_shard_view_plain_tensor():
    # A non-DTensor (replicated / unsharded) param: whole tensor is local, offset 0.
    t = torch.randn(64, 8, dtype=torch.bfloat16)
    local, offset, contributes = local_shard_view(t)
    assert offset == 0
    assert local.shape == (64 * 8,)
    assert torch.equal(local, t.reshape(-1))
    # outside a process group, rank 0 is assumed -> contributes
    assert contributes is True


def test_fsdp_shardspec_profiles():
    """The FSDP export's spec must satisfy both contract profiles coherently:
    translate(idx) == idx + flat_offset, and rebuild_dense == rank-order concat."""
    from verl.checkpoint_engine.delta_sync.spec import ShardSpec

    full = torch.arange(24, dtype=torch.float32).view(6, 4)
    shards = list(full.chunk(3, dim=0))  # 3 "ranks", Shard(0)
    offsets = [0, 8, 16]

    specs = []
    for r, sh in enumerate(shards):
        off = offsets[r]
        specs.append(ShardSpec(
            contributes=True, shard_shape=tuple(sh.shape),
            translate=lambda idx, _o=off: idx + _o,
            hf_name="w", full_shape=(6, 4), full_numel=24, dense_offset=off,
            rebuild_dense=lambda sl: [("w", torch.cat([x.reshape(-1) for x in sl]).view(6, 4))],
        ))

    # translate: local idx 0..n maps onto the flat full tensor
    for r, (sh, spec) in enumerate(zip(shards, specs)):
        idx = torch.arange(sh.numel())
        assert torch.equal(full.reshape(-1)[spec.translate(idx)], sh.reshape(-1))

    # rebuild_dense: pure permutation, NaN positions preserved
    nan_shards = [torch.full_like(sh.reshape(-1), float("nan")) for sh in shards]
    nan_shards[1][3] = 42.0
    (_, rebuilt), = specs[0].rebuild_dense(nan_shards)
    fl = rebuilt.reshape(-1)
    pos = (~torch.isnan(fl)).nonzero(as_tuple=False).view(-1)
    assert pos.tolist() == [offsets[1] + 3] and fl[pos[0]] == 42.0
