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


def test_spec_to_hf_chunk_preserves_nan_sentinels():
    """A dim-0-separable converter must preserve NaN sentinel positions -- the
    property the engine's sender-side non-NaN extraction relies on."""
    from verl.workers.engine.spec import ShardSpec

    def to_hf_chunk(dim0_start, segment):
        # pure slice + rename, one output per dim-0 row (identity permutation)
        return [(f"w.{dim0_start + i}", segment[i]) for i in range(segment.shape[0])]

    spec = ShardSpec(
        full_shape=(6, 4),
        to_hf_chunk=to_hf_chunk,
        hf_slots=[(f"w.{i}", (4,)) for i in range(6)],
    )
    seg = torch.full((1, 4), float("nan"), dtype=torch.float32)
    seg[0, 3] = 42.0
    ((name, out),) = spec.to_hf_chunk(2, seg)
    fl = out.reshape(-1)
    pos = (~torch.isnan(fl)).nonzero(as_tuple=False).view(-1)
    assert name == "w.2" and pos.tolist() == [3] and fl[pos[0]] == 42.0


def test_gather_slot_entries_sub_rounds_world1():
    """max_round_bytes splits the slot list into deterministic sub-rounds; the
    reassembled output must equal the single-round result (world=1 mechanics)."""
    import os

    import torch.distributed as dist

    from verl.checkpoint_engine.delta_sync.sparse_gather import gather_slot_entries_to_rank0

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

    ref = gather_slot_entries_to_rank0(idx, val, counts)
    per_elem = idx.element_size() + val.element_size()
    for budget_elems in (1, 4, 8):
        got = gather_slot_entries_to_rank0(idx, val, counts, max_round_bytes=budget_elems * per_elem)
        assert len(got) == k
        for (ri, rv), (gi, gv) in zip(ref, got, strict=True):
            assert torch.equal(ri, gi)
            assert torch.equal(rv, gv)


def test_prime_then_hf_delta_export_roundtrip():
    """The backend-side default delta strategy: prime_delta_snapshots pins the
    shards; a later pass through hf_delta_export yields a final-HF-coordinate
    entry with exactly the changed elements and refreshes the snapshot (a
    second delta pass yields a zero-count entry)."""
    from verl.workers.engine.spec import ShardSpec
    from verl.workers.engine.utils import _hf_entry_identity, hf_delta_export, prime_delta_snapshots

    w = torch.arange(12, dtype=torch.float32)
    spec = ShardSpec(full_shape=(12,))
    snaps: dict = {}

    prime_delta_snapshots(iter([("w", w.clone(), spec)]), snaps)
    assert "w" in snaps and snaps["w"].numel() == 12

    w2 = w.clone()
    w2[3] = -1.0
    w2[7] = 42.0

    ((slots, dtype_str, counts, hf_idx, hf_val, pg),) = list(
        hf_delta_export(iter([("w", w2, spec)]), snaps, _hf_entry_identity)
    )
    assert slots == [("w", (12,))] and dtype_str == "float32" and pg is None
    assert counts.tolist() == [2]
    assert hf_idx.dtype == torch.int32 and hf_idx.tolist() == [3, 7]
    assert hf_val.tolist() == [-1.0, 42.0]

    ((_, _, counts2, hf_idx2, _, _),) = list(
        hf_delta_export(iter([("w", w2.clone(), spec)]), snaps, _hf_entry_identity)
    )
    assert counts2.tolist() == [0] and hf_idx2.numel() == 0


def test_hf_delta_export_converter_param():
    """A converter param's delta entry carries hf_slots-keyed final coordinates:
    the NaN probe maps each touched element through to_hf_chunk (the converter
    machinery is veomni's own -- see verl.workers.engine.veomni.utils)."""
    import importlib.util
    import pathlib

    # veomni/utils.py is torch-only, but the veomni package __init__ pulls in the
    # full engine (heavy deps); load the module by file path to keep this a CPU
    # unit test.
    import verl.workers.engine as _eng
    from verl.workers.engine.spec import BlockPlacement, ShardSpec
    from verl.workers.engine.utils import hf_delta_export, prime_delta_snapshots

    _p = pathlib.Path(_eng.__file__).parent / "veomni" / "utils.py"
    _spec = importlib.util.spec_from_file_location("_veomni_delta_utils", _p)
    _m = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    hf_entry_converter = _m.hf_entry_converter

    def to_hf_chunk(dim0_start, segment):
        return [(f"w.{dim0_start + i}", segment[i]) for i in range(segment.shape[0])]

    spec = ShardSpec(
        full_shape=(3, 4),
        place=BlockPlacement((3, 4), (0, 0), (3, 4)),
        to_hf_chunk=to_hf_chunk,
        hf_slots=[(f"w.{i}", (4,)) for i in range(3)],
    )
    w = torch.arange(12, dtype=torch.float32)
    snaps: dict = {}
    prime_delta_snapshots(iter([("w", w.clone(), spec)]), snaps)

    w2 = w.clone()
    w2[5] = -1.0  # row 1, col 1
    w2[11] = 42.0  # row 2, col 3
    ((slots, _dt, counts, hf_idx, hf_val, _pg),) = list(
        hf_delta_export(iter([("w", w2, spec)]), snaps, hf_entry_converter)
    )
    assert slots == spec.hf_slots
    assert counts.tolist() == [0, 1, 1]
    assert hf_idx.tolist() == [1, 3] and hf_val.tolist() == [-1.0, 42.0]


def test_hf_delta_export_requires_seed():
    """A delta export without a prior prime must fail loud, not diff against
    garbage."""
    import pytest

    from verl.workers.engine.spec import ShardSpec
    from verl.workers.engine.utils import _hf_entry_identity, hf_delta_export

    def raw():
        yield "w", torch.zeros(4), ShardSpec(full_shape=(4,))

    with pytest.raises(AssertionError, match="seed snapshot"):
        list(hf_delta_export(raw(), {}, _hf_entry_identity))
