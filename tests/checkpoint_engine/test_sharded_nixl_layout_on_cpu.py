# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""CPU tests of the ``sharded_nixl`` bucket layout and write planning (``sharded_nixl_layout``)."""

import functools

import pytest
import torch

from verl.checkpoint_engine.sharded_nixl_layout import (
    BoxSegment,
    BucketLayout,
    LayoutBuilder,
    ParamEntry,
    build_bucket_layout,
    clip_segments,
    contribution_range,
    expand_fused_moe_entry,
    index_runs,
    local_entry,
    local_shapes_key,
    merge_box_plans,
    normalize_slices,
    owned_name_filter,
    plan_box_writes,
    plan_indexed_writes,
    plan_rank_local_indexed_writes,
    plan_rank_local_writes,
    plan_writes,
    slices_digest,
)
from verl.workers.engine.fsdp.utils import unfuse_moe_params
from verl.workers.engine.spec import BlockPlacement, ShardSpec
from verl.workers.rollout.base import RolloutLayout

BF16 = torch.bfloat16


def _entries(spec):
    """``ParamEntry`` list of plain tensors from ``(name, shape, dtype)`` tuples."""
    return [ParamEntry(n, tuple(s), d, n, 0) for n, s, d in spec]


def test_plan_indexed_writes_compacts_runs_across_buckets():
    builder = LayoutBuilder(bucket_size=8, align=2)
    entry = ParamEntry("w", (10,), BF16, "source")
    builder.add(entry)
    layout = builder.build()

    writes = plan_indexed_writes(layout, [entry], index_runs(torch.tensor([6]), torch.tensor([1, 2, 5, 6, 7, 9])))

    assert [(w.bucket_id, w.dst_offset, w.src_offset, w.length) for w in writes] == [
        (0, 2, 0, 4),
        (1, 2, 4, 6),
        (2, 2, 10, 2),
    ]


# ---------------------------------------------------------------- layout


def test_layout_first_fit_and_alignment():
    bucket = 1024
    layout = build_bucket_layout(_entries([("a", (100,), BF16), ("b", (100,), BF16), ("c", (300,), BF16)]), bucket, 256)
    a, b, c = layout.slots
    assert (a.bucket_id, a.offset) == (0, 0)
    assert (b.bucket_id, b.offset) == (0, 256)  # 200 bytes -> aligned up to 256
    assert (c.bucket_id, c.offset) == (1, 0)  # 600 bytes do not fit after 456
    assert layout.num_buckets == 2
    assert layout.total_bytes == 1000


def test_layout_chunks_tensor_larger_than_bucket():
    """A tensor larger than a bucket starts a fresh bucket and is cut into bucket-sized chunks."""
    bucket = 1024
    layout = build_bucket_layout(
        _entries([("s", (10,), BF16), ("big", (1500,), BF16), ("t", (10,), BF16)]), bucket, 256
    )
    slots = layout.slots_by_name["big"]
    assert [(s.bucket_id, s.chunk_offset, s.chunk_size, s.offset) for s in slots] == [
        (1, 0, 1024, 0),
        (2, 1024, 1024, 0),
        (3, 2048, 952, 0),
    ]
    t = layout.slots_by_name["t"][0]
    assert (t.bucket_id, t.offset) == (4, 0)  # 952 aligns up to 1024: no room left in bucket 3
    assert layout.num_buckets == 5


def test_layout_duplicate_names_raise():
    b = LayoutBuilder(1024)
    b.add(ParamEntry("x", (2,), BF16, "x", 0))
    with pytest.raises(ValueError):
        b.add(ParamEntry("x", (2,), BF16, "x", 0))


def test_layout_digest_and_wire_roundtrip():
    e = _entries([("a", (4, 4), BF16), ("b", (7,), torch.float32)])
    l1 = build_bucket_layout(e, 4096)
    l2 = build_bucket_layout(e, 4096)
    l3 = build_bucket_layout(e, 8192)
    assert l1.digest == l2.digest != l3.digest
    rt = BucketLayout.from_wire(l1.to_wire())
    assert rt.digest == l1.digest and rt.slots == l1.slots and rt.entries == l1.entries


# ---------------------------------------------------------------- MoE expansion


def test_expand_fused_moe_matches_unfuse_moe_params():
    """Every per-expert entry's byte range in the fused tensor equals ``unfuse_moe_params``."""
    E, inter, H = 4, 6, 5
    gate_up = torch.randn(E, 2 * inter, H, dtype=BF16)
    down = torch.randn(E, H, inter, dtype=BF16)
    named = [("model.layers.0.mlp.experts.gate_up_proj", gate_up), ("model.layers.0.mlp.experts.down_proj", down)]
    reference = dict(unfuse_moe_params(iter(named), "qwen3_moe"))
    for src_name, t in named:
        entries = expand_fused_moe_entry(src_name, tuple(t.shape), t.dtype, "qwen3_moe")
        flat = t.reshape(-1)
        assert all(e.name in reference for e in entries)
        for e in entries:
            ref = reference[e.name]
            assert tuple(ref.shape) == e.shape
            assert torch.equal(flat[e.src_elem_offset : e.src_elem_offset + e.numel].view(e.shape), ref)
    # gpt_oss keeps the fused layout for inference
    assert (
        len(expand_fused_moe_entry("model.layers.0.mlp.experts.gate_up_proj", (E, 2 * inter, H), BF16, "gpt_oss")) == 1
    )
    assert expand_fused_moe_entry("model.norm.weight", (H,), BF16, "qwen3_moe")[0].name == "model.norm.weight"


# ---------------------------------------------------------------- contribution range / write planning


def _block(full_shape, local_shape, global_offset, contributes=True):
    return ShardSpec(
        full_shape=tuple(full_shape),
        place=BlockPlacement(tuple(local_shape), tuple(global_offset), tuple(full_shape)),
        contributes=contributes,
    )


def test_contribution_range_shard0_and_replicated():
    """Shard(0) block, HSDP replica (no contribution), replicated tensor, unsupported dim-1 cut."""
    itemsize = 2
    spec = _block((8, 4), (2, 4), (4, 0))  # rank holding rows 4..6 of an 8x4 tensor
    assert contribution_range(spec, 8, itemsize) == (4 * 4 * itemsize, 8 * itemsize)
    assert contribution_range(_block((8, 4), (2, 4), (4, 0), contributes=False), 8, itemsize) is None
    rep = ShardSpec(full_shape=(3, 3), place=0, contributes=True)
    assert contribution_range(rep, 9, 4) == (0, 36)
    with pytest.raises(NotImplementedError):
        contribution_range(_block((8, 4), (8, 2), (0, 2)), 16, itemsize)  # dim-1 cut


def _simulate(full_tensors: dict[str, torch.Tensor], world: int, bucket_size: int, fused_model_type=None, owns=None):
    """Shard every tensor along dim 0 across ``world`` ranks and apply each rank's planned writes
    to fake buckets, asserting no two writes overlap. Returns (layout, buckets, entries_by_src)."""
    entries_by_src = {}
    builder = LayoutBuilder(bucket_size, 256)
    for name, t in full_tensors.items():
        es = expand_fused_moe_entry(name, tuple(t.shape), t.dtype, fused_model_type)
        entries_by_src[name] = es
        for e in es:
            builder.add(e)
    layout = builder.build()
    buckets = [bytearray(bucket_size) for _ in range(layout.num_buckets)]
    written = [bytearray(bucket_size) for _ in range(layout.num_buckets)]
    for rank in range(world):
        for name, t in full_tensors.items():
            rows = t.shape[0]
            per = (rows + world - 1) // world
            r0, r1 = min(rank * per, rows), min((rank + 1) * per, rows)
            if r0 >= r1:
                continue
            local = t[r0:r1].contiguous()
            spec = _block(t.shape, local.shape, (r0,) + (0,) * (t.dim() - 1))
            rng = contribution_range(spec, local.numel(), t.element_size())
            u8 = local.reshape(-1).view(torch.uint8).numpy().tobytes()
            for s in plan_writes(layout, entries_by_src[name], rng):
                if owns is not None and not owns(s.name):
                    continue
                buckets[s.bucket_id][s.dst_offset : s.dst_offset + s.length] = u8[
                    s.src_offset : s.src_offset + s.length
                ]
                assert not any(written[s.bucket_id][s.dst_offset : s.dst_offset + s.length]), "overlapping writes"
                written[s.bucket_id][s.dst_offset : s.dst_offset + s.length] = b"\x01" * s.length
    return layout, buckets, entries_by_src


def _reassemble(layout, buckets, name):
    """Concatenate a tensor's chunks from the fake buckets."""
    e = layout.entry_by_name[name]
    raw = bytearray()
    for s in layout.slots_by_name[name]:
        raw += buckets[s.bucket_id][s.offset : s.offset + s.chunk_size]
    return torch.frombuffer(bytes(raw), dtype=e.dtype).view(e.shape) if e.numel else torch.empty(e.shape, dtype=e.dtype)


@pytest.mark.parametrize("world", [1, 2, 3, 4])
def test_end_to_end_reassembly_dense(world):
    torch.manual_seed(0)
    tensors = {
        "embed": torch.randn(37, 16, dtype=BF16),  # uneven split across ranks
        "w1": torch.randn(8, 24, dtype=BF16),
        "big": torch.randn(600, 8, dtype=BF16),  # 9600 bytes > bucket -> chunked
        "bias": torch.randn(5, dtype=torch.float32),
        "tiny": torch.randn(1, 3, dtype=BF16),  # fewer rows than ranks
    }
    layout, buckets, _ = _simulate(tensors, world, bucket_size=4096)
    for name, t in tensors.items():
        assert torch.equal(_reassemble(layout, buckets, name), t), name


@pytest.mark.parametrize("world", [1, 2, 4])
def test_end_to_end_reassembly_fused_moe(world):
    """Sharded fused MoE tensors reassemble into the per-expert tensors of ``unfuse_moe_params``."""
    torch.manual_seed(1)
    E, inter, H = 8, 6, 5
    tensors = {
        "model.layers.0.mlp.experts.gate_up_proj": torch.randn(E, 2 * inter, H, dtype=BF16),
        "model.layers.0.mlp.experts.down_proj": torch.randn(E, H, inter, dtype=BF16),
        "model.layers.0.mlp.gate.weight": torch.randn(E, H, dtype=BF16),
    }
    ref = dict(unfuse_moe_params(iter(tensors.items()), "qwen3_moe"))
    layout, buckets, _ = _simulate(tensors, world, bucket_size=2048, fused_model_type="qwen3_moe")
    assert set(ref) == set(layout.entry_by_name)
    for name, t in ref.items():
        assert torch.equal(_reassemble(layout, buckets, name), t), name


def test_ownership_filter_ep_and_pp():
    rl = RolloutLayout(ep_size=2, ep_rank=1, local_experts={0: [2, 3], 1: [2, 3]}, layer_range=(0, 2))
    owns = owned_name_filter(rl)
    assert owns("model.embed_tokens.weight")
    assert owns("model.layers.0.mlp.experts.2.gate_proj.weight")
    assert not owns("model.layers.0.mlp.experts.0.gate_proj.weight")
    assert owns("model.layers.1.self_attn.q_proj.weight")
    assert not owns("model.layers.2.self_attn.q_proj.weight")  # other pipeline stage
    assert owns("model.layers.5.mlp.experts.0.up_proj.weight") is False
    assert not owns("model.language_model.layers.2.self_attn.q_proj.weight")
    assert owns("model.vision_tower.vision_model.encoder.layers.5.mlp.fc1.weight")  # not pipeline-split
    assert owned_name_filter(None)("anything")
    d = rl.to_dict()
    assert RolloutLayout.from_dict(d) == rl


@functools.cache
def _load_sharded_nixl_engine_mod():
    """Import ``sharded_nixl_checkpoint_engine`` even when ``nixl``/``cupy`` are absent (CPU CI)."""
    try:
        from verl.checkpoint_engine import sharded_nixl_checkpoint_engine as mod

        return mod
    except ImportError:
        pass

    import dataclasses
    import importlib.util
    import sys
    import types
    from pathlib import Path

    from verl.checkpoint_engine.base import CheckpointEngineRegistry

    stub_name = "verl.checkpoint_engine.nixl_checkpoint_engine"
    stub = types.ModuleType(stub_name)

    @dataclasses.dataclass
    class NixlAgentMetadata:
        agent_name: str
        agent_metadata: bytes
        zmq_ip: str
        zmq_port: int

    class NixlAgent:
        pass

    stub.NixlAgentMetadata = NixlAgentMetadata
    stub.NixlAgent = NixlAgent
    path = Path(__file__).resolve().parents[2] / "verl/checkpoint_engine/sharded_nixl_checkpoint_engine.py"
    spec = importlib.util.spec_from_file_location("_sharded_nixl_ce_cpu_test", path)
    mod = importlib.util.module_from_spec(spec)
    prev = sys.modules.get(stub_name)
    try:
        sys.modules[stub_name] = stub
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    finally:
        # the stub-backed class must not stay registered: other tests expect the recorded import error
        CheckpointEngineRegistry._registry.pop("sharded_nixl", None)
        sys.modules.pop(spec.name, None)
        if prev is None:
            sys.modules.pop(stub_name, None)
        else:
            sys.modules[stub_name] = prev
    return mod


def test_slice_hash_is_64_bit_and_alignment_free():
    """The per-slice checksum is the XOR of the slice's 8-byte words (an 8-bit XOR of its bytes
    misses swapped bytes) and does not depend on where the slice sits in the buffer."""
    _hash_u8_many = _load_sharded_nixl_engine_mod()._hash_u8_many

    torch.manual_seed(0)
    data = torch.randint(0, 256, (37,), dtype=torch.uint8)
    data[0], data[1] = 1, 2
    expect = 0
    for word in torch.cat([data, torch.zeros(3, dtype=torch.uint8)]).view(torch.int64).tolist():
        expect ^= word & (2**64 - 1)
    buf = torch.zeros(64, dtype=torch.uint8)
    buf[3:40] = data
    assert _hash_u8_many(buf, [(3, 37), (0, 0)]) == [expect, 0]
    assert _hash_u8_many(data, [(0, 37)]) == [expect]
    swapped = data.clone()
    swapped[0], swapped[1] = data[1], data[0]
    assert _hash_u8_many(swapped, [(0, 37)]) != [expect]


def test_ownership_filtered_writes_only_touch_owned_slots():
    """Experts the generator does not own are never written."""
    torch.manual_seed(2)
    E, inter, H = 4, 4, 4
    tensors = {
        "model.layers.0.mlp.experts.gate_up_proj": torch.randn(E, 2 * inter, H, dtype=BF16),
        "model.layers.0.mlp.gate.weight": torch.randn(E, H, dtype=BF16),
    }
    owns = owned_name_filter(RolloutLayout(ep_size=2, ep_rank=0, local_experts={0: [0, 1]}))
    layout, buckets, _ = _simulate(tensors, 2, bucket_size=1024, fused_model_type="qwen3_moe", owns=owns)
    ref = dict(unfuse_moe_params(iter(tensors.items()), "qwen3_moe"))
    for name, t in ref.items():
        got = _reassemble(layout, buckets, name)
        if owns(name):
            assert torch.equal(got, t), name
        else:
            assert not torch.equal(got, t) or t.numel() == 0, f"{name} should not have been written"


# ---------------------------------------------------------------- rank-local (TP) boxes
#
# Brute-force reference: fill the full tensor with its own flat indices (torch.arange), so every
# byte that lands anywhere identifies exactly which source element it came from.

I32 = torch.int32


def _gather(src: bytes, segments) -> bytes:
    """Apply BoxSegments to a rank's local shard bytes (what the engine's pack loop does)."""
    out = bytearray()
    for s in segments:
        for j in range(s.nrows):
            o = s.src_offset + j * s.stride
            out += src[o : o + s.row_bytes]
    return bytes(out)


def _flat_ranges(numel: int, world: int, row_len: int | None):
    """Per-rank flat element intervals: FSDP2 Shard(0)-like (row-aligned) when ``row_len`` is
    given, otherwise an even split of the flat elements (not row-aligned)."""
    if row_len:
        rows = numel // row_len
        per = (rows + world - 1) // world
        return [(min(r * per, rows) * row_len, min((r + 1) * per, rows) * row_len) for r in range(world)]
    per = (numel + world - 1) // world
    return [(min(r * per, numel), min((r + 1) * per, numel)) for r in range(world)]


def _check_box(shape, box, world, row_aligned):
    full = torch.arange(int(torch.tensor(shape).prod()), dtype=I32).view(shape)
    e = ParamEntry("w", tuple(shape), I32, "w", 0)
    expected = full[tuple(slice(s, s + n) for s, n in box)].contiguous() if box is not None else full
    exp_bytes = expected.reshape(-1).numpy().tobytes()
    got = bytearray(len(exp_bytes))
    hit = bytearray(len(exp_bytes))
    row_len = shape[-1] if (row_aligned and len(shape) == 2) else None
    flat_u8 = full.reshape(-1).numpy().tobytes()
    for f0, f1 in _flat_ranges(full.numel(), world, row_len):
        if f0 >= f1:
            continue
        run = plan_box_writes(e, box, (f0 * 4, (f1 - f0) * 4))
        # brute force: which box elements (box order) come from [f0, f1)
        want = [i for i, v in enumerate(expected.reshape(-1).tolist()) if f0 <= v < f1]
        if not want:
            assert run is None
            continue
        assert run is not None
        assert len(run.segments) <= 3, run.segments
        assert want == list(range(want[0], want[-1] + 1)), "box ∩ interval must be contiguous in box order"
        assert (run.dst_offset, run.length) == (want[0] * 4, len(want) * 4)
        data = _gather(flat_u8[f0 * 4 : f1 * 4], run.segments)
        assert len(data) == run.length
        got[run.dst_offset : run.dst_offset + run.length] = data
        assert not any(hit[run.dst_offset : run.dst_offset + run.length]), "overlapping runs"
        hit[run.dst_offset : run.dst_offset + run.length] = b"\x01" * run.length
    assert all(hit), "box not fully covered"
    assert bytes(got) == exp_bytes


@pytest.mark.parametrize("world", [1, 2, 3, 4, 5, 7])
@pytest.mark.parametrize("row_aligned", [True, False])
@pytest.mark.parametrize(
    "shape, box",
    [
        ((8, 6), ((2, 3), (0, 6))),  # dim-0 (column-parallel) box
        ((8, 6), ((0, 8), (2, 2))),  # dim-1 (row-parallel) box
        ((8, 6), ((0, 8), (0, 3))),  # dim-1, first half
        ((8, 6), ((0, 8), (3, 3))),  # dim-1, second half
        ((9, 7), ((1, 5), (2, 4))),  # interior box
        ((9, 7), None),  # full tensor
        ((9, 7), ((0, 9), (0, 7))),  # explicit full box
        ((13,), ((4, 6),)),  # 1-D box
        ((5, 1), ((1, 3), (0, 1))),  # single column
        ((1, 11), ((0, 1), (3, 5))),  # single row
    ],
)
def test_plan_box_writes_brute_force(shape, box, world, row_aligned):
    """plan_box_writes vs brute force for every rank of a flat split: one contiguous destination
    run per rank equal to the box elements its interval holds, <= 3 source segments, and the
    union of all ranks' runs reconstructs ``full[box]`` exactly once."""
    _check_box(shape, box, world, row_aligned)


def test_plan_box_writes_random_intervals():
    """Arbitrary (not even-split) intervals over random boxes."""
    g = torch.Generator().manual_seed(0)
    for _ in range(300):
        R, C = (int(x) for x in torch.randint(1, 9, (2,), generator=g))
        r0 = int(torch.randint(0, R, (1,), generator=g))
        c0 = int(torch.randint(0, C, (1,), generator=g))
        nr = int(torch.randint(1, R - r0 + 1, (1,), generator=g))
        nc = int(torch.randint(1, C - c0 + 1, (1,), generator=g))
        cuts = sorted({0, R * C, *(int(x) for x in torch.randint(0, R * C, (3,), generator=g))})
        full = torch.arange(R * C, dtype=I32).view(R, C)
        e = ParamEntry("w", (R, C), I32, "w", 0)
        expected = full[r0 : r0 + nr, c0 : c0 + nc].reshape(-1)
        got = torch.full_like(expected, -1)
        flat = full.reshape(-1).numpy().tobytes()
        for f0, f1 in zip(cuts, cuts[1:], strict=False):
            run = plan_box_writes(e, ((r0, nr), (c0, nc)), (f0 * 4, (f1 - f0) * 4))
            if run is None:
                continue
            assert len(run.segments) <= 3
            data = torch.frombuffer(bytearray(_gather(flat[f0 * 4 : f1 * 4], run.segments)), dtype=I32)
            got[run.dst_offset // 4 : (run.dst_offset + run.length) // 4] = data
        assert torch.equal(got, expected)


def test_plan_rank_local_indexed_writes_matches_boxes():
    """HF-indexed (Megatron) contributions cut into generator boxes: two column-parallel trainer
    ranks against matching column boxes and a row box; back-to-back runs collapse to one slice."""
    R, C = 4, 8
    full = torch.arange(R * C, dtype=I32).view(R, C)
    e = ParamEntry("w", (R, C), I32, "w")
    for box in (((0, R), (0, 4)), ((0, R), (4, 4)), ((1, 2), (0, C))):
        builder = LayoutBuilder(bucket_size=1 << 20, align=8)
        builder.add(local_entry(e, box))
        layout = builder.build()
        boxes = normalize_slices({"w": {"full_shape": [R, C], "box": [list(b) for b in box]}})
        expected = full[box[0][0] : box[0][0] + box[0][1], box[1][0] : box[1][0] + box[1][1]].reshape(-1)
        got = torch.full_like(expected, -1)
        for r in range(2):
            indices = (torch.arange(R)[:, None] * C + torch.arange(r * 4, r * 4 + 4)).reshape(-1)
            values = full.reshape(-1)[indices].numpy().tobytes()
            runs = {"w": index_runs(torch.tensor([indices.numel()]), indices)}
            plan = plan_rank_local_indexed_writes({"w": [e]}, runs, layout, boxes).get("w", [])
            if box[1] == (r * 4, 4):
                assert len(plan) == 1 and len(plan[0].segments) == 1
            for w in plan:
                data = torch.frombuffer(bytearray(_gather(values, w.segments)), dtype=I32)
                got[w.dst_offset // 4 : (w.dst_offset + w.length) // 4] = data
        assert torch.equal(got, expected)


def test_plan_rank_local_indexed_writes_strided_box_is_one_segment():
    """A box narrower than the trainer's column shard (vLLM TP=8 against trainer TP=4) arrives as
    one HF run per row; the rows must coalesce into one strided segment, not one copy per row."""
    R, C = 6, 8
    full = torch.arange(R * C, dtype=I32).view(R, C)
    e = ParamEntry("w", (R, C), I32, "w")
    indices = (torch.arange(R)[:, None] * C + torch.arange(4)).reshape(-1)  # trainer rank 0: columns 0-3
    values = full.reshape(-1)[indices].numpy().tobytes()
    runs = {"w": index_runs(torch.tensor([indices.numel()]), indices)}
    for box in (((0, R), (0, 2)), ((0, R), (2, 2)), ((1, 4), (1, 2))):
        builder = LayoutBuilder(bucket_size=1 << 20, align=8)
        builder.add(local_entry(e, box))
        boxes = normalize_slices({"w": {"full_shape": [R, C], "box": [list(b) for b in box]}})
        (w,) = plan_rank_local_indexed_writes({"w": [e]}, runs, builder.build(), boxes)["w"]
        assert len(w.segments) == 1 and w.segments[0].nrows == box[0][1], w.segments
        expected = full[box[0][0] : box[0][0] + box[0][1], box[1][0] : box[1][0] + box[1][1]].reshape(-1)
        assert torch.equal(torch.frombuffer(bytearray(_gather(values, w.segments)), dtype=I32), expected)


def test_plan_box_writes_fused_moe_entry():
    """A per-expert entry inside a fused source tensor (src_elem_offset != 0): boxes are over the
    expert's own 2-D shape, sources are offsets into the rank's shard of the FUSED tensor."""
    E, inter, H = 3, 4, 6
    fused = torch.arange(E * 2 * inter * H, dtype=I32).view(E, 2 * inter, H)
    entries = expand_fused_moe_entry("model.layers.0.mlp.experts.gate_up_proj", (E, 2 * inter, H), I32, "qwen3_moe")
    flat = fused.reshape(-1).numpy().tobytes()
    world = 4
    for e in entries:
        ref = fused.reshape(-1)[e.src_elem_offset : e.src_elem_offset + e.numel].view(e.shape)
        for box in (((1, 2), (0, H)), ((0, inter), (3, 3))):
            expected = ref[box[0][0] : box[0][0] + box[0][1], box[1][0] : box[1][0] + box[1][1]].reshape(-1)
            got = torch.full_like(expected, -1)
            for f0, f1 in _flat_ranges(fused.numel(), world, None):
                run = plan_box_writes(e, box, (f0 * 4, (f1 - f0) * 4))
                if run is None:
                    continue
                data = torch.frombuffer(bytearray(_gather(flat[f0 * 4 : f1 * 4], run.segments)), dtype=I32)
                got[run.dst_offset // 4 : (run.dst_offset + run.length) // 4] = data
            assert torch.equal(got, expected), (e.name, box)


def test_clip_segments_matches_bytes():
    """clip_segments(segs, lo, hi) gathers exactly bytes [lo, hi) of the dense gather of segs."""
    src = bytes(range(256)) * 2
    segs = (BoxSegment(3, 1, 5, 5), BoxSegment(20, 4, 6, 16), BoxSegment(90, 1, 7, 7))
    dense = _gather(src, segs)
    for lo in range(len(dense)):
        for hi in range(lo + 1, len(dense) + 1):
            clipped = clip_segments(segs, lo, hi)
            assert _gather(src, clipped) == dense[lo:hi], (lo, hi, clipped)
            assert sum(s.nbytes for s in clipped) == hi - lo


def _tp_slices(shapes: dict, tp: int, rank: int, num_kv_heads: int, head_dim: int) -> dict:
    """Tiny Qwen-like TP rules: q/gate/embed dim 0, o/down dim 1, k/v replicated per kv head
    when tp > num_kv_heads, norms absent (full)."""
    out = {}
    for name, shape in shapes.items():
        if name.endswith(("k_proj.weight", "v_proj.weight")):
            if tp >= num_kv_heads:
                blk = rank // (tp // num_kv_heads)
                box = [[blk * head_dim, head_dim], [0, shape[1]]]
            else:
                n = shape[0] // tp
                box = [[rank * n, n], [0, shape[1]]]
        elif name.endswith(("q_proj.weight", "gate_proj.weight", "embed_tokens.weight")):
            n = shape[0] // tp
            box = [[rank * n, n], [0, shape[1]]]
        elif name.endswith(("o_proj.weight", "down_proj.weight")):
            n = shape[1] // tp
            box = [[0, shape[0]], [rank * n, n]]
        else:
            continue
        out[name] = {"full_shape": list(shape), "box": box}
    return out


@pytest.mark.parametrize("trainers", [1, 2, 3])
@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("row_aligned", [True, False])
def test_rank_local_end_to_end(trainers, tp, row_aligned):
    """Full rank-local pipeline on fake buckets: rank-local layout from local shapes (one tensor
    larger than a bucket), per-generator plans for a TP=tp replica plus a second (DP) replica,
    dedupe across generators (merge_box_plans), pack each distinct piece once per trainer, write
    every generator's table entries from the packed pieces, reassemble -> each generator holds
    exactly ``full[box]`` of every tensor. Also checks the dedupe: k/v pieces are shared by
    the TP ranks with identical boxes (num_kv_heads=2 < tp=4) and by the DP replica."""
    torch.manual_seed(3)
    nkv, hd, H = 2, 4, 8
    shapes = {
        "model.embed_tokens.weight": (16, H),
        "model.layers.0.self_attn.q_proj.weight": (4 * hd, H),
        "model.layers.0.self_attn.k_proj.weight": (nkv * hd, H),
        "model.layers.0.self_attn.v_proj.weight": (nkv * hd, H),
        "model.layers.0.self_attn.o_proj.weight": (H, 4 * hd),
        "model.layers.0.mlp.gate_proj.weight": (96, H),  # local 48 x 8 x 4B > 1 KiB bucket at tp=2
        "model.layers.0.mlp.down_proj.weight": (H, 96),
        "model.layers.0.input_layernorm.weight": (H,),
    }
    full = {n: torch.randn(s).to(torch.float32) for n, s in shapes.items()}
    entries_by_src = {n: [ParamEntry(n, tuple(s), torch.float32, n, 0)] for n, s in shapes.items()}
    bucket = 1024
    gens = [_tp_slices(shapes, tp, r, nkv, hd) for r in range(tp)] * 2  # two TP replicas (DP=2)
    assert len({local_shapes_key(s) for s in gens}) == 1
    assert len({slices_digest(s) for s in gens}) == tp
    boxes0 = normalize_slices(gens[0])
    builder = LayoutBuilder(bucket, 256)
    for n, (e,) in entries_by_src.items():
        builder.add(local_entry(e, boxes0[n][1] if n in boxes0 else None))
    layout = builder.build()
    assert any(len(v) > 1 for v in layout.slots_by_name.values()) or tp == 4

    recv = [[bytearray(bucket) for _ in range(layout.num_buckets)] for _ in gens]
    hit = [[bytearray(bucket) for _ in range(layout.num_buckets)] for _ in gens]
    for t in range(trainers):
        ranges, local_u8 = {}, {}
        for n, x in full.items():
            row_len = x.shape[-1] if (row_aligned and x.dim() == 2) else None
            f0, f1 = _flat_ranges(x.numel(), trainers, row_len)[t]
            ranges[n] = (f0 * 4, (f1 - f0) * 4) if f1 > f0 else None
            local_u8[n] = x.reshape(-1)[f0:f1].numpy().tobytes()
        plans = [plan_rank_local_writes(entries_by_src, ranges, layout, normalize_slices(s)) for s in gens]
        merged = merge_box_plans(plans)
        # dedupe: no two merged pieces share a key; never more pieces than the per-gen sum
        keys = [w.key for ws in merged.values() for w in ws]
        assert len(keys) == len(set(keys))
        assert len(keys) <= sum(len(ws) for p in plans for ws in p.values())
        packed = {}
        for src, ws in merged.items():
            for w in ws:
                packed[w.key] = _gather(local_u8[src], w.segments)
                assert len(packed[w.key]) == w.length
        for g, plan in enumerate(plans):
            for ws in plan.values():
                for w in ws:
                    data = packed[w.key]  # every table entry is served from the shared pieces
                    b = recv[g][w.bucket_id]
                    assert not any(hit[g][w.bucket_id][w.dst_offset : w.dst_offset + w.length]), "overlap"
                    hit[g][w.bucket_id][w.dst_offset : w.dst_offset + w.length] = b"\x01" * w.length
                    b[w.dst_offset : w.dst_offset + w.length] = data
        if tp == 4 and trainers == 1:
            k_pieces = merged["model.layers.0.self_attn.k_proj.weight"]
            # 4 TP ranks x 2 replicas share nkv=2 distinct k boxes
            assert len({w.box_key for w in k_pieces}) == nkv
    for g, sl in enumerate(gens):
        for n, x in full.items():
            spec = sl.get(n)
            ref = x[tuple(slice(s, s + c) for s, c in spec["box"])] if spec else x
            got = _reassemble(layout, recv[g], n)
            assert got.shape == ref.shape and torch.equal(got, ref.contiguous()), (g, n)


def test_slices_validation_and_rolloutlayout_roundtrip():
    with pytest.raises(ValueError):
        normalize_slices({"w": {"full_shape": [4, 4], "box": [[2, 3], [0, 4]]}})  # out of bounds
    with pytest.raises(ValueError):
        normalize_slices({"w": {"full_shape": [4, 4], "box": [[0, 2]]}})  # wrong ndim
    sl = {"w": {"full_shape": [4, 4], "box": [[0, 2], [0, 4]]}}
    rl = RolloutLayout(
        tp_size=2, wire_formats=["named_tensors", "rank_local_named_tensors"], slices=sl, slice_digest=slices_digest(sl)
    )
    assert RolloutLayout.from_dict(rl.to_dict()) == rl
    old = RolloutLayout.from_dict({"tp_size": 1, "tp_rank": 0, "local_experts": {}})  # no rank-local fields
    assert old.wire_formats == ["named_tensors"] and old.slices is None and old.slice_digest == ""
    with pytest.raises(ValueError):
        plan_rank_local_writes(
            {"w": [ParamEntry("w", (4, 6), BF16, "w", 0)]}, {"w": (0, 48)}, {"w": []}, normalize_slices(sl)
        )


# ---------------------------------------------------------------- retry quiescence & guards


class _FakeNixlAgent:
    """Minimal ``NixlAgent`` stand-in backed by an inproc ZMQ PULL socket for ``prepare`` tests."""

    def __init__(self, name: str):
        import asyncio
        from collections import defaultdict, deque

        import zmq
        import zmq.asyncio

        self.agent_name = name
        self._ctx = zmq.asyncio.Context()
        self.socket = self._ctx.socket(zmq.PULL)
        self.socket.bind(f"inproc://snx-test-{name}-{id(self)}")
        self.messages: dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        self._notifs: dict[str, deque[bytes]] = defaultdict(deque)

    def close(self):
        self.socket.close(0)
        self._ctx.term()

    def get_agent_metadata(self):
        mod = _load_sharded_nixl_engine_mod()
        return mod.NixlAgentMetadata(self.agent_name, b"meta", "127.0.0.1", 0)

    def add_remote_agent(self, meta):
        return meta.agent_name

    def remove_remote_agent(self, name):
        pass

    def get_new_notifs(self):
        out = {k: list(v) for k, v in self._notifs.items() if v}
        self._notifs.clear()
        return out

    async def read_message(self, peer: str):
        return await self.messages[peer].get()


def _cpu_engine(name: str, **kwargs):
    """A ``device="cpu"`` engine on a ``_FakeNixlAgent``, with its staging buffers set so ``prepare``
    allocates none (pinned host memory needs a GPU driver)."""
    eng = _load_sharded_nixl_engine_mod().ShardedNixlCheckpointEngine(bucket_size=1024, device="cpu", **kwargs)
    eng.agent = _FakeNixlAgent(name)
    eng.bufs = [torch.zeros(1024, dtype=torch.uint8) for _ in range(eng.num_buffers)]
    eng.buf_descs = [(buf.data_ptr(), 1024, 0) for buf in eng.bufs]
    return eng


def test_quiesce_waits_for_proc_transfers_before_reuse():
    """``prepare`` on retry must wait for active ``PROC`` WRITEs to finish before releasing handles
    or draining notifications, so staging buffers are not reused under an in-flight DMA."""
    mod = _load_sharded_nixl_engine_mod()
    eng = _cpu_engine("trainer0", notif_timeout_s=1.0)
    agent = eng.agent
    try:
        polls = 0
        released = []
        handle = object()

        def check_xfer_state(h):
            nonlocal polls
            assert h is handle
            polls += 1
            if polls < 3:
                return "PROC"
            # Emulate UCX posting the trailing notification as the WRITE completes.
            agent._notifs["gen0"].append(b"ACK" + mod._ACK_HDR.pack(1, 0))
            return "DONE"

        def release_xfer_handle(h):
            assert h is handle and polls >= 3, "released while the WRITE was still running"
            released.append(h)

        agent.check_xfer_state = check_xfer_state
        agent.release_xfer_handle = release_xfer_handle
        eng._inflight[0] = [handle]

        meta = eng.prepare()

        assert polls == 3 and released == [handle] and eng._inflight == {}
        assert agent.get_new_notifs() == {}
        assert meta.attempt == 1
    finally:
        agent.close()


def test_quiesce_keeps_running_transfer_and_blocks_retry():
    """A WRITE still running at the deadline is never released (UCX can only try to cancel it, and it
    may still land): ``prepare`` raises and keeps the handle until a later ``prepare`` finds it stopped."""
    eng = _cpu_engine("trainer0", notif_timeout_s=0.01)
    try:
        handle, state, released = object(), ["PROC"], []
        eng.agent.check_xfer_state = lambda h: state[0]
        eng.agent.release_xfer_handle = released.append
        eng._inflight[0] = [handle]

        for _ in range(2):
            with pytest.raises(RuntimeError, match="still running"):
                eng.prepare()
            assert eng._inflight == {0: [handle]} and released == []

        state[0] = "DONE"
        eng.prepare()
        assert eng._inflight == {} and released == [handle]
    finally:
        eng.agent.close()


def test_retry_same_global_steps_uses_distinct_sync_id_and_ignores_late_notifs():
    """Retrying the same ``global_steps`` advances the attempt half of the 64-bit sync ID so delayed
    notifications and ZMQ messages from the failed attempt cannot match the retry, while a step
    mismatch within the same attempt still reaches the callers' sync-id checks."""
    import asyncio

    mod = _load_sharded_nixl_engine_mod()
    cls = mod.ShardedNixlCheckpointEngine
    t_eng = _cpu_engine("t0", is_master=True, notif_timeout_s=1.0)
    g_eng = _cpu_engine("g0", notif_timeout_s=1.0)
    g_agent = g_eng.agent

    def _connect():
        metas = [t_eng.prepare(), g_eng.prepare()]
        tkw, gkw = cls.build_topology(1, 1, metas)
        t_eng.init_process_group(**{k: v[0] for k, v in tkw.items()})
        g_eng.init_process_group(**{k: v[0] for k, v in gkw.items()})

    try:
        _connect()
        s1_t, s1_g = t_eng._next_sync_id(global_steps=1), g_eng._next_sync_id(global_steps=1)
        assert s1_t == s1_g

        # Retry the same global_steps=1 after a failure.
        _connect()
        s2_t, s2_g = t_eng._next_sync_id(global_steps=1), g_eng._next_sync_id(global_steps=1)
        assert s2_t == s2_g and s2_t != s1_t and (s2_t >> 32) > (s1_t >> 32)

        # Attempt-1 traffic arriving after prepare() is dropped; attempt-2 traffic for another step is
        # kept (parked or returned), so a version mismatch still fails fast instead of timing out.
        stale_snx = b"SNX" + mod._SNX_HDR.pack(s1_t, 0, 0, 0)
        other_step_snx = b"SNX" + mod._SNX_HDR.pack(s2_t + 1, 0, 0, 0)
        fresh_snx = b"SNX" + mod._SNX_HDR.pack(s2_t, 0, 0, 0)
        g_agent._notifs["t0"].extend([stale_snx, other_step_snx, fresh_snx])
        assert asyncio.run(g_eng._wait_notif("t0", b"SNX", s2_g, 0, 1.0)) == fresh_snx
        assert list(g_eng._notifs.values()) == [other_step_snx]

        g_agent.messages["t0"].put_nowait({"type": "hashes", "sync": s1_t, "k": 0, "rank": 0, "hashes": []})
        g_agent.messages["t0"].put_nowait({"type": "sync", "sync": s2_t + 1})
        assert asyncio.run(g_eng._read("t0", "sync", 1.0))["sync"] == s2_t + 1
    finally:
        t_eng.agent.close()
        g_agent.close()


def test_sharded_nixl_rejects_quantized_rollout_and_qat():
    """``sharded_nixl``'s shard export bypasses the trainer-side rollout weight quantizer, so a quantized
    checkpoint (also one under ``text_config``) or QAT must fail at startup; BF16 and other backends pass."""
    from types import SimpleNamespace

    from verl.utils.qat import dispatch_rollout_weight_quantizer

    def _rollout(backend="sharded_nixl"):
        return SimpleNamespace(
            name="vllm",
            checkpoint_engine=SimpleNamespace(backend=backend),
            quantization=SimpleNamespace(linear=None, moe=None, ignore=[], config_file=None),
        )

    def _model(quantization_config=None, text_config=None, model_type="deepseek_v4"):
        hf_config = SimpleNamespace(
            model_type=model_type, quantization_config=quantization_config, text_config=text_config
        )
        return SimpleNamespace(local_path="/model", hf_config=hf_config)

    fp8 = {"quant_method": "fp8"}
    assert dispatch_rollout_weight_quantizer(model_config=_model(), rollout_config=_rollout()) is None
    # DeepSeek-V4's native FP8 still passes through for the other backends
    assert dispatch_rollout_weight_quantizer(model_config=_model(fp8), rollout_config=_rollout("nixl")) is not None
    vlm = _model(text_config=SimpleNamespace(model_type="qwen3", quantization_config=fp8), model_type="qwen3_vl")
    for model, qat in ((_model(fp8), None), (vlm, None), (_model(fp8), SimpleNamespace(enable=True, mode="fp8"))):
        with pytest.raises(NotImplementedError, match="sharded_nixl"):
            dispatch_rollout_weight_quantizer(model_config=model, rollout_config=_rollout(), qat_config=qat)
