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
"""CPU tests for the block-placement machinery behind FSDP+EP support.

``BlockPlacement`` describes a rank's local shard as one hyper-rectangular block
of the full tensor (e.g. automodel's grouped experts on the ``(ep, ep_shard)``
mesh with ``(Shard(0), Shard(1))``). These tests pin down the two pure-math
pieces the engine relies on -- the local->global flat index translation, and the
NaN-sentinel full-tensor rebuild feeding a qwen3-moe-style ``to_hf`` permutation
(per-expert slice + gate/up split + transpose) -- without torch.distributed.
"""

import itertools

import pytest
import torch

from verl.workers.engine.spec import BlockPlacement, translate_flat_indices


def _blocks_for_grid(full_shape, grid):
    """Split ``full_shape`` into an even grid of blocks; yield BlockPlacements.

    ``grid`` maps tensor dim -> number of splits (even division assumed here;
    uneven splits are covered by the distributed test via DTensor itself).
    """
    per_dim = []
    for d, size in enumerate(full_shape):
        n = grid.get(d, 1)
        assert size % n == 0
        step = size // n
        per_dim.append([(o, step) for o in range(0, size, step)])
    for combo in itertools.product(*per_dim):
        goff = tuple(o for o, _ in combo)
        lshape = tuple(sz for _, sz in combo)
        yield BlockPlacement(lshape, goff, tuple(full_shape))


def test_translate_int_fast_path():
    lidx = torch.tensor([0, 3, 17], dtype=torch.int64)
    assert torch.equal(translate_flat_indices(lidx, 0), lidx)
    assert torch.equal(translate_flat_indices(lidx, 100), lidx + 100)


@pytest.mark.parametrize(
    "full_shape,grid",
    [
        ((8, 4, 6), {0: 4, 1: 2}),  # automodel experts: ep=4 x ep_shard=2 (Shard(0), Shard(1))
        ((16, 8), {1: 4}),  # plain Shard(1)
        ((6, 5, 7), {2: 7}),  # shard the last dim, odd sizes
        ((12,), {0: 3}),  # 1-D block degenerates to the flat case
    ],
)
def test_translate_matches_full_coordinates(full_shape, grid):
    torch.manual_seed(0)
    full = torch.randn(full_shape)
    flat = full.reshape(-1)
    for place in _blocks_for_grid(full_shape, grid):
        slices = tuple(slice(o, o + sz) for o, sz in zip(place.global_offset, place.local_shape, strict=False))
        local = full[slices].contiguous().reshape(-1)
        lidx = torch.arange(local.numel(), dtype=torch.int64)
        gidx = translate_flat_indices(lidx, place)
        assert torch.equal(flat[gidx], local), f"block {place} mistranslated"


def test_translate_sparse_subset():
    full_shape = (8, 4, 6)
    torch.manual_seed(1)
    full = torch.randn(full_shape)
    place = BlockPlacement((2, 2, 6), (4, 2, 0), full_shape)
    slices = tuple(slice(o, o + sz) for o, sz in zip(place.global_offset, place.local_shape, strict=False))
    local = full[slices].contiguous().reshape(-1)
    lidx = torch.tensor([0, 5, 11, 23], dtype=torch.int64)
    gidx = translate_flat_indices(lidx, place)
    assert torch.equal(full.reshape(-1)[gidx], local[lidx])


def _qwen3_style_to_hf(full_shape, inter_dim):
    """Mimic the automodel expert closure: fused [n, dim, 2I] -> per-expert
    gate/up [I, dim] via slice + split + transpose -- a pure permutation."""

    def to_hf(shards):
        (full_flat,) = shards
        fused = full_flat.view(full_shape)
        out = []
        for e in range(full_shape[0]):
            w = fused[e]
            out.append((f"experts.{e}.gate_proj.weight", w[:, :inter_dim].transpose(0, 1)))
            out.append((f"experts.{e}.up_proj.weight", w[:, inter_dim:].transpose(0, 1)))
        return out

    return to_hf


def test_block_nan_rebuild_matches_full_convert_then_diff():
    """Replicate the engine's block converter profile end to end on CPU:
    per-block local diff -> translate-free local NaN rebuild -> slice-assign into
    a full NaN tensor -> to_hf -> isnan scan == diff of the converted tensors."""
    torch.manual_seed(2)
    n_experts, dim, inter = 8, 4, 3
    full_shape = (n_experts, dim, 2 * inter)
    to_hf = _qwen3_style_to_hf(full_shape, inter)

    old = torch.randn(full_shape, dtype=torch.bfloat16)
    new = old.clone()
    new[0, 1, 2] += 1.0
    new[3, 0, 5] -= 0.5
    new[7, 3, 0] += 0.25

    # ep=4 x ep_shard=2 grid of blocks, like the (Shard(0), Shard(1)) expert mesh
    full_nan = torch.full(full_shape, float("nan"), dtype=old.dtype)
    for place in _blocks_for_grid(full_shape, {0: 4, 1: 2}):
        slices = tuple(slice(o, o + sz) for o, sz in zip(place.global_offset, place.local_shape, strict=False))
        lo = old[slices].contiguous().reshape(-1)
        ln = new[slices].contiguous().reshape(-1)
        changed = (lo.view(torch.uint8).view(lo.numel(), -1) != ln.view(torch.uint8).view(ln.numel(), -1)).any(dim=-1)
        lidx = changed.nonzero(as_tuple=False).view(-1)
        buf = torch.full((lo.numel(),), float("nan"), dtype=old.dtype)
        buf[lidx] = ln[lidx]
        full_nan[slices] = buf.view(place.local_shape)

    seen = 0
    ref_new = dict(to_hf([new.reshape(-1)]))
    ref_old = dict(to_hf([old.reshape(-1)]))
    for hf_name, hf_tensor in to_hf([full_nan.reshape(-1)]):
        fl = hf_tensor.reshape(-1)
        pos = (~torch.isnan(fl)).nonzero(as_tuple=False).view(-1)
        rn, ro = ref_new[hf_name].reshape(-1), ref_old[hf_name].reshape(-1)
        ref_changed = (
            (rn.view(torch.uint8).view(rn.numel(), -1) != ro.view(torch.uint8).view(ro.numel(), -1))
            .any(dim=-1)
            .nonzero(as_tuple=False)
            .view(-1)
        )
        assert torch.equal(pos, ref_changed), f"{hf_name}: positions diverge"
        assert torch.equal(fl[pos], rn[pos]), f"{hf_name}: values diverge"
        seen += int(pos.numel())
    assert seen == 3


def _veomni_style_to_hf(full_shape):
    """Mimic the veomni expert closure: fused [n, 2I, H] gate_up -> per-expert
    gate/up via chunk(dim=1) + per-expert slice + rename -- pure permutation,
    mirroring verl.workers.engine.veomni.utils.default_moe_param_handler."""

    def to_hf(shards):
        full_flat = shards[0] if len(shards) == 1 else torch.cat([sh.reshape(-1) for sh in shards])
        fused = full_flat.view(full_shape)
        gate, up = fused.chunk(2, dim=1)
        out = []
        for i in range(full_shape[0]):
            out.append((f"mlp.experts.{i}.gate_proj.weight", gate[i]))
            out.append((f"mlp.experts.{i}.up_proj.weight", up[i]))
        return out

    return to_hf


def test_explicit_place_block_rebuild_veomni_style():
    """veomni experts: manual ep split on dim0 (outside DTensor) + Shard(1)-style
    block on dim1. The exporter synthesizes BlockPlacement in a virtual full
    tensor; verify the engine-style rebuild -> to_hf -> isnan scan equals the
    reference per-expert delta."""
    torch.manual_seed(3)
    n_global, two_i, h = 8, 6, 4
    ep, fsdp = 4, 2
    n_local = n_global // ep
    full_shape = (n_global, two_i, h)
    to_hf = _veomni_style_to_hf(full_shape)

    old = torch.randn(full_shape, dtype=torch.bfloat16)
    new = old.clone()
    new[1, 2, 3] += 1.0
    new[5, 0, 0] -= 0.5

    full_nan = torch.full(full_shape, float("nan"), dtype=old.dtype)
    for ep_rank in range(ep):
        for f_rank in range(fsdp):
            # manual ep split on dim0, fsdp Shard(1) on the local block's dim1
            d1 = two_i // fsdp
            place = BlockPlacement(
                (n_local, d1, h),
                (ep_rank * n_local, f_rank * d1, 0),
                full_shape,
            )
            slices = tuple(slice(o, o + sz) for o, sz in zip(place.global_offset, place.local_shape, strict=False))
            lo = old[slices].contiguous().reshape(-1)
            ln = new[slices].contiguous().reshape(-1)
            changed = (lo.view(torch.uint8).view(lo.numel(), -1) != ln.view(torch.uint8).view(ln.numel(), -1)).any(
                dim=-1
            )
            lidx = changed.nonzero(as_tuple=False).view(-1)
            buf = torch.full((lo.numel(),), float("nan"), dtype=old.dtype)
            buf[lidx] = ln[lidx]
            full_nan[slices] = buf.view(place.local_shape)

    seen = 0
    ref_new = dict(to_hf([new.reshape(-1)]))
    ref_old = dict(to_hf([old.reshape(-1)]))
    for hf_name, hf_tensor in to_hf([full_nan.reshape(-1)]):
        fl = hf_tensor.reshape(-1)
        pos = (~torch.isnan(fl)).nonzero(as_tuple=False).view(-1)
        rn, ro = ref_new[hf_name].reshape(-1), ref_old[hf_name].reshape(-1)
        ref_changed = (
            (rn.view(torch.uint8).view(rn.numel(), -1) != ro.view(torch.uint8).view(ro.numel(), -1))
            .any(dim=-1)
            .nonzero(as_tuple=False)
            .view(-1)
        )
        assert torch.equal(pos, ref_changed), f"{hf_name}: positions diverge"
        assert torch.equal(fl[pos], rn[pos]), f"{hf_name}: values diverge"
        seen += int(pos.numel())
    assert seen == 2


def test_explicit_place_passthrough():
    """ShardSpec.place / gather_group short-circuit derive_placement."""
    from verl.workers.engine.spec import ShardSpec, derive_placement

    block = BlockPlacement((2, 3), (4, 0), (8, 3))
    sentinel = object()
    spec = ShardSpec(full_shape=(8, 3), place=block, gather_group=sentinel)
    place, contributes, group = derive_placement(spec)
    assert place is block and contributes and group is sentinel
