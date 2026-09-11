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
"""Megatron-side delta export machinery, built on Megatron-Bridge param mappings.

The delta engine consumes final HF-coordinate entries; everything mcore-specific
lives here: enumerating parameters through :meth:`AutoBridge.get_conversion_tasks`,
routing each parameter's entries to its wire merge group, and probing the
bridge's own ``megatron_to_hf`` converters with NaN sentinels to translate a
shard-local delta into HF coordinates.

The probe never runs a collective, yet executes the mapping's REAL parallel
code paths: a "group" carries two separable meanings -- its SIZE/RANK (which
value math like Mamba's ``local_dim = global // tp_size`` consumes) and its
COMMUNICATION. The probe copies keep the first faithful and replace only the
second: groups become :class:`_ProbeGroup` (real size/rank, any actual use for
communication raises), and the bridge's comm helpers are stubbed with local
synthesis -- ``gather_from_tp_ranks`` returns this rank's shard at its true
rank index with NaN placeholders for every other rank (their contributions are
exported by those ranks' own probes). Feeding ``megatron_to_hf`` the LOCAL
shard as a NaN buffer with the rank's own delta scattered in yields HF tensors
whose non-NaN survivors are exactly this rank's contributions in final HF
coordinates.

Scope (asserted in the exporter): TP + EP + PP/VPP, no LoRA. Under PP the
bridge's conversion tasks already enumerate the GLOBAL parameter directory
(identical order on every rank, tied embeddings deduped, placeholders for
other stages); non-owner ranks ship zero-count lockstep rows and every
param merges over the WORLD group, so the wire master needs no relay.

Safety boundary: communication must stay confined to the four stubbed helpers
(``gather_from_tp_ranks`` / ``gather_from_ep_ranks[_scale]`` / the PP
broadcasts, whose ``pp_size == 1`` fast path the probe keeps) -- anything else
touching a probe group raises instead of skewing silently. The remaining
assumption is that transforms REARRANGE elements rather than arithmetically
BLEND chunks (blending would eat the NaN sentinels); the TP>1 differential
test (real ``megatron_to_hf`` vs probe assembly, bitwise) is the regression
oracle for both assumptions on every Megatron-Bridge upgrade.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch

from verl.workers.engine.spec import ShardSpec

logger = logging.getLogger(__name__)


class _ProbeGroup:
    """Size/rank-faithful stand-in for a process group on probe copies: value
    math inside ``megatron_to_hf`` (e.g. Mamba's per-rank de-interleave dims)
    sees the REAL parallel sizes, while any attempt to actually communicate
    through the group fails loud (the probe stubs the bridge's comm helpers;
    anything else reaching a group is an unstubbed communication pattern)."""

    def __init__(self, size: int, rank: int):
        self._size = int(size)
        self._rank = int(rank)

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank

    def __getattr__(self, name):
        raise RuntimeError(
            f"probe process group asked for {name!r}: this mapping communicates outside "
            "the stubbed helpers (gather_from_tp_ranks / gather_from_ep_ranks[_scale] / "
            "pp broadcasts) -- extend make_probe's comm stubs before trusting its export"
        )


_NAN_POOL: dict = {}


def _nan_block(shape, dtype, device) -> torch.Tensor:
    """Read-only all-NaN placeholder for another rank's gather chunk, pooled by
    (shape, dtype, device): the gather stub hands these out on every probe call
    for every param, and they are only ever READ (the transforms are functional),
    so one block per distinct shape serves the whole model for the run's
    lifetime instead of a fresh cudaMalloc+fill per param per sync."""
    key = (tuple(shape), dtype, str(device))
    t = _NAN_POOL.get(key)
    if t is None:
        t = torch.full(tuple(shape), float("nan"), dtype=dtype, device=device)
        _NAN_POOL[key] = t
    return t


def _warm_lazy_mappings(mapping, module) -> None:
    """Force AutoMapping's lazy concrete delegate into existence: AutoMapping
    resolves its Column/Row/Replicated delegate on first use, snapshotting
    whatever process groups exist at that moment -- if that first use happened
    inside the probe, the delegate would be born with the REAL groups and
    gather for real (the qkv-bias double-size bug). Self-only on purpose:
    ``_inject`` warms every child right before copying it, so each node of the
    mapping tree is warmed exactly once."""
    if hasattr(mapping, "_detect_parallelism_type") and getattr(mapping, "_mapping", None) is None:
        try:
            t = mapping._detect_parallelism_type(module)
            mapping._mapping = mapping._get_or_create_mapping(t)
            mapping._detected_type = t
        except Exception as e:  # pragma: no cover - defensive; probe falls back to real groups
            logger.warning("could not warm lazy mapping %s: %s", type(mapping).__name__, e)


def make_probe(mapping, module):
    """Copy a Megatron-Bridge param mapping tree and turn ``megatron_to_hf``
    into a communication-free LOCAL transform that still runs the REAL
    (tp_size > 1) code paths: every copy's groups become size-faithful
    :class:`_ProbeGroup` stand-ins and the bridge's comm helpers are stubbed
    with local synthesis. The copy is recursive (composite mappings delegate to
    inner mappings -- QKVMapping._tp_mapping is an AutoMapping which itself
    delegates to a lazily-created concrete mapping; none of these receive the
    outer stubbing on their own)."""
    from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping
    from megatron.core.utils import get_pg_rank, get_pg_size

    _warm_lazy_mappings(mapping, module)

    def _stub(c):
        # size-faithful groups, sizes/ranks read from the ORIGINAL (real) groups
        # BEFORE replacement. The probe only ever runs on the stage that OWNS
        # the param, where a pp broadcast is the identity -- a (1, 0) group
        # keeps the pp helpers on that fast path regardless of the real pp size.
        c.pp_group = _ProbeGroup(1, 0)
        for attr in ("ep_group", "_tp_group", "_etp_group"):
            g = getattr(c, attr, None)
            setattr(c, attr, _ProbeGroup(get_pg_size(g), get_pg_rank(g)))

        def _gather_tp(tensor, _c=c):
            # what a real all_gather over the tp group would produce, minus the
            # other ranks' data: our shard rides at its true rank index, every
            # other chunk is a pooled read-only NaN block (those ranks export
            # their own contributions; transforms never write into their inputs,
            # which the TP>1 bitwise differential revalidates).
            nan = _nan_block(tensor.shape, tensor.dtype, tensor.device)
            out = [nan] * _c.tp_size
            out[_c.tp_rank] = tensor
            return out

        c.gather_from_tp_ranks = _gather_tp
        # each rank's probe emits only its own local experts, under the hf name
        # bound with the GLOBAL expert id at task construction; the engine
        # merges entries over the etp x ep group, so the ep fan-out reduces to
        # the bound name.
        c.gather_from_ep_ranks = lambda w, mod, name: {str(name): w}
        # mirrors the real helper's tail (unsqueeze(0) ... squeeze().unsqueeze(-1))
        c.gather_from_ep_ranks_scale = lambda w, mod, name: {str(name): w.unsqueeze(0).squeeze().unsqueeze(-1)}
        return c

    def _inject(m):
        c = _stub(copy.copy(m))
        for attr, value in list(vars(c).items()):
            if isinstance(value, MegatronParamMapping):
                # warm BEFORE copying the child: the lazy delegate must exist so
                # the child copy's vars() include it and the recursion reaches it.
                _warm_lazy_mappings(value, module)
                setattr(c, attr, _inject(value))
        return c

    return _inject(mapping)


@dataclass
class McoreParamExport:
    """One mcore parameter's export record: geometry + probe + module handle.
    ``param is None`` marks a parameter owned by another pipeline stage: the
    rank ships a zero-count lockstep row for it (probe/module unused)."""

    megatron_name: str
    param: Optional[torch.Tensor]
    spec: ShardSpec
    probe: Any  # comm-stubbed mapping copy (local megatron_to_hf evaluator)
    module: Any  # module handle megatron_to_hf reads config from
    # GLOBAL slot table for this directory row (set by the index-build
    # exchange): identical on every rank of the row's merge group.
    slots: Optional[list] = None


def build_export_index(bridge, megatron_model, slot_cache: dict | None = None) -> list[McoreParamExport]:
    """Enumerate every mcore parameter through the bridge's conversion tasks
    and precompute its probe + wire routing.

    No shard geometry is hand-computed here: the comm-stubbed probe emits final
    HF coordinates straight from the local shard, so the engine only needs to
    know WHICH group's entries to merge per parameter (the spec's
    ``gather_group``; ``full_shape`` is the nominal local shape) and which
    ranks contribute. The index is built once (parameter sets are static) and
    reused by both the shard export and the delta entry hook.

    Lockstep: ``get_conversion_tasks`` already enumerates the GLOBAL parameter
    list in an order identical on every rank (the bridge allgathers and sorts
    names across pp ranks, dedupes tied embeddings, and leaves
    ``param_weight=None`` placeholders for parameters owned by other pipeline
    stages). Under PP>1 those placeholders become zero-count lockstep rows and
    every param's entries merge over the WORLD group (owner-stage ranks
    contribute, dp/cp replicas and other stages stay empty); their slot tables
    are pre-seeded once via :func:`_preseed_slot_tables`. Under PP=1 the
    routing is unchanged (tp / etp x ep subgroups, replicated rank-0 direct).
    """
    from megatron.core import parallel_state as mpu

    pp_world = mpu.get_pipeline_model_parallel_world_size()
    tp_group = mpu.get_tensor_model_parallel_group()
    tp_world = torch.distributed.get_world_size(group=tp_group)
    ep_size = mpu.get_expert_model_parallel_world_size()
    world = torch.distributed.group.WORLD

    tasks = bridge.get_conversion_tasks(megatron_model)
    index: list[McoreParamExport] = []
    for task in tasks:
        mapping = task.mapping
        param = task.param_weight
        name = task.global_param_name
        if param is None:
            if pp_world == 1:
                # not a pp placeholder (e.g. a skipped mapping); nothing to ship
                continue
            # owned by another pipeline stage: zero-count lockstep row -- the
            # rank walks the same global directory and contributes nothing.
            index.append(
                McoreParamExport(
                    megatron_name=name,
                    param=None,
                    spec=ShardSpec(full_shape=(0,), place=0, gather_group=world, contributes=False),
                    probe=None,
                    module=None,
                )
            )
            continue
        module = task.megatron_module
        local_shape = tuple(int(x) for x in param.shape)

        is_expert = mapping.is_expert and (ep_size > 1 or tp_world > 1)
        is_tp_sharded = getattr(param, "tensor_model_parallel", False) and tp_world > 1

        if pp_world > 1:
            # single WORLD merge group for every param: the wire master (global
            # rank 0) sits in every gather regardless of which stage owns the
            # param, so no relay hop is needed. Owner-stage ranks contribute
            # their shard pieces; dp/cp replicas dedupe via ``contributes``
            # (identical copies -- exactly one replica set ships).
            if is_expert:
                contributes = mpu.get_expert_data_parallel_rank() == 0
            elif is_tp_sharded:
                contributes = mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            else:
                contributes = (
                    mpu.get_tensor_model_parallel_rank() == 0
                    and mpu.get_data_parallel_rank(with_context_parallel=True) == 0
                )
            spec = ShardSpec(full_shape=local_shape, place=0, gather_group=world, contributes=contributes)
        elif is_expert:
            # every rank holding a piece of this expert set contributes; the
            # engine merges their probe entries over the joint etp x ep group.
            spec = ShardSpec(
                full_shape=local_shape,
                place=0,
                gather_group=mpu.get_expert_tensor_and_model_parallel_group(),
            )
        elif is_tp_sharded:
            spec = ShardSpec(full_shape=local_shape, place=0, gather_group=tp_group)
        else:
            # replicated: engine's pg=None path (rank 0 consumes its own entry
            # directly, replicas stay in lockstep via zero counts).
            spec = ShardSpec(full_shape=local_shape)

        index.append(
            McoreParamExport(
                megatron_name=name,
                param=param,
                spec=spec,
                probe=make_probe(mapping, module),
                module=module,
            )
        )

    _exchange_slot_tables(index, slot_cache)
    # a row with an empty union has no owner on ANY rank: the bridge skipped
    # it everywhere (no mapping / missing HF key), so it is outside the
    # bridge's own export scope -- exactly what PP=1 does by skipping
    # param_weight=None tasks. Drop it (all ranks agree: the exchange result
    # is identical) instead of shipping unsized rows.
    unowned = [rec.megatron_name for rec in index if rec.slots is None]
    if unowned:
        logger.warning(
            "delta export drops %d row(s) the bridge skipped on every rank "
            "(no mapping or missing HF key; out of export scope): %s",
            len(unowned),
            unowned[:8],
        )
        index = [rec for rec in index if rec.slots is not None]
    return index


def _exchange_slot_tables(index: list[McoreParamExport], slot_cache: dict) -> None:
    """One-time GLOBAL slot-table exchange, run at every world size, keyed by
    DIRECTORY ROW (not by param name).

    The engine's batched gather merges entries BY SLOT POSITION within each
    directory row and rank 0 names the merged pieces from ITS OWN list, so
    the per-row list must be identical on every rank of the row's merge
    group. Name-keyed merging is not enough on mcore: expert param NAMES
    embed the expert ids (``...experts.linear_fc1.weight0`` on ep rank 0 vs
    ``weight1`` on ep rank 1), so the same row carries different names per
    rank while still gathering together. Each rank therefore probes a zero
    delta through its OWNED rows to reveal its local ``(hf_name, hf_shape)``
    list, one ``all_gather_object`` over WORLD exchanges the ordered row
    lists, and every row's table becomes the rank-order first-seen union --
    identical everywhere by construction. Dense/TP rows (same list on every
    rank) dedup to themselves; expert rows concatenate the per-ep-rank
    lists; PP placeholder rows inherit the owners' union."""
    local_rows: list = []
    for rec in index:
        if rec.param is None:
            local_rows.append(None)
            continue
        if rec.megatron_name not in slot_cache:
            empty_idx = torch.empty(0, dtype=torch.int64, device=rec.param.device)
            empty_val = torch.empty(0, dtype=torch.bfloat16, device=rec.param.device)
            mcore_hf_delta_entry(rec, 0, empty_idx, empty_val, slot_cache)
        local_rows.append(slot_cache[rec.megatron_name])

    world = torch.distributed.get_world_size()
    gathered: list = [None] * world
    torch.distributed.all_gather_object(gathered, local_rows)
    n_rows = len(local_rows)
    assert all(len(rows) == n_rows for rows in gathered), (
        f"directory row counts diverge across ranks: {[len(r) for r in gathered]} -- "
        "the bridge's global enumeration is expected to be structurally parallel"
    )
    for k, rec in enumerate(index):
        union: dict = {}
        for rows in gathered:  # rank order -> identical result on every rank
            row = rows[k]
            if row is None:
                continue
            for slot in row:
                union[(slot[0], tuple(slot[1]))] = None  # ordered-set semantics
        rec.slots = [(n, tuple(shape)) for (n, shape) in union.keys()] if union else None


def mcore_hf_delta_entry(rec: McoreParamExport, _place, lidx: torch.Tensor, lval: torch.Tensor, slot_cache: dict):
    """Probe one mcore param's shard-local delta into its final HF-coordinate
    entry ``(slots, dtype_str, counts, hf_idx, hf_val)``.

    Scatters the delta into a NaN buffer of the LOCAL shard shape (exactly what
    the real ``megatron_to_hf`` receives), runs the comm-stubbed probe -- real
    group sizes, gathers synthesized locally, so the mapping executes its real
    TP>1 code paths -- and extracts each output slot's surviving positions.
    The slot list is cached after the first call (the converter's output names
    are deterministic, so every rank's cache agrees and the batched gather
    stays aligned)."""
    dtype_str = str(lval.dtype).replace("torch.", "")
    assert lval.numel() == 0 or lval.is_floating_point(), (
        f"{rec.megatron_name}: NaN sentinels require a floating-point param, got {lval.dtype}"
    )

    cached = rec.slots if rec.slots is not None else slot_cache.get(rec.megatron_name)
    if rec.param is None:
        # owned by another pipeline stage: pure lockstep row. The slot table
        # was pre-seeded from the owner stage (fail loud if not -- an
        # unsized entry would silently misalign the batched gather).
        assert cached is not None, f"{rec.megatron_name}: slot table not pre-seeded for a non-owned PP param"
        assert lidx.numel() == 0, f"{rec.megatron_name}: delta reported for a param this rank does not own"
    if lidx.numel() == 0 and cached is not None:
        # empty delta: the slot table froze after the first probe, so the
        # zero-count lockstep entry needs no probe run at all -- skip the
        # buffer build, the transform and the full-output NaN scan.
        return (
            cached,
            dtype_str,
            torch.zeros(len(cached), dtype=torch.int64),
            torch.empty(0, dtype=torch.int32, device=lval.device),
            torch.empty(0, dtype=lval.dtype, device=lval.device),
        )

    buf = torch.full(tuple(rec.param.shape), float("nan"), dtype=lval.dtype, device=lval.device)
    if lidx.numel():
        buf.view(-1)[lidx] = lval

    outs = rec.probe.megatron_to_hf(buf, rec.module)

    key = rec.megatron_name
    slots = rec.slots if rec.slots is not None else slot_cache.get(key)
    if slots is None:
        # only reachable from the index-build exchange itself, which probes a
        # zero delta to reveal this rank's local list before the row unions
        # are installed on the recs; steady entries always find rec.slots.
        slots = [(n, tuple(int(x) for x in t.shape)) for n, t in outs.items()]
        slot_cache[key] = slots
    unknown = set(outs) - {n for n, _ in slots}
    assert not unknown, (
        f"{key}: probe emitted slots missing from the global table {sorted(unknown)[:4]} -- "
        "non-deterministic converter naming would misalign the batched gather"
    )
    counts = torch.zeros(len(slots), dtype=torch.int64)
    idx_pieces: list[torch.Tensor] = []
    val_pieces: list[torch.Tensor] = []
    for s_i, (sname, _sshape) in enumerate(slots):
        out = outs.get(sname)
        if out is None:
            continue  # another rank's slot (e.g. its ep-local experts): zero count here
        fl = out.reshape(-1)
        p_ = (~torch.isnan(fl)).nonzero(as_tuple=False).view(-1)
        if p_.numel():
            counts[s_i] = p_.numel()
            idx_pieces.append(p_.to(torch.int32))
            val_pieces.append(fl[p_])
    if idx_pieces:
        hf_idx = torch.cat(idx_pieces)
        hf_val = torch.cat(val_pieces)
    else:
        hf_idx = torch.empty(0, dtype=torch.int32, device=lval.device)
        hf_val = torch.empty(0, dtype=lval.dtype, device=lval.device)
    return slots, dtype_str, counts, hf_idx, hf_val


def _local_block_owner_mask(t: torch.Tensor, block: tuple[int, int], full_shape: tuple[int, int]) -> torch.Tensor:
    """Return the HF block cells containing at least one value owned here.

    The communication-free Megatron probe represents other TP/ETP ranks with
    NaNs in the final HF-coordinate tensor.  Ownership is therefore structural
    (non-NaN), not value based: an all-zero but locally owned block must still
    have an owner.
    """
    from verl.utils.kernel.fp8_kernel import ceil_div

    bm, bn = block
    m, n = full_shape
    assert t.dim() == 2 and tuple(t.shape) == full_shape, (
        f"probe output shape {tuple(t.shape)} does not match HF slot {full_shape}; owner positions would be ambiguous"
    )
    n_br, n_bc = ceil_div(m, bm), ceil_div(n, bn)
    valid = ~torch.isnan(t)
    pad_m, pad_n = n_br * bm - int(t.shape[0]), n_bc * bn - n
    if pad_m or pad_n:
        valid = torch.nn.functional.pad(valid, (0, pad_n, 0, pad_m), value=False)
    return valid.view(n_br, bm, n_bc, bn).any(dim=(1, 3))


def _block_owner_plan(
    local_masks: list[torch.Tensor],
    group,
    group_rank: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Build static ``(cross_positions, scale_positions)`` per HF slot.

    A full owner-count exchange happens once when the export plan is first
    used.  Steady syncs communicate amax values only at ``cross_positions``.
    Singleton blocks keep their amax and scale on their unique owner; a
    cross-rank block's reduced scale is emitted by group rank zero.
    """
    import torch.distributed as dist

    if not local_masks:
        return []
    if group is None:
        return [
            (
                torch.empty(0, dtype=torch.int32, device=mask.device),
                mask.reshape(-1).nonzero(as_tuple=False).view(-1).to(torch.int32),
            )
            for mask in local_masks
        ]
    sizes = [int(mask.numel()) for mask in local_masks]
    flat_local = torch.cat([mask.reshape(-1) for mask in local_masks])
    counts = flat_local.to(torch.int32)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=group)
    assert bool(torch.all(counts > 0)), "quantized HF block has no contributing trainer rank"

    out = []
    off = 0
    for mask, n in zip(local_masks, sizes, strict=True):
        count = counts[off : off + n]
        local = mask.reshape(-1)
        cross = count > 1
        scale_owner = (local & (count == 1)) | (cross & (group_rank == 0))
        out.append(
            (
                cross.nonzero(as_tuple=False).view(-1).to(torch.int32),
                scale_owner.nonzero(as_tuple=False).view(-1).to(torch.int32),
            )
        )
        off += n
    return out


def _reduce_cross_block_amax(
    grids: list[torch.Tensor],
    plan: list[tuple[torch.Tensor, torch.Tensor]],
    group,
) -> int:
    """MAX-reduce only block cells whose owner count is greater than one.

    Returns the number of FP32 cells placed on the collective data plane.  A
    zero return performs no collective at all, which is the common PP/EP-only
    case because those parallelisms own whole HF tensors/experts.
    """
    import torch.distributed as dist

    pieces = [g.reshape(-1)[cross.long()] for g, (cross, _scale) in zip(grids, plan, strict=True)]
    n_cross = sum(int(piece.numel()) for piece in pieces)
    if not n_cross:
        return 0
    assert group is not None, "cross-rank HF blocks require a merge process group"
    flat = torch.cat(pieces)
    dist.all_reduce(flat, op=dist.ReduceOp.MAX, group=group)
    off = 0
    for grid, (cross, _scale) in zip(grids, plan, strict=True):
        n = int(cross.numel())
        if n:
            grid.reshape(-1)[cross.long()] = flat[off : off + n]
        off += n
    return n_cross


def quant_shard_stream(engine, quant_spec):
    """Quant-domain shard exporter with the SAME contract as
    ``get_per_tensor_param_shard``: yields ``(name, local_flat, ShardSpec)``
    triples that the GENERIC ``hf_delta_export`` / ``prime_delta_snapshots``
    machinery consumes -- the quant path and the bf16 path share one diff
    implementation by construction.

    Per export-index record: run the comm-stubbed probe on the FULL local
    shard, exchange static block ownership once, compact/reduce only amax cells
    shared by more than one owner, quantize locally, and yield up to four
    dtype-homogeneous groups (concatenated across the record's union
    slots, zero-length segments for rows other ranks own -- lockstep by
    construction):

        ``{megatron_name}::c``  fp8 codes        (contributes: always)
        ``{megatron_name}::s``  fp32 scale cells (contributes: per-block owner)
        ``{megatron_name}::b``  bf16 passthrough (contributes: always)
        ``{megatron_name}::f``  fp32 passthrough (contributes: always)

    Slot metadata for the entry builder is registered on
    ``engine._quant_group_meta[name]`` as
    ``(slots, local_sizes, dtype, optional_full_slot_positions)``.
    """
    import torch.distributed as dist
    from megatron.core import parallel_state as mpu

    from verl.utils.fp8_sharded import (
        _ABSMAX_EPS,
        local_blockwise_absmax,
        quantize_shard_with_descale,
        sticky_ue8m0_descale,
    )
    from verl.utils.kernel.fp8_kernel import FP8_DTYPE, FP8_MAX, ceil_div

    helper_should_quantize = quant_spec.should_quantize
    # Seed and steady MUST agree on the dialect AND the ckpt-scale preference:
    # they write the same tensors on alternating syncs, and the delta only
    # resends changed positions, so any split leaves the other rule's stale
    # scales in place forever.
    scale_fmt = getattr(quant_spec, "scale_fmt", None)
    ckpt_scales = getattr(quant_spec, "ckpt_scales", None)
    fp32_pred = getattr(quant_spec, "fp32_predicate", None)
    block = list(quant_spec.weight_block_size)
    bm, bn = int(block[0]), int(block[1])
    index = engine._mcore_export_index()
    slot_cache = engine._delta_slot_cache
    meta = engine._quant_group_meta = getattr(engine, "_quant_group_meta", {})
    owner_plans = engine._quant_block_owner_plans = getattr(engine, "_quant_block_owner_plans", {})
    planned_cells = 0
    planned_cross_cells = 0

    for rec in index:
        if rec.probe is None:
            outs = {}
            dev = torch.device(torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        else:
            # DSv4's fp32 special parameters must retain their mantissa through
            # the HF mapping transform. Quantized and bf16 parameters are cast
            # at their own group sites below.
            src = rec.param.data
            if src.dtype != torch.float32:
                src = src.to(torch.bfloat16)
            outs = rec.probe.megatron_to_hf(src, rec.module)
            dev = rec.param.device
        pg = rec.spec.gather_group
        group_rank = dist.get_rank(pg) if pg is not None else dist.get_rank()
        slots = rec.slots if rec.slots is not None else slot_cache.get(rec.megatron_name)
        if slots is None:
            slots = [(n, tuple(int(x) for x in t.shape)) for n, t in outs.items()]
            slot_cache[rec.megatron_name] = slots

        quantizable = [(sname, sshape) for sname, sshape in slots if len(sshape) == 2 and helper_should_quantize(sname)]

        # Deduplicate DP/CP replicas before both value and scale ownership are
        # planned.  Otherwise identical replicas would make every local block
        # look cross-rank and defeat boundary-only amax exchange.
        _tp_world = torch.distributed.get_world_size(group=mpu.get_tensor_model_parallel_group())
        _ep_size = mpu.get_expert_model_parallel_world_size()
        _is_expert = ".experts." in rec.megatron_name and (_ep_size > 1 or _tp_world > 1)
        _is_tp_sharded = rec.param is not None and getattr(rec.param, "tensor_model_parallel", False) and _tp_world > 1
        if _is_expert:
            owns_replica = mpu.get_expert_data_parallel_rank() == 0
        elif _is_tp_sharded:
            owns_replica = mpu.get_data_parallel_rank(with_context_parallel=True) == 0
        else:
            owns_replica = (
                mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            )

        # The probe's NaN pattern is a static ownership map in final HF block
        # coordinates.  Exchange owner counts once, then cache only compact
        # positions: steady syncs reduce amax values for genuinely shared
        # boundary blocks and nothing else.
        plan_key = (rec.megatron_name, tuple(block), tuple(quantizable))
        plan = owner_plans.get(plan_key)
        if plan is None:
            local_masks = []
            for sname, sshape in quantizable:
                t = outs.get(sname)
                if t is None or not owns_replica:
                    local_masks.append(
                        torch.zeros(
                            ceil_div(int(sshape[0]), bm),
                            ceil_div(int(sshape[1]), bn),
                            dtype=torch.bool,
                            device=dev,
                        )
                    )
                else:
                    local_masks.append(_local_block_owner_mask(t, (bm, bn), tuple(sshape)))
            plan = _block_owner_plan(local_masks, pg, group_rank)
            owner_plans[plan_key] = plan
            planned_cells += sum(int(mask.numel()) for mask in local_masks)
            planned_cross_cells += sum(int(cross.numel()) for cross, _scale in plan)

        grids = []
        for sname, sshape in quantizable:
            t = outs.get(sname)
            if t is None:
                g = torch.zeros(
                    ceil_div(int(sshape[0]), bm), ceil_div(int(sshape[1]), bn), dtype=torch.float32, device=dev
                )
            else:
                g = local_blockwise_absmax(t.to(torch.bfloat16), block, 0, tuple(sshape))
            grids.append(g)
        _reduce_cross_block_amax(grids, plan, pg)

        # Codes and passthrough values follow model ownership.  Scale cells
        # follow the block plan above and may therefore have several ranks
        # contributing disjoint positions to one HF scale tensor.
        groups = {
            "c": {"slots": [], "pieces": [], "dtype": FP8_DTYPE, "contributes": owns_replica},
            "s": {"slots": [], "pieces": [], "positions": [], "dtype": torch.float32, "contributes": True},
            "b": {"slots": [], "pieces": [], "dtype": torch.bfloat16, "contributes": owns_replica},
            "f": {"slots": [], "pieces": [], "dtype": torch.float32, "contributes": owns_replica},
        }
        qi = 0
        for sname, sshape in slots:
            t = outs.get(sname)
            if len(sshape) == 2 and helper_should_quantize(sname):
                if scale_fmt == "ue8m0":
                    descale = sticky_ue8m0_descale(grids[qi], ckpt_scales.get(sname) if ckpt_scales else None)
                else:
                    descale = grids[qi].clamp(min=_ABSMAX_EPS) / FP8_MAX
                qi += 1
                if t is not None:
                    codes = quantize_shard_with_descale(t.to(torch.bfloat16), descale, block, 0)
                else:
                    codes = torch.empty(0, dtype=FP8_DTYPE, device=dev)
                groups["c"]["slots"].append((sname, tuple(sshape)))
                groups["c"]["pieces"].append(codes.reshape(-1))
                groups["s"]["slots"].append((sname + "_scale_inv", tuple(descale.shape)))
                scale_pos = plan[qi - 1][1]
                groups["s"]["positions"].append(scale_pos)
                groups["s"]["pieces"].append(descale.reshape(-1)[scale_pos.long()])
            elif fp32_pred is not None and fp32_pred(sname):
                # fp32 passthrough group: same fidelity rule as the seed wire --
                # DSv4's special params are fp32 on disk and in the serving
                # engine; folding them to bf16 costs 16 mantissa bits every
                # time they change. Routed by the CHECKPOINT-derived predicate,
                # never by the local tensor's presence or dtype: group slot
                # layouts must be identical on every rank, and non-owner ranks
                # see t is None here.
                flat = (
                    t.to(torch.float32).reshape(-1)
                    if t is not None
                    else torch.empty(0, dtype=torch.float32, device=dev)
                )
                groups["f"]["slots"].append((sname, tuple(sshape)))
                groups["f"]["pieces"].append(flat)
            else:
                flat = (
                    t.to(torch.bfloat16).reshape(-1)
                    if t is not None
                    else torch.empty(0, dtype=torch.bfloat16, device=dev)
                )
                groups["b"]["slots"].append((sname, tuple(sshape)))
                groups["b"]["pieces"].append(flat)

        for kind, g in groups.items():
            if not g["slots"]:
                continue
            name = f"{rec.megatron_name}::{kind}"
            sizes = [int(pc.numel()) for pc in g["pieces"]]
            positions = g.get("positions")
            meta[name] = (g["slots"], sizes, str(g["dtype"]).replace("torch.", ""), positions)
            flat = torch.cat(g["pieces"]) if g["pieces"] else torch.empty(0, dtype=g["dtype"], device=dev)
            spec = ShardSpec(
                full_shape=(int(flat.numel()),),
                place=0,
                contributes=g["contributes"] and bool(flat.numel()),
                gather_group=pg,
            )
            yield name, flat, spec

    if planned_cells:
        logger.warning(
            "quant amax owner plan rank=%d: scale_cells=%d cross_rank_cells=%d steady_max_payload_bytes=%d",
            dist.get_rank(),
            planned_cells,
            planned_cross_cells,
            planned_cross_cells * 4,
        )


def quant_delta_entry(engine):
    """Entry builder for the quant shard stream, closed over the engine's group
    metadata: splits the concatenated group's delta positions back into
    per-slot counts and slot-local indices -- the exact wire entry shape the
    bf16 path produces."""

    def _entry(name, spec, place, lidx, lval):
        slots, sizes, dtype_str, positions = engine._quant_group_meta[name]
        bounds = torch.tensor(sizes, device=lidx.device).cumsum(0)
        seg = torch.searchsorted(bounds, lidx, right=True)
        counts = torch.bincount(seg, minlength=len(sizes)).to("cpu")
        offsets = torch.cat([torch.zeros(1, dtype=bounds.dtype, device=lidx.device), bounds[:-1]])
        local_idx = (lidx - offsets[seg]).to(torch.int32)
        if positions is not None and lidx.numel():
            mapped = torch.empty_like(local_idx)
            for i, pos in enumerate(positions):
                sel = seg == i
                if bool(sel.any()):
                    mapped[sel] = pos.to(lidx.device)[local_idx[sel].long()]
            local_idx = mapped
        return (slots, dtype_str, counts, local_idx, lval)

    return _entry
