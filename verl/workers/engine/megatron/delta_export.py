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
import math
import re
from dataclasses import dataclass
from typing import Any, Optional

import torch

from verl.workers.engine.spec import ShardSpec

logger = logging.getLogger(__name__)

_EXPERT_ID_PATTERNS = (
    r"local_experts\.(\d+)",
    r"(?:weight|bias)(\d+)",
    r"experts\.(\d+)",
)


def _extract_expert_id(param_name: str) -> int:
    """Extract the global expert index from a Megatron global parameter name.

    Mirrors ``megatron.bridge.utils.common_utils.extract_expert_number_from_param``
    with a regex fallback so CPU unit tests do not require Megatron-Bridge.
    """
    try:
        from megatron.bridge.utils.common_utils import extract_expert_number_from_param

        return int(extract_expert_number_from_param(param_name))
    except ImportError:
        for pattern in _EXPERT_ID_PATTERNS:
            match = re.search(pattern, param_name)
            if match:
                return int(match.group(1))
        raise ValueError(f"No expert number found in parameter name: {param_name}") from None


def _get_num_moe_experts(megatron_model, module) -> int:
    """Resolve ``num_moe_experts`` from the task module or unwrapped model config."""
    candidates = [module]
    if isinstance(megatron_model, list | tuple):
        candidates.extend(megatron_model)
    elif megatron_model is not None:
        candidates.append(megatron_model)
    for obj in candidates:
        cur = obj
        while cur is not None:
            cfg = getattr(cur, "config", None)
            num_experts = getattr(cfg, "num_moe_experts", None)
            if num_experts is not None:
                num_experts = int(num_experts)
                if num_experts <= 0:
                    raise ValueError(f"Expected positive num_moe_experts, got {num_experts}")
                return num_experts
            cur = getattr(cur, "module", None)
    raise ValueError("Could not determine num_moe_experts for grouped-export MoE parameter")


# Sentinel cache entry recording that every shard header has already been walked.
_SCANNED_SHARDS = "\0scanned-shards"


def _get_hf_expected_shape(
    bridge,
    key: str,
    cache: dict[str, Optional[tuple[int, ...]]],
) -> Optional[tuple[int, ...]]:
    """Read an HF tensor's expected shape without loading tensor bytes when possible.

    For ``is_grouped_export=True`` mappings this is the only external ground truth the
    probe has: it both mirrors ``MegatronModelBridge._accumulate_grouped_export``'s
    adaptive transpose check and lets ``mcore_hf_delta_entry`` verify that the stacked
    slot shape it reports actually matches the checkpoint.
    """
    if key in cache:
        return cache[key]
    shape: Optional[tuple[int, ...]] = None
    hf_pretrained = getattr(bridge, "hf_pretrained", None)
    hf_state = getattr(hf_pretrained, "state", None) if hf_pretrained is not None else None
    if hf_state is not None:
        source = getattr(hf_state, "source", None)
        if source is not None and hasattr(source, "path") and hasattr(source, "key_to_filename_map"):
            try:
                from safetensors import safe_open

                file_path = None
                filename = (source.key_to_filename_map or {}).get(key)
                if filename:
                    candidate = source.path / filename
                    if candidate.exists():
                        file_path = candidate
                if file_path is not None:
                    with safe_open(file_path, framework="pt", device="cpu") as f:
                        if key in f.keys():
                            shape = tuple(int(d) for d in f.get_slice(key).get_shape())
                elif _SCANNED_SHARDS not in cache:
                    # No index to resolve this key, so walk the shards once and cache
                    # every header shape rather than re-globbing for each key.
                    cache[_SCANNED_SHARDS] = None
                    for candidate in sorted(source.path.glob("*.safetensors")):
                        with safe_open(candidate, framework="pt", device="cpu") as f:
                            for k in f.keys():
                                cache.setdefault(k, tuple(int(d) for d in f.get_slice(k).get_shape()))
                    shape = cache.get(key)
            except Exception as e:  # pragma: no cover - defensive fallback to state dict
                logger.debug("safetensors header shape lookup failed for %s: %s", key, e)
        if shape is None:
            try:
                if key in hf_state:
                    shape = tuple(int(d) for d in hf_state[key].shape)
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("hf_state shape lookup failed for %s: %s", key, e)
    cache[key] = shape
    return shape


def _slot_idx_dtype(slots: list[tuple[str, tuple[int, ...]]]) -> torch.dtype:
    """Choose the flat-index dtype for a directory row from its global slot table.

    Every rank shares the identical ``rec.slots`` table after ``_exchange_slot_tables``,
    so all ranks agree on ``idx_dtype`` even for zero-count or PP placeholder rows.
    Keeps ``int32`` whenever every slot fits in signed 32-bit offsets and promotes to
    ``int64`` if any stacked tensor has ``>= 2**31`` elements to avoid silent wraparound.
    """
    return torch.int64 if any(math.prod(shape) >= (1 << 31) for _, shape in slots) else torch.int32


@dataclass(frozen=True)
class _GroupedExportMeta:
    """Metadata for a Megatron-Bridge ``is_grouped_export=True`` parameter.

    Mappings such as ``FusedGatedExpertMapping`` / ``FusedExpertMapping``
    (Qwen3.5-MoE, Qwen3-VL-MoE, Gemma4(-VL), stacked-key GLM4.5(-V), Step3.5/3.7)
    emit one per-expert slice under a shared HF tensor name and rely on
    ``MegatronModelBridge._accumulate_grouped_export`` to stack all
    ``num_experts`` slices along dim 0 (and optionally transpose the last two
    dims when ``transpose_on_export=True``).

    Because the probe invokes ``megatron_to_hf`` directly per task, each task
    promotes its per-expert slice into stacked ``(num_experts, *slice_shape)``
    coordinates by shifting flat indices by ``expert_id * slice_numel``.
    """

    expert_id: int
    num_experts: int
    transpose_on_export: bool = False
    expected_shapes: Optional[dict[str, tuple[int, ...]]] = None


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
    # Set for Megatron-Bridge ``is_grouped_export=True`` expert parameters so
    # ``mcore_hf_delta_entry`` promotes each per-expert slice into stacked
    # ``(num_experts, *slice_shape)`` coordinates.
    grouped_export: Optional[_GroupedExportMeta] = None


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

    if slot_cache is None:
        slot_cache = {}

    pp_world = mpu.get_pipeline_model_parallel_world_size()
    tp_group = mpu.get_tensor_model_parallel_group()
    tp_world = torch.distributed.get_world_size(group=tp_group)
    ep_size = mpu.get_expert_model_parallel_world_size()
    etp_size = (
        mpu.get_expert_tensor_parallel_world_size() if hasattr(mpu, "get_expert_tensor_parallel_world_size") else 1
    )
    world = torch.distributed.group.WORLD

    tasks = bridge.get_conversion_tasks(megatron_model)
    hf_shape_cache: dict[str, Optional[tuple[int, ...]]] = {}
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

        is_expert = mapping.is_expert and (ep_size > 1 or tp_world > 1 or etp_size > 1)
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

        grouped_export: Optional[_GroupedExportMeta] = None
        if getattr(mapping, "is_grouped_export", False):
            expert_id = _extract_expert_id(name)
            num_experts = _get_num_moe_experts(megatron_model, module)
            assert 0 <= expert_id < num_experts, (
                f"{name}: extracted global expert_id={expert_id} out of range [0, {num_experts})"
            )
            transpose_on_export = bool(getattr(mapping, "transpose_on_export", False))
            # Always resolve the checkpoint shape, not just for transpose_on_export:
            # it is the only check that can catch a wrong num_experts or a slice that
            # already carries a leading expert dim. Header-only reads, cached per key.
            expected_shapes: dict[str, tuple[int, ...]] = {}
            candidate_keys: list[str] = []
            hf_param = getattr(mapping, "hf_param", None)
            if isinstance(hf_param, str):
                candidate_keys.append(hf_param)
            elif isinstance(hf_param, dict):
                candidate_keys.extend(v for v in hf_param.values() if isinstance(v, str))
            group_key = getattr(mapping, "group_key", None)
            if isinstance(group_key, str) and group_key not in candidate_keys:
                candidate_keys.append(group_key)
            for hf_key in candidate_keys:
                exp_shape = _get_hf_expected_shape(bridge, hf_key, hf_shape_cache)
                if exp_shape is not None:
                    expected_shapes[hf_key] = exp_shape
            grouped_export = _GroupedExportMeta(
                expert_id=expert_id,
                num_experts=num_experts,
                transpose_on_export=transpose_on_export,
                expected_shapes=expected_shapes or None,
            )

        index.append(
            McoreParamExport(
                megatron_name=name,
                param=param,
                spec=spec,
                probe=make_probe(mapping, module),
                module=module,
                grouped_export=grouped_export,
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
        idx_dtype = _slot_idx_dtype(cached)
        return (
            cached,
            dtype_str,
            torch.zeros(len(cached), dtype=torch.int64),
            torch.empty(0, dtype=idx_dtype, device=lval.device),
            torch.empty(0, dtype=lval.dtype, device=lval.device),
        )

    buf = torch.full(tuple(rec.param.shape), float("nan"), dtype=lval.dtype, device=lval.device)
    if lidx.numel():
        buf.view(-1)[lidx] = lval

    outs = rec.probe.megatron_to_hf(buf, rec.module)
    ge = rec.grouped_export
    if ge is not None and ge.transpose_on_export:
        # Mirror MegatronModelBridge._accumulate_grouped_export: transpose the
        # last two dims of each per-expert slice before indexing so flat
        # coordinates match the stacked+transposed HF tensor layout.
        transformed: dict[str, torch.Tensor] = {}
        for sname, out in outs.items():
            if out.ndim >= 2:
                stacked_shape = (ge.num_experts, *(int(x) for x in out.shape))
                transposed_shape = (
                    ge.num_experts,
                    *(int(x) for x in out.shape[:-2]),
                    int(out.shape[-1]),
                    int(out.shape[-2]),
                )
                expected = ge.expected_shapes.get(sname) if ge.expected_shapes else None
                if expected is None or (stacked_shape != expected and transposed_shape == expected):
                    out = out.transpose(-1, -2).contiguous()
            transformed[sname] = out
        outs = transformed

    key = rec.megatron_name
    slots = rec.slots if rec.slots is not None else slot_cache.get(key)
    if slots is None:
        # only reachable from the index-build exchange itself, which probes a
        # zero delta to reveal this rank's local list before the row unions
        # are installed on the recs; steady entries always find rec.slots.
        if ge is not None:
            slots = [(n, (ge.num_experts, *(int(x) for x in t.shape))) for n, t in outs.items()]
        else:
            slots = [(n, tuple(int(x) for x in t.shape)) for n, t in outs.items()]
        slot_cache[key] = slots
    unknown = set(outs) - {n for n, _ in slots}
    assert not unknown, (
        f"{key}: probe emitted slots missing from the global table {sorted(unknown)[:4]} -- "
        "non-deterministic converter naming would misalign the batched gather"
    )
    idx_dtype = _slot_idx_dtype(slots)
    counts = torch.zeros(len(slots), dtype=torch.int64)
    idx_pieces: list[torch.Tensor] = []
    val_pieces: list[torch.Tensor] = []
    for s_i, (sname, sshape) in enumerate(slots):
        out = outs.get(sname)
        if out is None:
            continue  # another rank's slot (e.g. its ep-local experts): zero count here
        if ge is not None:
            expected_slot_shape = (ge.num_experts, *(int(x) for x in out.shape))
            assert tuple(sshape) == expected_slot_shape, (
                f"{key}/{sname}: grouped-export slice shape {tuple(out.shape)} with "
                f"num_experts={ge.num_experts} does not match slot shape {tuple(sshape)}"
            )
            # The assert above only proves this rank agrees with the shared slot
            # table, which was built from the same probe; the checkpoint header is
            # the only independent witness that num_experts is right and that the
            # slice is a single expert rather than an already-stacked tensor.
            checkpoint_shape = ge.expected_shapes.get(sname) if ge.expected_shapes else None
            assert checkpoint_shape is None or tuple(sshape) == checkpoint_shape, (
                f"{key}/{sname}: stacked slot shape {tuple(sshape)} (num_experts="
                f"{ge.num_experts} x slice {tuple(out.shape)}) does not match the "
                f"checkpoint tensor shape {checkpoint_shape}"
            )
        fl = out.reshape(-1)
        p_ = (~torch.isnan(fl)).nonzero(as_tuple=False).view(-1)
        if p_.numel():
            counts[s_i] = p_.numel()
            val_pieces.append(fl[p_])
            if ge is not None:
                p_ = p_ + ge.expert_id * fl.numel()
            idx_pieces.append(p_.to(idx_dtype))
    if idx_pieces:
        hf_idx = torch.cat(idx_pieces)
        hf_val = torch.cat(val_pieces)
    else:
        hf_idx = torch.empty(0, dtype=idx_dtype, device=lval.device)
        hf_val = torch.empty(0, dtype=lval.dtype, device=lval.device)
    return slots, dtype_str, counts, hf_idx, hf_val
