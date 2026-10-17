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
"""Static bucket layout and write planning for the ``sharded_nixl`` checkpoint engine.

Pure planning, no transfers. Every trainer rank exports its local shard of every parameter in
the same order (``engine.get_per_tensor_param_shard()``), so every rank computes the same
:class:`BucketLayout` from the ``(name, full_shape, dtype)`` sequence without communicating.
A flat-contiguous shard (FSDP2 ``Shard(0)``, replicated, HSDP) is one byte interval of the
flattened tensor, so each rank's part of a bucket is a few :class:`WriteSlice` objects and the
generator's bucket holds the full tensors once every trainer has written.

Fused MoE parameters are expanded into the per-expert HF names vLLM loads, mirroring
``verl.workers.engine.fsdp.utils.unfuse_moe_params``; each expert is a contiguous sub-range of
the fused tensor, so expansion only relabels byte ranges.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Iterable, Optional

import torch

if TYPE_CHECKING:
    from verl.workers.engine.spec import ShardSpec
    from verl.workers.rollout.base import RolloutLayout

DEFAULT_ALIGN = 256

__all__ = [
    "DEFAULT_ALIGN",
    "Box",
    "BoxRun",
    "BoxSegment",
    "BoxWriteSlice",
    "BucketLayout",
    "ChunkSlot",
    "LayoutBuilder",
    "ParamEntry",
    "WriteSlice",
    "align_up",
    "box_local_shape",
    "clip_segments",
    "contribution_range",
    "expand_fused_moe_entry",
    "index_runs",
    "layout_digest",
    "local_entry",
    "local_shapes_key",
    "merge_box_plans",
    "normalize_slices",
    "owned_name_filter",
    "plan_box_writes",
    "plan_indexed_writes",
    "plan_rank_local_indexed_writes",
    "plan_rank_local_writes",
    "plan_writes",
    "slices_digest",
    "split_box_run",
]


def align_up(x: int, align: int) -> int:
    return (x + align - 1) // align * align


@dataclass(frozen=True)
class ParamEntry:
    """One HF tensor as the generator loads it. Element ``i`` of the entry is element
    ``src_elem_offset + i`` of the flattened exported tensor ``src_name`` (several entries share
    one source for a fused MoE tensor)."""

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    src_name: str
    src_elem_offset: int = 0

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n

    @property
    def itemsize(self) -> int:
        return torch.empty(0, dtype=self.dtype).element_size()

    @property
    def nbytes(self) -> int:
        return self.numel * self.itemsize

    @property
    def src_byte_offset(self) -> int:
        return self.src_elem_offset * self.itemsize

    def key(self) -> tuple:
        return (self.name, tuple(int(d) for d in self.shape), str(self.dtype))


@dataclass(frozen=True)
class ChunkSlot:
    """Bytes ``[chunk_offset, chunk_offset + chunk_size)`` of tensor ``name`` stored at byte
    ``offset`` of bucket ``bucket_id``; mirrors :class:`~verl.checkpoint_engine.base.TensorMeta`
    so the receiver can reuse ``merge_weight_chunks``."""

    name: str
    chunk_offset: int
    chunk_size: int
    bucket_id: int
    offset: int

    @property
    def end(self) -> int:
        return self.chunk_offset + self.chunk_size


@dataclass(frozen=True)
class WriteSlice:
    """Copy ``length`` bytes at ``src_offset`` of the local flat shard to ``dst_offset`` of bucket
    ``bucket_id``."""

    bucket_id: int
    dst_offset: int
    src_offset: int
    length: int
    name: str


@dataclass
class BucketLayout:
    """Every entry and the slots its bytes occupy; identical on every rank, identified by
    ``digest``."""

    bucket_size: int
    align: int
    entries: list[ParamEntry]
    slots: list[ChunkSlot]
    num_buckets: int
    digest: str
    slots_by_name: dict[str, list[ChunkSlot]] = field(default_factory=dict)
    slots_by_bucket: list[list[ChunkSlot]] = field(default_factory=list)
    entry_by_name: dict[str, ParamEntry] = field(default_factory=dict)

    def __post_init__(self):
        if not self.slots_by_name:
            for s in self.slots:
                self.slots_by_name.setdefault(s.name, []).append(s)
        if not self.slots_by_bucket:
            self.slots_by_bucket = [[] for _ in range(self.num_buckets)]
            for s in self.slots:
                self.slots_by_bucket[s.bucket_id].append(s)
        if not self.entry_by_name:
            self.entry_by_name = {e.name: e for e in self.entries}

    @property
    def total_bytes(self) -> int:
        return sum(e.nbytes for e in self.entries)

    def to_wire(self) -> dict:
        return {
            "bucket_size": self.bucket_size,
            "align": self.align,
            "num_buckets": self.num_buckets,
            "digest": self.digest,
            "entries": [(e.name, tuple(e.shape), str(e.dtype), e.src_name, e.src_elem_offset) for e in self.entries],
            "slots": [(s.name, s.chunk_offset, s.chunk_size, s.bucket_id, s.offset) for s in self.slots],
        }

    @classmethod
    def from_wire(cls, d: dict) -> BucketLayout:
        entries = [
            ParamEntry(name=n, shape=tuple(shape), dtype=_dtype_from_str(dt), src_name=sn, src_elem_offset=off)
            for n, shape, dt, sn, off in d["entries"]
        ]
        slots = [ChunkSlot(*s) for s in d["slots"]]
        return cls(
            bucket_size=d["bucket_size"],
            align=d["align"],
            entries=entries,
            slots=slots,
            num_buckets=d["num_buckets"],
            digest=d["digest"],
        )


def _dtype_from_str(s: str) -> torch.dtype:
    return getattr(torch, s.replace("torch.", ""))


def layout_digest(entries: Iterable[ParamEntry], bucket_size: int, align: int) -> str:
    h = hashlib.sha256()
    h.update(f"{bucket_size}:{align}".encode())
    for e in entries:
        h.update(repr(e.key()).encode())
    return h.hexdigest()


class LayoutBuilder:
    """Packs entries into buckets in stream order, with the packing rule of the ``nixl`` engine's
    ``split_weight_chunks``:

    * each tensor starts at an ``align``-aligned offset, in a new bucket if it does not fit;
    * a tensor larger than a bucket is split into ``bucket_size`` pieces, each starting a fresh
      bucket at offset 0 (``merge_weight_chunks`` expects pieces in order from chunk offset 0);
    * a duplicate HF name raises.
    """

    def __init__(self, bucket_size: int, align: int = DEFAULT_ALIGN):
        assert bucket_size > 0 and align > 0 and bucket_size % align == 0, (bucket_size, align)
        self.bucket_size = bucket_size
        self.align = align
        self.entries: list[ParamEntry] = []
        self.slots: list[ChunkSlot] = []
        self._names: set[str] = set()
        self._bucket = 0
        self._offset = 0
        self._started = False

    def add(self, entry: ParamEntry) -> list[ChunkSlot]:
        if entry.name in self._names:
            raise ValueError(f"duplicate HF tensor name in shard export: {entry.name!r}")
        self._names.add(entry.name)
        self.entries.append(entry)
        nbytes = entry.nbytes
        out: list[ChunkSlot] = []
        if nbytes == 0:
            # empty tensors still get a slot so the receiver yields one tensor per name
            off = align_up(self._offset, self.align) if self._started else 0
            out.append(ChunkSlot(entry.name, 0, 0, self._bucket, off))
            self._offset = off
        elif nbytes <= self.bucket_size:
            off = align_up(self._offset, self.align) if self._started else 0
            if off + nbytes > self.bucket_size:
                self._bucket += 1
                off = 0
            out.append(ChunkSlot(entry.name, 0, nbytes, self._bucket, off))
            self._offset = off + nbytes
        else:
            if self._started and self._offset > 0:
                self._bucket += 1
            chunk_offset = 0
            while chunk_offset < nbytes:
                chunk_size = min(self.bucket_size, nbytes - chunk_offset)
                out.append(ChunkSlot(entry.name, chunk_offset, chunk_size, self._bucket, 0))
                chunk_offset += chunk_size
                if chunk_offset < nbytes:
                    self._bucket += 1
            self._offset = out[-1].chunk_size
        self._started = True
        self.slots.extend(out)
        return out

    def build(self) -> BucketLayout:
        num_buckets = (self._bucket + 1) if self._started else 0
        return BucketLayout(
            bucket_size=self.bucket_size,
            align=self.align,
            entries=list(self.entries),
            slots=list(self.slots),
            num_buckets=num_buckets,
            digest=layout_digest(self.entries, self.bucket_size, self.align),
        )


def build_bucket_layout(entries: Iterable[ParamEntry], bucket_size: int, align: int = DEFAULT_ALIGN) -> BucketLayout:
    b = LayoutBuilder(bucket_size, align)
    for e in entries:
        b.add(e)
    return b.build()


# ---------------------------------------------------------------------------
# MoE expansion (manifest-level mirror of unfuse_moe_params)
# ---------------------------------------------------------------------------


def expand_fused_moe_entry(
    src_name: str, full_shape: tuple[int, ...], dtype: torch.dtype, model_type: Optional[str] = None
) -> list[ParamEntry]:
    """The generator-side entries of one exported parameter.

    * ``*.mlp.experts.gate_up_proj`` ``[E, 2I, H]``: per expert ``gate_proj`` and ``up_proj``
      ``[I, H]``, consecutive in the fused tensor.
    * ``*.mlp.experts.down_proj`` ``[E, H, I]``: per expert ``down_proj`` ``[H, I]``.
    * ``gpt_oss`` keeps its fused 3-D experts; everything else maps to itself.

    Must match ``unfuse_moe_params`` byte for byte.
    """
    shape = tuple(int(d) for d in full_shape)
    if model_type == "gpt_oss" and ".mlp.experts." in src_name and len(shape) == 3:
        return [ParamEntry(src_name, shape, dtype, src_name, 0)]
    if src_name.endswith(".mlp.experts.gate_up_proj") and len(shape) == 3:
        E, two_i, H = shape
        assert two_i % 2 == 0, f"{src_name}: dim 1 ({two_i}) is not 2*intermediate"
        inter = two_i // 2
        base = src_name.removesuffix(".gate_up_proj")
        out = []
        for e in range(E):
            start = e * two_i * H
            out.append(ParamEntry(f"{base}.{e}.gate_proj.weight", (inter, H), dtype, src_name, start))
            out.append(ParamEntry(f"{base}.{e}.up_proj.weight", (inter, H), dtype, src_name, start + inter * H))
        return out
    if src_name.endswith(".mlp.experts.down_proj") and len(shape) == 3:
        E, H, inter = shape
        base = src_name.removesuffix(".down_proj")
        return [
            ParamEntry(f"{base}.{e}.down_proj.weight", (H, inter), dtype, src_name, e * H * inter) for e in range(E)
        ]
    return [ParamEntry(src_name, shape, dtype, src_name, 0)]


# ---------------------------------------------------------------------------
# Trainer-side contribution range and write planning
# ---------------------------------------------------------------------------


def contribution_range(spec: ShardSpec, local_numel: int, itemsize: int) -> Optional[tuple[int, int]]:
    """This rank's ``(byte_offset, nbytes)`` in the flattened full tensor, or ``None`` when it
    does not contribute (e.g. an HSDP replica). Only flat-contiguous placements are supported
    (a dim-0 ``_StridedShard`` is one); cuts on dims > 0 raise ``NotImplementedError``."""
    from verl.workers.engine.spec import BlockPlacement, derive_dtensor_placement

    if spec.place is not None:
        place, contributes = spec.place, spec.contributes
    else:
        place, contributes, _ = derive_dtensor_placement(spec)
    if not contributes:
        return None
    nbytes = local_numel * itemsize
    if isinstance(place, int):
        return place * itemsize, nbytes
    assert isinstance(place, BlockPlacement), type(place)
    if place.is_flat_contiguous:
        return place.flat_offset * itemsize, nbytes
    raise NotImplementedError(
        "sharded_nixl supports only flat-contiguous shards (FSDP2 Shard(0), replicated, HSDP); "
        f"got local_shape={place.local_shape} global_offset={place.global_offset} full_shape={place.full_shape}"
    )


def plan_writes(
    layout: BucketLayout | dict[str, list[ChunkSlot]], entries_for_src: list[ParamEntry], src_range: tuple[int, int]
) -> list[WriteSlice]:
    """The writes for this rank's ``src_range`` of one source tensor: the range intersected with
    each entry's bytes, then with each of the entry's slots. ``layout`` may be a bare
    ``{name: [ChunkSlot]}`` while the layout is still being built."""
    slots_by_name = layout.slots_by_name if isinstance(layout, BucketLayout) else layout
    src_start, src_nbytes = src_range
    src_end = src_start + src_nbytes
    out: list[WriteSlice] = []
    for entry in entries_for_src:
        e_start = entry.src_byte_offset
        e_end = e_start + entry.nbytes
        lo, hi = max(src_start, e_start), min(src_end, e_end)
        if lo >= hi:
            continue
        in_lo, in_hi = lo - e_start, hi - e_start
        for slot in slots_by_name[entry.name]:
            s_lo, s_hi = max(in_lo, slot.chunk_offset), min(in_hi, slot.end)
            if s_lo >= s_hi:
                continue
            out.append(
                WriteSlice(
                    bucket_id=slot.bucket_id,
                    dst_offset=slot.offset + (s_lo - slot.chunk_offset),
                    src_offset=(e_start + s_lo) - src_start,
                    length=s_hi - s_lo,
                    name=entry.name,
                )
            )
    return out


def index_runs(counts: torch.Tensor, indices: torch.Tensor) -> list[list[tuple[int, int, int]]]:
    """Per entry ``[(hf_elem, value_elem, n)]``: the contiguous runs of compact values paired with
    per-entry flat HF ``indices`` (``counts`` values per entry, entry-major)."""
    out: list[list[tuple[int, int, int]]] = []
    value_start = 0
    for count in counts.tolist():
        count = int(count)
        entry_indices = indices[value_start : value_start + count]
        if count == 0:
            out.append([])
            continue
        breaks = (entry_indices[1:] != entry_indices[:-1] + 1).nonzero().flatten().cpu().tolist()
        starts = [0, *(i + 1 for i in breaks)]
        ends = [*(i + 1 for i in breaks), count]
        firsts = entry_indices[starts].tolist()
        out.append([(h, value_start + s, e - s) for h, s, e in zip(firsts, starts, ends, strict=True)])
        value_start += count
    assert value_start == indices.numel()
    return out


def plan_indexed_writes(
    layout: BucketLayout, entries: list[ParamEntry], runs: list[list[tuple[int, int, int]]]
) -> list[WriteSlice]:
    """Plan writes from compact values described by :func:`index_runs`."""
    out: list[WriteSlice] = []
    for entry, entry_runs in zip(entries, runs, strict=True):
        isz = entry.itemsize
        for hf, value, n in entry_runs:
            dst_start, dst_end = hf * isz, (hf + n) * isz
            for slot in layout.slots_by_name[entry.name]:
                lo, hi = max(dst_start, slot.chunk_offset), min(dst_end, slot.end)
                if lo >= hi:
                    continue
                out.append(
                    WriteSlice(
                        bucket_id=slot.bucket_id,
                        dst_offset=slot.offset + lo - slot.chunk_offset,
                        src_offset=value * isz + lo - dst_start,
                        length=hi - lo,
                        name=entry.name,
                    )
                )
    return out


# ---------------------------------------------------------------------------
# Rank-local slices (``rank_local_named_tensors``)
# ---------------------------------------------------------------------------
#
# A TP rollout rank reports, per HF name, the row-major box of the full tensor it keeps and
# receives that box as a dense tensor of the box's shape. A box is ``((start0, len0), (start1,
# len1))`` over a tensor viewed as ``(R, C)`` (ndim <= 2; 1-D as ``(1, N)``). The part of a
# trainer's flat interval inside a box is one contiguous range of the box's row-major order, so
# each (trainer rank, tensor, box) is one destination run from at most three strided source
# pieces: the partial first row, the whole rows (stride C), the partial last row.

Box = tuple[tuple[int, int], ...]  # per-dim (start, length) over the full tensor


@dataclass(frozen=True)
class BoxSegment:
    """``nrows`` rows of ``row_bytes`` bytes, ``stride`` bytes apart, from byte ``src_offset`` of
    the local flat shard; packed back to back they form a dense destination range."""

    src_offset: int
    nrows: int
    row_bytes: int
    stride: int

    @property
    def nbytes(self) -> int:
        return self.nrows * self.row_bytes


@dataclass(frozen=True)
class BoxRun:
    """One rank's part of one box: bytes ``[dst_offset, dst_offset + length)`` of the box-shaped
    local tensor, gathered from ``segments``."""

    dst_offset: int
    length: int
    segments: tuple[BoxSegment, ...]


@dataclass(frozen=True)
class BoxWriteSlice:
    """Rank-local :class:`WriteSlice`. ``box_key`` (``None`` = full tensor) tells apart TP ranks
    whose slices share ``(bucket_id, dst_offset, length)`` in the common rank-local layout but
    carry different bytes."""

    bucket_id: int
    dst_offset: int
    length: int
    segments: tuple[BoxSegment, ...]
    name: str
    box_key: Optional[Box] = None

    @property
    def key(self) -> tuple:
        return (self.bucket_id, self.dst_offset, self.length, self.box_key)


def _as_2d(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) == 0:
        return 1, 1
    if len(shape) == 1:
        return 1, int(shape[0])
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    n = 1
    for d in shape:
        n *= int(d)
    return 1, n


def _box_2d(shape: tuple[int, ...], box: Optional[Box]) -> tuple[int, int, int, int]:
    """Box as ``(r0, nr, c0, nc)`` over the ``_as_2d`` view of ``shape`` (``None`` = full)."""
    R, C = _as_2d(shape)
    if box is None:
        return 0, R, 0, C
    if len(shape) == 1:
        ((c0, nc),) = box
        return 0, 1, int(c0), int(nc)
    if len(shape) == 2:
        (r0, nr), (c0, nc) = box
        return int(r0), int(nr), int(c0), int(nc)
    if tuple(box) == full_box(shape):
        return 0, R, 0, C
    raise NotImplementedError(f"rank-local boxes support ndim <= 2 (or full boxes); got shape={shape} box={box}")


def full_box(shape: tuple[int, ...]) -> Box:
    return tuple((0, int(d)) for d in shape)


def box_local_shape(box: Box) -> tuple[int, ...]:
    return tuple(int(n) for _s, n in box)


def normalize_slices(slices: Optional[dict]) -> dict[str, tuple[tuple[int, ...], Box]]:
    """``RolloutLayout.slices`` -> validated ``{name: (full_shape, box)}``."""
    out: dict[str, tuple[tuple[int, ...], Box]] = {}
    for name, spec in (slices or {}).items():
        shape = tuple(int(d) for d in spec["full_shape"])
        box = tuple((int(s), int(n)) for s, n in spec["box"])
        if len(box) != len(shape):
            raise ValueError(f"slice {name!r}: box {box} does not match full_shape {shape}")
        for (s, n), d in zip(box, shape, strict=True):
            if s < 0 or n < 0 or s + n > d:
                raise ValueError(f"slice {name!r}: box {box} out of bounds for full_shape {shape}")
        out[name] = (shape, box)
    return out


def slices_digest(slices: Optional[dict]) -> str:
    if not slices:
        return ""
    norm = normalize_slices(slices)
    return hashlib.sha256(json.dumps(sorted((k, v) for k, v in norm.items())).encode()).hexdigest()


def local_shapes_key(slices: Optional[dict]) -> str:
    """Hash of the local shapes a slice table implies. Generators with equal keys share one
    rank-local bucket layout."""
    norm = normalize_slices(slices)
    items = sorted((k, shape, box_local_shape(box)) for k, (shape, box) in norm.items())
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()


def local_entry(entry: ParamEntry, box: Optional[Box]) -> ParamEntry:
    """``entry`` with its box's local shape (``box=None``: full shape)."""
    if box is None:
        return entry
    return ParamEntry(entry.name, box_local_shape(box), entry.dtype, entry.src_name, entry.src_elem_offset)


def _coalesce(segs: list[BoxSegment]) -> tuple[BoxSegment, ...]:
    """Merge adjacent rows of equal width/stride into blocks, then contiguous blocks into runs."""
    out: list[BoxSegment] = []
    for s in segs:
        if s.nrows == 0 or s.row_bytes == 0:
            continue
        if out and out[-1].row_bytes == s.row_bytes:
            p = out[-1]
            # a single row has no stride of its own: it takes its neighbour's, or the gap between
            # two single rows (per-row HF runs of a box narrower than the trainer's shard)
            stride = p.stride if p.nrows > 1 else s.stride if s.nrows > 1 else s.src_offset - p.src_offset
            if (
                stride >= p.row_bytes
                and (s.nrows == 1 or s.stride == stride)
                and s.src_offset == p.src_offset + p.nrows * stride
            ):
                out[-1] = BoxSegment(p.src_offset, p.nrows + s.nrows, p.row_bytes, stride)
                continue
        out.append(s)
    flat: list[BoxSegment] = []
    for s in out:
        if s.nrows > 1 and s.row_bytes == s.stride:
            s = BoxSegment(s.src_offset, 1, s.nbytes, s.nbytes)
        if s.nrows == 1:
            s = BoxSegment(s.src_offset, 1, s.row_bytes, s.row_bytes)
            if flat and flat[-1].nrows == 1 and flat[-1].src_offset + flat[-1].row_bytes == s.src_offset:
                p = flat[-1]
                n = p.row_bytes + s.row_bytes
                flat[-1] = BoxSegment(p.src_offset, 1, n, n)
                continue
        flat.append(s)
    return tuple(flat)


def plan_box_writes(entry: ParamEntry, box: Optional[Box], src_range: tuple[int, int]) -> Optional[BoxRun]:
    """This rank's run of ``box`` over ``entry`` (``None`` = whole entry) from its source byte
    range ``src_range``, or ``None`` if it holds none of the box. Segment offsets are relative to
    the start of ``src_range``."""
    isz = entry.itemsize
    s0, sn = src_range
    e0 = entry.src_byte_offset
    lo, hi = max(s0, e0), min(s0 + sn, e0 + entry.nbytes)
    if lo >= hi:
        return None
    assert (lo - e0) % isz == 0 and (hi - e0) % isz == 0, (entry.name, src_range)
    a, b = (lo - e0) // isz, (hi - e0) // isz  # element interval [a, b) of the entry
    _R, C = _as_2d(entry.shape)
    r0, nr, c0, nc = _box_2d(entry.shape, box)
    if nr == 0 or nc == 0:
        return None
    ra, rb = a // C, (b - 1) // C  # first / last row touched by [a, b)
    i0, i1 = max(ra, r0), min(rb, r0 + nr - 1)
    if i0 > i1:
        return None

    def window(i: int) -> tuple[int, int]:
        clo, chi = c0, c0 + nc
        if i == ra:
            clo = max(clo, a - ra * C)
        if i == rb:
            chi = min(chi, b - rb * C)
        return clo, chi

    # (row, col_lo, nrows, ncols): first row, block of whole-window rows, last row
    pieces: list[tuple[int, int, int, int]] = []
    flo, fhi = window(i0)
    pieces.append((i0, flo, 1, fhi - flo))
    if i1 - i0 > 1:
        pieces.append((i0 + 1, c0, i1 - i0 - 1, nc))
    if i1 > i0:
        llo, lhi = window(i1)
        pieces.append((i1, llo, 1, lhi - llo))
    pieces = [p for p in pieces if p[3] > 0]
    if not pieces:
        return None
    row, clo, _n, _w = pieces[0]
    dst = ((row - r0) * nc + (clo - c0)) * isz
    length = sum(n * w for _r, _c, n, w in pieces) * isz
    segs = [BoxSegment(e0 + (r * C + c) * isz - s0, n, w * isz, C * isz) for r, c, n, w in pieces]
    return BoxRun(dst, length, _coalesce(segs))


def clip_segments(segments: tuple[BoxSegment, ...], lo: int, hi: int) -> tuple[BoxSegment, ...]:
    """The segments producing bytes ``[lo, hi)`` of the dense concatenation of ``segments``."""
    out: list[BoxSegment] = []
    pos = 0
    for s in segments:
        u0, u1 = max(lo, pos) - pos, min(hi, pos + s.nbytes) - pos
        pos += s.nbytes
        if u0 >= u1:
            continue
        rb = s.row_bytes
        j0, j1 = u0 // rb, (u1 - 1) // rb
        if j0 == j1:
            out.append(BoxSegment(s.src_offset + j0 * s.stride + (u0 - j0 * rb), 1, u1 - u0, s.stride))
            continue
        out.append(BoxSegment(s.src_offset + j0 * s.stride + (u0 - j0 * rb), 1, (j0 + 1) * rb - u0, s.stride))
        if j1 - j0 > 1:
            out.append(BoxSegment(s.src_offset + (j0 + 1) * s.stride, j1 - j0 - 1, rb, s.stride))
        out.append(BoxSegment(s.src_offset + j1 * s.stride, 1, u1 - j1 * rb, s.stride))
    return _coalesce(out)


def split_box_run(run: BoxRun, slots: list[ChunkSlot], name: str, box_key: Optional[Box]) -> list[BoxWriteSlice]:
    """Map a :class:`BoxRun` onto the rank-local layout's slots of ``name``."""
    out = []
    r_lo, r_hi = run.dst_offset, run.dst_offset + run.length
    for slot in slots:
        lo, hi = max(r_lo, slot.chunk_offset), min(r_hi, slot.end)
        if lo >= hi:
            continue
        out.append(
            BoxWriteSlice(
                bucket_id=slot.bucket_id,
                dst_offset=slot.offset + (lo - slot.chunk_offset),
                length=hi - lo,
                segments=clip_segments(run.segments, lo - r_lo, hi - r_lo),
                name=name,
                box_key=box_key,
            )
        )
    return out


def plan_rank_local_writes(
    entries_by_src: dict[str, list[ParamEntry]],
    src_ranges: dict[str, Optional[tuple[int, int]]],
    local_slots: BucketLayout | dict[str, list[ChunkSlot]],
    boxes: dict[str, tuple[tuple[int, ...], Box]],
    owns=None,
) -> dict[str, list[BoxWriteSlice]]:
    """One rank's rank-local plan for one generator, ``{src_name: [BoxWriteSlice]}``, from its
    :func:`normalize_slices` table ``boxes``. A box whose ``full_shape`` differs from the
    trainer's tensor raises."""
    slots_by_name = local_slots.slots_by_name if isinstance(local_slots, BucketLayout) else local_slots
    out: dict[str, list[BoxWriteSlice]] = {}
    for src, entries in entries_by_src.items():
        rng = src_ranges.get(src)
        if rng is None:
            continue
        ws: list[BoxWriteSlice] = []
        for e in entries:
            if owns is not None and not owns(e.name):
                continue
            box = _entry_box(e, boxes)
            run = plan_box_writes(e, box, rng)
            if run is not None:
                ws.extend(split_box_run(run, slots_by_name[e.name], e.name, box))
        if ws:
            out[src] = ws
    return out


def _entry_box(e: ParamEntry, boxes: dict[str, tuple[tuple[int, ...], Box]]) -> Optional[Box]:
    spec = boxes.get(e.name)
    if spec is None:
        return None
    shape, box = spec
    if tuple(shape) != tuple(e.shape):
        raise ValueError(f"rank-local slice {e.name!r}: full_shape {shape} != trainer shape {e.shape}")
    return None if box == full_box(e.shape) else box


def plan_rank_local_indexed_writes(
    entries_by_src: dict[str, list[ParamEntry]],
    runs_by_src: dict[str, list[list[tuple[int, int, int]]]],
    local_slots: BucketLayout | dict[str, list[ChunkSlot]],
    boxes: dict[str, tuple[tuple[int, ...], Box]],
    owns=None,
) -> dict[str, list[BoxWriteSlice]]:
    """:func:`plan_rank_local_writes` for HF-indexed exports (Megatron): ``runs_by_src`` holds each
    source's :func:`index_runs`, so segment offsets index the rank's compact values. Runs that land
    back to back in a box become one slice (row-parallel o_proj: one per box, not one per row)."""
    slots_by_name = local_slots.slots_by_name if isinstance(local_slots, BucketLayout) else local_slots
    out: dict[str, list[BoxWriteSlice]] = {}
    for src, entries in entries_by_src.items():
        ws: list[BoxWriteSlice] = []
        for e, runs in zip(entries, runs_by_src.get(src, [[]] * len(entries)), strict=True):
            if not runs or (owns is not None and not owns(e.name)):
                continue
            assert e.src_elem_offset == 0, e
            box, isz = _entry_box(e, boxes), e.itemsize
            groups: list[list] = []  # [dst_offset, length, segments]
            for hf, value, n in runs:
                run = plan_box_writes(e, box, (hf * isz, n * isz))
                if run is None:
                    continue
                segs = [replace(s, src_offset=s.src_offset + value * isz) for s in run.segments]
                if groups and groups[-1][0] + groups[-1][1] == run.dst_offset:
                    groups[-1][1] += run.length
                    groups[-1][2].extend(segs)
                else:
                    groups.append([run.dst_offset, run.length, segs])
            for dst, length, segs in groups:
                ws.extend(split_box_run(BoxRun(dst, length, _coalesce(segs)), slots_by_name[e.name], e.name, box))
        if ws:
            out[src] = ws
    return out


def merge_box_plans(plans: Iterable[dict[str, list[BoxWriteSlice]]]) -> dict[str, list[BoxWriteSlice]]:
    """Union of several generators' plans, each distinct piece (``BoxWriteSlice.key``) once (DP
    replicas and replicated k/v heads share pieces), sorted by bucket then destination."""
    seen: dict[str, dict[tuple, BoxWriteSlice]] = {}
    for plan in plans:
        for src, ws in plan.items():
            d = seen.setdefault(src, {})
            for w in ws:
                d.setdefault(w.key, w)
    return {
        src: sorted(d.values(), key=lambda w: (w.bucket_id, w.dst_offset, w.length, repr(w.box_key)))
        for src, d in seen.items()
    }


# ---------------------------------------------------------------------------
# Generator ownership filter (EP experts, PP layer range)
# ---------------------------------------------------------------------------

# decoder layers only (model.layers.N, model.language_model.layers.N, language_model.model.layers.N):
# a vision tower's encoder.layers.N is not pipeline-split with them
_LAYER_RE = re.compile(r"(?:^|\.)(?:model|language_model)\.layers\.(\d+)\.")
_EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")


def owned_name_filter(layout_info: Optional[RolloutLayout]):
    """``owns(hf_name) -> bool`` for one generator: every name when ``layout_info`` is ``None``;
    otherwise layers outside ``layer_range`` and experts not in ``local_experts`` are skipped, and
    names without a layer index (embeddings, final norm, lm_head) are always owned."""
    if layout_info is None:
        return lambda name: True
    layer_range = layout_info.layer_range
    local_experts = {int(k): set(v) for k, v in (layout_info.local_experts or {}).items()}

    def owns(name: str) -> bool:
        m = _LAYER_RE.search(name)
        if m is None:
            return True
        layer = int(m.group(1))
        if layer_range is not None and not (layer_range[0] <= layer < layer_range[1]):
            return False
        em = _EXPERT_RE.search(name)
        if em is not None and layer in local_experts:
            return int(em.group(1)) in local_experts[layer]
        return True

    return owns
