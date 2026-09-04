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
"""Delta weight-sync checkpoint engine (NCCL transport) for DISAGGREGATED rollout.

Puts the delta on the trainer->rollout wire: the trainer byte-diffs against a
pinned-CPU snapshot and broadcasts only the changed ``(position, value)`` pairs
over the same ``ray.util.collective`` NCCL group the full-weight
:class:`NCCLCheckpointEngine` uses (actor rank0 -> rollout CheckpointEngineWorkers).
With SGLang, each rollout worker hands its local copy of the sparse payload to
its colocated SGLang TP worker via same-GPU ``update_weights_from_tensor`` IPC,
where the verl-shipped :mod:`verl.workers.rollout.sglang_rollout.delta_loader`
(registered through SGLang's stock ``--custom-weight-loader`` hook — no SGLang
fork or patch needed) decodes and masked-applies it *in place* onto the live
weights. No full-model mirror is staged anywhere on the rollout side: receiver
peak memory is one bucket plus one decode chunk, independent of model size.

With vLLM, the rollout adapter forwards the same payload over same-GPU IPC to
VERL's registered weight-transfer backend. It decodes each flush into vLLM
checkpoint patches, whose loader reuses native ``model.load_weights()`` mapping.

The first (seed) sync streams the backend's FULL HF export (``get_per_tensor_
param()``) over the values-only wire -- every backend already knows how to
assemble and convert its own full tensors, so resume works by construction and
the seed inherits Megatron/veomni assembly for free. After the seed the caller
primes the backend's pinned shard snapshots; every later sync ships the
backend-computed sparse HF delta.

Data ladder (sender side, steady) -- the names in this file anchor to these::

    backend HF delta ENTRY (slots, dtype, counts, hf_idx, hf_val, group)
    --_GatherQueue--> per-SLOT delta on rank 0 --_bucket_*--> _FlushPiece
    --_FlushBucket--> FLUSH (DeltaFlush) --_publish_flush--> wire

Wire encodings: ``indices`` (fixed-width absolute positions + values; every
steady sync) and ``values`` (values only; the seed). The values meta tag is
``"dense"`` for protocol continuity with the receiver's delta_loader.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from unittest.mock import patch

import ray.util.collective as collective
import torch
import zmq

from verl.utils.fusion_groups import DEEPSEEK_V4_FUSION_GROUPS

with patch("importlib.metadata.distributions", return_value=[]):
    import cupy as cp

from .base import CheckpointEngineRegistry
from .delta_sync.encode import DeltaFlush, DeltaParam, absolute_index_width, pack_absolute_indices
from .delta_sync.encode import checksum as _checksum
from .delta_sync.sparse_gather import gather_slot_entries_to_rank0
from .nccl_checkpoint_engine import MasterMetadata, NCCLCheckpointEngine, WorkerMetadata

logger = logging.getLogger(__name__)


def _prodshape(shape) -> int:
    n = 1
    for x in shape:
        n *= int(x)
    return n


@dataclass(slots=True)
class _FlushPiece:
    """One (possibly sliced) per-parameter piece buffered for a pending indices flush."""

    name: str
    dtype_str: str
    shape: list
    idx: torch.Tensor
    val: torch.Tensor


@dataclass(slots=True)
class _ValuesPiece:
    """One whole parameter's flat values buffered for the seed sync's values-only flush."""

    name: str
    dtype_str: str
    shape: list
    flat: torch.Tensor


class _FlushBucket:
    """One-flush-lookahead bucket pipeline, shared by the steady loop and both
    seed streams. Pieces accumulate until ``cap`` bytes; ``seal`` assembles them
    into the single pending flush, first emitting the previous pending with
    ``is_last=False`` (the lookahead: only the caller's finale knows which flush
    is last and emits it with ``is_last=True``). ``assemble`` and ``publish``
    carry the only real differences between the streams -- the wire format
    (indexed flush vs values-only flush) and the flush counters."""

    __slots__ = ("cap", "pieces", "nbytes", "pending", "_assemble", "_publish")

    def __init__(self, cap: int, assemble, publish):
        self.cap = int(cap)
        self.pieces: list = []
        self.nbytes = 0
        self.pending = None
        self._assemble = assemble
        self._publish = publish

    def add(self, piece, nbytes: int) -> None:
        self.pieces.append(piece)
        self.nbytes += int(nbytes)
        if self.nbytes >= self.cap:
            self.seal()

    def add_atomic(self, sized_pieces: list[tuple]) -> None:
        """Add several pieces that must not be split across flushes.

        The cap check happens BEFORE the group goes in (seal the current bucket
        first if it would overflow), so the boundary can only fall between
        groups -- never inside one. A group larger than ``cap`` becomes its own
        oversized flush, which is correct if not ideal; the fused DSv4 params
        this exists for are a few MiB.
        """
        if not sized_pieces:
            return
        total = sum(int(nb) for _, nb in sized_pieces)
        if self.pieces and self.nbytes + total > self.cap:
            self.seal()
        for piece, nbytes in sized_pieces:
            self.pieces.append(piece)
            self.nbytes += int(nbytes)
        if self.nbytes >= self.cap:
            self.seal()

    def seal(self) -> None:
        if not self.pieces:
            return
        self.emit(is_last=False)
        self.pending = self._assemble(self.pieces)
        self.pieces, self.nbytes = [], 0

    def emit(self, is_last: bool) -> None:
        if self.pending is not None:
            self._publish(self.pending, is_last)
            self.pending = None


# Destination params that the ROLLOUT-side loader rebuilds by concatenating two
# separately-named checkpoint tensors. sglang's DSv4 loader buffers the halves in
# a cache created inside ``load_weights`` and asserts it empty on return, so both
# members have to arrive in the SAME call -- i.e. the same flush. Bucketing by
# bytes alone splits them sooner or later and the assert fires; that is not
# delta-specific, plain full NCCL sync hits it too.
#
# Suffixes are spelled out through ``.self_attn.`` on purpose: a bare
# ``.wkv.weight`` would also match ``.compressor.wkv.weight`` and
# ``.indexer.compressor.wkv.weight``. ``_FusionStager._match`` asserts a name
# never matches two groups, so an ambiguity introduced later fails loudly here
# rather than silently mis-grouping.
#
# fp8 splits into two groups because sglang keys its cache on the destination
# param name: ``wqkv_a.weight`` and ``wqkv_a.weight_scale_inv`` are separate
# entries and each needs its own pair.
# The attention block is spelled ``self_attn`` in the names that reach sglang
# (the loader's cache key ``model.layers.N.self_attn.wqkv_a.weight`` back-derives
# an incoming ``...self_attn.wq_a.weight``), but Megatron-Bridge's DSv4 mapping
# writes ``layers.N.attn.*``. Carry both rather than bet on which layer renames:
# the two are mutually exclusive under ``endswith`` (``self_attn`` has no dot
# before ``attn``), so listing both cannot create an ambiguity.
_FUSION_GROUPS = tuple((str(index), group) for index, group in enumerate(DEEPSEEK_V4_FUSION_GROUPS))


class _FusionStager:
    """Hold the members of a fused destination param until the group is complete,
    then release them together so they ride one flush.

    Two things have to be true at the receiver for a fused param to survive a
    sparse sync, and this covers both:

    * **completeness** -- a member with no changed elements is released as an
      EMPTY entry instead of being dropped. The receiver densifies it to an
      all-NaN full-shape tensor, and since ``_masked_copy`` keeps the
      destination wherever the source is NaN, cat-ing that half into the fused
      param is a no-op for it. (Verified: ``_decode_one`` already returns pure
      NaN for a zero-length entry, both in the fp8 byte path and the float path,
      so the receiver needs no change.)
    * **co-location** -- see ``_FlushBucket.add_atomic``; being complete does not
      help if the two halves land in different ``load_weights`` calls.

    Params outside any group pass straight through.
    """

    __slots__ = ("_pending", "n_groups", "n_filled")

    def __init__(self) -> None:
        self._pending: dict[tuple[str, str], dict] = {}
        self.n_groups = 0  # groups released with at least one changed member
        self.n_filled = 0  # halves materialised as all-NaN because nothing changed

    @staticmethod
    def _match(name: str):
        hits = [(key, sfx) for key, sfxs in _FUSION_GROUPS for sfx in sfxs if name.endswith(sfx)]
        assert len(hits) <= 1, f"{name!r} matches multiple fusion groups: {hits}"
        return hits[0] if hits else None

    def offer(self, name: str, dtype_str: str, shape, aidx, aval):
        """Return ``(entries, is_group)``, or ``None`` while a group is incomplete.

        A non-member yields itself with ``is_group=False`` -- the caller keeps
        dropping unchanged non-members, so this costs nothing for the ~99% of
        params that are not fused. A member yields ``None`` until its siblings
        arrive, then the whole group in declared order with ``is_group=True`` --
        or ``([], True)`` if no member of the group changed at all, since there
        is then nothing to send.
        """
        matched = self._match(name)
        if matched is None:
            return [(name, dtype_str, shape, aidx, aval)], False

        key, sfx = matched
        suffixes = next(s for k, s in _FUSION_GROUPS if k == key)
        slot = self._pending.setdefault((name[: -len(sfx)], key), {})
        assert sfx not in slot, f"duplicate fusion member {name!r} for group {key!r}"
        slot[sfx] = (name, dtype_str, shape, aidx, aval)
        if len(slot) < len(suffixes):
            return None

        self._pending.pop((name[: -len(sfx)], key))
        members = [slot[s] for s in suffixes]
        if all(e[3] is None or e[3].numel() == 0 for e in members):
            return [], True
        # Materialise the absent halves. Device/dtype come from a member that did
        # change, so the empties cat cleanly with the rest of the flush.
        donor = next(e for e in members if e[3] is not None and e[3].numel())
        dev = donor[3].device
        out = []
        for m_name, m_dtype, m_shape, m_idx, m_val in members:
            if m_idx is None or m_idx.numel() == 0:
                m_idx = torch.empty(0, dtype=torch.int32, device=dev)
                m_val = torch.empty(0, dtype=getattr(torch, m_dtype), device=dev)
                self.n_filled += 1
            out.append((m_name, m_dtype, m_shape, m_idx, m_val))
        self.n_groups += 1
        return out, True

    def offer_piece(self, name: str, piece, nbytes: int):
        """Seed-path variant: co-locate a group's members, nothing else.

        A full export contains every member by construction, so there is no
        absent half to materialise -- only the flush boundary matters. Returns
        ``(list_of_(piece, nbytes), is_group)`` or ``None`` while incomplete.
        """
        matched = self._match(name)
        if matched is None:
            return [(piece, nbytes)], False
        key, sfx = matched
        suffixes = next(s for k, s in _FUSION_GROUPS if k == key)
        slot = self._pending.setdefault((name[: -len(sfx)], key), {})
        assert sfx not in slot, f"duplicate fusion member {name!r} for group {key!r}"
        slot[sfx] = (piece, nbytes)
        if len(slot) < len(suffixes):
            return None
        self._pending.pop((name[: -len(sfx)], key))
        self.n_groups += 1
        return [slot[s] for s in suffixes], True

    def assert_drained(self) -> None:
        assert not self._pending, (
            f"fusion groups never completed: {sorted(self._pending)}. Every member listed in "
            f"_FUSION_GROUPS must appear in the export stream, including unchanged ones."
        )


def _slice_pieces(name: str, dtype_str: str, shape, aidx: torch.Tensor, aval: torch.Tensor) -> list[tuple]:
    """Slice one param's (idx, val) delta into <= MAX_ENTRY_ELEMS ``(piece, nbytes)``
    pairs (bounds the receiver-side decode transient; the masked apply is sequential,
    so splitting is transparent). Bucket bytes = actual wire bytes (int32 positions
    + values).

    An empty delta yields ONE empty piece rather than none: a zero-length range()
    would emit nothing, but fusion-group members with no changed elements must
    still reach the receiver so it can densify them to all-NaN (see _FusionStager).
    """
    if aidx.numel() == 0:
        return [(_FlushPiece(name, dtype_str, list(shape), aidx, aval), 0)]
    max_elems = DeltaShardedCheckpointEngine.MAX_ENTRY_ELEMS
    out = []
    for s in range(0, aidx.numel(), max_elems):
        e = min(s + max_elems, aidx.numel())
        out.append(
            (
                _FlushPiece(name, dtype_str, list(shape), aidx[s:e], aval[s:e]),
                (e - s) * (4 + aval.element_size()),
            )
        )
    return out


def _bucket_sliced(
    bkt: _FlushBucket, name: str, dtype_str: str, shape, aidx: torch.Tensor, aval: torch.Tensor
) -> None:
    for piece, nbytes in _slice_pieces(name, dtype_str, shape, aidx, aval):
        bkt.add(piece, nbytes)


class _GatherQueue:
    """Per-gather-group batching of slot-keyed queue entries
    ``(slots, dtype_str, counts, idx, val)``. Entries carry FINAL-coordinate
    payloads (identity specs: one slot = the param itself; converter specs: the
    spec's hf_slots), so rank 0 never converts -- ``consume`` receives assembled
    per-slot pieces straight from the gather.

    One queue per ProcessGroup: separate queues stop pg alternation (dense fsdp
    group vs expert world group per layer) from shattering batches. The flush
    trigger is COUNT-ONLY: entry counts are identical on every rank while byte
    totals are not, so a count trigger is the only one that keeps the collective
    sequence identical across ranks (a per-rank byte trigger desyncs the gathers
    and deadlocks NCCL). Byte bounding happens INSIDE the batched gather via
    ``max_round_bytes``, decided from the all-gathered counts every rank sees."""

    __slots__ = ("batch_k", "max_round_bytes", "is_r0", "_consume", "_queues")

    def __init__(self, batch_k: int, max_round_bytes: int, is_r0: bool, consume):
        self.batch_k = max(int(batch_k), 1)
        self.max_round_bytes = int(max_round_bytes)
        self.is_r0 = is_r0
        self._consume = consume
        self._queues: dict[int, tuple] = {}  # id(pg) -> (pg, [entries])

    def put(self, pg, slots: list, dtype_str: str, counts: torch.Tensor, idx: torch.Tensor, val: torch.Tensor):
        # one queue per (group, value dtype): batches concatenate values, so a
        # batch must be dtype-homogeneous (fp8 codes / fp32 scales / bf16 mix
        # under quant mode). Entry order and dtypes are identical on every
        # rank, so the partition stays in lockstep.
        _pg, entries = self._queues.setdefault((id(pg), val.dtype), (pg, []))
        entries.append((slots, dtype_str, counts, idx, val))
        if len(entries) >= self.batch_k:
            self._flush(pg, entries)

    def flush_all(self) -> None:
        for pg, entries in self._queues.values():
            self._flush(pg, entries)

    def _flush(self, pg, entries: list) -> None:
        """One gather round for one group's queue."""
        if not entries:
            return
        batch = list(entries)
        entries.clear()
        if pg is None:
            # unsharded/replicated params: rank 0's local delta already is global
            if self.is_r0:
                for slots, dtype_str, counts, idx, val in batch:
                    off = 0
                    for (name, shape), c in zip(slots, counts.tolist(), strict=True):
                        self._consume(
                            name, dtype_str, tuple(shape), _prodshape(shape), idx[off : off + c], val[off : off + c]
                        )
                        off += c
            return
        dev = batch[0][3].device
        counts_concat = torch.cat([c for _, _, c, _, _ in batch]).to(dev)
        idx_concat = torch.cat([i for _, _, _, i, _ in batch])
        val_concat = torch.cat([v for _, _, _, _, v in batch])
        gathered = gather_slot_entries_to_rank0(
            idx_concat, val_concat, counts_concat, group=pg, max_round_bytes=self.max_round_bytes
        )
        if self.is_r0 and gathered is not None:
            slot_i = 0
            for slots, dtype_str, _counts, _i, _v in batch:
                for name, shape in slots:
                    aidx, aval = gathered[slot_i]
                    slot_i += 1
                    self._consume(name, dtype_str, tuple(shape), _prodshape(shape), aidx, aval)


@CheckpointEngineRegistry.register("delta_sharded")
class DeltaShardedCheckpointEngine(NCCLCheckpointEngine):
    """Sparse delta weight sync over NCCL, diffed on each rank's local shard.

    Reuses NCCLCheckpointEngine's group/zmq machinery but moves only changed
    positions+values: each actor rank keeps a pinned-CPU snapshot of only *its*
    shard, byte-diffs the shard, and only the changed ``(position, value)`` pairs
    are gathered to rank 0 and streamed to the rollout side -- no rank ever holds
    a full-model snapshot.

    ``send_weights`` takes the TRAINING ENGINE and drives the sync itself: the
    seed (first sync) streams the backend's full ``get_per_tensor_param()``
    export values-only and pins the diff base (``prime_delta_snapshots``); every
    steady sync consumes the backend's HF delta export
    ``get_per_tensor_param_delta_shard()`` — per-parameter FINAL-HF-coordinate
    entries ``(slots, dtype_str, counts, hf_idx, hf_val, gather_group)``. Naming,
    to-HF conversion, diff and snapshot all live on the backend side (see
    :mod:`verl.workers.engine.utils`); this engine only batches, gathers,
    buckets and ships, and so serves any backend that can produce HF deltas.
    """

    # Cap on changed elements per DeltaParam entry. The receiver-side decode
    # densifies per entry with an int64 index transient (8 B/element), so an
    # uncapped entry (e.g. a 7B model's whole embedding on the full seed, ~545M
    # elements) would spike several GiB at once. Oversized per-param deltas are
    # sliced into multiple entries (the masked apply is sequential, so splitting
    # is transparent); 64M elements bounds the transient to ~512 MiB.
    MAX_ENTRY_ELEMS = 64 << 20

    wire_format = "delta_flush"

    def prepare(self) -> WorkerMetadata:
        # Delta broadcasts small per-flush buffers directly, so skip the parent's
        # 2 * bucket_size fixed buffers. Still hand back the master zmq endpoint
        # that build_topology() distributes to the rollout workers.
        #
        # Only rank 0 broadcasts here (see the assert in send_weights) and this engine has no relay
        # path, so it always takes the parent's single-sender topology.
        master = (
            MasterMetadata(zmq_ip=self.ip, zmq_port=self.listen_port, multi_sender=False) if self.is_master else None
        )
        return WorkerMetadata(node_id=self.get_node_id(), master=master)

    # ---- trainer side ----
    # ---- shared STREAMING wire ----
    # Stream each flush as soon as it is produced so trainer peak memory stays near two buckets,
    # including during the full seed. Each ZMQ manifest + NCCL broadcast carries ``is_last``;
    # rollout adapters then forward that flush over same-GPU IPC to their backend-specific consumer.
    def _publish_flush(self, flush: DeltaFlush, first: bool, is_last: bool) -> None:
        meta = {
            "is_full": first,
            "encoding": self.encoding,
            "is_last": is_last,
            "terminal_empty": False,
            "pos_numel": int(flush.positions_cpu.numel()),
            "val_numel": int(flush.values_gpu.numel()),
            "val_dtype": str(flush.values_gpu.dtype).replace("torch.", ""),
            "spec": {
                "encoding": self.encoding,
                "values_bytes": self.quantize_fp8,
                # sparse flushes carry the quant config too: the receiver's
                # handshake (incl. the seed-required sentinel guard) must be
                # reachable on the steady path, not only on the dense seed.
                "quant_config": getattr(self, "_fp8_quant_cfg", None),
                "params": [vars(p) for p in flush.params],
                "checksum": int(flush.checksum),
            },
        }
        self.socket.send_string(self.topic, flags=zmq.SNDMORE)
        self.socket.send_pyobj(meta)
        pos_u8 = flush.positions_cpu.to("cuda", non_blocking=True).contiguous().view(torch.uint8)
        val_u8 = flush.values_gpu.contiguous().view(torch.uint8)
        # Stage into cupy-owned buffers: ray's NCCL broadcast is enqueued on a separate
        # stream with no recordStream on its inputs, so broadcasting a zero-copy view of
        # these torch tensors (freed right after this call) would race with allocator reuse.
        pos_cp = cp.empty(pos_u8.numel(), dtype=cp.uint8)
        val_cp = cp.empty(val_u8.numel(), dtype=cp.uint8)
        pos_cp[:] = cp.asarray(pos_u8)
        val_cp[:] = cp.asarray(val_u8)
        collective.broadcast(pos_cp, src_rank=0, group_name=self.group_name)
        collective.broadcast(val_cp, src_rank=0, group_name=self.group_name)

    def _publish_values_flush(
        self,
        params: list[DeltaParam],
        values: torch.Tensor,
        is_last: bool,
        verify: bool = False,
        values_bytes: bool = False,
    ) -> None:
        """Publish a values-only (full-coverage, positions-free) flush -- used by the first
        sync. The wire encoding tag stays ``"dense"`` -- it is protocol, shared with the
        receiver's delta_loader decode."""
        values = values.contiguous()
        empty_pos = torch.empty(0, dtype=torch.uint8, device=values.device)
        meta = {
            "is_full": True,
            "encoding": "dense",
            "is_last": is_last,
            "terminal_empty": False,
            "pos_numel": 0,
            "val_numel": int(values.numel()),
            "val_dtype": str(values.dtype).replace("torch.", ""),
            "spec": {
                "encoding": "dense",
                "verify": verify,
                "is_last": is_last,
                "values_bytes": values_bytes,
                "quant_config": getattr(self, "_fp8_quant_cfg", None),
                "params": [vars(p) for p in params],
                "checksum": int(_checksum(empty_pos, values)),
            },
        }
        self.socket.send_string(self.topic, flags=zmq.SNDMORE)
        self.socket.send_pyobj(meta)
        val_u8 = values.view(torch.uint8)
        # cupy-owned staging: same lifetime rationale as _publish_flush.
        val_cp = cp.empty(val_u8.numel(), dtype=cp.uint8)
        val_cp[:] = cp.asarray(val_u8)
        collective.broadcast(val_cp, src_rank=0, group_name=self.group_name)

    def _release_staging_pool(self, phase: str) -> None:
        """Return the cupy staging pool's blocks to CUDA and log the evidence:
        ``held`` is what the pool would have kept from the device without this
        release, and the device-free delta shows the memory actually coming
        back (warning level so the default worker log level records it)."""
        from verl.utils.device import get_torch_device

        pool = cp.get_default_memory_pool()
        held = pool.total_bytes()
        free_before, _ = get_torch_device().mem_get_info()
        pool.free_all_blocks()
        free_after, _ = get_torch_device().mem_get_info()
        logger.warning(
            "cupy staging pool after %s send: held %.2fGB; device free %.2f->%.2fGB on release",
            phase,
            held / (1 << 30),
            free_before / (1 << 30),
            free_after / (1 << 30),
        )

    def _publish_terminal(self, first: bool) -> None:
        """End-of-stream marker when zero flushes were produced (no broadcast, just a signal)."""
        meta = {"is_full": first, "encoding": self.encoding, "is_last": True, "terminal_empty": True}
        self.socket.send_string(self.topic, flags=zmq.SNDMORE)
        self.socket.send_pyobj(meta)

    # ---- rollout worker side ----
    def receive_weights(self, global_steps: int | None = None) -> Iterator[tuple[list[tuple[str, torch.Tensor]], bool]]:
        """Yield delta flushes for the rollout adapter to consume.

        Each ``(named_tensors, is_last)`` item contains a JSON manifest, optional
        int32 position bytes, and patch values. The generator keeps at most one
        received GPU flush live at a time.
        """
        assert self.rank > 0, "Rank 0 should not receive weights."
        applied = 0
        while True:
            self.socket.recv_string()
            meta = self.socket.recv_pyobj()
            if meta.get("terminal_empty"):
                break

            dense = meta.get("encoding") == "dense"
            val_dtype = getattr(torch, meta["val_dtype"])
            elem = torch.empty(0, dtype=val_dtype).element_size()
            val_u8 = torch.empty(meta["val_numel"] * elem, dtype=torch.uint8, device="cuda")
            if dense:
                pos = None
                collective.broadcast(val_u8, src_rank=0, group_name=self.group_name)
            else:
                pos = torch.empty(meta["pos_numel"], dtype=torch.uint8, device="cuda")
                collective.broadcast(pos, src_rank=0, group_name=self.group_name)
                collective.broadcast(val_u8, src_rank=0, group_name=self.group_name)
            val = val_u8.view(val_dtype)
            spec_bytes = json.dumps(meta["spec"]).encode()
            spec_t = torch.frombuffer(bytearray(spec_bytes), dtype=torch.uint8).to("cuda")
            named = [("__delta_spec__", spec_t), ("__values__", val)]
            if pos is not None:
                named.insert(1, ("__positions__", pos))
            is_last = bool(meta["is_last"])
            yield named, is_last
            applied += 1
            del pos, val_u8, val, spec_t
            if is_last:
                break
        logger.info("delta recv v=%s flushes=%d (yielded to server adapter)", global_steps, applied)

    def __init__(
        self,
        *args,
        encoding: str = "indices",
        batch_gather: int = 32,
        verify_every: int = 0,
        verify_seed: bool = False,
        quantize_fp8: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert encoding == "indices", f"delta_sharded ships only the 'indices' position encoding; got {encoding!r}"
        self.encoding = encoding
        # SGLang supports verify_every > 0; vLLM rejects it at startup.
        self.verify_every = int(verify_every)
        self.verify_seed = bool(verify_seed)
        # fp8 rollout mode: quantize on the trainer and ship the rollout's
        # exact state (fp8 codes + blockwise scale_inv tensors). Currently
        # full-resync per sync (the quant-domain sparse steady path lands
        # next); the wire is already half the bf16 bytes per element.
        self.quantize_fp8 = bool(quantize_fp8)
        self._shard_seeded = False
        # Gather the per-param sparse deltas in groups of this many parameters
        # (one count-matrix all_gather + two padded gathers per group instead of
        # three collectives per parameter).
        self.batch_gather = int(batch_gather)

    def _verify_due(self) -> bool:
        """True on every K-th steady sync (``verify_every=K``), and ALWAYS on the
        first steady sync -- the runtime fuse must exist even when periodic
        verification is off (verify_every=0 keeps only that one mandatory sweep)."""
        self._steady_count = getattr(self, "_steady_count", 0) + 1
        if self._steady_count == 1:
            return True
        if self.verify_every <= 0:
            return False
        return self._steady_count % self.verify_every == 0

    def _fp8_spec(self, engine=None):
        """Distill the serving engine's quant config into a rollout-agnostic
        :class:`~verl.utils.fp8_sharded.QuantSpec` for the backend."""
        from verl.utils.fp8_sharded import QuantSpec

        h = self._fp8_helper(engine)
        scale_fmt = h.quant_config.get("scale_fmt")
        # ue8m0 checkpoints carry per-block headroom that is unrecoverable from
        # the dequantized master; hand the quantizers the checkpoint's own
        # scales so unchanged blocks reproduce the checkpoint's bytes exactly.
        ckpt_scales = None
        ckpt_path = getattr(getattr(engine, "model_config", None), "local_path", None)
        if scale_fmt == "ue8m0":
            if ckpt_path:
                from verl.utils.fp8_sharded import load_ckpt_scales

                ckpt_scales = load_ckpt_scales(ckpt_path)
            else:
                raise ValueError("ue8m0 delta sync requires model_config.local_path to read checkpoint scales")
        # fp32 wire fidelity for the checkpoint's non-quantized fp32 families
        # (DSv4: hc_*, ape, attn_sink, e_score_correction_bias). Header-only
        # read, memoised per process like the fp8 predicate.
        fp32_predicate = None
        if ckpt_path:
            from verl.utils.fp8_ckpt_dtypes import build_ckpt_fp32_predicate

            fp32_predicate = build_ckpt_fp32_predicate(ckpt_path)
        return QuantSpec(
            weight_block_size=tuple(h.quant_config.get("weight_block_size", [128, 128])),
            should_quantize=self._quant_predicate(h, ckpt_path),
            scale_fmt=scale_fmt,
            ckpt_scales=ckpt_scales,
            fp32_predicate=fp32_predicate,
        )

    def _quant_predicate(self, helper, checkpoint_path: str | None):
        """Select quantized tensors from checkpoint metadata when available."""
        from verl.utils.fp8_ckpt_dtypes import build_ckpt_fp8_predicate

        if not checkpoint_path:
            logger.warning(
                "fp8 selection: no model path is available; falling back to the name allowlist"
            )
            return helper.should_quantize_param
        pred = build_ckpt_fp8_predicate(checkpoint_path)
        if pred is None:
            logger.warning(
                "fp8 selection: checkpoint at %s could not answer; using the name allowlist", checkpoint_path
            )
            return helper.should_quantize_param
        logger.warning("fp8 selection: using CHECKPOINT dtypes from %s (not the name allowlist)", checkpoint_path)
        return pred

    def _fp8_helper(self, engine=None):
        """Build the quantizer helper from the SAME inputs the rollout used --
        the model's hf_config (real ignored_layers / modules_to_not_convert)
        -- instead of guessing bare defaults, and guard the supported mode."""
        from verl.utils.sglang.sglang_fp8_utils import SGLangFP8QuantizerHelper, build_sglang_fp8_quant_config

        h = getattr(self, "_fp8_helper_inst", None)
        if h is None:
            hf_config = None
            if engine is not None:
                model_config = getattr(engine, "model_config", None)
                hf_config = getattr(model_config, "hf_config", None)
            if hf_config is None:
                logger.warning(
                    "fp8 delta: no hf_config available; quant config built from bare defaults "
                    "(ignored_layers/modules_to_not_convert from the checkpoint will be missed)"
                )
            cfg = build_sglang_fp8_quant_config(hf_config)
            # supported-mode guards: this engine ships blockwise fp8 with a
            # plain fp32 scale_inv grid, nothing else.
            assert cfg.get("quant_method") == "fp8", f"unsupported quant_method {cfg.get('quant_method')!r}"
            if cfg.get("weight_block_size") is None:
                raise NotImplementedError("per-tensor fp8 is not supported by the delta engine (blockwise only)")
            # ue8m0 is handled (both quantizers switch on QuantSpec.scale_fmt);
            # anything else is still a hard stop.
            sf = cfg.get("scale_fmt")
            assert sf in (None, "ue8m0"), f"unsupported scale_fmt {sf!r} in quant config: {cfg}"
            assert "deepgemm" not in cfg, f"unsupported scale format flag 'deepgemm' in quant config: {cfg}"
            self._fp8_quant_cfg = cfg
            h = SGLangFP8QuantizerHelper(cfg)
            self._fp8_helper_inst = h
        return h

    def _assemble_flush(self, per_param: list[_FlushPiece]) -> DeltaFlush:
        """Build one DeltaFlush (indices encoding) from rank 0's gathered per-param deltas.

        ``per_param``: :class:`_FlushPiece` entries whose ``idx`` are within-parameter
        flat positions (== what the receiver decodes).

        Positions stay on the GPU end to end (int32 pieces -> one cat -> uint8 view);
        the wire broadcasts from the GPU anyway, and a host round-trip here
        (``.cpu().numpy().tobytes()`` + join) dominated the whole send at scale
        (~2.4s/sync at 7B steady state, ~83s on the full seed).
        """
        bytes_wire = self.quantize_fp8  # mixed dtypes (fp8 codes + fp32 scales + bf16)
        idx_pieces: list[torch.Tensor] = []
        val_pieces: list[torch.Tensor] = []
        params: list[DeltaParam] = []
        pos_off = val_off = 0
        for piece in per_param:
            nnz = int(piece.idx.numel())
            # 24-bit absolute indices cover most model tensors while keeping
            # the format fixed-width and branch-free on decode. Larger tensors
            # retain the established int32 layout.
            param_numel = _prodshape(piece.shape)
            pos_width = absolute_index_width(param_numel)
            idx_pieces.append(pack_absolute_indices(piece.idx, pos_width))
            val = piece.val.contiguous().view(torch.uint8) if bytes_wire else piece.val
            val_pieces.append(val)
            n_val = int(val.numel())  # elements, or bytes in bytes_wire mode
            params.append(
                DeltaParam(
                    name=piece.name,
                    dtype=piece.dtype_str,
                    shape=list(piece.shape),
                    pos_start=pos_off,
                    pos_end=pos_off + nnz * pos_width,
                    pos_width=pos_width,
                    val_start=val_off,
                    val_end=val_off + n_val,
                )
            )
            pos_off += nnz * pos_width
            val_off += n_val

        values_gpu = torch.cat(val_pieces) if val_pieces else torch.empty(0, dtype=self.rollout_dtype, device="cuda")
        positions_u8 = (
            torch.cat(idx_pieces).contiguous().view(torch.uint8)
            if idx_pieces
            else torch.empty(0, dtype=torch.uint8, device=values_gpu.device)
        )
        cks = _checksum(positions_u8, values_gpu)
        return DeltaFlush(
            encoding=self.encoding, params=params, positions_cpu=positions_u8, values_gpu=values_gpu, checksum=cks
        )

    def _seed_verify_sweep(self, engine, spec, global_steps: int | None = None) -> None:
        """Verify sweep straight after the seed, inside the seed's receive
        session (the seed held is_last for us). Collective on every rank: the
        full export assembles per tensor. Same producer as the steady path's
        sweep, so the two sweeps judge seed and steady on identical terms."""
        if spec is not None:
            full, _ = engine.get_per_tensor_param(quant_spec=spec)
            self._send_full_seed(
                full,
                global_steps,
                verify=True,
                bytes_wire=True,
                fp32_predicate=getattr(spec, "fp32_predicate", None),
            )
        else:
            full, _ = engine.get_per_tensor_param()
            self._send_full_seed(full, global_steps, verify=True)

    def _send_full_seed_sharded(
        self, engine, spec, global_steps: int | None = None, hold_last: bool = False
    ) -> dict[str, float] | None:
        """Seed from the STEADY shard stream over the values-only wire.

        One producer for the first and every later sync: the shard stream's
        comm-stubbed mapping transforms and sticky-ue8m0 quantizer make the
        tensors, a dense per-record gather (values only, sequential P2P)
        assembles them on rank 0, and the pairs feed the SAME values-only
        bucketing as the legacy seed -- so the receiver sees a wire format it
        has always known. Snapshots are primed inline from the very flats that
        shipped, so the next steady diff base equals the shipped state by
        construction and the separate prime pass disappears.
        """
        from verl.checkpoint_engine.delta_sync.sparse_gather import dense_gather_group
        from verl.utils.device import is_cuda_available

        gen, _ = engine.get_per_tensor_param_shard(quant_spec=spec)
        engine._delta_shard_snap = getattr(engine, "_delta_shard_snap", {})
        snaps = engine._delta_shard_snap

        def pairs():
            meta = None
            for name, flat, sspec in gen:
                flat = flat.detach().contiguous().view(-1)
                # prime the steady diff base inline (same layout/pinning as
                # prime_delta_snapshots)
                snap = snaps.get(name)
                if snap is None or snap.numel() != flat.numel():
                    snap = torch.empty_like(flat, device="cpu", pin_memory=is_cuda_available)
                    snaps[name] = snap
                snap.copy_(flat, non_blocking=True)
                if meta is None:
                    meta = engine._quant_group_meta
                slots, sizes, dtype_str = meta[name]
                # replicas do not contribute: zero their size vector so the
                # gather's exactly-one-owner assert sees the true ownership map
                sizes_eff = sizes if sspec.contributes else [0] * len(sizes)
                pieces = dense_gather_group(flat, sizes_eff, sspec.gather_group)
                if pieces is None:
                    continue
                dtype = getattr(torch, dtype_str)
                for (slot_name, shape), piece in zip(slots, pieces, strict=True):
                    yield slot_name, piece.view(dtype).reshape(shape)

        return self._send_full_seed(
            pairs(),
            global_steps,
            bytes_wire=True,
            fp32_predicate=getattr(spec, "fp32_predicate", None),
            hold_last=hold_last,
        )

    def _send_full_seed(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
        verify: bool = False,
        bytes_wire: bool = False,
        fp32_predicate=None,
        hold_last: bool = False,
    ) -> dict[str, float] | None:
        """First sync: stream the backend's FULL HF export over the values-only wire.

        ``weights`` is ``get_per_tensor_param()`` -- every backend already knows how
        to assemble and convert its own full tensors (FSDP all-gather, veomni expert
        restack, Megatron TP/PP fusion), so the seed inherits all of that for free
        and this engine only buckets and broadcasts. Every trainer rank iterates the
        generator (the per-tensor assembly is collective); rank 0 buckets. Resume
        works by construction: whatever the trainer restored is what ships."""
        is_r0 = self.is_master
        t0 = time.time()
        n_flushes = 0
        total_elems = 0
        wire_bytes = 0

        def _assemble_values(pieces: list[_ValuesPiece]):
            params = []
            val_off = 0
            for piece in pieces:
                n = int(piece.flat.numel())
                params.append(
                    DeltaParam(
                        name=piece.name,
                        dtype=piece.dtype_str,
                        shape=list(piece.shape),
                        pos_start=0,
                        pos_end=0,
                        pos_width=4,
                        val_start=val_off,
                        val_end=val_off + n,
                    )
                )
                val_off += n
            return params, torch.cat([piece.flat for piece in pieces])

        def _publish_values(pending, is_last: bool) -> None:
            nonlocal n_flushes, wire_bytes
            params, values = pending
            self._publish_values_flush(params, values, is_last=is_last, verify=verify, values_bytes=bytes_wire)
            n_flushes += 1
            wire_bytes += int(values.nbytes)

        bkt = _FlushBucket(self.bucket_size, _assemble_values, _publish_values)
        # The seed streams the FULL export, so every fused member is present --
        # but byte bucketing splits pairs here exactly like it does in the steady
        # path, and the seed is the FIRST sync, so without this the run dies
        # before the steady staging is ever exercised.
        stager = _FusionStager()

        seen_names: set = set()
        for name, tensor in weights:
            # duplicate names in a full export mean two source params mapped to
            # the same HF tensor -- the receiver would apply whichever came last
            # and the delta diff base would silently disagree with the trainer
            # (observed: NemotronH A_log). Fail loud instead.
            assert name not in seen_names, f"full export yields duplicate HF tensor {name!r}"
            seen_names.add(name)
            tensor = tensor.detach()
            # fp8 codes and their fp32 scale_inv tensors ARE the rollout state:
            # never fold them into the bf16 wire dtype.
            # fp32 sources stay fp32: DSv4 stores its sensitive special params
            # (hyper-connection coefficients, ape tables, attention sinks) in
            # fp32, and folding them to the bf16 wire dtype silently costs 16
            # mantissa bits (measured: rel err p50 1.35e-3, max 3.9e-3 across
            # all five families) -- a train/serve fidelity gap the verify sweep
            # cannot see, because the replay folds identically. These tensors
            # total ~68 MB fp32 against a 267 GB seed.
            keep_dtype = (
                tensor.element_size() == 1
                or name.endswith("_scale_inv")
                or (fp32_predicate is not None and fp32_predicate(name))
            )
            if tensor.is_floating_point() and tensor.dtype != self.rollout_dtype and not keep_dtype:
                tensor = tensor.to(self.rollout_dtype)
            if not is_r0:
                del tensor
                continue
            flat = tensor.contiguous().reshape(-1)
            total_elems += int(flat.numel())
            if bytes_wire:
                # mixed-dtype flushes: pack every piece as raw bytes; offsets
                # in the spec become BYTE offsets and the receiver reinterprets
                # per-param via ``values_bytes``.
                flat = flat.view(torch.uint8)
            offered = stager.offer_piece(
                name,
                _ValuesPiece(name, str(tensor.dtype).replace("torch.", ""), list(tensor.shape), flat),
                flat.nbytes,
            )
            if offered is None:
                continue
            released, is_group = offered
            if is_group:
                bkt.add_atomic(released)
            else:
                for piece, nbytes in released:
                    bkt.add(piece, nbytes)

        if not is_r0:
            return
        stager.assert_drained()
        logger.info("seed fusion staging: groups=%d", stager.n_groups)
        bkt.seal()
        # hold_last: a verify sweep follows INSIDE this same receive session,
        # so its finale -- not ours -- carries is_last (the steady+sweep
        # contract: only the sync's final flush terminates the stream).
        if bkt.pending is not None:
            bkt.emit(is_last=not hold_last)
        elif not hold_last:
            self._publish_terminal(True)
        # warning level on purpose: worker default log level swallows info, and the
        # one-off seed cost is the number people ask for when sizing a run.
        # the cupy staging pool does not return its blocks to CUDA on its own;
        # after streaming up to 2x bucket_size through it, give the memory back
        # so the trainer's next optimizer/forward pass can use it (raw cudaMalloc
        # OOMs on tight mcore shapes otherwise).
        self._release_staging_pool("seed")
        logger.warning(
            "delta-sharded FULL-%s v=%s done in %.1fs (flushes=%d elems=%d wire=%.1fGB)",
            "VERIFY" if verify else "SEED",
            global_steps,
            time.time() - t0,
            n_flushes,
            total_elems,
            wire_bytes / (1 << 30),
        )
        if not total_elems:
            return None
        return {
            "checkpoint_engine/changed_ratio": 1.0,
            "checkpoint_engine/changed_elems": float(total_elems),
            "checkpoint_engine/payload_mbytes": wire_bytes / (1 << 20),
            "checkpoint_engine/flushes": float(n_flushes),
        }

    async def send_weights(
        self,
        engine,
        global_steps: int | None = None,
    ) -> dict[str, float] | None:
        """Drive one weight sync from the TRAINING ENGINE (unlike the full-sync
        engines, which consume a weights iterator): the seed/steady phase choice
        and the snapshot prime are this engine's own state machine, so the worker
        stays delta-agnostic. The seed (first sync) streams the backend's full
        ``get_per_tensor_param()`` export over the values-only wire, then pins the
        diff base via ``engine.prime_delta_snapshots()``; every later sync consumes
        ``get_per_tensor_param_delta_shard()`` (backend-computed final-HF deltas).
        """
        # All actor ranks participate (gather-v is collective); only torch rank 0 broadcasts.
        # rank 0 accumulates the gathered per-param deltas into bucket_size-sized flushes and streams
        # each one as soon as it fills (then frees it), so peak memory is ~2 buckets rather than the
        # whole model.
        assert self.rank <= 0, "Trainer workers other than rank 0 should not send weights."
        seeding = not self._shard_seeded
        # The quantized seed comes from the SAME shard stream as every steady delta -- same mapping
        # transforms, same sticky quantizer -- and travel the values-only wire
        # via a dense gather (no positions exist at any point; the sparse
        # transport's per-element position staging is unnecessary at 100% coverage).
        # ``verify_seed`` appends a dense receiver comparison before training begins.
        verify_after_seed = self.verify_seed
        if seeding and self.quantize_fp8:
            spec = self._fp8_spec(engine)
            metrics = self._send_full_seed_sharded(engine, spec, global_steps, hold_last=verify_after_seed)
            # The seed is complete only after every quantized shard has shipped
            # and its baseline snapshot has been captured.
            self._shard_seeded = True
            if verify_after_seed:
                self._seed_verify_sweep(engine, spec, global_steps)
            return metrics
        if seeding:
            # BF16 seed: stream the bridge's full export over the values-only
            # wire, then prime the sharded snapshots.
            full, _ = engine.get_per_tensor_param()
            metrics = self._send_full_seed(full, global_steps, hold_last=verify_after_seed)
            engine.prime_delta_snapshots()
            # The seed is complete only after its baseline snapshots are ready.
            self._shard_seeded = True
            if verify_after_seed:
                self._seed_verify_sweep(engine, None, global_steps)
            return metrics
        # the BACKEND owns delta production for every dtype: with a quant spec
        # it yields quant-domain entries (codes + scale grids diffed against
        # engine-held snapshots), without one the bf16 shard deltas.
        _spec = self._fp8_spec(engine) if self.quantize_fp8 else None
        weights, _ = engine.get_per_tensor_param_delta_shard(quant_spec=_spec)
        is_r0 = self.is_master
        n_flushes = 0
        changed_elems = 0
        total_elems = 0
        wire_bytes = 0

        def _publish_steady(flush, is_last: bool) -> None:
            nonlocal n_flushes
            self._publish_flush(flush, first=False, is_last=is_last)
            n_flushes += 1

        bkt = _FlushBucket(self.bucket_size, self._assemble_flush, _publish_steady)
        stager = _FusionStager()

        batch_k = self.batch_gather

        def _bucket_slot_delta(
            name: str,
            dtype_str: str,
            full_shape: tuple,
            full_numel: int,
            aidx: torch.Tensor | None,
            aval: torch.Tensor | None,
        ) -> None:
            nonlocal total_elems, changed_elems, wire_bytes
            total_elems += int(full_numel)
            # Members of a fused destination param are held back until the group
            # is whole, then emitted as one indivisible run of pieces. Everything
            # else falls through unchanged. Note the accounting below runs on the
            # RELEASED entries, so an empty half contributes 0 changed elements
            # and 0 wire bytes -- it costs one entry, not one tensor.
            offered = stager.offer(name, dtype_str, full_shape, aidx, aval)
            if offered is None:
                return
            released, is_group = offered
            sized: list[tuple] = []
            for e_name, e_dtype, e_shape, e_idx, e_val in released:
                if e_idx is None or (e_idx.numel() == 0 and not is_group):
                    continue  # unchanged and not fused -- drop it, as before
                changed_elems += int(e_idx.numel())
                if e_idx.numel():
                    pos_width = absolute_index_width(_prodshape(e_shape))
                    wire_bytes += int(e_idx.numel()) * (pos_width + e_val.element_size())
                sized.extend(_slice_pieces(e_name, e_dtype, e_shape, e_idx, e_val))
            if is_group:
                bkt.add_atomic(sized)
            else:
                for piece, nbytes in sized:
                    bkt.add(piece, nbytes)

        gq = _GatherQueue(batch_k, self.bucket_size, is_r0, _bucket_slot_delta)

        # ``weights`` is the BACKEND's HF delta stream (hf_delta_export): entries
        # already carry final HF coordinates -- naming, conversion, diff and
        # snapshot all happened on the backend side. This engine only batches,
        # gathers and ships.
        for slots, dtype_str, counts, hf_idx, hf_val, pg in weights:
            gq.put(pg, slots, dtype_str, counts, hf_idx, hf_val)
        gq.flush_all()
        # A half still parked here means its sibling never came through the export
        # stream, so the receiver would have been handed an unpairable member and
        # died inside sglang's loader with a far less informative message.
        stager.assert_drained()
        # Log unconditionally, including the zero case: if the export ever renames
        # these params, every suffix stops matching and the staging silently
        # degrades to a no-op -- which looks exactly like "no fused params in this
        # model". The count is the only thing that tells the two apart. DSv4 should
        # report 4 groups x 43 layers.
        if is_r0:
            logger.info(
                "delta fusion staging: groups=%d nan_filled_halves=%d",
                stager.n_groups,
                stager.n_filled,
            )

        # For SGLang, verify_every=K appends a dense state-verification sweep to
        # every K-th steady sync inside the same receive stream. The steady bucket
        # keeps ``is_last`` unset, and the sweep's final flush carries it. The
        # receiver bit-compares each destination before overwriting it and fails
        # on any mismatch (see delta_loader._verify_dense).
        verify = self._verify_due()
        if is_r0:
            bkt.seal()  # seal the final partial bucket into the pending flush
            if bkt.pending is not None:
                bkt.emit(is_last=not verify)
            elif not verify:
                self._publish_terminal(False)
        if verify:
            # collective on every rank: the full export assembles per tensor.
            if self.quantize_fp8:
                vspec = self._fp8_spec(engine)
                full, _ = engine.get_per_tensor_param(quant_spec=vspec)
                self._send_full_seed(
                    full,
                    global_steps,
                    verify=True,
                    bytes_wire=True,
                    fp32_predicate=getattr(vspec, "fp32_predicate", None),
                )
            else:
                full, _ = engine.get_per_tensor_param()
                self._send_full_seed(full, global_steps, verify=True)
        if not is_r0:
            return
        self._release_staging_pool("steady")  # return staging blocks to CUDA between syncs
        logger.info("delta-sharded send v=%s delta flushes=%d (streamed)", global_steps, n_flushes)
        if not total_elems:
            return None
        return {
            "checkpoint_engine/changed_ratio": changed_elems / total_elems,
            "checkpoint_engine/changed_elems": float(changed_elems),
            "checkpoint_engine/payload_mbytes": wire_bytes / (1 << 20),
            "checkpoint_engine/flushes": float(n_flushes),
        }
