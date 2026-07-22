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
Each rollout worker then hands its local copy of the sparse payload to its
colocated SGLang TP worker via same-GPU ``update_weights_from_tensor`` IPC, where
the verl-shipped :mod:`verl.workers.rollout.sglang_rollout.delta_loader` (registered through SGLang's
stock ``--custom-weight-loader`` hook — no SGLang fork or patch needed) decodes
and masked-applies it *in place* onto the live weights. No full-model mirror is
staged anywhere on the rollout side: receiver peak memory is one bucket plus one
decode chunk, independent of model size.

The first sync broadcasts a full delta (every position) so a dummy-initialized
rollout gets a correct base; subsequent syncs are sparse.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from unittest.mock import patch

import ray.util.collective as collective
import torch
import zmq

with patch("importlib.metadata.distributions", return_value=[]):
    import cupy as cp

from verl.workers.engine.spec import BlockPlacement, ShardSpec, derive_placement, translate_flat_indices

from .base import CheckpointEngineRegistry
from .delta_sync.encode import DeltaFlush, DeltaParam
from .delta_sync.encode import checksum as _checksum
from .delta_sync.sparse_gather import (
    assemble_dense_param_on_rank0,
    gather_slot_entries_to_rank0,
    shard_delta_indices,
)
from .nccl_checkpoint_engine import MasterMetadata, NCCLCheckpointEngine

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
class _DensePiece:
    """One full dense parameter buffered for the seed sync's dense flush."""

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
    (sparse indices flush vs values-only dense flush) and the flush counters."""

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


def _add_sliced(bkt: _FlushBucket, name: str, dtype_str: str, shape, aidx: torch.Tensor, aval: torch.Tensor) -> None:
    """Slice one param's (idx, val) delta into <= MAX_ENTRY_ELEMS pieces and bucket
    them (bounds the receiver-side decode transient; the masked apply is sequential,
    so splitting is transparent). Bucket bytes = actual wire bytes (int32 positions
    + values)."""
    max_elems = DeltaShardedCheckpointEngine.MAX_ENTRY_ELEMS
    for s in range(0, aidx.numel(), max_elems):
        e = min(s + max_elems, aidx.numel())
        bkt.add(
            _FlushPiece(name, dtype_str, list(shape), aidx[s:e], aval[s:e]),
            (e - s) * (4 + aval.element_size()),
        )


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
        _pg, entries = self._queues.setdefault(id(pg), (pg, []))
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

    ``send_weights`` consumes the SHARDED generator ``get_per_tensor_param_shard()``:
    ``(name, local_shard, ShardSpec)`` per local parameter (see
    :mod:`verl.workers.engine.spec`). All layout knowledge lives in the
    spec, so this one engine serves any trainer backend that can describe its shards.
    """

    # Cap on changed elements per DeltaParam entry. The receiver-side decode
    # densifies per entry with an int64 index transient (8 B/element), so an
    # uncapped entry (e.g. a 7B model's whole embedding on the full seed, ~545M
    # elements) would spike several GiB at once. Oversized per-param deltas are
    # sliced into multiple entries (the masked apply is sequential, so splitting
    # is transparent); 64M elements bounds the transient to ~512 MiB.
    MAX_ENTRY_ELEMS = 64 << 20

    # Bound on the rank-0 rebuild transient for dim-0-separable converter params
    # (``spec.to_hf_chunk``): the NaN rebuild and the dense seed assemble/convert in
    # dim-0 segments of at most this many elements instead of materializing the full
    # logical tensor (a Kimi-K2.5-scale fused expert stack is ~21 GB in bf16).
    REBUILD_CHUNK_ELEMS = 64 << 20

    wire_format = "delta_flush"

    def prepare(self) -> MasterMetadata | None:
        # Delta broadcasts small per-flush buffers directly, so skip the parent's
        # 2 * bucket_size fixed buffers. Still hand back the master zmq endpoint
        # that build_topology() distributes to the rollout workers.
        return MasterMetadata(zmq_ip=self.ip, zmq_port=self.listen_port) if self.is_master else None

    # ---- trainer side ----
    # ---- shared STREAMING wire ----
    # Broadcast each flush the moment it is produced and free it, instead of materializing every
    # flush up front. Peak device memory stays ~2 buckets (like NCCLCheckpointEngine's send/recv
    # buffers) rather than the whole model -- required for large models where the first (full-seed)
    # sync would otherwise hold the entire delta on rank 0. Wire: one zmq manifest + NCCL broadcast
    # per flush, with an ``is_last`` flag so the receiver loops until the stream ends. Each
    # CheckpointEngineWorker then hands its local copy of the sparse payload to its colocated
    # SGLang TP worker (same-GPU IPC), where the verl-shipped custom weight loader applies it
    # in place -- no full-model staging anywhere on the rollout side.
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

    def _publish_dense_flush(self, params: list[DeltaParam], values: torch.Tensor, is_last: bool) -> None:
        """Publish a dense (full-coverage, positions-free) flush -- used by the first sync."""
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

    def _publish_terminal(self, first: bool) -> None:
        """End-of-stream marker when zero flushes were produced (no broadcast, just a signal)."""
        meta = {"is_full": first, "encoding": self.encoding, "is_last": True, "terminal_empty": True}
        self.socket.send_string(self.topic, flags=zmq.SNDMORE)
        self.socket.send_pyobj(meta)

    # ---- rollout worker side ----
    def receive_weights(self, global_steps: int | None = None) -> Iterator[tuple[list[tuple[str, torch.Tensor]], bool]]:
        """Yield the sparse flushes for the server adapter to apply in place.

        Each item is ``(named_tensors, is_last)``: the sentinel-encoded flush
        (``__delta_spec__`` json bytes, optional ``__positions__``, ``__values__``)
        received into this worker's own GPU buffer -- one flush resident at a
        time, freed as soon as the consumer drops it. The sglang server adapter
        forwards each flush over same-GPU CUDA IPC to the verl-shipped
        ``delta_loader.apply_delta`` (registered through the custom-weight-loader
        hook), which decodes and masked-applies it against SGLang's live weights;
        no full-model mirror is staged anywhere.
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

    def __init__(self, *args, encoding: str = "indices", batch_gather: int = 32, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert encoding == "indices", f"delta_sharded ships only the 'indices' position encoding; got {encoding!r}"
        self.encoding = encoding
        self._shard_snap: dict[str, torch.Tensor] = {}  # name -> pinned-CPU shard snapshot
        self._shard_seeded = False
        self._placements: dict[str, tuple] = {}  # name -> (flat_offset, contributes, group)
        # Gather the per-param sparse deltas in groups of this many parameters
        # (one count-matrix all_gather + two padded gathers per group instead of
        # three collectives per parameter).
        self.batch_gather = int(batch_gather)

    def _placement(self, name: str, spec: ShardSpec) -> tuple[int | BlockPlacement, bool, object]:
        """Derive (and cache) this rank's ``(place, contributes, gather_group)`` from the spec."""
        got = self._placements.get(name)
        if got is None:
            got = derive_placement(spec)
            self._placements[name] = got
        return got

    def _assemble_flush(self, per_param: list[_FlushPiece]) -> DeltaFlush:
        """Build one DeltaFlush (indices encoding) from rank 0's gathered per-param deltas.

        ``per_param``: :class:`_FlushPiece` entries whose ``idx`` are within-parameter
        flat positions (== what the receiver decodes).

        Positions stay on the GPU end to end (int32 pieces -> one cat -> uint8 view);
        the wire broadcasts from the GPU anyway, and a host round-trip here
        (``.cpu().numpy().tobytes()`` + join) dominated the whole send at scale
        (~2.4s/sync at 7B steady state, ~83s on the full seed).
        """
        idx_pieces: list[torch.Tensor] = []
        val_pieces: list[torch.Tensor] = []
        params: list[DeltaParam] = []
        pos_off = val_off = 0
        for piece in per_param:
            nnz = int(piece.idx.numel())
            # positions ride the wire as int32 (pos_width=4); a parameter bigger than
            # 2^31 elements would silently wrap, so fail loud instead. DeltaParam
            # carries pos_width for a future 8-byte escalation if a model needs it.
            assert _prodshape(piece.shape) < (1 << 31), (
                f"{piece.name}: {_prodshape(piece.shape)} elements exceeds the int32 position encoding"
            )
            idx_pieces.append(piece.idx.to(torch.int32))
            val_pieces.append(piece.val)
            params.append(
                DeltaParam(
                    name=piece.name,
                    dtype=piece.dtype_str,
                    shape=list(piece.shape),
                    pos_start=pos_off,
                    pos_end=pos_off + nnz * 4,
                    pos_width=4,
                    val_start=val_off,
                    val_end=val_off + nnz,
                )
            )
            pos_off += nnz * 4
            val_off += nnz

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

    def _send_dense_seed(
        self,
        weights: Generator[tuple[str, torch.Tensor, ShardSpec], None, None],
        global_steps: int | None = None,
    ) -> dict[str, float] | None:
        """First sync: assemble and broadcast the raw weights, bucketed, positions-free.

        Explicit dense semantics instead of the previous bitflip-forced full diff:
        no per-element indices on the wire (values only), no whole-parameter
        (idx, val) spike on rank 0 -- its peak is one assembled parameter plus a
        bucket -- and the snapshot is populated as the stream goes.
        """
        is_r0 = self.is_master
        n_flushes = 0
        total_elems = 0
        wire_bytes = 0

        def _assemble_dense(pieces: list[_DensePiece]):
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

        def _publish_dense(pending, is_last: bool) -> None:
            nonlocal n_flushes, wire_bytes
            params, values = pending
            self._publish_dense_flush(params, values, is_last=is_last)
            n_flushes += 1
            wire_bytes += int(values.nbytes)

        def _publish_sparse(flush, is_last: bool) -> None:
            nonlocal n_flushes, wire_bytes
            self._publish_flush(flush, True, is_last=is_last)
            n_flushes += 1
            wire_bytes += int(flush.values_gpu.nbytes) + int(flush.positions_cpu.numel())

        dense = _FlushBucket(self.bucket_size, _assemble_dense, _publish_dense)
        # Sparse (indices-encoded) seed flushes for sender-side-converted slot params:
        # positions ride the wire for these params (a one-off ~3x on their bytes) so
        # every rank ships its own column stripes and nobody assembles anything.
        sparse = _FlushBucket(self.bucket_size, self._assemble_flush, _publish_sparse)

        def _bucket_dense(hf_name: str, tensor: torch.Tensor) -> None:
            nonlocal total_elems
            flat = tensor.reshape(-1)
            total_elems += int(flat.numel())
            dense.add(
                _DensePiece(hf_name, str(flat.dtype).replace("torch.", ""), list(tensor.shape), flat), flat.nbytes
            )

        for name, local, spec in weights:
            local = local.detach().contiguous().view(-1)
            snap = self._shard_snap.get(name)
            if snap is None or snap.numel() != local.numel():
                snap = torch.empty_like(local, device="cpu", pin_memory=True)
            snap.copy_(local, non_blocking=True)
            self._shard_snap[name] = snap

            place, contributes, pg = self._placement(name, spec)
            shard = local if contributes else torch.empty(0, dtype=local.dtype, device=local.device)
            if spec.to_hf_chunk is None:
                # the spec's placements map the shard straight into the full tensor
                if pg is None:
                    if is_r0 and contributes:
                        _bucket_dense(name, local.view(spec.full_shape))
                elif isinstance(place, BlockPlacement):
                    full = assemble_dense_param_on_rank0(shard, place, group=pg)
                    if is_r0 and full is not None:
                        _bucket_dense(name, full.view(spec.full_shape))
                else:
                    raise NotImplementedError(
                        f"{name}: plain-int explicit placements are no longer supported by the "
                        "unified sender-side engine (attach a BlockPlacement instead)"
                    )
            elif isinstance(place, BlockPlacement) and spec.hf_slots is not None:
                # SENDER-SIDE seed: every rank converts its own rows (full
                # coverage) segment by segment and ships slot-keyed sparse
                # entries -- no rank assembles or converts anyone else's data.
                # Segments bound both the NaN window and the per-round gather
                # blob (threshold scaled by group size).
                full_shape = spec.full_shape
                inner = max(_prodshape(full_shape[1:]), 1)
                n_rows = int(full_shape[0])
                slots_per_row = len(spec.hf_slots) // n_rows
                world = torch.distributed.get_world_size(pg) if pg is not None else 1
                seg_rows = max(1, self.REBUILD_CHUNK_ELEMS // max(inner * world, 1))
                dtype_str = str(local.dtype).replace("torch.", "")
                lview = local.view(tuple(int(x) for x in place.local_shape))
                o0, l0 = int(place.global_offset[0]), int(place.local_shape[0])
                inner_box = BlockPlacement(
                    tuple(int(x) for x in place.local_shape[1:]),
                    tuple(int(x) for x in place.global_offset[1:]),
                    tuple(int(x) for x in full_shape[1:]),
                )
                row_numel = _prodshape(place.local_shape[1:])
                iidx = translate_flat_indices(torch.arange(row_numel, device=local.device), inner_box)
                if is_r0:
                    total_elems += _prodshape(full_shape)
                for row0 in range(0, n_rows, seg_rows):
                    rows = min(seg_rows, n_rows - row0)
                    r_lo, r_hi = max(o0, row0), min(o0 + l0, row0 + rows)
                    counts = torch.zeros(rows * slots_per_row, dtype=torch.int64)
                    idx_pieces: list[torch.Tensor] = []
                    val_pieces: list[torch.Tensor] = []
                    for r in range(r_lo, r_hi):
                        row_vals = lview[r - o0].reshape(-1)
                        for s_i, pidx, pval in self._convert_row_slots(name, spec, r, iidx, row_vals, local):
                            counts[(r - row0) * slots_per_row + s_i] = pidx.numel()
                            idx_pieces.append(pidx)
                            val_pieces.append(pval)
                    my_idx = (
                        torch.cat(idx_pieces) if idx_pieces else torch.empty(0, dtype=torch.int32, device=local.device)
                    )
                    my_val = (
                        torch.cat(val_pieces) if val_pieces else torch.empty(0, dtype=local.dtype, device=local.device)
                    )
                    gathered = gather_slot_entries_to_rank0(
                        my_idx, my_val, counts.to(local.device), group=pg, max_round_bytes=self.bucket_size
                    )
                    if is_r0 and gathered is not None:
                        for k_i, (aidx, aval) in enumerate(gathered):
                            sname, sshape = spec.hf_slots[row0 * slots_per_row + k_i]
                            if aidx is not None and aidx.numel():
                                _add_sliced(sparse, sname, dtype_str, sshape, aidx, aval)
            else:
                raise NotImplementedError(
                    f"{name}: converter specs without an enumerable slot table (hf_slots) are "
                    "not supported by the unified sender-side engine; rewrite the converter as a "
                    "dim-0-separable to_hf_chunk + hf_slots (see #7060)"
                )

        self._shard_seeded = True
        if not is_r0:
            return
        dense.seal()
        sparse.seal()
        if sparse.pending is not None:
            dense.emit(is_last=False)
            sparse.emit(is_last=True)
        elif dense.pending is not None:
            dense.emit(is_last=True)
        else:
            self._publish_terminal(True)
        logger.info(
            "delta-sharded send v=%s DENSE-SEED flushes=%d elems=%d",
            global_steps,
            n_flushes,
            total_elems,
        )
        if not total_elems:
            return None
        return {
            "checkpoint_engine/changed_ratio": 1.0,
            "checkpoint_engine/changed_elems": float(total_elems),
            "checkpoint_engine/payload_mbytes": wire_bytes / (1 << 20),
            "checkpoint_engine/flushes": float(n_flushes),
        }

    def _convert_row_slots(
        self,
        name: str,
        spec: ShardSpec,
        r: int,
        pos_in_row: torch.Tensor,
        vals: torch.Tensor,
        ref: torch.Tensor,
    ) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
        """NaN-probe one dim-0 row through the spec's converter: scatter the row's
        (within-row position, value) pairs into a NaN-filled row buffer, run
        ``to_hf_chunk`` on it, and extract each output slot's surviving positions
        and values (the converter is a pure permutation, so non-NaN survivors are
        exactly the input pairs in final HF coordinates). Shared by the steady
        loop (touched subset of a row) and the seed (every owned element of a
        row). Returns ``[(slot_offset_in_row, idx_int32, val), ...]``, skipping
        empty slots. ``ref`` supplies dtype/device."""
        full_shape = spec.full_shape
        inner = max(_prodshape(full_shape[1:]), 1)
        slots_per_row = len(spec.hf_slots) // int(full_shape[0])
        buf = torch.full((inner,), float("nan"), dtype=ref.dtype, device=ref.device)
        buf[pos_in_row] = vals
        outs = spec.to_hf_chunk(int(r), buf.view(1, *full_shape[1:]))
        assert len(outs) == slots_per_row, (
            f"{name}: to_hf_chunk gave {len(outs)} outputs/row, slot table expects {slots_per_row}"
        )
        res = []
        for s_i, (_hf_name, hf_tensor) in enumerate(outs):
            fl = hf_tensor.reshape(-1)
            p_ = (~torch.isnan(fl)).nonzero(as_tuple=False).view(-1)
            if p_.numel():
                res.append((s_i, p_.to(torch.int32), fl[p_]))
        return res

    def _convert_block_delta(
        self,
        name: str,
        spec: ShardSpec,
        place: BlockPlacement,
        lidx: torch.Tensor,
        lval: torch.Tensor,
        local: torch.Tensor,
    ) -> tuple[list, str, torch.Tensor, torch.Tensor, torch.Tensor]:
        """SENDER-SIDE scoped conversion for one converter param: turn this rank's
        shard-local delta into final HF-coordinate slot entries. Only the touched
        dim-0 rows are converted (NaN row window -> ``to_hf_chunk`` -> non-NaN
        extraction); rank 0 does no conversion at all. Every rank enumerates the
        same slot list (zero counts when untouched), so the batched gather stays
        aligned across ranks. Returns the queue entry ``(slots, dtype_str, counts,
        idx_concat, val_concat)``."""
        full_shape = spec.full_shape
        inner = max(_prodshape(full_shape[1:]), 1)
        n_rows = int(full_shape[0])
        K = len(spec.hf_slots)
        slots_per_row = K // n_rows
        counts = torch.zeros(K, dtype=torch.int64)
        idx_pieces: list[torch.Tensor] = []
        val_pieces: list[torch.Tensor] = []
        if lidx.numel():
            g = translate_flat_indices(lidx, place)
            order = torch.argsort(g)
            g, gv = g[order], lval[order]
            rows = torch.div(g, inner, rounding_mode="floor")
            urows, rcounts = torch.unique_consecutive(rows, return_counts=True)
            pos = 0
            for r, cnt in zip(urows.tolist(), rcounts.tolist(), strict=False):
                sel_g = g[pos : pos + cnt]
                sel_v = gv[pos : pos + cnt]
                pos += cnt
                for s_i, pidx, pval in self._convert_row_slots(name, spec, r, sel_g - r * inner, sel_v, local):
                    counts[int(r) * slots_per_row + s_i] = pidx.numel()
                    idx_pieces.append(pidx)
                    val_pieces.append(pval)
        if idx_pieces:
            my_idx = torch.cat(idx_pieces)
            my_val = torch.cat(val_pieces)
        else:
            my_idx = torch.empty(0, dtype=torch.int32, device=local.device)
            my_val = torch.empty(0, dtype=local.dtype, device=local.device)
        return spec.hf_slots, str(local.dtype).replace("torch.", ""), counts, my_idx, my_val

    def _identity_delta(
        self,
        name: str,
        spec: ShardSpec,
        place: int | BlockPlacement,
        lidx: torch.Tensor,
        lval: torch.Tensor,
        local: torch.Tensor,
    ) -> tuple[list, str, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Identity profile (dense DTensor / explicit blocks / unsharded): the
        param IS its own single slot -- translate the shard-local delta to
        within-param coordinates and ride the same unified queue as converter
        params. int32 positions: the wire is int32 anyway and ``_assemble_flush``
        asserts the range. Returns the queue entry ``(slots, dtype_str, counts,
        idx, val)``."""
        gidx = (translate_flat_indices(lidx, place) if lidx.numel() else lidx).to(torch.int32)
        counts = torch.zeros(1, dtype=torch.int64)
        counts[0] = int(gidx.numel())
        return [(name, tuple(spec.full_shape))], str(local.dtype).replace("torch.", ""), counts, gidx, lval

    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor, ShardSpec], None, None],
        global_steps: int | None = None,
    ) -> dict[str, float] | None:
        # All actor ranks participate (gather-v is collective); only torch rank 0 broadcasts.
        # rank 0 accumulates the gathered per-param deltas into bucket_size-sized flushes and streams
        # each one as soon as it fills (then frees it), so peak memory is ~2 buckets rather than the
        # whole model.
        assert self.rank <= 0, "Trainer workers other than rank 0 should not send weights."
        if not self._shard_seeded:
            return self._send_dense_seed(weights, global_steps)
        is_r0 = self.is_master
        first = False  # the dense first sync is handled by _send_dense_seed
        n_flushes = 0
        changed_elems = 0
        total_elems = 0
        wire_bytes = 0

        def _publish_steady(flush, is_last: bool) -> None:
            nonlocal n_flushes
            self._publish_flush(flush, first, is_last=is_last)
            n_flushes += 1

        bkt = _FlushBucket(self.bucket_size, self._assemble_flush, _publish_steady)

        batch_k = self.batch_gather

        def _consume(
            name: str,
            dtype_str: str,
            full_shape: tuple,
            full_numel: int,
            aidx: torch.Tensor | None,
            aval: torch.Tensor | None,
        ) -> None:
            nonlocal total_elems, changed_elems, wire_bytes
            total_elems += int(full_numel)
            if aidx is None or aidx.numel() == 0:
                return
            changed_elems += int(aidx.numel())
            wire_bytes += int(aidx.numel()) * (4 + aval.element_size())
            _add_sliced(bkt, name, dtype_str, full_shape, aidx, aval)

        gq = _GatherQueue(batch_k, self.bucket_size, is_r0, _consume)

        for name, local, spec in weights:
            local = local.detach().contiguous().view(-1)
            place, contributes, pg = self._placement(name, spec)
            # The seed sync allocated every param's snapshot; a missing name or a
            # numel drift means the rollout side was never seeded for this shard --
            # diffing against a fresh (garbage) buffer would silently ship a bogus
            # near-full delta, so fail loud instead.
            snap = self._shard_snap[name]
            assert snap.numel() == local.numel(), (
                f"{name}: shard numel changed since seed ({snap.numel()} -> {local.numel()})"
            )
            if contributes:
                base = snap.to(local.device, non_blocking=True)
                lidx, lval = shard_delta_indices(local, base, 0)  # shard-local coordinates
            else:
                # replicated param owned by another rank; contribute nothing but keep lockstep.
                lidx = torch.empty(0, dtype=torch.int64, device=local.device)
                lval = torch.empty(0, dtype=local.dtype, device=local.device)
            snap.copy_(local, non_blocking=True)  # update snapshot to current shard
            self._shard_snap[name] = snap

            if spec.to_hf_chunk is not None and isinstance(place, BlockPlacement) and spec.hf_slots is not None:
                entry = self._convert_block_delta(name, spec, place, lidx, lval, local)
            elif spec.to_hf_chunk is not None:
                raise NotImplementedError(
                    f"{name}: converter specs without an enumerable slot table (hf_slots) are "
                    "not supported by the unified sender-side engine; rewrite the converter as a "
                    "dim-0-separable to_hf_chunk + hf_slots (see #7060)"
                )
            else:
                entry = self._identity_delta(name, spec, place, lidx, lval, local)
            gq.put(pg, *entry)
        gq.flush_all()

        self._shard_seeded = True
        if not is_r0:
            return
        bkt.seal()  # seal the final partial bucket into the pending flush
        if bkt.pending is not None:
            bkt.emit(is_last=True)
        else:
            self._publish_terminal(first)
        logger.info(
            "delta-sharded send v=%s %s flushes=%d (streamed)",
            global_steps,
            "FULL" if first else "delta",
            n_flushes,
        )
        if not total_elems:
            return None
        return {
            "checkpoint_engine/changed_ratio": changed_elems / total_elems,
            "checkpoint_engine/changed_elems": float(changed_elems),
            "checkpoint_engine/payload_mbytes": wire_bytes / (1 << 20),
            "checkpoint_engine/flushes": float(n_flushes),
        }
