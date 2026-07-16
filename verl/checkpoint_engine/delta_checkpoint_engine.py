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
from unittest.mock import patch

import ray.util.collective as collective
import torch
import zmq

with patch("importlib.metadata.distributions", return_value=[]):
    import cupy as cp

from verl.workers.engine.spec import derive_placement

from .base import CheckpointEngineRegistry
from .delta_sync.encode import DeltaFlush, DeltaParam
from .delta_sync.encode import checksum as _checksum
from .delta_sync.sparse_gather import (
    gather_dense_to_rank0,
    gather_shards_to_rank0,
    gather_v_batched_to_rank0,
    shard_delta_indices,
)
from .nccl_checkpoint_engine import MasterMetadata, NCCLCheckpointEngine

logger = logging.getLogger(__name__)


def _prodshape(shape) -> int:
    n = 1
    for x in shape:
        n *= int(x)
    return n


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
    def receive_weights(self, global_steps: int | None = None):
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

    def _placement(self, name, spec):
        """Derive (and cache) this rank's placement facts from the declarative spec."""
        got = self._placements.get(name)
        if got is None:
            got = derive_placement(spec)
            self._placements[name] = got
        return got

    def _assemble_flush(self, per_param: list) -> DeltaFlush:
        """Build one DeltaFlush (indices encoding) from rank 0's gathered per-param deltas.

        ``per_param``: list of ``(name, dtype_str, shape, global_idx, values)`` where
        ``global_idx`` are within-parameter flat positions (== what the receiver decodes).

        Positions stay on the GPU end to end (int32 pieces -> one cat -> uint8 view);
        the wire broadcasts from the GPU anyway, and a host round-trip here
        (``.cpu().numpy().tobytes()`` + join) dominated the whole send at scale
        (~2.4s/sync at 7B steady state, ~83s on the full seed).
        """
        idx_pieces: list[torch.Tensor] = []
        val_pieces: list[torch.Tensor] = []
        params: list[DeltaParam] = []
        pos_off = val_off = 0
        for name, dtype_str, shape, idx, val in per_param:
            nnz = int(idx.numel())
            # positions ride the wire as int32 (pos_width=4); a parameter bigger than
            # 2^31 elements would silently wrap, so fail loud instead. DeltaParam
            # carries pos_width for a future 8-byte escalation if a model needs it.
            assert _prodshape(shape) < (1 << 31), (
                f"{name}: {_prodshape(shape)} elements exceeds the int32 position encoding"
            )
            idx_pieces.append(idx.to(torch.int32))
            val_pieces.append(val)
            params.append(
                DeltaParam(
                    name=name,
                    dtype=dtype_str,
                    shape=list(shape),
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

    def _send_dense_seed(self, weights, global_steps=None):
        """First sync: assemble and broadcast the raw weights, bucketed, positions-free.

        Explicit dense semantics instead of the previous bitflip-forced full diff:
        no per-element indices on the wire (values only), no whole-parameter
        (idx, val) spike on rank 0 -- its peak is one assembled parameter plus a
        bucket -- and the snapshot is populated as the stream goes.
        """
        is_r0 = self.is_master
        bucket: list = []  # (name, dtype_str, full_shape, flat_full_tensor)
        bucket_bytes = 0
        pending = None  # (params, values) awaiting emission (1-flush lookahead for is_last)
        n_flushes = 0
        total_elems = 0
        wire_bytes = 0

        def _emit(is_last):
            nonlocal pending, n_flushes, wire_bytes
            if pending is not None:
                self._publish_dense_flush(pending[0], pending[1], is_last=is_last)
                n_flushes += 1
                wire_bytes += int(pending[1].nbytes)
                pending = None

        def _seal():
            nonlocal bucket, bucket_bytes, pending
            if not bucket:
                return
            _emit(is_last=False)
            params = []
            val_off = 0
            for name, dtype_str, full_shape, flat in bucket:
                n = int(flat.numel())
                params.append(
                    DeltaParam(
                        name=name,
                        dtype=dtype_str,
                        shape=list(full_shape),
                        pos_start=0,
                        pos_end=0,
                        pos_width=4,
                        val_start=val_off,
                        val_end=val_off + n,
                    )
                )
                val_off += n
            values = torch.cat([flat for *_, flat in bucket])
            pending = (params, values)
            bucket = []
            bucket_bytes = 0

        def _bucket_dense(hf_name, tensor):
            nonlocal total_elems, bucket_bytes
            flat = tensor.reshape(-1)
            total_elems += int(flat.numel())
            bucket.append((hf_name, str(flat.dtype).replace("torch.", ""), list(tensor.shape), flat))
            bucket_bytes += int(flat.nbytes)
            if bucket_bytes >= self.bucket_size:
                _seal()

        for name, local, spec in weights:
            local = local.detach().contiguous().view(-1)
            snap = self._shard_snap.get(name)
            if snap is None or snap.numel() != local.numel():
                snap = torch.empty_like(local, device="cpu", pin_memory=True)
            snap.copy_(local, non_blocking=True)
            self._shard_snap[name] = snap

            offset, contributes, pg = self._placement(name, spec)
            shard = local if contributes else torch.empty(0, dtype=local.dtype, device=local.device)
            if spec.to_hf is None:
                # the spec's placements map the shard straight into the full tensor
                if pg is None:
                    if is_r0 and contributes:
                        _bucket_dense(name, local.view(spec.full_shape))
                else:
                    full = gather_dense_to_rank0(
                        shard, offset if contributes else 0, _prodshape(spec.full_shape), group=pg
                    )
                    if is_r0 and full is not None:
                        _bucket_dense(name, full.view(spec.full_shape))
            else:
                # converter profile (e.g. Megatron): rebuild dense shards via to_hf
                shards = gather_shards_to_rank0(shard, group=pg)
                if is_r0 and shards is not None:
                    for hf_name, hf_tensor in spec.to_hf(shards):
                        _bucket_dense(hf_name, hf_tensor)

        self._shard_seeded = True
        if not is_r0:
            return
        _seal()
        if pending is not None:
            _emit(is_last=True)
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

    async def send_weights(self, weights, global_steps=None):
        # All actor ranks participate (gather-v is collective); only torch rank 0 broadcasts.
        # rank 0 accumulates the gathered per-param deltas into bucket_size-sized flushes and streams
        # each one as soon as it fills (then frees it), so peak memory is ~2 buckets rather than the
        # whole model.
        assert self.rank <= 0, "Trainer workers other than rank 0 should not send weights."
        if not self._shard_seeded:
            return self._send_dense_seed(weights, global_steps)
        is_r0 = self.is_master
        first = False  # the dense first sync is handled by _send_dense_seed
        bucket: list = []
        bucket_bytes = 0
        pending = None  # a DeltaFlush awaiting emission (1-flush lookahead so the last flush is is_last)
        n_flushes = 0
        changed_elems = 0
        total_elems = 0
        wire_bytes = 0

        def _emit(is_last):
            nonlocal pending, n_flushes
            if pending is not None:
                self._publish_flush(pending, first, is_last=is_last)
                n_flushes += 1
                pending = None

        def _seal():
            nonlocal bucket, bucket_bytes, pending
            if not bucket:
                return
            _emit(is_last=False)  # another bucket follows, so the prior one is not the last
            pending = self._assemble_flush(bucket)
            bucket = []
            bucket_bytes = 0

        batch_k = self.batch_gather
        group: list = []  # (name, dtype_str, full_shape, full_numel, gidx, gval)

        def _consume(name, dtype_str, full_shape, full_numel, aidx, aval):
            nonlocal total_elems, changed_elems, wire_bytes, bucket_bytes
            total_elems += int(full_numel)
            if aidx is None or aidx.numel() == 0:
                return
            changed_elems += int(aidx.numel())
            wire_bytes += int(aidx.numel()) * (4 + aval.element_size())
            # Slice oversized per-param deltas so one entry never exceeds
            # MAX_ENTRY_ELEMS (bounds the receiver-side decode transient).
            for s in range(0, aidx.numel(), self.MAX_ENTRY_ELEMS):
                e = min(s + self.MAX_ENTRY_ELEMS, aidx.numel())
                bucket.append((name, dtype_str, list(full_shape), aidx[s:e], aval[s:e]))
                bucket_bytes += (e - s) * 8 + (e - s) * aval.element_size()
                if bucket_bytes >= self.bucket_size:
                    _seal()

        def _flush_group():
            nonlocal group
            if not group:
                return
            pg = group[0][6]
            if pg is None:
                # unsharded params: this rank's delta already is the global delta
                if is_r0:
                    for name, dtype_str, full_shape, full_numel, gi, gv, _pg in group:
                        _consume(name, dtype_str, full_shape, full_numel, gi, gv)
                group = []
                return
            dev = group[0][4].device
            idx_concat = torch.cat([g[4] for g in group])
            val_concat = torch.cat([g[5] for g in group])
            counts = torch.tensor([int(g[4].numel()) for g in group], dtype=torch.int64, device=dev)
            gathered = gather_v_batched_to_rank0(idx_concat, val_concat, counts, group=pg)
            if is_r0 and gathered is not None:
                for (name, dtype_str, full_shape, full_numel, _gi, _gv, _pg), (aidx, aval) in zip(
                    group, gathered, strict=False
                ):
                    _consume(name, dtype_str, full_shape, full_numel, aidx, aval)
            group = []

        nan_group: list = []  # rebuild-profile params batched per gather_group

        def _flush_nan_group():
            nonlocal nan_group, total_elems
            if not nan_group:
                return
            pg = nan_group[0][2]
            dev = nan_group[0][5].device
            idx_concat = torch.cat([g[5] for g in nan_group])
            val_concat = torch.cat([g[6] for g in nan_group])
            counts = torch.tensor([int(g[5].numel()) for g in nan_group], dtype=torch.int64, device=dev)
            gathered = gather_v_batched_to_rank0(idx_concat, val_concat, counts, group=pg, grouped=True)
            if is_r0 and gathered is not None:
                for (name, local_dtype, _pg, spec, shard_shape, _gi, _gv), per_rank in zip(
                    nan_group, gathered, strict=False
                ):
                    shard_list = []
                    for gi, gv in per_rank:
                        buf = torch.full(shard_shape, float("nan"), dtype=local_dtype, device=dev)
                        buf.view(-1)[gi.to(torch.int64)] = gv  # index ops need Long
                        shard_list.append(buf)
                    for hf_name, hf_tensor in spec.to_hf(shard_list):
                        fl = hf_tensor.reshape(-1)
                        total_elems += int(fl.numel())
                        pos = (~torch.isnan(fl)).nonzero(as_tuple=False).view(-1)
                        if pos.numel():
                            # total_elems is added inside _consume too; compensate
                            total_elems -= int(fl.numel())
                            _consume(
                                hf_name,
                                str(fl.dtype).replace("torch.", ""),
                                tuple(hf_tensor.shape),
                                int(fl.numel()),
                                pos,
                                fl[pos],
                            )
            nan_group = []

        for name, local, spec in weights:
            local = local.detach().contiguous().view(-1)
            offset, contributes, pg = self._placement(name, spec)
            snap = self._shard_snap.get(name)
            if snap is None or snap.numel() != local.numel():
                snap = torch.empty_like(local, device="cpu", pin_memory=True)
            if contributes:
                base = snap.to(local.device, non_blocking=True)
                lidx, lval = shard_delta_indices(local, base, 0)  # shard-local coordinates
            else:
                # replicated param owned by another rank; contribute nothing but keep lockstep.
                lidx = torch.empty(0, dtype=torch.int64, device=local.device)
                lval = torch.empty(0, dtype=local.dtype, device=local.device)
            snap.copy_(local, non_blocking=True)  # update snapshot to current shard
            self._shard_snap[name] = snap

            if spec.to_hf is not None:
                # converter profile: boundary-preserving gather within the spec's group,
                # NaN-sentinel rebuild through the trainer's converter on rank 0.
                if nan_group and nan_group[0][2] is not pg:
                    _flush_nan_group()
                # shard-local coordinates are bounded by the shard's numel, so the
                # full-tensor <2^31 assert in _assemble_flush covers them too.
                nan_group.append((name, local.dtype, pg, spec, tuple(local.shape), lidx.to(torch.int32), lval))
                if len(nan_group) >= max(batch_k, 1):
                    _flush_nan_group()
                continue

            # placement profile: translate rank-locally, gather without boundaries.
            # int32 halves the gather's position bytes; the wire is int32 anyway and
            # the per-parameter <2^31 assert in _assemble_flush covers the range.
            gidx = ((lidx + offset) if lidx.numel() else lidx).to(torch.int32)
            gval = lval
            full_numel = _prodshape(spec.full_shape)
            if group and group[-1][6] is not pg:
                _flush_group()
            group.append((name, str(local.dtype).replace("torch.", ""), spec.full_shape, full_numel, gidx, gval, pg))
            if len(group) >= max(batch_k, 1):
                _flush_group()
        _flush_group()
        _flush_nan_group()

        self._shard_seeded = True
        if not is_r0:
            return
        _seal()  # seal the final partial bucket into `pending`
        if pending is not None:
            _emit(is_last=True)
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
