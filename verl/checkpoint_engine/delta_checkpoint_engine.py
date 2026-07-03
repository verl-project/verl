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
:class:`NCCLCheckpointEngine` uses (actor rank0 -> rollout workers). Following
verl's design, the rollout *worker* reconstructs full tensors from the delta and
hands them to ``server_adapter.update_weights`` for the local push into SGLang;
the changed-byte payload is what crosses the wire, so the disaggregated
trainer->rollout transfer drops to the sparsity ratio.

The first sync broadcasts a full delta (every position) so a dummy-initialized
rollout gets a correct base; subsequent syncs are sparse.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator, Generator
from unittest.mock import patch

import ray.util.collective as collective
import torch
import torch.distributed as dist
import zmq

with patch("importlib.metadata.distributions", return_value=[]):
    import cupy as cp

from .delta_sync import DeltaState, iter_delta_flushes
from .delta_sync.encode import DeltaParam, checksum as _checksum, decode_chunk
from .delta_sync.sharded import gather_v_to_rank0, shard_delta_indices
from .delta_sync.wrapper import DeltaFlush

from .base import CheckpointEngineRegistry
from .nccl_checkpoint_engine import MasterMetadata, NCCLCheckpointEngine

logger = logging.getLogger(__name__)


def _bitflip_like(t: torch.Tensor) -> torch.Tensor:
    """A tensor whose every byte differs from ``t`` (forces a full diff)."""
    flat = t.detach().contiguous().view(-1).view(torch.uint8)
    return (flat ^ 0xFF).view(t.dtype).view(t.shape)


@CheckpointEngineRegistry.register("delta")
class DeltaCheckpointEngine(NCCLCheckpointEngine):
    """NCCL delta transport. Reuses NCCLCheckpointEngine's group/zmq machinery;
    overrides send/receive to move only changed positions+values."""

    def __init__(self, *args, encoding: str = "indices", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoding = encoding
        self._state = DeltaState()  # trainer-side snapshot for diffing
        self._mirror: dict[str, torch.Tensor] = {}  # rollout-side full-weight mirror

    def prepare(self) -> MasterMetadata | None:
        # Delta broadcasts small per-flush buffers directly, so skip the parent's
        # 2 * bucket_size fixed buffers. Still hand back the master zmq endpoint
        # that build_topology() distributes to the rollout workers.
        return MasterMetadata(zmq_ip=self.ip, zmq_port=self.listen_port) if self.is_master else None

    # ---- trainer side ----
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ):
        assert self.rank <= 0, "Trainer workers other than rank 0 should not send weights."
        if self.rank < 0:
            # Non-src actor ranks must still iterate to drive the FSDP all-gather.
            for _name, _w in weights:
                pass
            return

        first = not self._state.seeded
        if first:
            materialized = list(weights)
            self._state.seed([(n, _bitflip_like(t)) for n, t in materialized])
            weights_iter = iter(materialized)
        else:
            weights_iter = weights

        flushes = list(
            iter_delta_flushes(
                weights_iter, self._state, encoding=self.encoding, bucket_bytes=self.bucket_size
            )
        )

        # 1. publish the per-sync manifest over the zmq side-channel. The receiver
        #    needs it to size the recv buffers and decode the position blob.
        meta = {
            "is_full": first,
            "encoding": self.encoding,
            "flushes": [
                {
                    "params": [vars(p) for p in f.params],
                    "pos_numel": int(f.positions_cpu.numel()),
                    "val_numel": int(f.values_gpu.numel()),
                    "val_dtype": str(f.values_gpu.dtype).replace("torch.", ""),
                    "checksum": int(f.checksum),
                }
                for f in flushes
            ],
        }
        self.socket.send_string(self.topic, flags=zmq.SNDMORE)
        self.socket.send_pyobj(meta)

        # 2. broadcast each flush's positions then values over NCCL as raw bytes.
        #    Master uses cupy-owned buffers (mirrors NCCLCheckpointEngine) so NCCL
        #    can register them even under PYTORCH_CUDA_ALLOC_CONF=expandable_segments.
        total = 0
        for f in flushes:
            pos_u8 = f.positions_cpu.to("cuda", non_blocking=True).contiguous().view(torch.uint8)
            val_u8 = f.values_gpu.contiguous().view(torch.uint8)
            pos_cp = cp.empty(pos_u8.numel(), dtype=cp.uint8)
            val_cp = cp.empty(val_u8.numel(), dtype=cp.uint8)
            pos_cp[:] = cp.asarray(pos_u8)
            val_cp[:] = cp.asarray(val_u8)
            collective.broadcast(pos_cp, src_rank=0, group_name=self.group_name)
            collective.broadcast(val_cp, src_rank=0, group_name=self.group_name)
            total += int(f.nnz)
        logger.info(
            "delta-nccl send v=%s %s flushes=%d nnz=%d",
            global_steps, "FULL" if first else "delta", len(flushes), total,
        )

    # ---- rollout worker side ----
    async def receive_weights(
        self,
        global_steps: int | None = None,
    ) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        assert self.rank > 0, "Rank 0 should not receive weights."
        self.socket.recv_string()
        meta = self.socket.recv_pyobj()

        for fmeta in meta["flushes"]:
            val_dtype = getattr(torch, fmeta["val_dtype"])
            elem = torch.empty(0, dtype=val_dtype).element_size()
            pos = torch.empty(fmeta["pos_numel"], dtype=torch.uint8, device="cuda")
            val_u8 = torch.empty(fmeta["val_numel"] * elem, dtype=torch.uint8, device="cuda")
            collective.broadcast(pos, src_rank=0, group_name=self.group_name)
            collective.broadcast(val_u8, src_rank=0, group_name=self.group_name)
            val = val_u8.view(val_dtype)

            got = _checksum(pos, val)
            if got != fmeta["checksum"]:
                raise RuntimeError(
                    f"delta-nccl checksum mismatch on rank {self.rank} "
                    f"v={global_steps}: got {got}, expected {fmeta['checksum']}"
                )

            params = [DeltaParam(**p) for p in fmeta["params"]]
            decoded = decode_chunk(
                meta["encoding"], pos.cpu().numpy().tobytes(), val, params
            )
            for name, dt in decoded.items():
                mask = ~torch.isnan(dt)
                cur = self._mirror.get(name)
                if cur is None:
                    # first sync (full): the decode covers every position.
                    cur = dt.clone()
                    cur[~mask] = 0
                else:
                    cur = cur.to(dt.device)
                    cur[mask] = dt[mask]
                self._mirror[name] = cur
                yield name, cur


@CheckpointEngineRegistry.register("delta_sharded")
class DeltaShardedCheckpointEngine(DeltaCheckpointEngine):
    """Delta over NCCL, but the diff is computed on each rank's local FSDP shard.

    Instead of ``full_tensor()``-ing every parameter and diffing on rank 0 (parent), each
    actor rank keeps a pinned-CPU snapshot of only *its* shard, byte-diffs the shard, and
    only the changed ``(within-param position, value)`` pairs are gathered to rank 0 -- so
    the all-gather volume drops to the sparsity ratio and rank 0 no longer holds a
    full-model snapshot. The assembled result is bit-identical to the parent's diff, so the
    receiver (:meth:`DeltaCheckpointEngine.receive_weights`) is reused unchanged.

    ``send_weights`` here expects the SHARDED generator ``get_per_tensor_param_shard()``
    (``(name, local_flat_shard, within_param_offset, full_numel, full_shape, contributes)``)
    rather than the full-tensor generator.
    """

    def __init__(self, *args, encoding: str = "indices", **kwargs) -> None:
        super().__init__(*args, encoding=encoding, **kwargs)
        self._shard_snap: dict[str, torch.Tensor] = {}  # name -> pinned-CPU shard snapshot
        self._shard_seeded = False

    def _assemble_flush(self, per_param: list) -> DeltaFlush:
        """Build one DeltaFlush (indices encoding) from rank 0's gathered per-param deltas.

        ``per_param``: list of ``(name, dtype_str, shape, global_idx_i32, values)`` where
        ``global_idx`` are within-parameter flat positions (== what the receiver decodes).
        """
        pos_pieces: list[bytes] = []
        val_pieces: list[torch.Tensor] = []
        params: list[DeltaParam] = []
        pos_off = val_off = 0
        for name, dtype_str, shape, idx_i32, val in per_param:
            nnz = int(idx_i32.numel())
            pos_bytes = idx_i32.to(torch.int32).cpu().numpy().tobytes()
            params.append(
                DeltaParam(name=name, dtype=dtype_str, shape=list(shape),
                           pos_start=pos_off, pos_end=pos_off + len(pos_bytes), pos_width=4,
                           val_start=val_off, val_end=val_off + nnz)
            )
            pos_pieces.append(pos_bytes)
            val_pieces.append(val)
            pos_off += len(pos_bytes)
            val_off += nnz

        merged = b"".join(pos_pieces)
        positions_cpu = (
            torch.frombuffer(bytearray(merged), dtype=torch.uint8) if merged
            else torch.empty(0, dtype=torch.uint8)
        )
        values_gpu = torch.cat(val_pieces) if val_pieces else torch.empty(0, dtype=self.rollout_dtype, device="cuda")
        positions_gpu = positions_cpu.to(values_gpu.device, non_blocking=True)
        cks = _checksum(positions_gpu, values_gpu)
        return DeltaFlush(encoding=self.encoding, params=params,
                          positions_cpu=positions_cpu, values_gpu=values_gpu, checksum=cks)

    async def send_weights(self, weights, global_steps=None):
        # All actor ranks participate (gather-v is collective); only torch rank 0 broadcasts.
        assert self.rank <= 0, "Trainer workers other than rank 0 should not send weights."
        is_r0 = self.is_master
        first = not self._shard_seeded
        per_param: list = []

        for name, local, offset, _full_numel, full_shape, contributes in weights:
            local = local.detach().contiguous().view(-1)
            snap = self._shard_snap.get(name)
            if snap is None or snap.numel() != local.numel():
                snap = torch.empty_like(local, device="cpu", pin_memory=True)
            if contributes:
                base = _bitflip_like(local) if first else snap.to(local.device, non_blocking=True)
                gidx, gval = shard_delta_indices(local, base, offset)
            else:
                # replicated param owned by another rank; contribute nothing but keep lockstep.
                gidx = torch.empty(0, dtype=torch.int64, device=local.device)
                gval = torch.empty(0, dtype=local.dtype, device=local.device)
            snap.copy_(local, non_blocking=True)  # update snapshot to current shard
            self._shard_snap[name] = snap

            aidx, aval = gather_v_to_rank0(gidx, gval)
            if is_r0 and aidx is not None and aidx.numel() > 0:
                per_param.append(
                    (name, str(local.dtype).replace("torch.", ""), list(full_shape), aidx, aval)
                )

        self._shard_seeded = True
        if not is_r0:
            return

        flush = self._assemble_flush(per_param)
        meta = {
            "is_full": first,
            "encoding": self.encoding,
            "flushes": [{
                "params": [vars(p) for p in flush.params],
                "pos_numel": int(flush.positions_cpu.numel()),
                "val_numel": int(flush.values_gpu.numel()),
                "val_dtype": str(flush.values_gpu.dtype).replace("torch.", ""),
                "checksum": int(flush.checksum),
            }],
        }
        self.socket.send_string(self.topic, flags=zmq.SNDMORE)
        self.socket.send_pyobj(meta)

        pos_u8 = flush.positions_cpu.to("cuda", non_blocking=True).contiguous().view(torch.uint8)
        val_u8 = flush.values_gpu.contiguous().view(torch.uint8)
        pos_cp = cp.empty(pos_u8.numel(), dtype=cp.uint8)
        val_cp = cp.empty(val_u8.numel(), dtype=cp.uint8)
        pos_cp[:] = cp.asarray(pos_u8)
        val_cp[:] = cp.asarray(val_u8)
        collective.broadcast(pos_cp, src_rank=0, group_name=self.group_name)
        collective.broadcast(val_cp, src_rank=0, group_name=self.group_name)
        logger.info(
            "delta-sharded send v=%s %s params=%d nnz=%d",
            global_steps, "FULL" if first else "delta", len(flush.params), int(flush.values_gpu.numel()),
        )
