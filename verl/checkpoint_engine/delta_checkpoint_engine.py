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
import zmq

with patch("importlib.metadata.distributions", return_value=[]):
    import cupy as cp

from .delta_sync import DeltaState, iter_delta_flushes
from .delta_sync.encode import DeltaParam, checksum as _checksum, decode_chunk

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
