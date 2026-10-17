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

The seed/steady state machine, bucketing and wire schema are shared with the
Ascend NPU (HCCL) variant via :mod:`verl.checkpoint_engine.delta_sync.pipeline`;
this module contributes only the NCCL/CUDA transport (ray collective group,
cupy staging) and the rollout-side receive loop.

Wire encodings: ``indices`` (int32 positions + values; every steady sync) and
``values`` (values only; the seed). The values meta tag is ``"dense"`` for
protocol continuity with the receiver's delta_loader.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from unittest.mock import patch

import ray.util.collective as collective
import torch

with patch("importlib.metadata.distributions", return_value=[]):
    import cupy as cp

from .base import CheckpointEngineRegistry
from .delta_sync.pipeline import DeltaSyncMixin
from .nccl_checkpoint_engine import MasterMetadata, NCCLCheckpointEngine, WorkerMetadata

logger = logging.getLogger(__name__)


@CheckpointEngineRegistry.register("delta_sharded")
class DeltaShardedCheckpointEngine(DeltaSyncMixin, NCCLCheckpointEngine):
    """Sparse delta weight sync over NCCL, diffed on each rank's local shard.

    Reuses NCCLCheckpointEngine's group/zmq machinery but moves only changed
    positions+values: each actor rank keeps a pinned-CPU snapshot of only *its*
    shard, byte-diffs the shard, and only the changed ``(position, value)`` pairs
    are gathered to rank 0 and streamed to the rollout side -- no rank ever holds
    a full-model snapshot. The seed/steady state machine lives in
    :class:`DeltaSyncMixin`; see its ``send_weights`` for the engine-driven
    contract with the training backend.
    """

    wire_format = "delta_flush"

    @property
    def _wire_device(self):
        return "cuda"

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

    # ---- trainer side transport ----
    def _broadcast_flush(self, pos_u8: torch.Tensor | None, val_u8: torch.Tensor) -> None:
        # Stage into cupy-owned buffers: ray's NCCL broadcast is enqueued on a separate
        # stream with no recordStream on its inputs, so broadcasting a zero-copy view of
        # these torch tensors (freed right after this call) would race with allocator reuse.
        val_cp = cp.empty(val_u8.numel(), dtype=cp.uint8)
        val_cp[:] = cp.asarray(val_u8)
        if pos_u8 is None:
            collective.broadcast(val_cp, src_rank=0, group_name=self.group_name)
            return
        pos_cp = cp.empty(pos_u8.numel(), dtype=cp.uint8)
        pos_cp[:] = cp.asarray(pos_u8)
        collective.broadcast(pos_cp, src_rank=0, group_name=self.group_name)
        collective.broadcast(val_cp, src_rank=0, group_name=self.group_name)

    def _release_after_round(self, phase: str) -> None:
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
