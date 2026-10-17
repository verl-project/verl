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
"""Delta weight-sync checkpoint engine (HCCL transport) for Ascend NPU rollouts.

Same seed/steady state machine, wire schema and flush manifest as the NCCL
variant (:mod:`verl.checkpoint_engine.delta_checkpoint_engine`); the transport
is a ``PyHcclCommunicator`` broadcast group whose consumers are the vLLM worker
processes themselves (the direct-receiver geometry):

* the trainer-side rank 0 builds its communicator lazily at the first publish
  and broadcasts each flush's positions+values after the zmq manifest, with an
  explicit ``torch.npu.synchronize()`` before the buffers go out of scope (no
  cupy staging: PyHccl enqueues on the current stream, so holding the tensor
  references across the sync point is sufficient);
* the rollout-side CheckpointEngineWorkers never join the broadcast group --
  ``receive_weights`` relays one rendezvous control item (zmq + stateless group
  endpoints, group geometry) to the rollout adapter, which drives the vLLM
  worker engines through their weight-transfer lifecycle RPCs;
* both sides keep the communicator across sync rounds (HCCL group setup is
  expensive; mirrors the dense HCCL engine's ``rebuild_group=False`` reuse).

The lazy rendezvous is gated by per-rank wire-ready receipts: every receiver
connects its zmq SUB, joins the group and raises a ``delta_wire_ready/{rank}``
flag in the group's TCPStore; the trainer polls those flags
(:func:`verl.checkpoint_engine.delta_sync.hccl_receiver.wait_delta_wire_ready`)
before publishing the first manifest. Code order alone cannot make the stream
lossless -- ``zmq.Socket.connect()`` is asynchronous -- while the flags order
the first publish after every receiver joined, giving each subscription
(initiated before its join) the whole join duration to reach the PUB socket.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator

import torch

from verl.utils.device import is_torch_npu_available

if not is_torch_npu_available(check_device=False):
    raise ImportError("DeltaHCCLCheckpointEngine is unavailable because the torch.npu module is not available.")

from verl.utils.distributed import stateless_init_process_group

from .base import CheckpointEngineRegistry
from .delta_sync.hccl_receiver import wait_delta_wire_ready
from .delta_sync.pipeline import DeltaSyncMixin
from .hccl_checkpoint_engine import HCCLCheckpointEngine, MasterMetadata

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@CheckpointEngineRegistry.register("delta_sharded")
class DeltaHCCLCheckpointEngine(DeltaSyncMixin, HCCLCheckpointEngine):
    """Sparse delta weight sync over HCCL, registered under the same
    ``delta_sharded`` backend name as the NCCL engine (mirroring how the dense
    HCCL engine registers ``nccl``): on NPU hosts the NCCL module fails to
    import (cupy), so this class owns the name; on CUDA hosts this module
    raises at import and the NCCL engine keeps it.
    """

    wire_format = "delta_flush"

    def __init__(self, *args, **kwargs) -> None:
        # DeltaSyncMixin.__init__ (next in MRO) consumes the delta kwargs
        # (encoding / batch_gather / verify_every) and forwards the rest to
        # HCCLCheckpointEngine.
        if kwargs.get("rebuild_group", False):
            raise NotImplementedError(
                "delta_sharded on NPU keeps one PyHccl broadcast group across sync rounds (the vLLM worker "
                "engines cache their communicators); rebuild_group=True is not supported"
            )
        super().__init__(*args, **kwargs)
        self._comm = None
        self._rendezvous: MasterMetadata | None = None
        self._group_world_size: int | None = None

    @property
    def _wire_device(self):
        return torch.device("npu", torch.npu.current_device())

    def prepare(self):
        # Delta broadcasts small per-flush buffers directly, so skip the parent's
        # 2 * bucket_size fixed buffers; hand back the HCCL rendezvous endpoints
        # (zmq manifest channel + stateless dist group) for build_topology.
        return (
            MasterMetadata(zmq_ip=self.ip, zmq_port=self.zmq_port, dist_ip=self.ip, dist_port=self.dist_port)
            if self.is_master
            else None
        )

    def init_process_group(self, rank: int, world_size: int, master_metadata: MasterMetadata):
        """Record the rendezvous only.

        The broadcast group's members are trainer rank 0 and the vLLM worker
        engines (which rendezvous inside their update RPC, long after this
        call), so nobody builds a communicator here: the trainer builds lazily
        at first publish, and the rollout CheckpointEngineWorkers never join
        the group at all (the direct-receiver geometry). ``rank``/``world_size`` still follow the dense
        single-sender topology: trainer rank 0 = 0, rollout workers 1..N with
        world = N + 1 -- exactly the geometry the vLLM workers take over.
        """
        self.rank = rank
        self.world_size = world_size
        self._rendezvous = master_metadata
        if rank == 0:
            self._group_world_size = world_size
        logger.info("delta-hccl init_process_group rank: %s, world_size: %s (rendezvous recorded)", rank, world_size)

    def finalize(self):
        """Clear the NPU cache only: the communicator is kept across sync
        rounds by design (``rebuild_group=True`` is rejected at construction),
        so the conditional destroy below mirrors the parent contract without
        ever firing (null-safe: the communicator is lazily built)."""
        if self.rebuild_group:
            if self._comm is not None:
                self._comm.destroyComm(self._comm.comm)
                self._comm = None
            self.rank = None
            self.world_size = None
            self._group_world_size = None
        torch.npu.empty_cache()

    def _ensure_communicator(self):
        if self._comm is None:
            assert self.rank == 0, "only trainer rank 0 broadcasts delta flushes"
            assert self._rendezvous is not None and self._group_world_size is not None, (
                "init_process_group must run before the first publish"
            )
            self._comm = stateless_init_process_group(
                self._rendezvous.dist_ip, self._rendezvous.dist_port, 0, self._group_world_size, self.device
            )
            # zmq SUB connect() is asynchronous, so a complete communicator does
            # not by itself imply every receiver is subscribed; the per-rank
            # wire-ready receipts order the first publish after every receiver
            # joined, and a residual miss still fails loud (manifest timeout or
            # per-flush checksum) rather than corrupting weights.
            wait_delta_wire_ready(self._comm.group.store, self._group_world_size - 1)
        return self._comm

    def _ensure_wire(self) -> None:
        self._ensure_communicator()

    # ---- trainer side transport ----
    async def send_weights(self, engine, global_steps: int | None = None):
        """Run the delta pipeline; on failure abort the HCCL group so the
        receivers fail fast instead of hanging forever (vLLM collective_rpc
        holds a worker's error until all workers return, so a one-sided abort
        is the only loud exit)."""
        try:
            return await super().send_weights(engine, global_steps=global_steps)
        except BaseException:
            logger.exception(
                "delta-hccl sender: weight-update session FAILED; aborting the communicator so receivers "
                "fail fast (full job restart required)"
            )
            if self._comm is not None:
                try:
                    self._comm.destroyComm(self._comm.comm)
                except Exception:
                    logger.exception("delta-hccl sender: communicator abort failed (best effort)")
                self._comm = None
            raise

    def _broadcast_flush(self, pos_u8: torch.Tensor | None, val_u8: torch.Tensor) -> None:
        comm = self._ensure_communicator()
        if pos_u8 is not None:
            comm.broadcast(pos_u8, src=0)
        comm.broadcast(val_u8, src=0)
        # PyHccl enqueues on the current stream; keep the flush tensors alive
        # (the caller holds them until this returns) and synchronize before the
        # caller frees or reuses the buffers.
        torch.npu.synchronize()

    def _release_after_round(self, phase: str) -> None:
        torch.npu.empty_cache()
        logger.info("delta-hccl %s send: npu cache cleared", phase)

    # ---- rollout worker side ----
    def receive_weights(self, global_steps: int | None = None) -> Iterator[tuple[list[tuple[str, torch.Tensor]], bool]]:
        """Relay the HCCL rendezvous to the rollout adapter.

        This worker never joins the broadcast group, so it yields exactly one
        control item carrying the rendezvous endpoints and group geometry. The
        adapter decodes it and drives the vLLM worker engines from there; the
        actual flush payloads flow trainer -> vLLM workers directly.
        ``engine_rank`` is this worker's slot in the dense single-sender
        topology -- direct receivers (tests) reuse it as their
        group rank, while production vLLM workers derive their own
        (replica-major / tp-minor) rank with the same numbering.
        """
        assert self.rank > 0, "Rank 0 should not receive weights."
        assert self._rendezvous is not None, "init_process_group must run before receive_weights"
        info = {
            "mode": "verl_delta_hccl",
            "zmq_ip": self._rendezvous.zmq_ip,
            "zmq_port": self._rendezvous.zmq_port,
            "dist_ip": self._rendezvous.dist_ip,
            "dist_port": self._rendezvous.dist_port,
            "topic": self.topic,
            "world_size": self.world_size,
            "engine_rank": self.rank,
        }
        payload = torch.frombuffer(bytearray(json.dumps(info).encode()), dtype=torch.uint8)
        yield [("__delta_rendezvous__", payload)], False
        logger.info("delta-hccl rendezvous relayed to server adapter (v=%s)", global_steps)
