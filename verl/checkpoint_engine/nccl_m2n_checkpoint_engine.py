# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""NCCL M2N checkpoint backend for layout-aware weight redistribution."""

from __future__ import annotations

import atexit
import logging
import os
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Generator

import ray
import torch
import zmq

try:
    from nccl.core import SUM, Communicator, UniqueId, get_unique_id
    from nccl.m2n import DistTensor, Handle, Mesh, Replicate, Shard

    _NCCL_IMPORT_ERROR = None
except ImportError as exc:  # Optional experimental dependency.
    _NCCL_IMPORT_ERROR = exc

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry
from verl.checkpoint_engine.reshard_layout import (
    LocalWeightDesc,
    ReshardLayout,
    build_reshard_layouts,
    local_weight_desc_from_shard_api,
)
from verl.models.transformers.hf_dense_decoder_tp import infer_dense_decoder_tp_shard_dim
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _require_nccl() -> None:
    if _NCCL_IMPORT_ERROR is not None:
        raise ImportError(
            "checkpoint_engine.backend='nccl_m2n' requires NCCL4Py with the nccl.m2n extension"
        ) from _NCCL_IMPORT_ERROR


@dataclass(frozen=True)
class NCCLM2NMasterMetadata:
    """Bootstrap data created by the trainer master.

    Attributes:
        unique_id: Serialized NCCL unique ID shared by all source and destination ranks.
        zmq_ip: IP address of the master's weight-metadata publisher.
        zmq_port: TCP port of the master's weight-metadata publisher.
        source_dp: Size of the replicated dimension of the source mesh.
        source_shard_size: Size of the sharded dimension of the source mesh.
        destination_dp: Size of the replicated dimension of the destination mesh.
        destination_shard_size: Size of the sharded dimension of the destination mesh.
    """

    unique_id: bytes
    zmq_ip: str
    zmq_port: int
    source_dp: int
    source_shard_size: int
    destination_dp: int
    destination_shard_size: int


def _placement_objects(layout: ReshardLayout) -> list[Any]:
    """Translate a reshard layout into NCCL M2N placement objects."""

    return [Replicate() if dim is None else Shard(dim) for dim in layout.placements]


def _nccl_stream(stream: Any) -> Any:
    """Use the raw handle for torch streams that lack ``__cuda_stream__``."""

    # NCCL4Py accepts integer handles on every supported release. Torch 2.9
    # exposes ``cuda_stream`` but no longer implements the older protocol that
    # the pinned NCCL4Py otherwise probes for non-cuda.core stream objects.
    raw_stream = getattr(stream, "cuda_stream", None)
    return int(raw_stream) if raw_stream is not None else stream


def _allocate_destination(shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
    """Allocate one destination rank's output tensor on its current CUDA device.

    Args:
        shape: Shape of the destination rank-local tensor.
        dtype: Data type of the destination rank-local tensor.

    Returns:
        A newly allocated CUDA tensor with ``shape`` and ``dtype``.
    """

    device = torch.device("cuda", torch.cuda.current_device())
    return torch.empty(shape, dtype=dtype, device=device)


@CheckpointEngineRegistry.register("nccl_m2n")
class NCCLM2NCheckpointEngine(CheckpointEngine):
    """Transfer source shards directly into destination rank-local shards.

    Source and destination ranks form separate two-dimensional meshes with shapes
    ``(source_dp, source_shard_size)`` and
    ``(destination_dp, destination_shard_size)``. The first dimension contains
    replicated model copies; the second is the rank-local sharding dimension.

    Args:
        bucket_size: Common checkpoint-engine bucket size in bytes. NCCL M2N
            manages its staging pool internally, so this value is retained only
            for checkpoint-engine interface compatibility.
        is_master: Whether this source rank creates the NCCL unique ID and publishes
            per-weight metadata. Exactly one source rank must be the master.
        source_dp: Number of replicated groups in the source mesh.
        source_shard_size: Number of ranks in the source mesh's sharded dimension.
        destination_dp: Number of replicated groups in the destination mesh.
        destination_shard_size: Number of ranks in the destination mesh's generic
            sharded dimension. A consumer such as vLLM may interpret this as TP.
    """

    wire_format = "rank_local_named_tensors"
    topic = "nccl_m2n_metadata"
    ready_topic_prefix = "nccl_m2n_ready:"

    def __init__(
        self,
        bucket_size: int,
        is_master: bool = False,
        source_dp: int | None = None,
        source_shard_size: int | None = None,
        destination_dp: int | None = None,
        destination_shard_size: int | None = None,
    ) -> None:
        topology = {
            "source_dp": source_dp,
            "source_shard_size": source_shard_size,
            "destination_dp": destination_dp,
            "destination_shard_size": destination_shard_size,
        }
        missing = [name for name, value in topology.items() if value is None]
        if missing:
            raise ValueError(f"NCCL M2N topology requires explicit values for {', '.join(missing)}")

        self.bucket_size = int(bucket_size)
        self.is_master = bool(is_master)
        self.source_dp = int(source_dp)
        self.source_shard_size = int(source_shard_size)
        self.destination_dp = int(destination_dp)
        self.destination_shard_size = int(destination_shard_size)
        if (
            min(
                self.bucket_size,
                self.source_dp,
                self.source_shard_size,
                self.destination_dp,
                self.destination_shard_size,
            )
            <= 0
        ):
            raise ValueError("NCCL M2N sizes must be positive")

        self.source_world_size = self.source_dp * self.source_shard_size
        self.reshard_world_size = self.source_world_size + self.destination_dp * self.destination_shard_size
        self.rank: int | None = None
        self.role: str | None = None
        self._master_metadata: NCCLM2NMasterMetadata | None = None
        self._comm = self._handle = self._transfer_stream = None
        self._socket = self._zmq_context = None
        self._closed = False
        if self.is_master:
            self._start_metadata_server()
        atexit.register(self.close)

    def _start_metadata_server(self) -> None:
        ip = ray.util.get_node_ip_address().strip("[]")
        port, _ = get_free_port(ip)
        context = zmq.Context()
        # XPUB exposes subscription notifications, allowing rank zero to wait
        # until every destination's metadata subscription has reached this socket.
        socket = context.socket(zmq.XPUB)
        socket.setsockopt(zmq.XPUB_VERBOSE, 1)
        address = f"tcp://[{ip}]:{port}" if is_valid_ipv6_address(ip) else f"tcp://{ip}:{port}"
        if is_valid_ipv6_address(ip):
            socket.setsockopt(zmq.IPV6, 1)
        socket.bind(address)
        self._zmq_context, self._socket = context, socket
        self._metadata_ip, self._metadata_port = ip, port

    def _connect_metadata_client(self, metadata: NCCLM2NMasterMetadata) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        address = (
            f"tcp://[{metadata.zmq_ip}]:{metadata.zmq_port}"
            if is_valid_ipv6_address(metadata.zmq_ip)
            else f"tcp://{metadata.zmq_ip}:{metadata.zmq_port}"
        )
        if is_valid_ipv6_address(metadata.zmq_ip):
            socket.setsockopt(zmq.IPV6, 1)
        socket.connect(address)
        # Subscribe to the common topic first. XPUB observes subscriptions from
        # one connection in order, so seeing the rank-specific readiness topic
        # proves that the common subscription is active at the publisher.
        socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)
        socket.setsockopt_string(zmq.SUBSCRIBE, f"{self.ready_topic_prefix}{self.rank}")
        self._zmq_context, self._socket = context, socket

    def _wait_for_metadata_subscribers(self) -> None:
        """Wait for every destination subscription and acknowledge each one."""

        expected = set(range(self.source_world_size, self.reshard_world_size))
        ready: set[int] = set()
        prefix = self.ready_topic_prefix.encode()
        while ready != expected:
            event = self._socket.recv()
            if not event:
                raise RuntimeError("received an empty NCCL M2N subscription event")
            subscription = event[1:]
            if not subscription.startswith(prefix):
                continue
            try:
                rank = int(subscription[len(prefix) :].decode("ascii"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise RuntimeError(f"invalid NCCL M2N readiness subscription: {subscription!r}") from exc
            if rank not in expected:
                raise RuntimeError(f"unexpected NCCL M2N destination readiness rank {rank}")
            if event[0] == 1:
                ready.add(rank)
            elif event[0] == 0:
                ready.discard(rank)
            else:
                raise RuntimeError(f"invalid NCCL M2N subscription action {event[0]}")
        for rank in sorted(expected):
            self._socket.send_string(f"{self.ready_topic_prefix}{rank}")

    def _wait_for_metadata_publisher(self) -> None:
        """Wait until rank zero acknowledges this destination's subscriptions."""

        expected = f"{self.ready_topic_prefix}{self.rank}"
        received = self._socket.recv_string()
        if received != expected:
            raise RuntimeError(f"unexpected NCCL M2N readiness acknowledgement {received!r}; expected {expected!r}")

    def prepare(self) -> NCCLM2NMasterMetadata | None:
        """Create the communicator bootstrap metadata on the master source rank.

        Returns:
            Cached bootstrap metadata on the master, or ``None`` on every other rank.
        """

        if not self.is_master:
            return None
        _require_nccl()
        if self._master_metadata is None:
            self._master_metadata = NCCLM2NMasterMetadata(
                bytes(get_unique_id()),
                self._metadata_ip,
                self._metadata_port,
                self.source_dp,
                self.source_shard_size,
                self.destination_dp,
                self.destination_shard_size,
            )
        return self._master_metadata

    @classmethod
    def build_topology(
        cls, trainer_world_size: int, rollout_world_size: int, metadata: list[Any]
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        """Build per-worker communicator arguments for source and destination ranks.

        Args:
            trainer_world_size: Number of trainer workers in the source mesh.
            rollout_world_size: Number of rollout workers in the destination mesh.
            metadata: Results of ``prepare`` ordered with trainer workers first.

        Returns:
            Trainer and rollout keyword-argument maps for ``init_process_group``.
            Every value is a per-worker list suitable for worker-group dispatch.

        Raises:
            ValueError: If there is not exactly one trainer master or either worker
                count does not match the configured mesh dimensions.
        """

        masters = [item for item in metadata[:trainer_world_size] if isinstance(item, NCCLM2NMasterMetadata)]
        if len(masters) != 1:
            raise ValueError(f"NCCL M2N requires exactly one trainer master, got {len(masters)}")
        master = masters[0]
        source_size = master.source_dp * master.source_shard_size
        destination_size = master.destination_dp * master.destination_shard_size
        world_size = source_size + destination_size
        if trainer_world_size != source_size or rollout_world_size != destination_size:
            raise ValueError(
                f"NCCL M2N requires exactly {source_size} trainer and {destination_size} rollout ranks, "
                f"got {trainer_world_size} and {rollout_world_size}"
            )
        return (
            {
                "rank": list(range(source_size)),
                "world_size": [world_size] * trainer_world_size,
                "master_metadata": [master] * trainer_world_size,
                "role": ["source"] * source_size,
            },
            {
                "rank": list(range(source_size, world_size)),
                "world_size": [world_size] * rollout_world_size,
                "master_metadata": [master] * rollout_world_size,
                "role": ["destination"] * rollout_world_size,
            },
        )

    def init_process_group(
        self,
        rank: int,
        world_size: int,
        master_metadata: NCCLM2NMasterMetadata,
        role: str,
    ) -> None:
        """Create the combined NCCL M2N communicator and runtime handle.

        Args:
            rank: This worker's rank in the source-plus-destination communicator.
            world_size: Total number of source and destination ranks.
            master_metadata: Bootstrap and mesh metadata returned by the master.
            role: This worker's role, either ``"source"`` or ``"destination"``.

        Raises:
            ValueError: If the rank, role, world size, or topology is invalid.
            RuntimeError: If an existing communicator has a different topology or
                communicator bootstrap fails.
        """

        self.rank, self.role = int(rank), role
        if self.rank < 0:
            raise ValueError(f"NCCL M2N rank must be non-negative, got {self.rank}")
        if role not in {"source", "destination"}:
            raise ValueError(f"NCCL M2N role must be source or destination, got {role!r}")

        _require_nccl()
        if world_size != self.reshard_world_size:
            raise ValueError(f"runtime world_size={world_size}, expected {self.reshard_world_size}")
        expected = (self.source_dp, self.source_shard_size, self.destination_dp, self.destination_shard_size)
        actual = (
            master_metadata.source_dp,
            master_metadata.source_shard_size,
            master_metadata.destination_dp,
            master_metadata.destination_shard_size,
        )
        if actual != expected:
            raise ValueError(f"NCCL M2N topology {actual} does not match local config {expected}")
        if self._comm is not None:
            if self._comm.rank != self.rank or self._comm.nranks != world_size:
                raise RuntimeError("cannot reuse an NCCL M2N communicator with a different topology")
            return

        self._comm = Communicator.init(world_size, self.rank, UniqueId.from_bytes(master_metadata.unique_id))
        self._handle = Handle.create()
        self._transfer_stream = torch.cuda.Stream(device=torch.cuda.current_device())
        if role == "destination":
            self._connect_metadata_client(master_metadata)

        sync = torch.ones(1, dtype=torch.int32, device="cuda")
        stream = torch.cuda.current_stream()
        self._comm.allreduce(sync, sync, SUM, stream=_nccl_stream(stream))
        stream.synchronize()
        if sync.item() != world_size:
            raise RuntimeError("NCCL M2N communicator bootstrap failed")
        if self.rank == 0:
            # CheckpointEngineManager waits for every init_process_group call, so
            # no weight can be published until this readiness barrier returns.
            self._wait_for_metadata_subscribers()
        elif self.role == "destination":
            self._wait_for_metadata_publisher()

    def _coerce_weight(self, exported: Any) -> LocalWeightDesc:
        if isinstance(exported, LocalWeightDesc):
            return exported

        weight = local_weight_desc_from_shard_api(
            exported,
            destination_shard_dim=infer_dense_decoder_tp_shard_dim(exported[0]),
        )
        if self.rank is not None and weight.source_shard_dim is not None:
            spec = exported[2]
            mesh_dim = next(axis for axis, placement in enumerate(spec.placements) if placement.is_shard())
            device_mesh_rank = int(spec.mesh.get_local_rank(mesh_dim=mesh_dim))
            m2n_mesh_rank = self.rank % self.source_shard_size
            if device_mesh_rank != m2n_mesh_rank:
                raise ValueError(
                    f"source DeviceMesh rank {device_mesh_rank} does not match M2N mesh rank {m2n_mesh_rank}"
                )
        return weight

    def _layouts(self, weight: LocalWeightDesc) -> tuple[ReshardLayout, ReshardLayout]:
        return build_reshard_layouts(
            weight,
            source_replica_size=self.source_dp,
            source_shard_size=self.source_shard_size,
            destination_replica_size=self.destination_dp,
            destination_shard_size=self.destination_shard_size,
        )

    def _descriptors(
        self,
        weight: LocalWeightDesc,
        source: torch.Tensor | None,
        destination: torch.Tensor | None,
    ) -> tuple[Any, Any]:
        source_layout, destination_layout = self._layouts(weight)
        return (
            DistTensor(
                source,
                local_shape=source_layout.local_shape,
                dtype=weight.tensor.dtype,
                mesh=Mesh(source_layout.mesh_dims, start_rank=source_layout.start_rank),
                placements=_placement_objects(source_layout),
            ),
            DistTensor(
                destination,
                local_shape=destination_layout.local_shape,
                dtype=weight.tensor.dtype,
                mesh=Mesh(destination_layout.mesh_dims, start_rank=destination_layout.start_rank),
                placements=_placement_objects(destination_layout),
            ),
        )

    def _publish(self, payload: dict[str, Any]) -> None:
        self._socket.send_string(self.topic, flags=zmq.SNDMORE)
        self._socket.send_pyobj(payload)

    def _receive(self) -> dict[str, Any]:
        if self._socket.recv_string() != self.topic:
            raise RuntimeError("received an unexpected NCCL M2N metadata topic")
        return self._socket.recv_pyobj()

    @torch.no_grad()
    async def send_weights(self, weights: Generator, global_steps: int | None = None) -> dict:
        """Reshard local source tensors into destination rank-local tensors.

        Args:
            weights: Generator yielding ``LocalWeightDesc`` objects or
                ``(name, local_tensor, ShardSpec)`` tuples.
            global_steps: Optional trainer step associated with the update. Reserved
                for checkpoint-engine interface compatibility.

        Returns:
            An empty metrics dictionary after all transfers have been enqueued.
            ``finalize`` waits for their completion.
        """

        del global_steps
        if self.rank is None:
            raise RuntimeError("NCCL M2N process group is not initialized")
        if self.role != "source" or any(resource is None for resource in (self._comm, self._handle)):
            raise RuntimeError(f"invalid NCCL M2N sender state for role={self.role!r}")

        caller_stream = torch.cuda.current_stream()
        stream = self._transfer_stream
        if stream is None:
            raise RuntimeError("NCCL M2N transfer stream is not initialized")
        for exported in weights:
            weight = self._coerce_weight(exported)
            stream.wait_stream(caller_stream)
            if self.rank == 0:
                self._publish(
                    {
                        "kind": "weight",
                        "name": weight.name,
                        "global_shape": tuple(weight.global_shape),
                        "dtype": weight.tensor.dtype,
                        "destination_shard_dim": weight.destination_shard_dim,
                        "source_shard_dim": weight.source_shard_dim,
                        "source_shard_size": weight.source_shard_size,
                    }
                )
            source_desc, destination_desc = self._descriptors(weight, weight.tensor, None)
            # This removes application-managed staging; M2N may still copy through
            # the internal staging-buffer pipeline used by Handle.reshard().
            self._handle.reshard(
                self._comm,
                source_desc,
                destination_desc,
                stream=_nccl_stream(stream),
            )
            # reshard reads the source asynchronously. Return the dependency to
            # its owning stream so allocator reuse stays ordered without
            # record_stream's nondeterministic lifetime tracking.
            caller_stream.wait_stream(stream)
        if self.rank == 0:
            self._publish({"kind": "end"})
        # finalize() fences every matching asynchronous M2N operation. The
        # handle-owned staging pool remains alive until close() destroys the handle.
        return {}

    @torch.no_grad()
    async def receive_weights(self, global_steps: int | None = None) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """Receive destination rank-local tensors in source publication order.

        Args:
            global_steps: Optional trainer step associated with the update. Reserved
                for checkpoint-engine interface compatibility.

        Yields:
            The parameter name and its newly allocated destination rank-local tensor.
        """

        del global_steps
        if self.rank is None or self.rank < 0 or self.role != "destination":
            raise RuntimeError(f"invalid NCCL M2N receiver state: rank={self.rank}, role={self.role!r}")
        if any(resource is None for resource in (self._comm, self._handle)):
            raise RuntimeError("NCCL M2N receiver resources are not initialized")

        caller_stream = torch.cuda.current_stream()
        stream = self._transfer_stream
        if stream is None:
            raise RuntimeError("NCCL M2N transfer stream is not initialized")
        while True:
            metadata = self._receive()
            if metadata.get("kind") == "end":
                break
            if metadata.get("kind") != "weight":
                raise RuntimeError(f"invalid NCCL M2N metadata: {metadata!r}")

            weight = LocalWeightDesc(
                metadata["name"],
                torch.empty(metadata["global_shape"], dtype=metadata["dtype"], device="meta"),
                torch.Size(metadata["global_shape"]),
                metadata["destination_shard_dim"],
                metadata["source_shard_dim"],
                int(metadata["source_shard_size"]),
            )
            _, destination_layout = self._layouts(weight)
            destination = _allocate_destination(destination_layout.local_shape, weight.tensor.dtype)
            source_desc, destination_desc = self._descriptors(weight, None, destination)
            stream.wait_stream(caller_stream)
            self._handle.reshard(
                self._comm,
                source_desc,
                destination_desc,
                stream=_nccl_stream(stream),
            )
            caller_stream.wait_stream(stream)
            yield weight.name, destination

    def finalize(self) -> None:
        """Host-wait for outstanding transfer and consumer work after an update."""

        if self.rank is not None and self.rank >= 0:
            if self._transfer_stream is not None:
                self._transfer_stream.synchronize()
            torch.cuda.current_stream().synchronize()

    def close(self) -> None:
        """Wait for M2N work and release the handle, communicator, and ZMQ resources."""

        if self._closed:
            return
        self._closed = True
        try:
            if self._transfer_stream is not None:
                self._transfer_stream.synchronize()
            if self._handle is not None:
                self._handle.destroy()
            if self._comm is not None:
                self._comm.destroy()
            if self._socket is not None:
                self._socket.close(linger=0)
            if self._zmq_context is not None:
                self._zmq_context.term()
        except Exception:
            logger.exception("failed to close NCCL M2N resources")
        finally:
            self._comm = self._handle = self._transfer_stream = None
            self._socket = self._zmq_context = None
