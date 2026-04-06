# Copyright 2025 Google LLC
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

"""WPI (Weight Propagation Interface) checkpoint engine for cross-node weight distribution.

Uses the WPI driver's NCCL broadcast (NodePropagate) for cross-node transfer and
shared VRAM + FD passing for zero-copy local access. This replaces the Ray collective
NCCL approach with a driver-managed broadcast that integrates with Kubernetes.

Key differences from NCCLCheckpointEngine:
- No Ray collective — WPI driver manages NCCL internally via NodePropagate gRPC
- Persistent VRAM buffer — allocated once via NodeStageWeight, reused across training steps
- FD-based memory mapping — consumers import shared CUDA memory via POSIX FD (SCM_RIGHTS)
- Notification-based sync — rollout workers wait for READY signal instead of NCCL participation

Architecture:
    Trainer Node                          Rollout Node(s)
    ┌──────────────────┐                  ┌──────────────────┐
    │  Actor Engine     │                  │  WPI Driver       │
    │  (FSDP/Megatron)  │                  │  (DaemonSet)      │
    │       │           │                  │       │           │
    │  send_weights()   │                  │  NCCL recv →      │
    │  ┌────▼────┐      │   NodePropagate  │  ┌────▼────┐      │
    │  │  VRAM   │──────┼──────────────────┤──│  VRAM   │      │
    │  │  Buffer │      │   NCCL bcast     │  │  Buffer │      │
    │  └─────────┘      │                  │  └────┬────┘      │
    └──────────────────┘                  │       │ FD import │
                                          │  receive_weights() │
                                          │       │           │
                                          │  ServerAdapter     │
                                          │  (vLLM/SGLang)    │
                                          └──────────────────┘
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Generator

import ray
import torch
import zmq

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry, TensorMeta
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address
from verl.utils.wpi_client import WPIClient

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class WPIMasterMetadata:
    """Metadata from the trainer master node, broadcast during topology building."""

    zmq_ip: str
    zmq_port: int
    node_ip: str  # IP of the trainer node for NodePropagate targeting


@CheckpointEngineRegistry.register("wpi")
class WPICheckpointEngine(CheckpointEngine):
    """WPI checkpoint engine for cross-node weight distribution.

    Uses the WPI driver's pre-allocated shared VRAM buffer and NCCL broadcast
    (NodePropagate) for transferring weights from trainer to rollout replicas.

    Lifecycle per training step:
        1. prepare()           — Stage VRAM buffer on WPI driver, map via FD import
        2. build_topology()    — Discover rollout node IPs from Ray, assign ranks
        3. init_process_group() — Init ZMQ PUB/SUB for metadata broadcast
        4. send_weights()      — Pack tensors into VRAM, broadcast metadata, trigger NodePropagate
        5. receive_weights()   — Wait for READY, read metadata, yield tensors from local VRAM
        6. finalize()          — Clean up (buffer persists for reuse)

    Args:
        bucket_size: Total VRAM buffer size in bytes. Sent to WPI driver via NodeStageWeight.
            Must be large enough to hold all model parameters.
        buffer_id: Unique WPI buffer identifier. Must be consistent across trainer and
            rollout nodes.
        claim_id: Kubernetes WeightClaim ID for lifecycle management. Defaults to buffer_id.
        socket_dir: Path to WPI UNIX socket directory. Defaults to /run/wpi/sockets.
        driver_port: WPI driver gRPC port. Defaults to 50051.
        is_master: True if this is the trainer rank 0 process. Only the master sends weights.
        rollout_dtype: Expected dtype for weights on rollout side. Defaults to bfloat16.
    """

    def __init__(
        self,
        bucket_size: int,
        buffer_id: str = "verl-weights",
        claim_id: str = "",
        socket_dir: str = "/run/wpi/sockets",
        driver_port: int = 50051,
        is_master: bool = False,
        rollout_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.bucket_size = bucket_size
        self.buffer_id = buffer_id
        self.claim_id = claim_id or buffer_id
        self.socket_dir = socket_dir
        self.driver_port = driver_port
        self.is_master = is_master
        self.rollout_dtype = rollout_dtype

        # Filled during prepare() / init_process_group()
        self.rank: int | None = None
        self.world_size: int | None = None
        self.vram_buffer: torch.Tensor | None = None
        self.target_node_ids: list[str] | None = None

        # WPI client for gRPC + FD + CUDA import
        self.wpi_client = WPIClient(
            socket_dir=socket_dir,
            driver_port=driver_port,
        )

        # ZMQ for metadata (offset table) broadcast — same pattern as NCCLCheckpointEngine
        self.topic = "wpi_bucket_metadata"
        self.socket: zmq.Socket | None = None
        if self.is_master:
            self._start_zmq_server()

        self._staged = False

    def _start_zmq_server(self):
        """Start ZMQ PUB socket on the trainer master for broadcasting tensor metadata."""
        self.ip = ray.util.get_node_ip_address().strip("[]")
        self.listen_port, _ = get_free_port(self.ip)

        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        if is_valid_ipv6_address(self.ip):
            address = f"tcp://[{self.ip}]:{self.listen_port}"
            self.socket.setsockopt(zmq.IPV6, 1)
        else:
            address = f"tcp://{self.ip}:{self.listen_port}"

        self.socket.bind(address)
        print(f"WPI master ZMQ PUB bound to {address}", flush=True)

    def _connect_zmq_client(self, metadata: WPIMasterMetadata):
        """Connect ZMQ SUB socket on rollout workers to receive tensor metadata."""
        assert not self.is_master, "Master process should not connect as ZMQ client."
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        if is_valid_ipv6_address(metadata.zmq_ip):
            address = f"tcp://[{metadata.zmq_ip}]:{metadata.zmq_port}"
            self.socket.setsockopt(zmq.IPV6, 1)
        else:
            address = f"tcp://{metadata.zmq_ip}:{metadata.zmq_port}"

        self.socket.connect(address)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)
        print(f"WPI rollout ZMQ SUB connected to {address}", flush=True)

    def prepare(self) -> dict[str, Any] | None:
        """Stage the VRAM buffer on the local WPI driver and map it into this process.

        For the trainer master (is_master=True):
            1. Calls NodeStageWeight to allocate VRAM on the local driver
            2. Receives FD from the driver via UNIX socket
            3. Imports CUDA memory via cuMemImportFromShareableHandle
            4. Wraps as PyTorch tensor for direct weight writing

        For rollout workers (is_master=False):
            1. Receives FD from the (already-staged) driver
            2. Imports CUDA memory
            3. Connects to the notify socket for READY signals

        Returns:
            Metadata dict for topology building (master returns ZMQ info + node IP,
            non-master returns node IP for target discovery).
        """
        try:
            import urllib.request
            req = urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/ip', headers={'Metadata-Flavor': 'Google'})
            node_ip = urllib.request.urlopen(req, timeout=1).read().decode().strip()
            print(f"WPI: Derived Node IP from metadata server: {node_ip}", flush=True)
        except Exception as e:
            print(f"WPI Warning: Failed to get Node IP from metadata server: {e}. Falling back to Ray node IP.", flush=True)
            node_ip = ray.util.get_node_ip_address().strip("[]")
        gpu_id = int(os.environ.get("LOCAL_RANK", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]))

        if not self._staged:
            # Every node must stage the buffer on its local WPI driver
            # so the driver creates the VRAM allocation + FD sockets.
            # For the trainer master, stage_weight allocates the source buffer.
            # For rollout workers, stage_weight allocates an empty receive buffer.
            self.wpi_client.stage_weight(
                buffer_id=self.buffer_id,
                size_bytes=self.bucket_size,
                claim_id=self.claim_id,
            )
            self._staged = True

        # Receive FD and import CUDA memory
        fd = self.wpi_client.receive_fd(self.buffer_id, gpu_id=gpu_id)
        device_ptr = self.wpi_client.import_cuda_memory(fd, self.bucket_size, device_id=gpu_id)
        self.vram_buffer = self.wpi_client.wrap_as_buffer(device_ptr, self.bucket_size)
        print(f"WPI: VRAM buffer mapped, shape={self.vram_buffer.shape}, device={self.vram_buffer.device}", flush=True)

        if not self.is_master:
            # Connect to notify socket for READY signals
            self.wpi_client.connect_notify_socket(self.buffer_id)

        if self.is_master:
            return WPIMasterMetadata(
                zmq_ip=self.ip,
                zmq_port=self.listen_port,
                node_ip=node_ip,
            )
        else:
            return {"node_ip": node_ip}

    @classmethod
    def build_topology(
        cls,
        trainer_world_size: int,
        rollout_world_size: int,
        metadata: list[dict],
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        """Build communication topology between trainer and rollout workers.

        Assigns ranks:
            - Rank 0: trainer master (the one that sends weights)
            - Rank -1: other trainer workers (they skip send)
            - Ranks 1..N: rollout workers (they receive weights)

        Also collects unique rollout node IPs for NodePropagate targeting.

        Args:
            trainer_world_size: Number of trainer workers.
            rollout_world_size: Number of rollout workers.
            metadata: List of metadata dicts from prepare() calls.
                First trainer_world_size entries are trainer metadata,
                remaining are rollout metadata.

        Returns:
            Tuple of (trainer_kwargs, rollout_kwargs) dicts for init_process_group().
        """
        master_metadata = metadata[0]  # Trainer rank 0 metadata
        trainer_node_ip = master_metadata.node_ip

        # Collect unique rollout node IPs for NodePropagate,
        # excluding the trainer's own node to avoid "Duplicate GPU" in NCCL.
        # When trainer and rollout workers share a node, the trainer's WPI driver
        # already has the buffer — no need to NCCL broadcast to itself.
        rollout_metadata = metadata[trainer_world_size:]
        rollout_node_ips = list({
            m["node_ip"] for m in rollout_metadata
            if m is not None and m["node_ip"] != trainer_node_ip
        })

        trainer_kwargs = {
            "rank": [0] + [-1] * (trainer_world_size - 1),
            "world_size": [rollout_world_size + 1] * trainer_world_size,
            "master_metadata": [master_metadata] * trainer_world_size,
            "target_node_ids": [rollout_node_ips] * trainer_world_size,
        }
        rollout_kwargs = {
            "rank": list(range(1, rollout_world_size + 1)),
            "world_size": [rollout_world_size + 1] * rollout_world_size,
            "master_metadata": [master_metadata] * rollout_world_size,
            "target_node_ids": [None] * rollout_world_size,  # Only master needs targets
        }
        return trainer_kwargs, rollout_kwargs

    def init_process_group(
        self,
        rank: int,
        world_size: int,
        master_metadata: WPIMasterMetadata,
        target_node_ids: list[str] | None = None,
    ):
        """Initialize communication channels.

        Unlike NCCLCheckpointEngine, this does NOT create a Ray collective group.
        WPI driver manages NCCL internally. We only set up ZMQ PUB/SUB for the
        tensor metadata (offset table) broadcast.

        Args:
            rank: Rank of this process (0=master, -1=skip, 1..N=rollout).
            world_size: Total number of participants.
            master_metadata: ZMQ connection info from the trainer master.
            target_node_ids: Rollout node IPs (only used by master for NodePropagate).
        """
        self.rank = rank
        self.world_size = world_size
        self.target_node_ids = target_node_ids

        # For trainer workers other than rank 0, nothing to do
        if rank < 0:
            return

        # For rollout workers, connect ZMQ SUB
        if rank > 0:
            self._connect_zmq_client(master_metadata)

        print(f"WPI: init_process_group rank={rank}, world_size={world_size}", flush=True)

    @torch.no_grad()
    async def send_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None]):
        """Pack weights into the WPI shared VRAM buffer and trigger cross-node broadcast.

        Steps:
            1. Pack (name, tensor) pairs into self.vram_buffer with offset tracking
            2. Broadcast the offset table (metadata) via ZMQ PUB to all rollout workers
            3. Call NodePropagate via gRPC to trigger WPI driver's NCCL broadcast
               (this also sends READY notification to rollout consumers)

        Args:
            weights: Generator yielding (param_name, tensor) tuples from the model engine.
        """
        assert self.rank is not None and self.rank <= 0, "Only trainer workers should call send_weights."

        # For trainer ranks other than 0, consume weights without sending
        if self.rank < 0:
            for name, weight in weights:
                pass
            return

        assert self.vram_buffer is not None, "VRAM buffer not initialized. Call prepare() first."
        assert self.target_node_ids is not None, "Target node IDs not set. Call init_process_group() first."

        start_time = time.time()
        bucket_meta: dict[str, TensorMeta] = {}
        offset = 0

        # Pack all weight tensors into the flat VRAM buffer
        for name, weight in weights:
            nbytes = weight.nbytes
            assert offset + nbytes <= self.bucket_size, (
                f"Weight {name}({weight.shape}, {weight.dtype}) would exceed buffer size. "
                f"Current offset: {offset}, weight size: {nbytes}, buffer size: {self.bucket_size}. "
                f"Increase update_weights_bucket_megabytes in checkpoint_engine config."
            )

            bucket_meta[name] = {
                "name": name,
                "shape": weight.shape,
                "dtype": weight.dtype,
                "offset": offset,
            }
            # Copy weight data into the shared VRAM buffer
            self.vram_buffer[offset : offset + nbytes].copy_(
                weight.view(-1).view(torch.uint8), non_blocking=True
            )
            offset += nbytes

        # Synchronize to ensure all copies are complete before broadcast
        torch.cuda.synchronize()

        # Broadcast metadata via ZMQ
        metadata = {"bucket_meta": bucket_meta, "total_bytes": offset}
        self.socket.send_string(self.topic, flags=zmq.SNDMORE)
        self.socket.send_pyobj(metadata)
        print(f"WPI: Metadata broadcast, {len(bucket_meta)} params, {offset} bytes", flush=True)

        # Trigger WPI driver to NCCL broadcast the buffer to all rollout nodes
        # This is a blocking gRPC call that completes after NCCL bcast + READY notification
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            self.wpi_client.propagate,
            self.buffer_id,
            self.target_node_ids,
        )

        elapsed = time.time() - start_time
        bandwidth_gbps = (offset / (1024**3)) / elapsed if elapsed > 0 else 0
        print(
            f"WPI: send_weights complete, {len(bucket_meta)} params, "
            f"{offset / (1024**2):.1f} MB in {elapsed:.2f}s ({bandwidth_gbps:.2f} GB/s)",
            flush=True
        )

    @torch.no_grad()
    async def receive_weights(self) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """Wait for READY notification and yield weight tensors from the local VRAM buffer.

        Steps:
            1. Wait for READY notification from WPI driver (indicates NCCL broadcast is done)
            2. Receive offset table (metadata) via ZMQ SUB
            3. Yield (name, tensor) tuples by slicing the local VRAM buffer

        Yields:
            (param_name, tensor) tuples matching the model's state_dict keys.
        """
        assert self.rank is not None and self.rank > 0, "Only rollout workers should call receive_weights."
        assert self.vram_buffer is not None, "VRAM buffer not initialized. Call prepare() first."

        start_time = time.time()

        # Wait for WPI driver to signal that NCCL broadcast is complete
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.wpi_client.wait_for_ready, 300.0)

        # Receive metadata (offset table) from trainer via ZMQ
        # The ZMQ message may have been sent slightly before or after READY
        self.socket.recv_string()  # topic filter
        metadata = self.socket.recv_pyobj()
        bucket_meta = metadata["bucket_meta"]
        total_bytes = metadata.get("total_bytes", 0)

        print(f"WPI: Received metadata, {len(bucket_meta)} params, {total_bytes} bytes", flush=True)

        # Yield tensors from the VRAM buffer
        for name, meta in bucket_meta.items():
            dtype = meta["dtype"]
            shape = meta["shape"]
            offset = meta["offset"]
            size = dtype.itemsize * shape.numel()

            tensor = self.vram_buffer[offset : offset + size].view(dtype=dtype).view(shape)
            yield name, tensor

        elapsed = time.time() - start_time
        print(
            f"WPI: receive_weights complete, {len(bucket_meta)} params in {elapsed:.2f}s",
            flush=True
        )

    def finalize(self):
        """Clean up after a weight sync step.

        The VRAM buffer is NOT freed — it persists for reuse across training steps.
        Only the per-step ZMQ connections are cleaned up if rebuild is needed.
        """
        # Don't free the VRAM buffer — it's persistent and reused
        # Don't unstage — the buffer stays allocated in the WPI driver
        torch.cuda.empty_cache()
        print("WPI: finalize() complete (buffer persists for reuse)", flush=True)

    def shutdown(self):
        """Full shutdown — release the WPI VRAM buffer and clean up all resources.

        Should be called when the training job is completely done.
        """
        if self._staged:
            try:
                self.wpi_client.unstage_weight(self.claim_id)
            except Exception as e:
                logger.warning(f"WPI: Error during unstage: {e}")
            self._staged = False

        self.wpi_client.close()
        self.vram_buffer = None

        if self.socket is not None:
            self.socket.close()
            self.socket = None

        print("WPI: shutdown() complete", flush=True)
