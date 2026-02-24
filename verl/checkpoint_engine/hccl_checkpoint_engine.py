# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import asyncio
import logging
import os
from dataclasses import dataclass

import ray
import torch
import zmq
from vllm.distributed.utils import StatelessProcessGroup

from verl.checkpoint_engine.base import (
    CheckpointEngineRegistry,
    CollectiveCheckpointEngine,
    TensorMeta,
)
from verl.utils.distributed import stateless_init_process_group
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class MasterMetadata:
    zmq_ip: str
    zmq_port: int
    dist_ip: str
    dist_port: int


class BroadcastOperation:
    """Async broadcast operation with HCCL in separate thread.

    Args:
        rank (int): The rank of the current process.
        group_name (str): The name of the HCCL process group.
        bucket (torch.Tensor): The tensor to broadcast.
        metadata (dict[str, TensorMeta]): The metadata of the tensor.
        socket (zmq.Socket): The zeromq socket to communicate with master.
        topic (str): The topic to subscribe.
    """

    def __init__(
        self,
        rank: int,
        process_group: StatelessProcessGroup | str,
        bucket: torch.Tensor,
        metadata: dict[str, TensorMeta],
        socket: zmq.Socket,
        topic: str,
    ) -> None:
        self.rank = rank
        self.pyhccl = process_group
        self.bucket = bucket
        self.metadata = metadata
        self.socket = socket
        self.topic = topic

        loop = asyncio.get_running_loop()
        self._task = loop.run_in_executor(None, self._run)

    def _run(self):
        # broadcast tensor meta via zeromq PUB/SUB
        if self.rank == 0:
            self.socket.send_string(self.topic, flags=zmq.SNDMORE)
            self.socket.send_pyobj(self.metadata)
        else:
            self.socket.recv_string()
            self.metadata = self.socket.recv_pyobj()

        # broadcast tensor via HCCL
        self.pyhccl.broadcast(self.bucket, src=0)

    async def wait_for_complete(self) -> dict[str, TensorMeta]:
        """Wait for the broadcast operation to complete.

        Returns:
            dict[str, TensorMeta]: The bucket meta after broadcast.
        """
        await self._task
        return self.metadata


@CheckpointEngineRegistry.register("hccl")
class HCCLCheckpointEngine(CollectiveCheckpointEngine):
    """HCCL checkpoint engine with collective communication.

    Args:
        bucket_size (int): Bucket size in bytes to transfer multiple weights at one time. Note that we use
            two buffer to send and recv weights at same time, so the device memory overhead is 2 * bucket_size.
        group_name (str): The name of the HCCL process group. Defaults to "default".
        rebuild_group (bool): Whether to rebuild the HCCL process group in each update. Defaults to False.
        is_master (bool): Whether the current process is the master process. Defaults to False.
        rollout_dtype (torch.dtype): The dtype of the weights received from rollout workers. Defaults to torch.bfloat16.
    """

    def __init__(
        self,
        bucket_size: int,
        group_name: str = "default",
        rebuild_group: bool = False,
        is_master: bool = False,
        rollout_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.bucket_size = bucket_size
        self.group_name = group_name
        self.rebuild_group = rebuild_group
        self.rollout_dtype = rollout_dtype
        self.pyhccl = None
        self.device = torch.npu.current_device()

        # start zeromq server for broadcasting bucket tensor metadata
        self.is_master = is_master
        self.topic = "bucket_metadata"
        if self.is_master:
            self._start_zmq_server()
            self.dist_port, _ = get_free_port(self.ip)

    @property
    def bucket_size(self) -> int:
        """Return the bucket size in bytes."""
        return self._bucket_size

    @bucket_size.setter
    def bucket_size(self, value: int):
        """Set the bucket size in bytes."""
        self._bucket_size = value

    @property
    def rank(self) -> int:
        """Return the rank of the current process."""
        return self._rank

    @rank.setter
    def rank(self, value: int):
        """Set the rank of the current process."""
        self._rank = value

    @property
    def send_buf(self):
        """Return the send buffer."""
        return self._send_buf

    @send_buf.setter
    def send_buf(self, value):
        """Set the send buffer."""
        self._send_buf = value

    @property
    def recv_buf(self):
        """Return the receive buffer."""
        return self._recv_buf

    @recv_buf.setter
    def recv_buf(self, value):
        """Set the receive buffer."""
        self._recv_buf = value

    def prepare(self) -> MasterMetadata:
        self.send_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device="npu")
        self.recv_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device="npu")

        return (
            MasterMetadata(zmq_ip=self.ip, zmq_port=self.zmq_port, dist_ip=self.ip, dist_port=self.dist_port)
            if self.is_master
            else None
        )

    def finalize(self):
        """Destroy the HCCL process group if rebuild_group is True."""
        if self.rebuild_group:
            if self.rank >= 0:
                self.pyhccl.destroyComm(self.pyhccl.comm)
                self.pyhccl = None
            self.rank = None
            self.world_size = None

        self.send_buf = None
        self.recv_buf = None

    @classmethod
    def build_topology(cls, trainer_world_size: int, rollout_world_size: int, metadata: list[dict]):
        trainer_kwargs = {
            "rank": [0] + [-1] * (trainer_world_size - 1),
            "world_size": [rollout_world_size + 1] * trainer_world_size,
            "master_metadata": [metadata[0]] * trainer_world_size,
        }
        rollout_kwargs = {
            "rank": list(range(1, rollout_world_size + 1)),
            "world_size": [rollout_world_size + 1] * rollout_world_size,
            "master_metadata": [metadata[0]] * rollout_world_size,
        }
        return trainer_kwargs, rollout_kwargs

    def _start_zmq_server(self):
        self.ip = ray.util.get_node_ip_address().strip("[]")
        self.zmq_port, _ = get_free_port(self.ip)

        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        if is_valid_ipv6_address(self.ip):
            address = f"tcp://[{self.ip}]:{self.zmq_port}"
            self.socket.setsockopt(zmq.IPV6, 1)
        else:
            address = f"tcp://{self.ip}:{self.zmq_port}"

        self.socket.bind(address)

    def _connect_zmq_client(self, metadata: MasterMetadata):
        assert not self.is_master, "Master process should not connect to other processes."
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        if is_valid_ipv6_address(metadata.zmq_ip):
            address = f"tcp://[{metadata.zmq_ip}]:{metadata.zmq_port}"
            self.socket.setsockopt(zmq.IPV6, 1)
        else:
            address = f"tcp://{metadata.zmq_ip}:{metadata.zmq_port}"

        self.socket.connect(address)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)

    def init_process_group(self, rank: int, world_size: int, master_metadata: MasterMetadata):
        """Initialize the HCCL process group.

        Args:
            rank (int): The rank of the current process.
            world_size (int): The total number of processes.
        """
        # For trainer workers other than rank 0, their rank should be -1.
        if rank < 0:
            self.rank = rank
            self.world_size = world_size
            return

        if self.rebuild_group or self.pyhccl is None:
            self.pyhccl = stateless_init_process_group(
                master_metadata.dist_ip, master_metadata.dist_port, rank, world_size, self.device
            )
            self.rank = rank
            self.world_size = world_size
        else:
            assert self.rank == rank, f"rank {rank} is not equal to self.rank {self.rank}"
            assert self.world_size == world_size, (
                f"world_size {world_size} is not equal to self.world_size {self.world_size}"
            )

        if self.rank > 0:
            self._connect_zmq_client(master_metadata)

        # barrier
        signal = torch.tensor([1], dtype=torch.int8, device=torch.npu.current_device())
        self.pyhccl.all_reduce(signal)

        logger.info(f"init_process_group rank: {self.rank}, world_size: {self.world_size}")

    def _synchronize(self):
        """Synchronize NPU operations."""
        torch.npu.synchronize()

    def _create_broadcast_send_op(self, bucket, metadata):
        """Create broadcast operation for sending weights."""
        return BroadcastOperation(
            rank=self.rank,
            process_group=self.pyhccl,
            bucket=bucket,
            metadata=metadata,
            socket=self.socket,
            topic=self.topic,
        )

    def _create_broadcast_recv_op(self, bucket):
        """Create broadcast operation for receiving weights."""
        return BroadcastOperation(
            rank=self.rank,
            process_group=self.pyhccl,
            bucket=bucket,
            metadata=None,
            socket=self.socket,
            topic=self.topic,
        )

    def _copy_to_buffer(self, buffer, tensor, offset):
        """Copy tensor to buffer."""
        buffer[offset : offset + tensor.nbytes] = tensor.view(-1).view(torch.uint8)
