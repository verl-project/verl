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
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Generator
from unittest.mock import patch

with patch("importlib.metadata.distributions", return_value=[]):
    import cupy as cp

import ray
import ray.util.collective as collective
import torch
import zmq

from verl.checkpoint_engine.base import (
    CheckpointEngine,
    CheckpointEngineRegistry,
    TensorMeta,
    align_bucket_offset,
    merge_weight_chunks,
    split_weight_chunks,
)
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class MasterMetadata:
    """Endpoint of the broadcast source (actor rank 0), handed to every worker in the group."""

    zmq_ip: str
    zmq_port: int
    multi_sender: bool


@dataclass
class WorkerMetadata:
    """What each worker reports from `prepare()` for `build_topology` to place it in the group.

    Only the source rank fills `master`; every worker reports `node_id` so it can be matched
    against the source's.
    """

    node_id: str
    master: MasterMetadata | None = None


class BroadcastOperation:
    """Async broadcast operation with NCCL in separate thread.

    Args:
        rank (int): The rank of the current process.
        group_name (str): The name of the NCCL process group.
        bucket (cp.ndarray | torch.Tensor): The tensor to broadcast.
        metadata (dict[str, TensorMeta]): The metadata of the tensor.
        socket (zmq.Socket): The zeromq socket to communicate with master.
        topic (str): The topic to subscribe.
    """

    def __init__(
        self,
        rank: int,
        group_name: str,
        bucket: cp.ndarray | torch.Tensor,
        metadata: dict[str, TensorMeta],
        socket: zmq.Socket,
        topic: str,
    ) -> None:
        self.rank = rank
        self.group_name = group_name
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
            self.bucket = self.bucket[: self.metadata["length"]]

        # broadcast tensor via NCCL
        collective.broadcast(self.bucket, src_rank=0, group_name=self.group_name)

    async def wait_for_complete(self) -> dict[str, TensorMeta]:
        """Wait for the broadcast operation to complete.
        (This does not guarantee that the NCCL kernel has finished, only that it has been enqueued.)

        Returns:
            dict[str, TensorMeta]: The bucket meta after broadcast.
        """
        await self._task
        return self.metadata


@CheckpointEngineRegistry.register("nccl")
class NCCLCheckpointEngine(CheckpointEngine):
    """NCCL checkpoint engine with collective communication.

    Args:
        bucket_size (int): Bucket size in bytes to transfer multiple weights at one time. Note that we use
            two buffer to send and recv weights at same time, so the device memory overhead is 2 * bucket_size.
        group_name (str): The name of the NCCL process group. Defaults to "default".
        rebuild_group (bool): Whether to rebuild the NCCL process group in each update. Defaults to False.
        is_master (bool): Whether the current process is the master process. Defaults to False.
        rollout_dtype (torch.dtype): The dtype of the weights received from rollout workers. Defaults to torch.bfloat16.
        multi_sender (bool): Whether to also admit the source's NVLink-local actor workers into the
            broadcast group as relays, widening the fan-out at the root. Defaults to False, which
            keeps the group at one sender (actor rank 0) plus the rollout workers.
    """

    def __init__(
        self,
        bucket_size: int,
        group_name: str = "default",
        rebuild_group: bool = False,
        is_master: bool = False,
        rollout_dtype: torch.dtype = torch.bfloat16,
        multi_sender: bool = True,
    ) -> None:
        self.bucket_size = bucket_size
        self.group_name = group_name
        self.rebuild_group = rebuild_group
        self.rollout_dtype = rollout_dtype
        self.multi_sender = multi_sender

        # start zeromq server for broadcasting bucket tensor metadata
        self.is_master = is_master
        self.topic = "bucket_metadata"
        if self.is_master:
            self._start_zmq_server()

    @staticmethod
    def get_node_id() -> str:
        """Identity of the node this GPU belongs to, used as a proxy for NVLink reachability.

        GPUs on the same node are NVLink-reachable, and a node never spans more than one NVLink
        domain, so matching on node id can only ever under-select peers, never over-select them.
        """
        return ray.get_runtime_context().get_node_id()

    def prepare(self) -> WorkerMetadata:
        # For master process, use cupy instead of torch to avoid memory register error
        # when `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
        if self.is_master:
            self.send_buf = cp.zeros(self.bucket_size, dtype=cp.uint8)
            self.recv_buf = cp.zeros(self.bucket_size, dtype=cp.uint8)
        else:
            self.send_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device="cuda")
            self.recv_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device="cuda")

        master = (
            MasterMetadata(zmq_ip=self.ip, zmq_port=self.listen_port, multi_sender=self.multi_sender)
            if self.is_master
            else None
        )
        return WorkerMetadata(node_id=self.get_node_id(), master=master)

    def finalize(self):
        """Destroy the NCCL process group if rebuild_group is True."""
        if self.rebuild_group:
            if self.rank >= 0:
                collective.destroy_collective_group(self.group_name)
            self.rank = None
            self.world_size = None

        self.send_buf = None
        self.recv_buf = None

        torch.cuda.empty_cache()

    @staticmethod
    def _single_sender_ranks(actor_wg_world_size: int) -> list[int]:
        """Only actor rank 0 joins the group; every other actor worker sits the broadcast out."""
        return [0] + [-1] * (actor_wg_world_size - 1)

    @staticmethod
    def _multi_sender_ranks(actor_wg_world_size: int, metadata: list[WorkerMetadata]) -> list[int]:
        """Actor rank 0 joins, plus every actor worker on its node, which are its NVLink peers.

        Those peers carry no data the rollout needs. They join so NCCL has somewhere to fan a
        bucket out to over NVLink, from which it can push on over each peer's own NIC. A peer on
        any other node would have to pull a full copy over the fabric to contribute nothing, so it
        gets rank -1 instead.
        """
        source_node = metadata[0].node_id

        # Ranks must stay contiguous, so number the survivors as we walk the actor workers.
        ranks, next_rank = [], 0
        for i in range(actor_wg_world_size):
            if i == 0 or metadata[i].node_id == source_node:
                ranks.append(next_rank)
                next_rank += 1
            else:
                ranks.append(-1)
        return ranks

    @classmethod
    def build_topology(cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list[WorkerMetadata]):
        master = metadata[0].master
        assert master is not None, "actor rank 0 must be the checkpoint engine master"

        if master.multi_sender:
            actor_ranks = cls._multi_sender_ranks(actor_wg_world_size, metadata)
        else:
            actor_ranks = cls._single_sender_ranks(actor_wg_world_size)

        # Multi-sender degrades to a single sender when rank 0 has no actor peers on its node.
        num_senders = sum(rank >= 0 for rank in actor_ranks)
        world_size = num_senders + rollout_world_size
        logger.info(
            f"build_topology: {num_senders} of {actor_wg_world_size} actor workers send, world_size {world_size}"
        )

        actor_wg_kwargs = {
            "rank": actor_ranks,
            "world_size": [world_size] * actor_wg_world_size,
            "master_metadata": [master] * actor_wg_world_size,
            "num_senders": [num_senders] * actor_wg_world_size,
        }
        rollout_kwargs = {
            "rank": list(range(num_senders, world_size)),
            "world_size": [world_size] * rollout_world_size,
            "master_metadata": [master] * rollout_world_size,
            "num_senders": [num_senders] * rollout_world_size,
        }
        return actor_wg_kwargs, rollout_kwargs

    def _start_zmq_server(self):
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

    def init_process_group(self, rank: int, world_size: int, master_metadata: MasterMetadata, num_senders: int):
        """Initialize the NCCL process group.

        Args:
            rank (int): The rank of the current process.
            world_size (int): The total number of processes.
            master_metadata (MasterMetadata): The endpoint of the broadcast source.
            num_senders (int): How many actor-side ranks the group starts with. Ranks below this
                send or relay weights; ranks at or above it consume them.
        """
        self.num_senders = num_senders

        # Actor workers left out of the group are given rank -1.
        if rank < 0:
            self.rank = rank
            self.world_size = world_size
            return

        if self.rebuild_group or not collective.is_group_initialized(self.group_name):
            collective.init_collective_group(world_size, rank, "nccl", self.group_name)
            self.rank = rank
            self.world_size = world_size
        else:
            assert self.rank == rank, f"rank {rank} is not equal to self.rank {self.rank}"
            assert self.world_size == world_size, (
                f"world_size {world_size} is not equal to self.world_size {self.world_size}"
            )

        # Only consumers read the bucket metadata stream; relays never touch the payload.
        if self.rank >= num_senders:
            self._connect_zmq_client(master_metadata)
        collective.barrier(self.group_name)

        logger.info(f"init_process_group rank: {self.rank}, world_size: {self.world_size}")

    @torch.no_grad()
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ):
        """Send the weights of the model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        assert self.rank < self.num_senders, "Rollout workers should not send weights."

        # Actor workers left out of the group still have to walk the generator: producing each
        # weight is itself collective over the actor group.
        if self.rank < 0:
            for _name, _weight in weights:
                pass
            return

        if self.rank > 0:
            await self._relay_weights(weights)
            return

        send_buf, recv_buf = self.send_buf, self.recv_buf
        broadcast_op = None

        # In the case of multi senders, the broadcast and all-gather kernels must
        # be queued in a deterministic order, otherwise the NCCL kernel may deadlock.
        # Moreover the wait_for_complete() function only waits for the NCCL kernel to be enqueued,
        # not for the kernel to finish.
        pipelined = self.num_senders == 1

        start_time = time.time()
        bucket_meta: dict[str, TensorMeta] = {}
        offset = 0
        async for tensor_meta, chunk in split_weight_chunks(weights, self.bucket_size):
            # start each tensor at an offset its dtype can be viewed at (mixed-dtype streams)
            offset = align_bucket_offset(offset, tensor_meta.dtype)
            # fill the tensor bucket
            if offset + tensor_meta.chunk_size > self.bucket_size:
                torch.cuda.synchronize()

                # wait previous broadcast op finish
                if pipelined and broadcast_op is not None:
                    await broadcast_op.wait_for_complete()

                broadcast_op = BroadcastOperation(
                    rank=self.rank,
                    group_name=self.group_name,
                    bucket=send_buf[:offset],
                    metadata={"bucket_meta": bucket_meta, "is_last": False, "length": offset},
                    socket=self.socket,
                    topic=self.topic,
                )

                # swap send_buf and recv_buf
                send_buf, recv_buf = recv_buf, send_buf
                bucket_meta = {}
                offset = 0

            assert offset + tensor_meta.chunk_size <= self.bucket_size
            assert tensor_meta.name not in bucket_meta

            tensor_meta.offset = offset
            bucket_meta[tensor_meta.name] = tensor_meta
            send_buf[offset : offset + tensor_meta.chunk_size] = cp.asarray(chunk)
            offset += tensor_meta.chunk_size

            # keep the relays in step: no-op once the bucket's broadcast has already been drained
            if not pipelined and broadcast_op is not None:
                await broadcast_op.wait_for_complete()

        # broadcast last bucket
        torch.cuda.synchronize()
        if pipelined and broadcast_op is not None:
            await broadcast_op.wait_for_complete()

        broadcast_op = BroadcastOperation(
            rank=self.rank,
            group_name=self.group_name,
            bucket=send_buf[:offset],
            metadata={"bucket_meta": bucket_meta, "is_last": True, "length": offset},
            socket=self.socket,
            topic=self.topic,
        )
        await broadcast_op.wait_for_complete()

        # the wait_for_complete() function just waits for the NCCL kernel to be enqueued,
        # not for the kernel to finish, hence we need to synchronize to make sure the
        # buffer does not get freed before the kernel finishes.
        torch.cuda.synchronize()

        logger.info(f"Rank {self.rank} send weights done, time cost: {time.time() - start_time:.2f}s")

    async def _relay_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None]):
        """Join every broadcast as an NVLink-local relay for rank 0, dropping the payload.

        A relay holds nothing the rollout needs, so it only has to match the root bucket for
        bucket. It recomputes the boundaries from its own copy of the weight stream under the same
        rule `send_weights` uses above, which keeps the two in step without a round trip -- and it
        has to be derived locally rather than read off the wire, because pulling a weight is itself
        collective over the actor group, so blocking on the wire would deadlock against rank 0's
        own gathers.
        """
        start_time = time.time()
        offset = 0

        # a single buffer suffices: nothing reads the bucket, so successive broadcasts can share it
        async for tensor_meta, _ in split_weight_chunks(weights, self.bucket_size, meta_only=True):
            if offset + tensor_meta.chunk_size > self.bucket_size:
                collective.broadcast(self.recv_buf[:offset], src_rank=0, group_name=self.group_name)
                offset = 0

            offset += tensor_meta.chunk_size

        # relay last bucket
        collective.broadcast(self.recv_buf[:offset], src_rank=0, group_name=self.group_name)

        # wait for the enqueued NCCL kernels, so finalize() cannot free the buffer under them
        torch.cuda.synchronize()

        logger.info(f"Rank {self.rank} relay weights done, time cost: {time.time() - start_time:.2f}s")

    @torch.no_grad()
    async def receive_weights(
        self,
        global_steps: int | None = None,
    ) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """Receive the weights of the model.

        Yields:
            A tuple of the name of the weight tensor and the tensor itself.
        """
        async for name, weight in merge_weight_chunks(self._receive_weight_chunks(), self.bucket_size):
            yield name, weight

    async def _receive_weight_chunks(self) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """Receive the weight chunks of the model.

        Yields:
            A tuple of the name of the weight tensor and the chunk itself.
        """
        assert self.rank > 0, "Rank 0 should not receive weights."
        send_buf, recv_buf = self.send_buf, self.recv_buf
        total_bytes, total_params = 0, 0

        # receive first bucket
        start_time = time.time()
        broadcast_op = BroadcastOperation(
            rank=self.rank,
            group_name=self.group_name,
            bucket=recv_buf,
            metadata=None,
            socket=self.socket,
            topic=self.topic,
        )
        metadata = await broadcast_op.wait_for_complete()
        total_bytes += metadata["length"]
        total_params += len(metadata["bucket_meta"])

        # wait for the NCCL broadcast kernel to finish before we yield the tensors
        # otherwise if the buffer is clone using a non-blocking copy, it may
        # lead to data corruption
        torch.cuda.synchronize()

        # swap send_buf and recv_buf
        send_buf, recv_buf = recv_buf, send_buf
        while not metadata["is_last"]:
            # 1. receive next bucket
            broadcast_op = BroadcastOperation(
                rank=self.rank,
                group_name=self.group_name,
                bucket=recv_buf,
                metadata=None,
                socket=self.socket,
                topic=self.topic,
            )

            # 2. yield tensor from send_buf
            for name, tensor_meta in metadata["bucket_meta"].items():
                tensor = send_buf[tensor_meta.offset : tensor_meta.offset + tensor_meta.chunk_size]
                yield tensor_meta, tensor

            # 3. wait for next bucket broadcast finish
            metadata = await broadcast_op.wait_for_complete()
            total_bytes += metadata["length"]
            total_params += len(metadata["bucket_meta"])

            # 4. swap send_buf and recv_buf
            torch.cuda.synchronize()  # sync non-blocking copy
            send_buf, recv_buf = recv_buf, send_buf

        # yield tensor from send_buf
        for name, tensor_meta in metadata["bucket_meta"].items():
            tensor = send_buf[tensor_meta.offset : tensor_meta.offset + tensor_meta.chunk_size]
            yield tensor_meta, tensor

        time_cost = time.time() - start_time
        bandwidth = total_bytes / time_cost / (1024 * 1024 * 1024)
        logger.info(
            f"Rank {self.rank} receive weights done, total_params: {total_params}, "
            f"time cost: {time_cost:.2f}s, bandwidth: {bandwidth:.2f} GB/s"
        )
