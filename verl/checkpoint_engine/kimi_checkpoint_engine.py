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
import concurrent.futures
import logging
import os
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Generator

import ray
import torch
from collections import defaultdict
from checkpoint_engine.ps import ParameterMeta, ParameterServer, _gen_h2d_buckets, _to_named_tensor, H2DBucket
import checkpoint_engine.distributed as dist

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry
from verl.utils.net_utils import get_free_port
from verl.utils.device import get_torch_device, get_nccl_backend

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def ckpt_get_named_tensor_buckets(
        iterable: Generator[tuple[str, torch.Tensor], None, None],
        bucket_bytes: int,
        world_size: int,
        rank_id: int,
        rollout_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    if bucket_bytes <= 0:
        raise ValueError(f"bucket_bytes must be greater than 0, got {bucket_bytes}")

    current_bucket = {}
    current_size = 0
    for tensor_idx, (name, tensor) in enumerate(iterable):
        tensor = tensor.to(rollout_dtype)
        if tensor_idx % world_size == rank_id:
            tensor_size = tensor.element_size() * tensor.numel()
            if current_size + tensor_size > bucket_bytes:
                if current_bucket:
                    yield current_bucket
                    current_bucket = {}
                    current_size = 0

            current_bucket[name] = tensor
            current_size += tensor_size

    if current_bucket:
        yield current_bucket


async def revice_tensor(
        self,
        checkpoint_name: str,
        ranks_group: int,
        ranks: list[int] | None = None,
        bucket_size: int = 2 << 30,
        disable_h2d_buffer: bool = False,
) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
    assert len(self._current_global_parameter_metas) != 0, "parameter metas is empty"
    assert dist.is_initialized(), "process group is not initialized"
    assert self._p2p_store is not None, "p2p store is not initialized"
    assert ranks, "ranks should be set"

    # first execute a barrier to avoid subsequent device oom
    dist.barrier(group=ranks_group)
    buckets = _gen_h2d_buckets(
        self._current_global_parameter_metas,
        bucket_size,
        self._local_rdma_devices,
        self._remote_rdma_devices,
        ranks,
    )
    h2d_buffer: torch.Tensor | None = (
        None
        if disable_h2d_buffer
        else torch.empty(bucket_size, dtype=torch.uint8, device=self.device_manager.device_type)
    )
    # p2p store need to register h2d_buffer to let other ranks read
    if ranks:
        h2d_buffer_name = "__h2d_buffer__"
        if h2d_buffer is not None and self._p2p_store is not None:
            self._p2p_store.register_named_tensors({h2d_buffer_name: h2d_buffer})
    receiver_rank_buckets: list[tuple[int, H2DBucket]] = []
    for receiver_rank, owner_rank, bucket in buckets:
        if receiver_rank != self._rank:
            continue
        receiver_rank_buckets.append((owner_rank, bucket))
    buffer = torch.empty(
        bucket_size * 2, dtype=torch.uint8, device=self.device_manager.device_type
    )
    buckets_by_receiver_rank: dict[int, list[H2DBucket]] = defaultdict(list)

    max_len = 0
    for receiver_rank, _, bucket in buckets:
        buckets_by_receiver_rank[receiver_rank].append(bucket)
        if len(buckets_by_receiver_rank[receiver_rank]) > max_len:
            max_len = len(buckets_by_receiver_rank[receiver_rank])
    gidx = 0
    metadata: list[ParameterMeta]
    try:
        for i in range(max_len):
            if i < len(receiver_rank_buckets) and not disable_h2d_buffer:
                self._copy_to_buffer(
                    checkpoint_name,
                    receiver_rank_buckets[i][1],
                    h2d_buffer,
                    receiver_rank_buckets[i][0] if ranks else None,
                )
            for receiver_rank, _buckets in buckets_by_receiver_rank.items():
                if i >= len(_buckets):
                    continue
                bucket = _buckets[i]
                start = gidx % 2 * bucket_size
                buffer_b: torch.Tensor = buffer[start: start + bucket.size]
                if receiver_rank == self._rank:
                    if disable_h2d_buffer:
                        self._copy_to_buffer(checkpoint_name, bucket, buffer_b)
                    else:
                        buffer_b.data.copy_(h2d_buffer[: bucket.size])
                broadcast_op = BroadcastOperation(
                    rank=receiver_rank,
                    ranks_group=ranks_group,
                    bucket=buffer_b,
                    metadata=bucket.items,
                )
                if gidx == 0:
                    metadata = await broadcast_op.wait_for_complete()
                    gidx += 1
                    continue
                meta_list = _to_named_tensor(metadata, (gidx - 1) % 2 * bucket_size)
                for item in meta_list:
                    shape = item["shape"]
                    if isinstance(shape, list | tuple):
                        shape = torch.Size(shape)
                    assert isinstance(shape, torch.Size)
                    dtype, offset = item["dtype"], item["offset"]
                    size = dtype.itemsize * shape.numel()
                    tensor = buffer[offset: offset + size].view(dtype=dtype).view(shape)
                    yield item["name"], tensor
                metadata = await broadcast_op.wait_for_complete()
                self.device_manager.device_module.synchronize()
                gidx += 1

        meta_list = _to_named_tensor(metadata, (gidx - 1) % 2 * bucket_size)
        for item in meta_list:
            shape = item["shape"]
            if isinstance(shape, list | tuple):
                shape = torch.Size(shape)
            assert isinstance(shape, torch.Size)
            dtype, offset = item["dtype"], item["offset"]
            size = dtype.itemsize * shape.numel()
            tensor = buffer[offset: offset + size].view(dtype=dtype).view(shape)
            yield item["name"], tensor

    finally:
        dist.barrier(group=ranks_group)
        if ranks and h2d_buffer is not None:
            self._p2p_store.unregister_named_tensors([h2d_buffer_name])
        self.device_manager.device_module.empty_cache()


@dataclass
class MasterMetadata:
    ip: str
    port: int


class BroadcastOperation:
    """Async broadcast operation with NCCL in separate thread.

    Args:
        rank (int): The rank of the current process.
        group_name (dist.ProcessGroup): The NCCL process group.
        bucket (torch.Tensor): The tensor to broadcast.
        metadata (list[ParameterMeta]): The metadata of the tensor.
    """

    def __init__(
            self,
            rank: int,
            ranks_group: int,
            bucket: torch.Tensor,
            metadata: list[ParameterMeta],
    ) -> None:
        self.rank = rank
        self.ranks_group = ranks_group
        self.bucket = bucket
        self.metadata = metadata

        loop = asyncio.get_running_loop()
        self._task = loop.run_in_executor(None, self._run)

    def _run(self):
        # broadcast tensor
        dist.broadcast(self.bucket, src=self.rank, group=self.ranks_group)

    async def wait_for_complete(self) -> list[ParameterMeta]:
        """Wait for the broadcast operation to complete.

        Returns:
            list[ParameterMeta]: The bucket meta after broadcast.
        """
        await self._task
        return self.metadata


@CheckpointEngineRegistry.register("kimi_ckpt_engine")
class KIMICheckpointEngine(CheckpointEngine):
    """NCCL checkpoint engine with collective communication.

    Args:
        bucket_size (int): Bucket size in bytes to transfer multiple weights at one time. Note that we use
            two buffer to send and recv weights at same time, so the device memory overhead is 2 * bucket_size.
        group_name (str): The name of the NCCL process group. Defaults to "default".
        rebuild_group (bool): Whether to rebuild the NCCL process group in each update. Defaults to False.
        is_master (bool): Whether the current process is the master process. Defaults to False.
        rollout_dtype (torch.dtype): The dtype of the weights received from rollout workers. Defaults to torch.bfloat16.
    """

    def __init__(
        self,
        train_world_size: int,
        rollout_world_size: int,
        bucket_size: int,
        rebuild_group: bool = False,
        is_master: bool = False,
        rollout_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.train_world_size = train_world_size
        self.rollout_world_size = rollout_world_size
        self.world_size = train_world_size + rollout_world_size

        self.bucket_size = bucket_size
        self.rebuild_group = rebuild_group
        self.rollout_dtype = rollout_dtype
        self.is_master = is_master
        self.initialized = False
        self.checkpoint_name = "kimi_checkpoint_engine"

    def prepare(self) -> MasterMetadata:
        if self.is_master:
            self.ip = ray.util.get_node_ip_address().strip("[]")
            self.listen_port, _ = get_free_port(self.ip)

        return MasterMetadata(ip=self.ip, port=self.listen_port) if self.is_master else None

    def finish(self):
        """Destroy the ckpt engine process group if rebuild_group is True."""
        if self.rebuild_group:
            dist.destroy_process_group()
            self.rank = None
            self.world_size = None
            self.initialized = False

    def init_process_group(self, rank: int, world_size: int, master_metadata: MasterMetadata):
        """Initialize the ckpt engine process group.

        Args:
            rank (int): The rank of the current process.
            world_size (int): The total number of processes.
        """
        if not self.initialized:
            self.parameter_server = ParameterServer(rank=rank, world_size=self.world_size, auto_pg=False, custom_dist=True)
            self.parameter_server.revice_tensor = revice_tensor
            dist.init_process_group(
                host=master_metadata.ip,
                port=master_metadata.port,
                rank=rank,
                world_size=world_size,
                backend=get_nccl_backend(),
            )

            self.rollout_ranks = list(range(self.train_world_size, self.world_size))
            self.rollout_group = dist.new_group(self.rollout_ranks)
            self.initialized = True

        dist.barrier()

    @torch.no_grad()
    async def send_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None]):
        """Send the weights of the model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """

        def offload_cpu(named_tensors: dict[str, torch.Tensor], name: str, tensor: torch.Tensor):
            named_tensors[name] = tensor.to("cpu", non_blocking=True)

        start_time = time.time()
        named_tensors = {}
        for named_tensors_gpu in ckpt_get_named_tensor_buckets(weights, self.bucket_size, self.train_world_size, self.rank,
                                                               self.rollout_dtype):
            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                futures = [
                    executor.submit(
                        offload_cpu,
                        named_tensors,
                        name,
                        tensor,
                    )
                    for name, tensor in named_tensors_gpu.items()
                ]
            for future in concurrent.futures.as_completed(futures):
                future.result()

        get_torch_device().synchronize()

        self.parameter_server.register_checkpoint(self.checkpoint_name, named_tensors=named_tensors)
        named_tensors = {}
        torch.cuda.empty_cache()
        logger.info(f"Rank {self.rank} offload and register, time cost: {time.time() - start_time:.2f}s")

        self.parameter_server.gather_metas(self.checkpoint_name)
        dist.barrier()
        self.parameter_server.unregister_checkpoint(self.checkpoint_name)
        logger.info(f"Rank {self.rank} send weights done, time cost: {time.time() - start_time:.2f}s")

    @torch.no_grad()
    async def receive_weights(self) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """Receive the weights of the model.

        Yields:
            A tuple of the name of the weight tensor and the tensor itself.
        """
        self.parameter_server.gather_metas(self.checkpoint_name)

        start_time = time.time()
        total_bytes = 0
        async for name, tensor in self.parameter_server.revice_tensor(self.checkpoint_name, self.rollout_group,
                                                                        self.rollout_ranks, self.bucket_size):
            total_bytes += tensor.element_size() * tensor.nelement()
            yield name, tensor
        dist.barrier()
        time_cost = time.time() - start_time
        bandwidth = total_bytes / time_cost / (1024 * 1024 * 1024)
        logger.info(
            f"Rank {self.rank} receive weights done,"
            f"time cost: {time_cost:.2f}s, bandwidth: {bandwidth:.2f} GB/s"
        )
