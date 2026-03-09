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
import logging
import os
from unittest.mock import patch

with patch("importlib.metadata.distributions", return_value=[]):
    import cupy as cp

import ray.util.collective as collective
import torch

from verl.checkpoint_engine.base import (
    CheckpointEngineRegistry,
    CollectiveCheckpointEngine,
    MasterMetadata,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@CheckpointEngineRegistry.register("nccl")
class NCCLCheckpointEngine(CollectiveCheckpointEngine):
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
        bucket_size: int,
        group_name: str = "default",
        rebuild_group: bool = False,
        is_master: bool = False,
        rollout_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(
            bucket_size=bucket_size,
            group_name=group_name,
            rebuild_group=rebuild_group,
            is_master=is_master,
            rollout_dtype=rollout_dtype,
        )
        self._async_broadcast_mode = True  # NCCL uses async broadcast

    def prepare(self) -> MasterMetadata | None:
        """Prepare checkpoint engine before each step send_weights/receive_weights."""
        # For master process, use cupy instead of torch to avoid memory register error
        # when `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
        if self.is_master:
            self._send_buf = cp.zeros(self.bucket_size, dtype=cp.uint8)
            self._recv_buf = cp.zeros(self.bucket_size, dtype=cp.uint8)
        else:
            self._send_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device="cuda")
            self._recv_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device="cuda")

        if self.is_master:
            return MasterMetadata(zmq_ip=self._ip, zmq_port=self._zmq_port)
        return None

    def finalize(self):
        """Finalize checkpoint engine after each step send_weights/receive_weights."""
        if self.rebuild_group:
            if self._rank is not None and self._rank >= 0:
                collective.destroy_collective_group(self.group_name)
            self._rank = None
            self._world_size = None

        self._send_buf = None
        self._recv_buf = None

        torch.cuda.empty_cache()

    def init_process_group(self, rank: int, world_size: int, master_metadata: MasterMetadata):
        """Initialize the NCCL process group.

        Args:
            rank: The rank of the current process.
            world_size: The total number of processes.
            master_metadata: The metadata from the master process.
        """
        # For trainer workers other than rank 0, their rank should be -1.
        if rank < 0:
            self._rank = rank
            self._world_size = world_size
            return

        if self.rebuild_group or not collective.is_group_initialized(self.group_name):
            collective.init_collective_group(world_size, rank, "nccl", self.group_name)
            self._rank = rank
            self._world_size = world_size
        else:
            assert self._rank == rank, f"rank {rank} is not equal to self.rank {self._rank}"
            assert self._world_size == world_size, (
                f"world_size {world_size} is not equal to self.world_size {self._world_size}"
            )

        if self._rank > 0:
            self._connect_zmq_client(master_metadata)
        collective.barrier(self.group_name)

        logger.info(f"init_process_group rank: {self._rank}, world_size: {self._world_size}")

    def _broadcast(self, bucket, src_rank: int):
        """Broadcast tensor using NCCL."""
        collective.broadcast(bucket, src_rank=src_rank, group_name=self.group_name)

    def _synchronize(self):
        """Synchronize CUDA operations."""
        torch.cuda.synchronize()

    def _copy_to_buffer(self, buffer, tensor, offset):
        """Copy tensor to buffer using cupy."""
        buffer[offset : offset + tensor.nbytes] = cp.asarray(tensor.view(-1).view(torch.uint8))
