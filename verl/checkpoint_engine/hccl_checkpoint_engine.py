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

import torch

from verl.checkpoint_engine.base import (
    CheckpointEngineRegistry,
    CollectiveCheckpointEngine,
    MasterMetadata,
)
from verl.utils.distributed import stateless_init_process_group
from verl.utils.net_utils import get_free_port

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


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
        self._pyhccl = None
        self._device = torch.npu.current_device()
        super().__init__(
            bucket_size=bucket_size,
            group_name=group_name,
            rebuild_group=rebuild_group,
            is_master=is_master,
            rollout_dtype=rollout_dtype,
        )
        if self.is_master:
            self._dist_port, _ = get_free_port(self._ip)

    def prepare(self) -> MasterMetadata | None:
        """Prepare checkpoint engine before each step send_weights/receive_weights."""
        self._send_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device="npu")
        self._recv_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device="npu")

        if self.is_master:
            return MasterMetadata(
                zmq_ip=self._ip,
                zmq_port=self._zmq_port,
                dist_ip=self._ip,
                dist_port=self._dist_port,
            )
        return None

    def finalize(self):
        """Finalize checkpoint engine after each step send_weights/receive_weights."""
        if self.rebuild_group:
            if self._rank is not None and self._rank >= 0:
                self._pyhccl.destroyComm(self._pyhccl.comm)
                self._pyhccl = None
            self._rank = None
            self._world_size = None

        self._send_buf = None
        self._recv_buf = None

    def init_process_group(self, rank: int, world_size: int, master_metadata: MasterMetadata):
        """Initialize the HCCL process group.

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

        if self.rebuild_group or self._pyhccl is None:
            self._pyhccl = stateless_init_process_group(
                master_metadata.dist_ip, master_metadata.dist_port, rank, world_size, self._device
            )
            self._rank = rank
            self._world_size = world_size
        else:
            assert self._rank == rank, f"rank {rank} is not equal to self.rank {self._rank}"
            assert self._world_size == world_size, (
                f"world_size {world_size} is not equal to self.world_size {self._world_size}"
            )

        if self._rank > 0:
            self._connect_zmq_client(master_metadata)

        # barrier
        signal = torch.tensor([1], dtype=torch.int8, device=torch.npu.current_device())
        self._pyhccl.all_reduce(signal)

        logger.info(f"init_process_group rank: {self._rank}, world_size: {self._world_size}")

    def _broadcast(self, bucket, src_rank: int):
        """Broadcast tensor using HCCL."""
        self._pyhccl.broadcast(bucket, src=src_rank)

    def _synchronize(self):
        """Synchronize NPU operations."""
        torch.npu.synchronize()

    def _copy_to_buffer(self, buffer, tensor, offset):
        """Copy tensor to buffer using torch."""
        buffer[offset : offset + tensor.nbytes] = tensor.view(-1).view(torch.uint8)
