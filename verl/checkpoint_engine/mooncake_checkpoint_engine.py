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
import gc
import logging
import os
import time
from typing import Any, AsyncGenerator, Generator

import ray
import torch
from mooncake.engine import TransferEngine
from vllm.distributed.utils import StatelessProcessGroup

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry, TensorMeta
from verl.utils.device import get_torch_device
from verl.utils.net_utils import get_free_port

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@CheckpointEngineRegistry.register("mooncake")
class MooncakeCheckpointEngine(CheckpointEngine):
    """Mooncake checkpoint engine with p2p communication using TransferEngine

    Args:
        bucket_size (int): Bucket size in bytes to transfer multiple weights at one time.
        device (str): The device to use for the checkpoint engine, "cpu" or "cuda".
        rollout_dtype (torch.dtype): The dtype of the weights received from rollout workers.
        device_name (str): Mooncake device name filter.
    """

    def __init__(
        self,
        bucket_size: int,
        device: str = "cuda",
        rollout_dtype: torch.dtype = torch.bfloat16,
        device_name: str = "",
        is_master: bool = False,
        rebuild_group: bool = False,
    ):
        self.bucket_size = bucket_size
        self.device = device
        self.rollout_dtype = rollout_dtype
        self.is_master = is_master
        self.rebuild_group = rebuild_group

        rank = int(os.environ["RANK"])
        device_count = get_torch_device().device_count()
        local_rank = rank % device_count
        get_torch_device().set_device(local_rank)

        self.engine = TransferEngine()
        hostname = ray.util.get_node_ip_address().strip("[]")
        ret = self.engine.initialize(
            hostname,
            "P2PHANDSHAKE",
            "ascend_direct" if self.device == "npu" else "rdma",
            device_name,
        )
        assert ret == 0, f"TransferEngine initialize failed ret={ret}"

        rpc_port = self.engine.get_rpc_port()
        self.session_id = f"{hostname}:{rpc_port}"
        self.hostname = hostname

        self.buf = torch.empty(2 * self.bucket_size, dtype=torch.uint8, device=self.device)
        self.magic_buf = torch.empty(4 * 1024, dtype=torch.uint8, device=self.device)
        ret = self.engine.batch_register_memory(
            [self.buf.data_ptr(), self.magic_buf.data_ptr()],
            [2 * self.bucket_size, 4 * 1024],
        )
        assert ret == 0, f"batch_register_memory failed ret={ret}"
        logger.info(f"__init__ session_id={self.session_id}")

    def prepare(self) -> dict[str, Any]:
        """Prepare send and recv buckets"""
        # self.buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device=self.device)
        # self.engine.register_memory(self.buf.data_ptr(), self.bucket_size)
        logger.info(
            f"prepare ptr={self.buf.data_ptr():#x} len={2 * self.bucket_size} "
            f"magic_buf_ptr={self.magic_buf.data_ptr():#x}"
        )
        port, _ = get_free_port(self.hostname)
        return {"addr": self.hostname, "port": port}

    @classmethod
    def build_topology(cls, trainer_world_size: int, rollout_world_size: int, metadatas: list[dict]):
        trainer_kwargs = {
            "rank": [0] + [-1] * (trainer_world_size - 1),
            "world_size": [rollout_world_size + 1] * trainer_world_size,
            "metadata": [metadatas[0]] * trainer_world_size,
        }
        rollout_kwargs = {
            "rank": list(range(1, rollout_world_size + 1)),
            "world_size": [rollout_world_size + 1] * rollout_world_size,
            "metadata": [metadatas[0]] * rollout_world_size,
        }
        return trainer_kwargs, rollout_kwargs

    def init_process_group(self, rank: int, world_size: int, metadata: dict[str, Any]):
        self.rank = rank
        self.world_size = world_size
        if rank < 0:
            logger.info(f"init_process_group rank={rank}")
            return

        self.store = StatelessProcessGroup.create(
            host=metadata["addr"],
            port=metadata["port"],
            rank=rank,
            world_size=world_size,
        )

        info = {
            "session_id": self.session_id,
            "ptr": self.buf.data_ptr(),
        }

        info_list = self.store.all_gather_obj(info)
        self.buffer_info = None if rank == 0 else info_list[rank - 1]

        logger.info(f"init_process_group rank={rank} world_size={world_size} buffer_info={self.buffer_info}")

    def finalize(self):
        """Cleanup communication and deregister memory"""
        self.store = None
        get_torch_device().empty_cache()
        gc.collect()
        logger.info(f"finalize rank={self.rank}")

    async def wait_for_complete(self, buf: torch.Tensor):
        magic = torch.tensor([0xAB, 0xDC, 0xEF, 0x88], dtype=torch.uint8, device=self.device)
        while True:
            if torch.equal(buf[:4], magic):
                break
            await asyncio.sleep(0)

    @torch.no_grad()
    async def send_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None]):
        """Send weights using Mooncake TransferEngine"""
        if self.rank < 0:
            for name, weight in weights:
                pass
            logger.info(f"send_weights rank={self.rank}")
            return

        total_bytes = 0
        start_time = time.time()
        bucket_meta: dict[str, TensorMeta] = {}
        offset = 0
        should_wait = False
        bufs = [self.buf[: self.bucket_size], self.buf[self.bucket_size :]]
        idx = 0
        current = bufs[idx]

        for name, weight in weights:
            weight = weight.to(self.rollout_dtype)

            if offset + weight.nbytes > self.bucket_size:
                total_bytes += offset
                get_torch_device().synchronize()
                info = {
                    "bucket_meta": bucket_meta,
                    "ptr": current.data_ptr(),
                    "len": offset,
                    "is_last": False,
                }
                # send to rank 1
                self.store.send_obj(info, 1)

                idx ^= 1
                current = bufs[idx]
                bucket_meta = {}
                offset = 0

                if should_wait:
                    await self.wait_for_complete(current)
                should_wait = True

            assert offset + weight.nbytes <= self.bucket_size, (
                f"Weight {name}({weight.shape}, {weight.dtype}) is too large to fit in the bucket."
            )

            bucket_meta[name] = {
                "name": name,
                "shape": weight.shape,
                "dtype": weight.dtype,
                "offset": offset,
            }
            current[offset : offset + weight.nbytes].copy_(weight.view(-1).view(torch.uint8), non_blocking=True)
            offset += weight.nbytes

        get_torch_device().synchronize()
        info = {
            "bucket_meta": bucket_meta,
            "ptr": current.data_ptr(),
            "len": offset,
            "is_last": True,
        }
        self.store.send_obj(info, 1)
        await self.wait_for_complete(current)

        time_cost = time.time() - start_time
        bandwidth = total_bytes / time_cost / (1024 * 1024 * 1024)
        logger.info(
            f"Rank {self.rank} send weights done, "
            f"total bytes: {total_bytes} time cost: {time_cost:.2f}s bandwidth: {bandwidth:.2f} GB/s"
        )

    @torch.no_grad()
    async def receive_weights(self) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """Receive weights using Mooncake TransferEngine"""
        start_time = time.time()
        total_bytes = 0
        bufs = [self.buf[: self.bucket_size], self.buf[self.bucket_size :]]
        idx = 0
        current = bufs[idx]
        self.magic_buf[:4] = torch.tensor([0xAB, 0xDC, 0xEF, 0x88], dtype=torch.uint8, device=self.device)

        while True:
            # 1 receive info from previous rank
            info = self.store.recv_obj(self.rank - 1)
            if idx >= 2 and self.rank < self.world_size - 1:
                await self.wait_for_complete(current)

            ptr = info["ptr"]
            ret = self.engine.transfer_sync_read(
                self.buffer_info["session_id"],
                current.data_ptr(),
                ptr,
                info["len"],
            )
            assert ret == 0, f"transfer_sync_read failed {ret}"
            total_bytes += info["len"]

            # 2 send info to next rank
            info["ptr"] = current.data_ptr()
            if self.rank < self.world_size - 1:
                self.store.send_obj(info, self.rank + 1)

            # 3 yield tensor from current buffer
            for name, meta in info["bucket_meta"].items():
                dtype, shape = meta["dtype"], meta["shape"]
                size = dtype.itemsize * shape.numel()
                tensor = current[meta["offset"] : meta["offset"] + size].view(dtype=dtype).view(shape)
                yield name, tensor

            # 4 write magic data to previous rank
            ret = self.engine.transfer_sync_write(
                self.buffer_info["session_id"],
                self.magic_buf.data_ptr(),
                ptr,
                4,
            )
            assert ret == 0, f"transfer_sync_write failed {ret}"

            # 5 swap buffer
            idx += 1
            current = bufs[idx % 2]
            get_torch_device().synchronize()

            if info["is_last"]:
                break

        time_cost = time.time() - start_time
        bandwidth = total_bytes / time_cost / (1024 * 1024 * 1024)
        logger.info(
            f"Rank {self.rank} receive weights done, time cost: {time_cost:.2f}s, bandwidth: {bandwidth:.2f} GB/s"
        )
