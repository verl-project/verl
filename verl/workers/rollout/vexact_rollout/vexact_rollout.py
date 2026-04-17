# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
from typing import Generator

import ray
import torch
from torch.distributed.device_mesh import DeviceMesh

from verl.utils.device import get_device_id, get_device_name, get_torch_device, is_support_ipc
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class ServerAdapter(BaseRollout):
    """
    VExact server adapter for async mode, serves as a client to request VExact server
    to resume/release/update_weights.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)
        self.server_handle: ray.actor.ActorHandle = None  # Lazy

        rank = int(os.environ["RANK"])
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        rollout_world_size = (
            self.config.tensor_model_parallel_size
            * self.config.data_parallel_size
            * self.config.pipeline_model_parallel_size
        )
        self.replica_rank = rank // rollout_world_size
        self.rollout_rank = rank % rollout_world_size
        self.node_rank = self.rollout_rank // local_world_size

        assert rollout_world_size == device_mesh.size() // device_mesh["dp"].size()
        assert self.replica_rank == device_mesh["dp"].get_local_rank()

        # ZMQ handle for weight transfer, must match Worker.receive_weights
        driver_id = f"verl_rollout_replica_{self.replica_rank}"
        self.zmq_handle = f"ipc:///tmp/vexact-weight-{driver_id}-{self.rollout_rank}.sock"
        logger.info(f"Sender:{self.zmq_handle=}:{get_torch_device().get_device_properties(get_device_id()).uuid}")

        self.use_shm = not is_support_ipc()
        if self.use_shm:
            logger.warning("IPC is not supported on your devices. Falling back to shared memory for weight transfer.")

    def _get_server_handle(self) -> ray.actor.ActorHandle:
        """Lazy init server handle because server is launched after hybrid engine."""
        if self.server_handle is None:
            # Async server handle, must match async server ray actor name
            self.server_handle = ray.get_actor(f"vexact_server_{self.replica_rank}_{self.node_rank}")
        return self.server_handle

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: List of resource tags to resume (e.g. ["weights", "kv_cache"]).
        """
        if self.config.free_cache_engine and self.rollout_rank == 0:
            await self._get_server_handle().wake_up.remote(tags=tags)

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if self.config.free_cache_engine and self.rollout_rank == 0:
            await self._get_server_handle().sleep.remote()

    @torch.no_grad()
    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        **kwargs,  # noqa: ARG002
    ):
        """Update model weights via bucketed IPC transfer to inference workers."""
        future = None
        if self.rollout_rank == 0:
            future = self._get_server_handle().receive_weights.remote()

        from verl.workers.rollout.vexact_rollout.bucketed_weight_transfer import BucketedWeightSender

        sender = BucketedWeightSender(
            zmq_handle=self.zmq_handle,
            device=torch.device(f"{get_device_name()}:{get_device_id()}"),
            use_shm=self.use_shm,
        )
        sender.send_weights(weights)

        if future is not None:
            await future

        if self.rollout_rank == 0:
            await self.server_handle.clear_kv_cache.remote()

    def generate_sequences(self, prompts):
        """Sync generation no longer supported."""
        raise NotImplementedError(
            "ServerAdapter does not support synchronous generate_sequences(). "
            "Use the async server interface via VExactReplica and VExactServer instead."
        )
