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

from omegaconf import DictConfig

from verl.single_controller.ray.base import RayResourcePool, split_resource_pool
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.ray_utils import auto_await
from verl.workers.config import DistillationConfig, DistillationTeacherModelConfig, HFModelConfig
from verl.workers.rollout.replica import get_rollout_replica_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TeacherModelManager:
    """Teacher model manager."""

    def __init__(
        self,
        config: DictConfig,
        resource_pool: RayResourcePool = None,
    ):
        """
        Initialize the teacher model manager.

        Args:
            config (DictConfig): Teacher model configuration.
            resource_pool (RayResourcePool, optional): Resource pool. Defaults to None.
        """

        # Need dataclass conversion for max_logprobs handling in post_init
        self.config: DistillationConfig = omega_conf_to_dataclass(config)
        self.resource_pool = resource_pool
        self._initialize_llm_servers()
        self._initialize_router()

        self.sleep()

    def _initialize_llm_servers(self):
        teacher_configs: dict[str, DistillationTeacherModelConfig] = self.config.teacher_models
        teacher_world_size = (
            self.config.inference.tensor_model_parallel_size
            * self.config.inference.data_parallel_size
            * self.config.inference.pipeline_model_parallel_size
        )
        world_size = (
            self.resource_pool.world_size
            if self.resource_pool  # colocate mode
            else self.config.n_gpus_per_node * self.config.nnodes  # standalone mode
        )
        num_replicas = world_size // teacher_world_size
        num_teachers = len(teacher_configs)
        if num_replicas < num_teachers:
            raise ValueError(
                f"Need at least one teacher replica per teacher, but got {num_replicas=} for {num_teachers=}."
            )
        if num_replicas % num_teachers != 0:
            raise ValueError(
                f"Teacher replicas ({num_replicas}) must be divisible by the number of teachers ({num_teachers})."
            )
        replicas_per_teacher = num_replicas // num_teachers

        self.rollout_replicas = []
        self.teacher_replicas_by_task = {}
        replica_rank = 0
        rollout_replica_class = get_rollout_replica_class(self.config.inference.name)
        for teacher_model_config in teacher_configs.values():
            model_config = HFModelConfig(path=teacher_model_config.model_path)

            teacher_replicas = [
                rollout_replica_class(
                    replica_rank=replica_rank + offset,
                    config=self.config.inference,
                    model_config=model_config,
                    gpus_per_node=self.config.n_gpus_per_node,
                    is_teacher_model=True,
                )
                for offset in range(replicas_per_teacher)
            ]
            self.rollout_replicas.extend(teacher_replicas)
            self.teacher_replicas_by_task[teacher_model_config.task] = teacher_replicas
            replica_rank += replicas_per_teacher

        if self.resource_pool:
            split_resource_pools = split_resource_pool(self.resource_pool, split_size=teacher_world_size)
            assert len(split_resource_pools) == len(self.rollout_replicas)
            self._run_all(
                [
                    server.init_colocated(resource_pool)
                    for server, resource_pool in zip(self.rollout_replicas, split_resource_pools, strict=True)
                ]
            )
        else:
            self._run_all([server.init_standalone() for server in self.rollout_replicas])

        self.server_handles_by_task = {
            task: [server._server_handle for server in teacher_replicas]
            for task, teacher_replicas in self.teacher_replicas_by_task.items()
        }
        self.server_addresses_by_task = {
            task: [server._server_address for server in teacher_replicas]
            for task, teacher_replicas in self.teacher_replicas_by_task.items()
        }
        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

    def _initialize_router(self):
        self.router_address_by_task = {}

        from ..reward_loop.router.naive_router import launch_router_process

        for task, server_addresses in self.server_addresses_by_task.items():
            worker_urls = [f"http://{server_address}" for server_address in server_addresses]
            self.router_address_by_task[task], _ = launch_router_process(worker_urls=worker_urls)

    def get_router_address(self, task: str | None = None):
        if task is None:
            return self.router_address_by_task
        return self.router_address_by_task[task]

    @auto_await
    async def wake_up(self):
        """Wake up all rollout replica instances."""
        await self._run_all([replica.wake_up() for replica in self.rollout_replicas])

    @auto_await
    async def sleep(self):
        """Sleep all rollout replica instances."""
        await self._run_all([replica.sleep() for replica in self.rollout_replicas])

    @auto_await
    async def _run_all(self, tasks: list[asyncio.Task]):
        await asyncio.gather(*tasks)
