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


@auto_await
async def _run_all(tasks: list[asyncio.Task]):
    await asyncio.gather(*tasks)


def _run_single(task):
    async def run():
        return await task

    return asyncio.run(run())


class TeacherModelManager:
    """Teacher model manager."""

    def __init__(
        self,
        teacher_model_config: DistillationTeacherModelConfig,
        resource_pool: RayResourcePool,
    ):
        """
        Initialize the teacher model manager.

        Args:
            teacher_model_config (DistillationTeacherModelConfig): Teacher model configuration.
            resource_pool (RayResourcePool): Resource pool.
        """

        # Need dataclass conversion for max_logprobs handling in post_init
        self.teacher_model_config = teacher_model_config
        self.resource_pool = resource_pool
        self._initialize_llm_servers()
        self._initialize_load_balancer_handle()
        self._initialize_router()

        self.sleep()

    def _initialize_llm_servers(self):
        teacher_model_config = self.teacher_model_config
        teacher_world_size = (
            teacher_model_config.inference.tensor_model_parallel_size
            * teacher_model_config.inference.data_parallel_size
            * teacher_model_config.inference.pipeline_model_parallel_size
        )
        world_size = (
            self.resource_pool.world_size
            if self.resource_pool  # colocate mode
            else teacher_model_config.n_gpus_per_node * teacher_model_config.nnodes  # standalone mode
        )
        num_replicas = world_size // teacher_world_size

        rollout_replica_class = get_rollout_replica_class(teacher_model_config.inference.name)
        rollout_config = teacher_model_config.inference
        model_config = HFModelConfig(path=teacher_model_config.model_path)
        self.tokenizer = model_config.get_processor()
        self.rollout_replicas = [
            rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=teacher_model_config.n_gpus_per_node,
                is_teacher_model=True,
            )
            for replica_rank in range(num_replicas)
        ]
        if self.resource_pool:
            split_resource_pools = split_resource_pool(self.resource_pool, split_size=teacher_world_size)
            assert len(split_resource_pools) == len(self.rollout_replicas)
            _run_all(
                [
                    server.init_colocated(resource_pool)
                    for server, resource_pool in zip(self.rollout_replicas, split_resource_pools, strict=True)
                ]
            )
        else:
            _run_all([server.init_standalone() for server in self.rollout_replicas])
        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

    def _initialize_load_balancer_handle(self):
        from verl.experimental.agent_loop.agent_loop import GlobalRequestLoadBalancer

        self.load_balancer_handle = GlobalRequestLoadBalancer.remote(
            server_actor_ids=self.server_addresses,
        )

    def _initialize_router(self):
        worker_urls = [f"http://{server_address}" for server_address in self.server_addresses]

        from ..reward_loop.router.naive_router import launch_router_process

        self.router_address, _ = launch_router_process(worker_urls=worker_urls)

    def get_router_address(self):
        return self.router_address

    @auto_await
    async def wake_up(self):
        """Wake up all rollout replica instances."""
        await _run_all([replica.wake_up() for replica in self.rollout_replicas])

    @auto_await
    async def sleep(self):
        """Sleep all rollout replica instances."""
        await _run_all([replica.sleep() for replica in self.rollout_replicas])


class MultiTeacherModelManager:
    """Multi Teacher model manager."""

    def __init__(
        self,
        config: DictConfig,
        resource_pool: RayResourcePool,
    ):
        """
        Initialize the teacher model manager.

        Args:
            config (DictConfig): Full configuration.
            resource_pool (RayResourcePool): Resource pool.
        """

        # Need dataclass conversion for max_logprobs handling in post_init
        self.config = config
        self.distillation_config: DistillationConfig = omega_conf_to_dataclass(config.distillation)

        self.resource_pool = resource_pool
        self._initialize_teacher_model_managers()
        self._initialize_async_server_manager()
        self.sleep()

    def _initialize_teacher_model_managers(self):
        """TODO: MOPD -- split resource pool across teachers and init one TeacherModelManager per teacher."""
        self.teacher_model_manager = TeacherModelManager(
            teacher_model_config=self.distillation_config.teacher_model, resource_pool=self.resource_pool
        )

    def _initialize_async_server_manager(self):
        """TODO: MOPD -- balancers/servers become a dict (one per teacher)"""
        from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager

        self.load_balancer_handle = self.teacher_model_manager.load_balancer_handle
        server_addresses = self.teacher_model_manager.server_addresses
        server_handles = self.teacher_model_manager.server_handles
        servers = list(zip(server_addresses, server_handles, strict=True))

        # In standalone mode, server manager is initialized in the agent loop
        if not self.distillation_config.teacher_model.enable_resource_pool:
            self.server_manager = AsyncTeacherLLMServerManager(
                config=self.config,
                servers=servers,
                load_balancer_handle=self.load_balancer_handle,
            )

    def get_router_address(self):
        """TODO: MOPD -- return dict of router addresses (one per teacher)"""
        return self.teacher_model_manager.router_address

    def compute_logprobs(self, data):
        self.wake_up()
        try:
            return _run_single(self.server_manager.compute_teacher_logprobs_batch(data))
        finally:
            self.sleep()

    @auto_await
    async def wake_up(self):
        """Wake up all rollout replica instances.
        TODO: MOPD -- wake up each of the teacher model managers.
        """
        await _run_all([manager.wake_up() for manager in [self.teacher_model_manager]])

    @auto_await
    async def sleep(self):
        """Sleep all rollout replica instances.
        TODO: MOPD -- sleep each of the teacher model managers.
        """
        await _run_all([manager.sleep() for manager in [self.teacher_model_manager]])
