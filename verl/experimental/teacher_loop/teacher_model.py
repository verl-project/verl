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


class TeacherModelManager:
    """Owns the rollout replicas for a single distillation teacher.

    Splits `resource_pool` into per-replica chunks sized by
    `teacher_model_config.inference` parallelism, launches a colocated inference replica
    on each chunk, and exposes the resulting server handles and addresses plus a
    `GlobalRequestLoadBalancer` for use by `AsyncTeacherLLMServerManager`.
    """

    def __init__(
        self,
        distillation_config: DistillationConfig,
        teacher_model_config: DistillationTeacherModelConfig,
        resource_pool: RayResourcePool,
    ):
        """
        Initialize the teacher model manager.

        Args:
            distillation_config (DistillationConfig): Full distillation config; stored
                for use alongside the per-teacher config.
            teacher_model_config (DistillationTeacherModelConfig): Config for this
                teacher (model path, routing key, inference parallelism).
            resource_pool (RayResourcePool): Sub-pool already sized for this teacher
                (`teacher_model_config.world_size` bundles).
        """

        # Need dataclass conversion for max_logprobs handling in post_init
        self.distillation_config = distillation_config
        self.teacher_model_config = teacher_model_config
        self.resource_pool = resource_pool
        self._initialize_llm_servers()
        self._initialize_load_balancer_handle()

    def _initialize_llm_servers(self):
        teacher_model_config = self.teacher_model_config
        teacher_world_size = (
            teacher_model_config.inference.tensor_model_parallel_size
            * teacher_model_config.inference.data_parallel_size
            * teacher_model_config.inference.pipeline_model_parallel_size
        )
        world_size = self.resource_pool.world_size
        if world_size % teacher_world_size != 0:
            raise ValueError(
                f"Teacher world size {teacher_world_size} must divide allocated resource pool size {world_size}."
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
                gpus_per_node=teacher_model_config.world_size,
                is_teacher_model=True,
            )
            for replica_rank in range(num_replicas)
        ]
        split_resource_pools = split_resource_pool(self.resource_pool, split_size=teacher_world_size)
        assert len(split_resource_pools) == len(self.rollout_replicas)
        _run_all(
            [
                server.init_colocated(resource_pool)
                for server, resource_pool in zip(self.rollout_replicas, split_resource_pools, strict=True)
            ]
        )
        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

    def _initialize_load_balancer_handle(self):
        from verl.experimental.agent_loop.agent_loop import GlobalRequestLoadBalancer

        self.load_balancer_handle = GlobalRequestLoadBalancer.remote(
            server_actor_ids=self.server_addresses,
        )

    @auto_await
    async def wake_up(self):
        """Wake up all rollout replica instances."""
        await _run_all([replica.wake_up() for replica in self.rollout_replicas])

    @auto_await
    async def sleep(self):
        """Sleep all rollout replica instances."""
        await _run_all([replica.sleep() for replica in self.rollout_replicas])


class MultiTeacherModelManager:
    """Manages one inner `TeacherModelManager` per teacher model, keyed by each teacher's `key`."""

    def __init__(
        self,
        config: DictConfig,
        resource_pool: RayResourcePool,
    ):
        """
        Initialize the multi-teacher model manager.

        Args:
            config (DictConfig): Full configuration (needed so AsyncTeacherLLMServerManager callers
                can read `distillation` / `teacher_key` without reconstructing the config).
            resource_pool (RayResourcePool): Combined resource pool for all teachers.
        """
        self.config = config
        self.distillation_config: DistillationConfig = omega_conf_to_dataclass(config.distillation)

        self.resource_pool = resource_pool
        self.teacher_model_managers: dict[str, TeacherModelManager] = {}
        self.server_addresses: dict[str, list[str]] = {}
        self.server_handles: dict[str, list] = {}
        self.load_balancer_handle: dict[str, object] = {}

        self._initialize_teacher_model_managers()

    def _initialize_teacher_model_managers(self):
        teacher_models = self.distillation_config.teacher_models
        split_sizes = [teacher.n_gpus_per_node for teacher in teacher_models.values()]
        split_pools = split_resource_pool(self.resource_pool, split_size=split_sizes)
        assert len(split_pools) == len(teacher_models), (
            f"split_resource_pool returned {len(split_pools)} pools for {len(teacher_models)} teachers."
        )

        for (_, teacher_model_config), teacher_pool in zip(teacher_models.items(), split_pools, strict=True):
            manager = TeacherModelManager(
                distillation_config=self.distillation_config,
                teacher_model_config=teacher_model_config,
                resource_pool=teacher_pool,
            )
            key = teacher_model_config.key
            self.teacher_model_managers[key] = manager
            self.server_addresses[key] = manager.server_addresses
            self.server_handles[key] = manager.server_handles
            self.load_balancer_handle[key] = manager.load_balancer_handle

    @auto_await
    async def wake_up(self):
        """Wake up every teacher's rollout replicas."""
        await _run_all([manager.wake_up() for manager in self.teacher_model_managers.values()])

    @auto_await
    async def sleep(self):
        """Sleep every teacher's rollout replicas."""
        await _run_all([manager.sleep() for manager in self.teacher_model_managers.values()])
