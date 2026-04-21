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
    """Teacher model manager."""

    def __init__(
        self,
        distillation_config: DistillationConfig,
        teacher_model_config: DistillationTeacherModelConfig,
        resource_pool: RayResourcePool,
    ):
        """
        Initialize the teacher model manager.

        Args:
            distillation_config (DistillationConfig): Distillation configuration.
            teacher_model_config (DistillationTeacherModelConfig): Teacher model configuration.
            resource_pool (RayResourcePool): Dedicated teacher resource pool.
        """

        # Need dataclass conversion for max_logprobs handling in post_init
        self.distillation_config = distillation_config
        self.teacher_model_config = teacher_model_config
        self.resource_pool = resource_pool
        self._initialize_llm_servers()
        self._initialize_load_balancer_handle()

    def _initialize_llm_servers(self):
        teacher_model_config = self.teacher_model_config
        per_replica_world_size = teacher_model_config.per_replica_world_size
        num_replicas = teacher_model_config.num_replicas
        expected_pool_size = num_replicas * per_replica_world_size
        if self.resource_pool.world_size != expected_pool_size:
            raise ValueError(
                f"Teacher {teacher_model_config.key!r} expected sub-pool of size "
                f"{expected_pool_size} (num_replicas={num_replicas} * "
                f"per_replica_world_size={per_replica_world_size}), but got "
                f"{self.resource_pool.world_size}."
            )

        gpus_per_node = self.distillation_config.n_gpus_per_node
        rollout_replica_class = get_rollout_replica_class(teacher_model_config.inference.name)
        rollout_config = teacher_model_config.inference
        model_config = HFModelConfig(path=teacher_model_config.model_path)
        name_suffix = (teacher_model_config.key or "").replace("/", "_")
        self.rollout_replicas = [
            rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=gpus_per_node,
                is_teacher_model=True,
                name_suffix=name_suffix,
            )
            for replica_rank in range(num_replicas)
        ]
        split_resource_pools = split_resource_pool(self.resource_pool, split_size=per_replica_world_size)
        assert len(split_resource_pools) == len(self.rollout_replicas)
        self._validate_replica_node_alignment(split_resource_pools, per_replica_world_size, gpus_per_node)
        _run_all(
            [
                server.init_colocated(resource_pool)
                for server, resource_pool in zip(self.rollout_replicas, split_resource_pools, strict=True)
            ]
        )
        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

    def _validate_replica_node_alignment(self, replica_pools, per_replica_world_size, gpus_per_node):
        """Verify that each replica occupies the expected number of nodes.

        `per_replica_world_size` (W below) is the GPU count of a *single* inference
        replica — the product of the replica's inference-time parallelism
        (tensor_model_parallel_size * data_parallel_size * pipeline_model_parallel_size).
        It is not the teacher's total GPU footprint (`num_replicas * W`).

        `split_resource_pool` walks bundles linearly and is oblivious to node
        boundaries, so a replica's sub-pool can end up touching more nodes than W
        implies when W does not divide the node layout cleanly.

        Example (P = n_gpus_per_node = 4, two teachers with W=3 and W=4):

                node 0                  node 1
                [0 1 2 3]               [4 5 6 7]         ← bundle idx

            teacher A (W=3):
                [A A A .]               [. . . .]          expected span 1, observed 1  ✓
            teacher B (W=4):
                [. . . B]               [B B B .]          expected span 1, observed 2  ✗

        Teacher B's one replica (W=4) is expected to stay on a single node, but the
        linear split dropped it on bundles 3-6 — straddling nodes 0 and 1.
        """
        key = self.teacher_model_config.key
        P = gpus_per_node
        W = per_replica_world_size
        expected_span = (W + P - 1) // P
        for i, sub_pool in enumerate(replica_pools):
            start = sub_pool.start_bundle_index
            first_node = start // P
            last_node = (start + W - 1) // P
            observed_span = last_node - first_node + 1
            if observed_span != expected_span:
                raise ValueError(
                    f"Teacher {key!r} replica {i} sub-pool bundles [{start}, {start + W}) "
                    f"span {observed_span} node(s) but per_replica_world_size {W} with "
                    f"n_gpus_per_node {P} expects {expected_span}. Reorder teachers or "
                    f"adjust num_replicas / inference parallelism so each replica sub-pool "
                    f"aligns to node boundaries."
                )

    def _initialize_load_balancer_handle(self):
        from verl.experimental.agent_loop.agent_loop import GlobalRequestLoadBalancer

        self.load_balancer_handle = GlobalRequestLoadBalancer.remote(
            server_actor_ids=self.server_addresses,
        )


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
            config (DictConfig): Full configuration.
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
        split_sizes = [teacher.world_size for teacher in teacher_models.values()]
        split_pools = split_resource_pool(self.resource_pool, split_size=split_sizes)

        for (key, teacher_model_config), teacher_pool in zip(teacher_models.items(), split_pools, strict=True):
            manager = TeacherModelManager(
                distillation_config=self.distillation_config,
                teacher_model_config=teacher_model_config,
                resource_pool=teacher_pool,
            )
            self.teacher_model_managers[key] = manager
            self.server_addresses[key] = manager.server_addresses
            self.server_handles[key] = manager.server_handles
            self.load_balancer_handle[key] = manager.load_balancer_handle
