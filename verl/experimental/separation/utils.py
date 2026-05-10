# Copyright 2025 Meituan Ltd. and/or its affiliates
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


import ray

from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, need_reference_policy


def create_resource_pool_manager(config, roles: list) -> ResourcePoolManager:
    """
    Create resource pool manager

    Args:
        config: Configuration object
        roles: List of roles that need to create resource pools

    Returns:
        ResourcePoolManager: Resource pool manager
    """
    resource_pool_spec = {}
    mapping = {}

    # Actor/Critic resource pool
    training_roles = [Role.Actor, Role.ActorRollout, Role.Critic, Role.RefPolicy]
    if any(role in roles for role in training_roles):
        assert config.trainer.n_gpus_per_node > 0, "config.trainer.n_gpus_per_node must be greater than 0"
        assert config.trainer.nnodes > 0, "config.trainer.nnodes must be greater than 0"

        trainer_pool = [config.trainer.n_gpus_per_node] * config.trainer.nnodes
        resource_pool_spec["trainer_pool"] = trainer_pool

        for role in training_roles:
            if role in roles:
                mapping[role] = "trainer_pool"

    # Rollout resource pool
    if Role.Rollout in roles:
        assert config.rollout.n_gpus_per_node > 0, "config.rollout.n_gpus_per_node must be greater than 0"
        assert config.rollout.nnodes > 0, "config.rollout.nnodes must be greater than 0"

    if Role.RewardModel in roles:
        rm_cfg = config.reward.reward_model
        assert rm_cfg.n_gpus_per_node > 0, "config.reward.reward_model.n_gpus_per_node must be greater than 0"
        assert rm_cfg.nnodes > 0, "config.reward.reward_model.nnodes must be greater than 0"

    # Teacher model resource pool (for distillation)
    if Role.TeacherModel in roles:
        distillation_cfg = config.get("distillation", {})
        n_gpus = distillation_cfg.get("n_gpus_per_node", 0)
        nnodes = distillation_cfg.get("nnodes", 1)
        assert n_gpus > 0, "distillation.n_gpus_per_node must be greater than 0 for TeacherModel"
        teacher_pool = [n_gpus] * nnodes
        resource_pool_spec["teacher_pool"] = teacher_pool
        mapping[Role.TeacherModel] = "teacher_pool"

    return ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)


def create_role_worker_mapping(config):
    """
    Create mapping from roles to worker classes

    Args:
        config: Configuration object

    Returns:
        dict: Mapping from roles to worker classes
    """
    # Always use the unified model engine worker implementation.
    from verl.experimental.separation.engine_workers import DetachActorWorker
    from verl.single_controller.ray import RayWorkerGroup
    from verl.workers.engine_workers import TrainingWorker

    ray_worker_group_cls = RayWorkerGroup

    train_role = Role.Actor
    if config.get("async_training", {}).get("use_trainer_do_validate", False):
        train_role = Role.ActorRollout

    role_worker_mapping = {
        train_role: ray.remote(DetachActorWorker),
        Role.Critic: ray.remote(TrainingWorker),
    }

    # Add reference policy (if KL loss or reward is required)
    if need_reference_policy(config):
        role_worker_mapping[Role.RefPolicy] = ray.remote(DetachActorWorker)

    return role_worker_mapping, ray_worker_group_cls
