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

import ray
from omegaconf import DictConfig

from verl.experimental.agent_loop import AgentLoopManager
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, RewardModelWorker


def init_agent_loop_manager(config: DictConfig) -> AgentLoopManager | RayWorkerGroup:
    # =========================== 1. Create hybrid ActorRollout workers ===========================
    actor_rollout_cls = (
        AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
    )
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(actor_rollout_cls),
    }
    if config.reward_model.enable:
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
    }
    if config.reward_model.enable_resource_pool:
        mapping[Role.RewardModel] = "reward_pool"
        if config.reward_model.n_gpus_per_node <= 0:
            raise ValueError("config.reward_model.n_gpus_per_node must be greater than 0")
        if config.reward_model.nnodes <= 0:
            raise ValueError("config.reward_model.nnodes must be greater than 0")

        reward_pool = [config.reward_model.n_gpus_per_node] * config.reward_model.nnodes
        resource_pool_spec["reward_pool"] = reward_pool
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    resource_pool_manager.create_resource_pool()
    resource_pool_to_cls = {pool: {} for pool in resource_pool_manager.resource_pool_dict.values()}

    # create actor and rollout
    resource_pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)
    actor_rollout_cls = RayClassWithInitArgs(
        cls=role_worker_mapping[Role.ActorRollout], config=config.actor_rollout_ref, role="actor_rollout"
    )
    resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls

    if config.reward_model.enable:
        # we create a RM here
        resource_pool = resource_pool_manager.get_resource_pool(Role.RewardModel)
        rm_cls = RayClassWithInitArgs(role_worker_mapping[Role.RewardModel], config=config.reward_model)
        resource_pool_to_cls[resource_pool]["rm"] = rm_cls

    all_wg = {}
    for resource_pool, class_dict in resource_pool_to_cls.items():
        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        wg_dict = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
        all_wg.update(spawn_wg)
    actor_rollout_wg = all_wg["actor_rollout"]
    actor_rollout_wg.init_model()

    if config.actor_rollout_ref.rollout.mode == "sync":
        raise ValueError("Agent loop tests require async rollout mode. Please set rollout.mode=async.")

    if config.reward_model.enable_resource_pool and config.reward_model.enable:
        rm_resource_pool = resource_pool_manager.get_resource_pool(Role.RewardModel)
    else:
        rm_resource_pool = None
    # =========================== 2. Create AgentLoopManager ===========================
    agent_loop_manager = AgentLoopManager(
        config=config,
        worker_group=actor_rollout_wg,
        rm_resource_pool=rm_resource_pool,
    )

    return agent_loop_manager
