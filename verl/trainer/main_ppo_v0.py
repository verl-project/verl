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
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other mpain.
"""

import os
import socket
import sys

from omegaconf import OmegaConf

from verl.runtime import Worker
from verl.trainer.distillation import is_distillation_enabled
from verl.trainer.ppo.model_config import PPOModelConfigs, PPORoleConfigs
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler, need_critic, need_reference_policy
from verl.utils.config import validate_config


class BaseTaskRunner(Worker):
    def __init__(self):
        super().__init__()
        self.role_worker_mapping = {}
        self.mapping = {}
        self.model_configs = None

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker using the unified model engine implementation."""
        from verl.trainer.ppo.ray_trainer import Role
        from verl.workers.engine_workers import ActorRolloutRefWorker

        actor_rollout_cls = ActorRolloutRefWorker

        actor_root = PPORoleConfigs.resolve(config, self.model_configs).actor
        lora_rank = actor_root.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = actor_root.model.get("lora_rank", 0)
        ref_in_actor = lora_rank > 0 or actor_root.model.get("lora_adapter_path") is not None
        # Ref policy is fused into ActorRolloutRefWorker unless LoRA is used with a dedicated ref model.
        if need_reference_policy(config, actor_root.actor) and not ref_in_actor:
            role = Role.ActorRolloutRef
        else:
            role = Role.ActorRollout
        self.role_worker_mapping[role] = actor_rollout_cls
        self.mapping[role] = "global_pool"
        return actor_rollout_cls

    def add_critic_worker(self, config):
        """Add critic worker to role mapping using the unified model engine implementation."""
        from verl.trainer.ppo.ray_trainer import Role
        from verl.workers.engine_workers import TrainingWorker

        # The model-engine TrainingWorker handles all critic backends (fsdp/fsdp2/megatron/...)
        # internally based on ``config.critic.strategy``.
        self.role_worker_mapping[Role.Critic] = TrainingWorker
        self.mapping[Role.Critic] = "global_pool"

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""

        from verl.runtime import current_runtime
        from verl.trainer.ppo.runtime_resource_pool import RuntimeResourcePoolManager
        from verl.trainer.ppo.utils import Role

        topology = current_runtime().topology
        if topology.models:
            actor_role = next(
                role for role in (Role.ActorRolloutRef, Role.ActorRollout) if role in self.role_worker_mapping
            )
            runtime = current_runtime()
            actor_model = self.model_configs.one("actor").model
            ref_bindings = self.model_configs.all("ref")
            actor_pool = runtime.model_resource_pool(actor_model.name)
            ref_pool = runtime.model_resource_pool(ref_bindings[0].model.name) if ref_bindings else None
            if ref_pool is not None and ref_pool is not actor_pool:
                raise NotImplementedError(
                    "the current fused ActorRolloutRefWorker requires actor and ref to share one placement"
                )
            mapping = {actor_role: actor_model.name}
            if Role.Critic in self.role_worker_mapping:
                mapping[Role.Critic] = self.model_configs.one("critic").model.name
            if Role.RewardModel in self.mapping:
                mapping[Role.RewardModel] = self.model_configs.one("rm").model.name
            self.mapping = mapping
            resource_pool_manager = RuntimeResourcePoolManager(
                resource_pool_spec={},
                mapping=self.mapping,
            )
            return resource_pool_manager

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        if config.reward.reward_model.enable_resource_pool:
            if config.reward.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward.reward_model.nnodes <= 0:
                raise ValueError("config.reward.reward_model.nnodes must be greater than 0")

            reward_pool = [config.reward.reward_model.n_gpus_per_node] * config.reward.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool
        else:
            config.reward.reward_model.nnodes = config.trainer.nnodes
            config.reward.reward_model.n_gpus_per_node = config.trainer.n_gpus_per_node

        distillation_config = config.get("distillation")
        if is_distillation_enabled(distillation_config):
            if distillation_config.n_gpus_per_node <= 0:
                raise ValueError("config.distillation.n_gpus_per_node must be greater than 0")
            if distillation_config.nnodes <= 0:
                raise ValueError("config.distillation.nnodes must be greater than 0")

            teacher_pool = [distillation_config.n_gpus_per_node] * distillation_config.nnodes
            resource_pool_spec["teacher_pool"] = teacher_pool

        resource_pool_manager = RuntimeResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=self.mapping,
        )
        return resource_pool_manager

    def add_reward_model_resource_pool(self, config):
        """Add reward model worker if enabled."""
        from verl.trainer.ppo.ray_trainer import Role

        reward_model_config = PPORoleConfigs.resolve(config, self.model_configs).reward_model
        if reward_model_config.enable:
            # we do not use reward model workers, so we only register reward model in resource pool
            # without continue to register reward model worker in role mapping
            if reward_model_config.enable_resource_pool:
                self.mapping[Role.RewardModel] = "reward_pool"
            else:
                self.mapping[Role.RewardModel] = "global_pool"

    def add_teacher_model_resource_pool(self, config):
        """Add teacher model worker if enabled."""
        from verl.trainer.ppo.ray_trainer import Role

        if is_distillation_enabled(config.get("distillation")):
            # we do not use teacher model workers, so we only register teacher model in resource pool
            # without registering a teacher model worker in role-worker mapping
            self.mapping[Role.TeacherModel] = "teacher_pool"

    def run(self, config):
        pass


class TaskRunner(BaseTaskRunner):
    """Worker for executing distributed PPO training tasks.

    This class encapsulates the main training logic and is instantiated as a
    one-rank WorkerGroup by the common Runtime.

    Attributes:
        role_worker_mapping: Dictionary mapping Role enums to Worker classes
        mapping: Dictionary mapping Role enums to resource pool IDs for GPU allocation
    """

    def __init__(self):
        super().__init__()

    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        from verl.runtime import current_runtime

        topology = current_runtime().topology
        self.model_configs = PPOModelConfigs(config, topology) if topology.models else None
        roles = PPORoleConfigs.resolve(config, self.model_configs)

        self.add_actor_rollout_worker(config)
        if need_critic(config, roles.critic):
            self.add_critic_worker(config)

        self.add_reward_model_resource_pool(config)

        self.add_teacher_model_resource_pool(config)

        # validate config
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config, roles.actor.actor),
            use_critic=need_critic(config, roles.critic),
            actor_model_config=roles.actor.model,
            actor_config=roles.actor.actor,
            rollout_config=roles.rollout.rollout,
            ref_config=roles.ref.ref,
            critic_config=roles.critic,
        )

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(roles.actor.model.path, use_shm=roles.actor.model.get("use_shm", False))

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        # Create training and validation datasets.
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize the PPO trainer.
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            model_configs=self.model_configs,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        # The trainer creates rollout server processes during init_workers().
        # Stop those servers before the enclosing Runtime tears down their
        # WorkerGroups; otherwise vLLM native EngineCore processes are killed
        # mid-cleanup and can terminate in C++ destructors.
        try:
            # Initialize the workers of the trainer.
            trainer.init_workers()
            # Start the training process.
            trainer.fit()
        finally:
            body_failed = sys.exc_info()[0] is not None
            llm_server_manager = getattr(trainer, "llm_server_manager", None)
            if llm_server_manager is not None:
                try:
                    llm_server_manager.close()
                except Exception:  # Cleanup must not replace the active training failure.
                    if not body_failed:
                        raise
