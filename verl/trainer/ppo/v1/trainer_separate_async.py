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
from enum import Enum

import ray
from omegaconf import DictConfig

from verl.checkpoint_engine import CheckpointEngineManager
from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.workers.rollout.llm_server import LLMServerManager

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class HybridEngineMode(Enum):
    TRAINER = "trainer"
    ROLLOUT = "rollout"


@register_trainer("separate_async")
class PPOTrainerSeparateAsync(PPOTrainer):
    """Asynchronous PPO trainer
    1. Trainer and rollout are separate, trainer may switch to rollout if idle.
    2. Partial rollout is enabled.
    """

    def __init__(self, config: DictConfig):
        raise NotImplementedError("separate async training is still under development")

        train_batch_size = config.data.train_batch_size
        ppo_mini_batch_size = config.actor_rollout_ref.actor.ppo_mini_batch_size
        assert train_batch_size == ppo_mini_batch_size, (
            f"train_batch_size must be equal to ppo_mini_batch_size in separate async training, "
            f"but got {train_batch_size} and {ppo_mini_batch_size}"
        )
        assert config.actor_rollout_ref.rollout.nnodes > 0, "nnodes must be > 0 in separate async training"
        assert config.actor_rollout_ref.rollout.n_gpus_per_node > 0, (
            "n_gpus_per_node must be > 0 in separate async training"
        )
        assert config.actor_rollout_ref.rollout.checkpoint_engine.backend != "naive", (
            "please use nccl/nixl/mooncake, etc. backend for separate async training"
        )

        # TODO: Support Decoupled PPO: https://arxiv.org/abs/2505.24298
        self.config.algorithm.rollout_correction.bypass_mode = True

        super().__init__(config)

    def _setup(self):
        super()._setup()

        # initialize standalone rollout
        # TODO: make initialization parallel with super().init()
        self.standalone_server_manager: LLMServerManager = LLMServerManager.create(config=self.config)

        # create checkpoint engine manager for trainer and standalone rollout
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.standalone_checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=self.standalone_server_manager.get_replicas(),
        )

        # hybrid engine is in rollout mode after initialization
        self.current_mode = HybridEngineMode.ROLLOUT
        self.add_replicas_to_balancer()

    def get_llm_client(self):
        # get server client from standalone rollout
        return self.standalone_server_manager.get_client()

    def on_init_end(self):
        # update weights after loading checkpoint
        self.standalone_checkpoint_manager.update_weights(self.global_steps)
        self.checkpoint_manager.update_weights(self.global_steps)

    def on_train_begin(self):
        num_warmup_batches = self.config.trainer.v1.separate_async.num_warmup_batches
        for _ in range(num_warmup_batches):
            self._add_batch_to_generate()
        logger.info(f"Added {num_warmup_batches} warmup batches to the agent loop manager")

    def on_validate_begin(self):
        if self.current_mode == HybridEngineMode.TRAINER:
            logger.info("Switching hybrid engine to rollout mode for validation")
            self.switch_to_rollout()

    def on_validate_end(self):
        if self.current_mode == HybridEngineMode.ROLLOUT:
            logger.info("Switching hybrid engine to trainer mode for training")
            self.switch_to_trainer()

    def on_sample_begin(self):
        if self.current_mode == HybridEngineMode.TRAINER and self.should_switch_to_rollout():
            logger.info("Switching hybrid engine to rollout mode for generation")
            self.switch_to_rollout()

    def on_sample_end(self):
        if self.current_mode == HybridEngineMode.ROLLOUT:
            logger.info("Switching hybrid engine to trainer mode for training")
            self.switch_to_trainer()

    def on_step_end(self):
        if self.global_steps % self.config.trainer.v1.separate_async.parameter_sync_step == 0:
            with marked_timer("update_weights", self.timing_raw, color="red"):
                # wake up all replicas to update weights
                self.standalone_checkpoint_manager.update_weights(self.global_steps)

    def switch_to_rollout(self):
        self.checkpoint_manager.update_weights(self.global_steps)
        self.checkpoint_manager.resume_generation_replicas()
        self.add_replicas_to_balancer()
        self.current_mode = HybridEngineMode.ROLLOUT

    def switch_to_trainer(self):
        self.remove_replicas_from_balancer()
        self.checkpoint_manager.abort_replicas()
        self.checkpoint_manager.sleep_replicas()
        self.current_mode = HybridEngineMode.TRAINER

    def add_replicas_to_balancer(self):
        global_load_balancer = self.standalone_server_manager.global_load_balancer
        servers = dict(
            zip(self.llm_server_manager.server_addresses, self.llm_server_manager.server_handles, strict=True)
        )
        ray.get(global_load_balancer.add_servers.remote(servers))

    def remove_replicas_from_balancer(self):
        global_load_balancer = self.standalone_server_manager.global_load_balancer
        ray.get(global_load_balancer.remove_servers.remote(self.llm_server_manager.server_addresses))

    def should_switch_to_rollout(self):
        # TODO: Implement switch strategy by checking replay buffer and switch overhead
        return False
