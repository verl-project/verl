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

"""DAPO-configured entry point for the current VERL legacy worker runner.

The DAPO recipe submodule still imports ``TaskRunner`` from
``verl.trainer.main_ppo``.  Current VERL moved that implementation to
``main_ppo_v0`` while reserving ``main_ppo.TaskRunnerV1`` for the transfer-
queue trainer.  Its custom trainer also imports APIs removed from current
VERL.  The current ``RayPPOTrainer`` natively implements GRPO and rollout
correction, so this adapter composes that maintained trainer with the DAPO
config and DAPO reward manager.
"""

import os
import socket
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import run_ppo
from verl.trainer.main_ppo_v0 import BaseTaskRunner
from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler, need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device


class DAPOTaskRunner(BaseTaskRunner):
    """Use current PPO trainer with DAPO config and unified engine workers."""

    def run(self, config):
        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_teacher_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        from verl.utils.config import omega_conf_to_dataclass
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.workers.config import HFModelConfig

        model_config: HFModelConfig = omega_conf_to_dataclass(config.actor_rollout_ref.model)
        tokenizer = model_config.tokenizer
        processor = model_config.processor
        resource_pool_manager = self.init_resource_pool_mgr(config)

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

        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        succeeded = False
        try:
            trainer.init_workers()
            trainer.fit()
            succeeded = True
        finally:
            tracking = getattr(trainer, "logger", None)
            if tracking is not None:
                tracking.finish(exit_code=0 if succeeded else 1)


@hydra.main(config_path="../../recipe/dapo/config", config_name="dapo_megatron_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(DAPOTaskRunner))


if __name__ == "__main__":
    main()
