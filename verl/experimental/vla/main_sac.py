# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
from pprint import pprint

import datasets
import hydra
import ray
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.vla.sac.sac_ray_trainer import RobRaySACTrainer
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_local_path_from_hdfs

logger = logging.getLogger(__name__)


def calculate_reward(data: DataProto, return_dict: bool = False) -> torch.Tensor:
    complete_tensor = data.batch["complete"]
    reward_per_step = complete_tensor.float()
    if return_dict:
        return {"reward_tensor": reward_per_step}
    else:
        return reward_per_step


@hydra.main(config_path="config", config_name="rob_sac_trainer", version_base=None)
def main(config):
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        logger.info(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))
    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    # print initial config
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.experimental.vla.workers.env.env_worker import EnvWorker
        from verl.single_controller.ray import RayWorkerGroup

        from .fsdp_workers import RobActorRolloutRefWorker

        ray_worker_group_cls = RayWorkerGroup
    else:
        raise NotImplementedError

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(RobActorRolloutRefWorker),
        Role.Env: ray.remote(EnvWorker),
    }

    # setup resource pool manager
    train_rollout_gpu_num = config.trainer.n_rollout_gpus_per_node
    train_rollout_nodes_num = config.trainer.nnodes
    env_gpu_num = config.trainer.n_env_gpus_per_node
    env_nodes_num = config.env.disagg_sim.nnodes if config.env.disagg_sim.enable else config.trainer.nnodes

    resource_pool_spec = {
        "train_rollout_pool": [train_rollout_gpu_num] * train_rollout_nodes_num,
        "env_gpu_pool": [env_gpu_num] * env_nodes_num,
    }
    mapping = {
        Role.ActorRollout: "train_rollout_pool",
        Role.Env: "env_gpu_pool",
    }
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    # create datasets
    train_dataset = datasets.load_dataset("parquet", data_files=config.data.train_files)["train"]
    val_dataset = datasets.load_dataset("parquet", data_files=config.data.val_files)["train"]

    # instantiate trainer and start training
    trainer = RobRaySACTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=calculate_reward,
        val_reward_fn=calculate_reward,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    main()
