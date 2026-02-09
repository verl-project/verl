# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import os
from pprint import pprint

import hydra
import ray
import torch
from omegaconf import OmegaConf

from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.megatron_utils import get_hf_model_checkpoint_path, load_megatron_model_to_gpu
from verl.workers.megatron_workers import ActorRolloutRefWorker

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class CustomSaveWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_merged_weights(self, hf_ckpt_path):
        import os

        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)

        torch.distributed.barrier()

        print(f"[Rank {os.environ.get('RANK', '?')}] Saving weights to {hf_ckpt_path}...")

        if self.vanilla_bridge:
            self.bridge.save_weights(
                self.actor_module, hf_ckpt_path, distributed_filesystem=True, memory_efficient=True
            )
        else:
            self.bridge.save_hf_weights(self.actor_module, hf_ckpt_path)

        return True


@hydra.main(config_path="../verl/trainer/config", config_name="ppo_megatron_trainer", version_base=None)
def main(config):
    assert config.actor_rollout_ref.model.lora.adapter_path is not None, "adapter_path must be specified"

    if (
        config.actor_rollout_ref.actor.optim.lr_decay_steps is None
        or config.actor_rollout_ref.actor.optim.lr_decay_steps < 1
    ):
        # set to bypass OptimizerParamScheduler checks
        config.actor_rollout_ref.actor.optim.lr_decay_steps = 100000

    run_merge(config)


def run_merge(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        default_runtime_env = {"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}}
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    ray_cls_with_init = RayClassWithInitArgs(
        cls=ray.remote(CustomSaveWorker), config=config.actor_rollout_ref, role="actor"
    )
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)

    worker = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls_with_init,
        device_name=config.trainer.device,
    )
    worker.init_model()

    adapter_path = config.actor_rollout_ref.model.lora.adapter_path
    hf_ckpt_path = get_hf_model_checkpoint_path(os.path.dirname(adapter_path))
    worker.save_merged_weights(hf_ckpt_path)


if __name__ == "__main__":
    """
    Use the same config as your training script, besides **specifying the adapter_path**.

    For example, your training script starts with: 
        `python3 -m verl.trainer.main_ppo --config-name=ppo_megatron_trainer ...`
    Now replace it with 
        `python3 ./scripts/megatron_merge_lora.py --config-name=ppo_megatron_trainer ...`
    """
    main()
