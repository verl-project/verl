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
import os

import pytest
import ray
from omegaconf import DictConfig
from openai import AsyncOpenAI

from tests.checkpoint_engine.test_utils import create_trainer_worker_group
from verl.checkpoint_engine import CheckpointEngineManager, CheckpointEngineWorker
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_name
from verl.workers.config import CheckpointEngineConfig, HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import get_rollout_replica_class


@pytest.fixture
def init_config() -> DictConfig:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose(config_name="ppo_trainer")

    config.trainer.n_gpus_per_node = 8
    config.trainer.nnodes = 1
    config.actor_rollout_ref.model.path = os.path.expanduser("~/models/Qwen/Qwen3-VL-2B-Instruct")
    config.actor_rollout_ref.rollout.name = os.environ["ROLLOUT_NAME"]
    config.actor_rollout_ref.rollout.skip_tokenizer_init = False
    config.actor_rollout_ref.rollout.max_num_seqs = 256
    config.actor_rollout_ref.rollout.checkpoint_engine.backend = "nccl" if get_device_name() == "cuda" else "hccl"

    return config


@pytest.mark.asyncio
async def test_server_adapter(init_config):
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
                "VLLM_DISABLE_COMPILE_CACHE": "1",
            }
        }
    )

    # 1. create trainer worker group
    model_config: HFModelConfig = omega_conf_to_dataclass(init_config.actor_rollout_ref.model)
    checkpoint_engine_config: CheckpointEngineConfig = omega_conf_to_dataclass(
        init_config.actor_rollout_ref.rollout.checkpoint_engine
    )
    trainer_pool = RayResourcePool(process_on_nodes=[4], max_colocate_count=3)
    trainer = create_trainer_worker_group(trainer_pool, model_config, checkpoint_engine_config)
    trainer.reset()

    # 2. create rollout replicas
    rollout_config: RolloutConfig = omega_conf_to_dataclass(init_config.actor_rollout_ref.rollout)

    # 2.1 create checkpoint engine worker group
    rollout_pool = RayResourcePool(process_on_nodes=[4], max_colocate_count=3)
    ray_cls_with_init = RayClassWithInitArgs(
        cls=ray.remote(CheckpointEngineWorker),
        model_config=model_config,
        rollout_config=rollout_config,
    )
    rollout = RayWorkerGroup(
        resource_pool=rollout_pool, ray_cls_with_init=ray_cls_with_init, device_name=get_device_name()
    )

    # 2.2 create rollout replicas
    rollout_replica_class = get_rollout_replica_class(rollout_config.name)
    rollout_replicas = [
        rollout_replica_class(
            replica_rank=replica_rank,
            config=rollout_config,
            model_config=model_config,
        )
        for replica_rank in range(2)
    ]
    await asyncio.gather(*[replica.init_hybrid(rollout) for replica in rollout_replicas])

    # 3. create checkpoint engine manager
    checkpoint_manager = CheckpointEngineManager(
        backend=checkpoint_engine_config.backend, trainer=trainer, replicas=rollout_replicas
    )
    for i in range(3):
        await checkpoint_manager.update_weights()

        server_addresses = rollout_replicas[i % len(rollout_replicas)].server_address
        client = AsyncOpenAI(
            api_key="123-abc",
            base_url=f"http://{server_addresses}/v1",
        )

        completion = await client.chat.completions.create(
            model=init_config.actor_rollout_ref.model.path,
            messages=[{"role": "user", "content": "What can you do?"}],
        )
        print("[OUTPUT]:", completion.choices[0].message.content)

    ray.shutdown()
