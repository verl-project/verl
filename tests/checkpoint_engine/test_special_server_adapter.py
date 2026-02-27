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

from tests.checkpoint_engine.test_utils import create_trainer_worker_group
from verl.checkpoint_engine import CheckpointEngineManager, CheckpointEngineWorker
from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager
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
    config.actor_rollout_ref.rollout.max_num_seqs = 256
    config.actor_rollout_ref.rollout.response_length = 4096
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
        config=checkpoint_engine_config, trainer=trainer, replicas=rollout_replicas
    )
    await checkpoint_manager.update_weights(global_steps=0)

    server_handles = [server._server_handle for server in rollout_replicas]
    server_manager = AsyncLLMServerManager(config=init_config, server_handles=server_handles)

    n = 16
    prompts = [
        [{"role": "user", "content": "Please write an article about the history of China, at least 1000 words."}],
        [{"role": "user", "content": "Please write an article about the history of America, at least 1000 words."}],
        [{"role": "user", "content": "Please write an article about the geography of China, at least 1000 words."}],
        [{"role": "user", "content": "Please write an article about the geography of America, at least 1000 words."}],
    ] * n
    for global_steps in range(1, 4):
        tasks = []
        for i, prompt in enumerate(prompts):
            prompt_ids = model_config.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)
            tasks.append(
                asyncio.create_task(
                    server_manager.generate(
                        request_id=f"test_{global_steps}_{i}",
                        prompt_ids=prompt_ids,
                        sampling_params={
                            "temperature": 1.0,
                            "logprobs": True,
                        },
                    )
                )
            )

        # wait a while and update weights to interrupt the generation
        await asyncio.sleep(3)
        await checkpoint_manager.update_weights(global_steps=global_steps)

        outputs = await asyncio.gather(*tasks)
        expected_steps = global_steps - 1
        num_aborted, num_completed = 0, 0
        for output in outputs:
            if output.stop_reason in ("aborted", "abort"):
                num_aborted += 1
            else:
                num_completed += 1
            assert output.global_steps == expected_steps, (
                f"output.global_steps is {output.global_steps}, expected {expected_steps}"
            )
        assert num_aborted > 0, f"num_aborted is {num_aborted}, expected > 0"
        print(f"========== [{global_steps=}] {num_aborted=}, {num_completed=} ==========")
        print("[RESPONSE]", model_config.tokenizer.decode(outputs[0].token_ids, skip_special_tokens=True))

    ray.shutdown()
