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
from transformers import PreTrainedTokenizer

from tests.checkpoint_engine.test_utils import create_trainer_worker_group
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AsyncLLMServerManager
from verl.experimental.fully_async_policy.agent_loop.agent_loop import FullyAsyncLLMServerManager
from verl.single_controller.ray import (
    RayResourcePool,
)
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import CheckpointEngineConfig, HFModelConfig


@pytest.fixture
def init_config() -> DictConfig:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "+async_training.partial_rollout=True",
            ],
        )

    config.actor_rollout_ref.model.path = os.path.expanduser("~/models/Qwen/Qwen3-VL-2B-Instruct")
    config.actor_rollout_ref.rollout.name = os.environ["ROLLOUT_NAME"]
    config.actor_rollout_ref.rollout.max_num_seqs = 256
    config.actor_rollout_ref.rollout.response_length = 4096
    config.actor_rollout_ref.rollout.checkpoint_engine.backend = "nccl"
    config.actor_rollout_ref.rollout.nnodes = 1
    config.trainer.n_gpus_per_node = 4
    config.trainer.nnodes = 1

    return config


async def _run_update_weights_with_global_steps_none(
    server_manager: AsyncLLMServerManager,
    checkpoint_manager: CheckpointEngineManager,
    tokenizer: PreTrainedTokenizer,
):
    await checkpoint_manager.update_weights(global_steps=None)
    prompt = [{"role": "user", "content": "How to make a sandwich?"}]
    prompt_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)
    output = await server_manager.generate(
        request_id="test_0",
        prompt_ids=prompt_ids,
        sampling_params={
            "temperature": 1.0,
            "logprobs": True,
        },
    )
    assert output.stop_reason not in ("aborted", "abort"), (
        f"output.stop_reason is {output.stop_reason}, expected not abort"
    )
    assert output.extra_fields["global_steps"] is None, (
        f"output.extra_fields['global_steps'] is {output.extra_fields['global_steps']}, expected None"
    )
    print("========== [update_weights with global_steps=None] ==========")
    print("[RESPONSE]", tokenizer.decode(output.token_ids, skip_special_tokens=True))


async def _run_server_manager_without_resume(
    initial_steps: int,
    train_steps: int,
    server_manager: AsyncLLMServerManager,
    checkpoint_manager: CheckpointEngineManager,
    prompts: list[list[dict]],
    tokenizer: PreTrainedTokenizer,
):
    for global_steps in range(initial_steps, initial_steps + train_steps):
        tasks = []
        for i, prompt in enumerate(prompts):
            prompt_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)
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
        await asyncio.sleep(2)
        await checkpoint_manager.update_weights(global_steps=global_steps)

        outputs = await asyncio.gather(*tasks)
        expected_steps = global_steps - 1
        for output in outputs:
            global_steps = output.extra_fields["global_steps"]
            assert output.stop_reason in ("aborted", "abort"), (
                f"output.stop_reason is {output.stop_reason}, expected in abort"
            )
            assert global_steps == expected_steps, f"output.global_steps is {global_steps}, expected {expected_steps}"
        print(f"========== [{initial_steps=}, {train_steps=}] ==========")
        print("[RESPONSE]", tokenizer.decode(outputs[0].token_ids, skip_special_tokens=True))


async def _run_server_manager_with_resume(
    initial_steps: int,
    train_steps: int,
    server_manager: FullyAsyncLLMServerManager,
    checkpoint_manager: CheckpointEngineManager,
    prompts: list[list[dict]],
    tokenizer: PreTrainedTokenizer,
):
    # 1. rollout generate responses
    tasks = []
    for i, prompt in enumerate(prompts):
        prompt_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)
        tasks.append(
            asyncio.create_task(
                server_manager.generate(
                    request_id=f"test_{initial_steps}_{i}",
                    prompt_ids=prompt_ids,
                    sampling_params={
                        "temperature": 1.0,
                        "logprobs": True,
                    },
                )
            )
        )

    # 2. trainer update weights to rollout multiple times
    for global_steps in range(initial_steps, initial_steps + train_steps):
        # wait a while and update weights to interrupt the generation
        await asyncio.sleep(2)
        await checkpoint_manager.update_weights(global_steps=global_steps)

    # 3. wait for rollout generate responses finished
    outputs = await asyncio.gather(*tasks)
    expected_min_steps = initial_steps - 1
    for output in outputs:
        min_global_steps = output.extra_fields["min_global_steps"]
        max_global_steps = output.extra_fields["max_global_steps"]
        assert min_global_steps == expected_min_steps, (
            f"output.min_global_steps is {min_global_steps}, expected {expected_min_steps}"
        )
        assert max_global_steps > expected_min_steps, (
            f"output.max_global_steps is {max_global_steps}, expected > {expected_min_steps}"
        )
        assert output.stop_reason not in ("aborted", "abort"), (
            f"output.stop_reason is {output.stop_reason}, expected not abort"
        )
    print(f"========== [{initial_steps=}, {train_steps=}] ==========")
    print("[RESPONSE]", tokenizer.decode(outputs[0].token_ids, skip_special_tokens=True))


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
    trainer_pool = RayResourcePool(process_on_nodes=[init_config.trainer.n_gpus_per_node], max_colocate_count=3)
    trainer = create_trainer_worker_group(trainer_pool, model_config, checkpoint_engine_config)
    trainer.reset()

    # 2. create standalone rollout with AgentLoopManager
    agent_loop_manager = await AgentLoopManager.create(config=init_config)
    servers = list(
        zip(
            agent_loop_manager.server_addresses,
            [server._server_handle for server in agent_loop_manager.rollout_replicas],
            strict=True,
        )
    )
    load_balancer_handle = agent_loop_manager.global_load_balancer

    # 3. create checkpoint engine manager
    checkpoint_manager = CheckpointEngineManager(
        config=checkpoint_engine_config, trainer=trainer, replicas=agent_loop_manager.rollout_replicas
    )

    n = 4
    prompts = [
        [{"role": "user", "content": "Please write an article about the history of China, at least 1000 words."}],
        [{"role": "user", "content": "Please write an article about the history of America, at least 1000 words."}],
        [{"role": "user", "content": "Please write an article about the geography of China, at least 1000 words."}],
        [{"role": "user", "content": "Please write an article about the geography of America, at least 1000 words."}],
    ] * n

    server_manager = AsyncLLMServerManager(
        config=init_config, servers=servers, load_balancer_handle=load_balancer_handle
    )

    # 4. test update_weights with global_steps=None
    await _run_update_weights_with_global_steps_none(
        server_manager=server_manager,
        checkpoint_manager=checkpoint_manager,
        tokenizer=model_config.tokenizer,
    )

    # 5. test AsyncLLMServerManager without partial rollout resume
    await checkpoint_manager.update_weights(global_steps=0)
    await _run_server_manager_without_resume(
        initial_steps=1,
        train_steps=3,
        server_manager=server_manager,
        checkpoint_manager=checkpoint_manager,
        prompts=prompts,
        tokenizer=model_config.tokenizer,
    )

    # 6. test FullyAsyncLLMServerManager with partial rollout resume
    server_manager = FullyAsyncLLMServerManager(
        config=init_config, servers=servers, load_balancer_handle=load_balancer_handle
    )
    await _run_server_manager_with_resume(
        initial_steps=4,
        train_steps=3,
        server_manager=server_manager,
        checkpoint_manager=checkpoint_manager,
        prompts=prompts,
        tokenizer=model_config.tokenizer,
    )

    ray.shutdown()
