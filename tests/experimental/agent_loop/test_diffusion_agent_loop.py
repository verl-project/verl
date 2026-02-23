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
import os

import numpy as np
import pytest
import ray
from omegaconf import DictConfig

from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.protocol import DataProto


@pytest.fixture
def init_config() -> DictConfig:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose(config_name="ppo_diffusion_trainer")

    model_path = os.path.expanduser("~/models/Qwen/Qwen-Image")
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.model.tokenizer_path = os.path.join(model_path, "tokenizer")
    config.actor_rollout_ref.rollout.name = "vllm_omni"
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.enforce_eager = True
    config.actor_rollout_ref.rollout.n = 4
    config.actor_rollout_ref.rollout.num_inference_steps = 10
    config.actor_rollout_ref.rollout.guidance_scale = 4.0
    config.actor_rollout_ref.rollout.agent.num_workers = 2
    config.actor_rollout_ref.rollout.skip_tokenizer_init = True
    config.actor_rollout_ref.rollout.agent.default_agent_loop = "diffusion_single_turn_agent"
    config.actor_rollout_ref.rollout.noise_level = 1.0
    config.actor_rollout_ref.rollout.sde_window_size = 2
    config.actor_rollout_ref.rollout.sde_window_range = [0, 5]

    qwen_pipeline = "verl.utils.vllm_omni.pipelines.QwenImagePipelineWithLogProb"
    config.actor_rollout_ref.rollout.engine_kwargs.vllm_omni = {"custom_pipeline": qwen_pipeline}
    config.reward.reward_manager.name = "image"
    config.trainer.n_gpus_per_node = 4

    tokenizer_max_length = 1024
    prompt_template_encode_start_idx = 34
    max_length = tokenizer_max_length + prompt_template_encode_start_idx

    config.data.apply_chat_template_kwargs = dict(max_length=max_length, padding=True, truncation=True)
    config.data.max_prompt_length = max_length
    config.actor_rollout_ref.rollout.max_model_len = max_length

    # TODO (mike): test with TP later
    config.actor_rollout_ref.rollout.tensor_model_parallel_size = 1
    return config


def test_single_turn(init_config):
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
            }
        }
    )

    agent_loop_manager = AgentLoopManager(init_config)

    system_prompt = (
        "Describe the image by detailing the color, shape, size, texture, quantity, text, "
        "spatial relationships of the objects and background:"
    )
    user_prompts = ["A photo of cute cat with long fur and big eyes.", "A photo of cute dog with short hair."]

    raw_prompts = []
    for user_prompt in user_prompts:
        raw_prompts.append(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

    raw_negative_prompts = []
    for user_prompt in user_prompts:
        raw_negative_prompts.append(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": " "},
            ]
        )

    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array(raw_prompts),
            "raw_negative_prompt": np.array(raw_negative_prompts),
            "data_source": np.array(["jpeg_compressibility"] * len(raw_prompts)),
            "reward_model": np.array([{"style": "rule", "ground_truth": ""}] * len(raw_prompts)),
        },
    )
    n = init_config.actor_rollout_ref.rollout.n
    batch = batch.repeat(n)
    result = agent_loop_manager.generate_sequences(prompts=batch)
    assert len(result) == len(raw_prompts) * n

    expected_batch_keys = [
        "responses",
        "all_latents",
        "all_timesteps",
        "prompt_embeds",
        "prompt_embeds_mask",
        "input_ids",
        "attention_mask",
    ]
    for key in expected_batch_keys:
        assert key in result.batch, f"Key {key} not found in result batch."

    # check turns
    num_turns = result.non_tensor_batch["__num_turns__"]
    assert np.all(num_turns == 2)

    print("Test passed!")
    ray.shutdown()
