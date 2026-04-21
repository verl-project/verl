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

import os

import ray
import torch
from hydra import compose, initialize_config_dir

from verl.experimental.reward_loop import RewardLoopManager
from verl.protocol import DataProto
from verl.utils import hf_tokenizer


def create_data_samples(tokenizer, data_source="ocr") -> DataProto:
    prompts = ['a photo of displaying "OCR"'] * 3
    responses = [torch.randn((3, 512, 512))] * 3
    data_source = [data_source] * len(responses)
    reward_info = [{"ground_truth": "OCR"}] * len(responses)
    extra_info = [{}] * len(responses)

    responses = torch.stack(responses)
    prompt_length = 1024
    pad_token_id = tokenizer.pad_token_id
    prompt_ids = []
    for prompt in prompts:
        prompt_tokens = tokenizer.encode(prompt)
        padded_prompt = [pad_token_id] * (prompt_length - len(prompt_tokens)) + prompt_tokens
        prompt_ids.append(torch.tensor(padded_prompt))
    prompt_ids = torch.stack(prompt_ids)

    data = DataProto.from_dict(
        tensors={
            "input_ids": prompt_ids,
            "responses": responses,
        },
        non_tensors={
            "data_source": data_source,
            "reward_model": reward_info,
            "extra_info": extra_info,
        },
    )
    return data


def test_reward_model_genrm():
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        }
    )
    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose(config_name="ppo_trainer")

    rollout_model_name = os.path.expanduser("~/models/tiny-random/Qwen-Image")
    reward_model_name = os.path.expanduser("~/models/tiny-random/qwen3-vl")

    config.actor_rollout_ref.model.path = rollout_model_name
    config.actor_rollout_ref.model.tokenizer_path = os.path.join(rollout_model_name, "tokenizer")
    config.reward.custom_reward_function.path = "examples/flowgrpo_trainer/reward_fn.py"
    config.reward.custom_reward_function.name = "compute_score_ocr"
    config.reward.num_workers = 1
    config.reward.reward_manager.name = "visual"
    config.reward.reward_model.enable = True
    config.reward.reward_model.enable_resource_pool = True
    config.reward.reward_model.n_gpus_per_node = 2
    config.reward.reward_model.nnodes = 1
    config.reward.reward_model.model_path = reward_model_name
    config.reward.reward_model.rollout.name = os.getenv("ROLLOUT_NAME", "vllm")
    config.reward.reward_model.rollout.gpu_memory_utilization = 0.9
    config.reward.reward_model.rollout.tensor_model_parallel_size = 2
    config.reward.reward_model.rollout.skip_tokenizer_init = False
    config.reward.reward_model.rollout.prompt_length = 2048
    config.reward.reward_model.rollout.response_length = 32

    # 1. init reward model manager
    reward_loop_manager = RewardLoopManager(config)

    # 2. init test data
    rollout_tokenizer = hf_tokenizer(config.actor_rollout_ref.model.tokenizer_path)
    data = create_data_samples(rollout_tokenizer)

    # 3. generate responses
    outputs = reward_loop_manager.compute_rm_score(data)

    for idx, output in enumerate(outputs):
        print(f"GRM Response {idx}:\n{output.non_tensor_batch['genrm_response']}\n")
        print(f"Score:\n{output.non_tensor_batch['score']}\n")
        print("=" * 50 + "\n")

    ray.shutdown()


def test_rule_reward():
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        }
    )
    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose(config_name="ppo_trainer")

    rollout_model_name = os.path.expanduser("~/models/tiny-random/Qwen-Image")

    config.actor_rollout_ref.model.path = rollout_model_name
    config.actor_rollout_ref.model.tokenizer_path = os.path.join(rollout_model_name, "tokenizer")
    config.reward.num_workers = 1
    config.reward.reward_manager.name = "visual"
    config.reward.reward_model.enable = False

    # 1. init reward model manager
    reward_loop_manager = RewardLoopManager(config)

    # 2. init test data
    rollout_tokenizer = hf_tokenizer(config.actor_rollout_ref.model.tokenizer_path)
    data = create_data_samples(rollout_tokenizer, data_source="jpeg_compressibility")

    # 3. generate responses
    outputs = reward_loop_manager.compute_rm_score(data)

    for idx, output in enumerate(outputs):
        print(f"Rule-based Reward Score:\n{output.batch['rm_scores']}\n")
        print("=" * 50 + "\n")

    ray.shutdown()
