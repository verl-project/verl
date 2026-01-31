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

import os

import ray
import torch
from hydra import compose, initialize_config_dir

from verl.experimental.reward_loop import RewardLoopManager
from verl.protocol import DataProto
from verl.utils import hf_tokenizer
from verl.utils.model import compute_position_id_with_mask


def create_data_samples(tokenizer) -> DataProto:
    convs = [
        [
            {
                "role": "user",
                "content": "What is the range of the numeric output of a sigmoid node in a neural network?",
            },
            {"role": "assistant", "content": "Between -1 and 1."},
        ],
        [
            {
                "role": "user",
                "content": "What is the range of the numeric output of a sigmoid node in a neural network?",
            },
            {"role": "assistant", "content": "Between 0 and 1."},
        ],
        [
            {"role": "user", "content": "What is the capital of Australia?"},
            {
                "role": "assistant",
                "content": "Canberra is the capital city of Australia.",
            },
        ],
        [
            {"role": "user", "content": "What is the capital of Australia?"},
            {
                "role": "assistant",
                "content": "Sydney is the capital of Australia.",
            },
        ],
    ]
    raw_prompt = [conv[:1] for conv in convs]
    data_source = ["gsm8k"] * len(convs)
    reward_info = [{"ground_truth": "Not Used"}] * len(convs)
    extra_info = [{"question": conv[0]["content"]} for conv in convs]

    prompt_length, response_length = 1024, 4096
    pad_token_id = tokenizer.pad_token_id
    prompts, responses, input_ids, attention_masks = [], [], [], []
    for conv in convs:
        prompt_tokens = tokenizer.apply_chat_template(conv[:1], tokenize=True)
        response_tokens = tokenizer.apply_chat_template(conv, tokenize=True)[len(prompt_tokens) :]

        padded_prompt = [pad_token_id] * (prompt_length - len(prompt_tokens)) + prompt_tokens
        padded_response = response_tokens + [pad_token_id] * (response_length - len(response_tokens))
        attention_mask = (
            [0] * (prompt_length - len(prompt_tokens))
            + [1] * len(prompt_tokens)
            + [1] * len(response_tokens)
            + [0] * (response_length - len(response_tokens))
        )
        prompts.append(torch.tensor(padded_prompt))
        responses.append(torch.tensor(padded_response))
        input_ids.append(torch.tensor(padded_prompt + padded_response))
        attention_masks.append(torch.tensor(attention_mask))

    prompts = torch.stack(prompts)
    responses = torch.stack(responses)
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    position_ids = compute_position_id_with_mask(attention_masks)

    data = DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "position_ids": position_ids,
        },
        non_tensors={
            "data_source": data_source,
            "reward_model": reward_info,
            "raw_prompt": raw_prompt,
            "extra_info": extra_info,
        },
    )
    return data, convs


def test_reward_model_manager():
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

    rollout_model_name = os.path.expanduser("~/models/Qwen/Qwen2.5-0.5B-Instruct")
    reward_model_name = os.path.expanduser("~/models/Qwen/Qwen2.5-1.5B-Instruct")

    config.actor_rollout_ref.model.path = rollout_model_name
    config.custom_reward_function.path = "tests/experimental/reward_loop/reward_fn.py"
    config.custom_reward_function.name = "compute_score_gsm8k"
    config.reward_model.reward_manager = "dapo"
    config.reward_model.enable = True
    config.reward_model.enable_resource_pool = True
    config.reward_model.n_gpus_per_node = 8
    config.reward_model.nnodes = 1
    config.reward_model.model.path = reward_model_name
    config.reward_model.rollout.name = os.getenv("ROLLOUT_NAME", "vllm")
    config.reward_model.rollout.gpu_memory_utilization = 0.9
    config.reward_model.rollout.tensor_model_parallel_size = 2
    config.reward_model.rollout.skip_tokenizer_init = False
    config.reward_model.rollout.prompt_length = 2048
    config.reward_model.rollout.response_length = 4096

    # 1. init reward model manager
    reward_loop_manager = RewardLoopManager(config)

    # 2. init test data
    rollout_tokenizer = hf_tokenizer(rollout_model_name)
    data, convs = create_data_samples(rollout_tokenizer)

    # 3. generate responses
    outputs = reward_loop_manager.compute_rm_score(data)

    for idx, (conv, output) in enumerate(zip(convs, outputs, strict=True)):
        print(f"Problem {idx}:\n{conv[0]['content']}\n")
        print(f"AI Solution {idx}:\n{conv[1]['content']}\n")
        print(f"GRM Response {idx}:\n{output.non_tensor_batch['genrm_response']}\n")
        print("=" * 50 + "\n")

    ray.shutdown()
