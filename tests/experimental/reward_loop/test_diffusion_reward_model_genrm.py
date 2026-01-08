# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
import ray
import torch
from hydra import compose, initialize_config_dir
from PIL import Image

from verl.experimental.reward_loop import DiffusionRewardLoopManager
from verl.protocol import DataProto
from verl.utils import hf_tokenizer


def create_data_samples(tokenizer) -> DataProto:
    images = ["assets/good.jpg", "assets/fair.jpg", "assets/poor.jpg", "assets/ocr.jpg"]
    prompts = ["a good photo", "a fair photo", "a poor photo", 'a photo of displaying "OCR"']
    pil_images = [np.array(Image.open(img).convert("RGB").resize((512, 512))) for img in images]
    responses = [torch.tensor(img).permute(2, 0, 1) / 255.0 for img in pil_images]
    data_source = ["ocr"] * len(images)
    reward_info = [{"ground_truth": "OCR"}] * len(images)
    extra_info = [{}] * len(images)

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
            "prompts": prompt_ids,
            "responses": responses,
        },
        non_tensors={
            "data_source": data_source,
            "reward_model": reward_info,
            "extra_info": extra_info,
        },
    )
    return data


def test_diffusion_reward_model_manager():
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

    rollout_model_name = os.path.expanduser("~/models/Qwen/Qwen-Image")
    reward_model_name = os.path.expanduser("~/models/Qwen/Qwen2.5-1.5B-Instruct")

    config.actor_rollout_ref.model.path = rollout_model_name
    config.custom_reward_function.path = "tests/experimental/reward_loop/reward_fn.py"
    config.custom_reward_function.name = "compute_score_ocr"
    config.reward_model.reward_manager = "diffusion"
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
    reward_loop_manager = DiffusionRewardLoopManager(config)

    # 2. init test data
    rollout_tokenizer = hf_tokenizer(rollout_model_name)
    data = create_data_samples(rollout_tokenizer)

    # 3. generate responses
    outputs = reward_loop_manager.compute_rm_score(data)

    for idx, output in enumerate(outputs):
        print(f"GRM Response {idx}:\n{output.non_tensor_batch['genrm_response']}\n")
        print(f"Score:\n{output.non_tensor_batch['score']}\n")
        print("=" * 50 + "\n")

    ray.shutdown()
