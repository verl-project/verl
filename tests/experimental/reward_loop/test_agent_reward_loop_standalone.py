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
from hydra import compose, initialize_config_dir
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.experimental.agent_loop import AgentLoopManager
from verl.experimental.reward_loop import RewardLoopManager
from verl.protocol import DataProto
from verl.trainer.main_ppo import create_rl_sampler
from verl.utils import hf_tokenizer
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn


def test_agent_reward_loop_standalone():
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

    rollout_model_path = os.path.expanduser("~/models/Qwen/Qwen2.5-0.5B-Instruct")
    reward_model_path = os.path.expanduser("~/models/Qwen/Qwen2.5-1.5B-Instruct")

    # actor_rollout_ref config
    config.data.return_raw_chat = True
    config.data.max_prompt_length = 1024
    config.data.max_response_length = 4096
    config.actor_rollout_ref.model.path = rollout_model_path
    config.actor_rollout_ref.actor.use_dynamic_bsz = True
    config.actor_rollout_ref.rollout.name = os.getenv("ROLLOUT_NAME", "vllm")
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.tensor_model_parallel_size = 2
    config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.9
    config.actor_rollout_ref.rollout.enforce_eager = True
    config.actor_rollout_ref.rollout.prompt_length = 1024
    config.actor_rollout_ref.rollout.response_length = 4096
    config.actor_rollout_ref.rollout.skip_tokenizer_init = True
    config.trainer.n_gpus_per_node = 4
    config.trainer.nnodes = 1

    config.reward_model.reward_manager.name = "dapo"
    config.reward_model.enable = True
    config.reward_model.enable_resource_pool = True
    config.reward_model.n_gpus_per_node = 4
    config.reward_model.nnodes = 1
    config.reward_model.model_path = reward_model_path
    config.reward_model.rollout.name = os.getenv("ROLLOUT_NAME", "vllm")
    config.reward_model.rollout.gpu_memory_utilization = 0.9
    config.reward_model.rollout.tensor_model_parallel_size = 2
    config.reward_model.rollout.skip_tokenizer_init = False
    config.reward_model.rollout.prompt_length = 5120
    config.reward_model.rollout.response_length = 4096
    config.custom_reward_function.path = "tests/experimental/reward_loop/reward_fn.py"
    config.custom_reward_function.name = "compute_score_gsm8k"

    # 1. init reward model manager
    reward_loop_manager = RewardLoopManager(config)
    agent_loop_manager = AgentLoopManager(
        config=config, reward_loop_worker_handles=reward_loop_manager.reward_loop_workers
    )

    # 2. init test data
    local_folder = os.path.expanduser("~/data/gsm8k/")
    data_files = [os.path.join(local_folder, "train.parquet")]
    tokenizer = hf_tokenizer(rollout_model_path)

    dataset = RLHFDataset(
        data_files=data_files,
        tokenizer=tokenizer,
        config=config.data,
        processor=None,
    )

    batch_size = 64
    sampler = create_rl_sampler(config.data, dataset)
    dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=config.data.dataloader_num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        sampler=sampler,
    )

    # 3. generate responses
    batch_dict = next(iter(dataloader))
    batch = DataProto.from_single_dict(batch_dict)

    # standalone reward model should wake up for agent_reward_loop
    gen_batch = agent_loop_manager.generate_sequences(prompts=batch)

    rm_scores = gen_batch.batch["rm_scores"]
    sample_scores = rm_scores.sum(dim=1)
    print(sample_scores)

    ray.shutdown()
