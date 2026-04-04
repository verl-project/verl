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
"""
Using FSDPTrainer
"""

import os

import hydra
import torch
from transformers import AutoTokenizer

from verl import DataProto
from verl.single_controller.torchrpc import torchrpc_remote
from verl.trainer.ppo.torchrpc_trainer import TorchRPCPPOTrainer
from verl.utils.fs import copy_to_local


def make_reward_function(tokenizer, num_examine):
    def arithmetic_sequence_reward_function(data: DataProto, return_dict: bool = False):
        from tests.e2e.envs.digit_completion.task import compute_reward

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        for i in range(data.batch.batch_size[0]):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            # extract raw prompt
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            # extract response
            response_ids = data_item.batch["responses"]
            response_length = response_ids.shape[-1]
            response_mask = data.batch["attention_mask"][i][-response_length:]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt = tokenizer.decode(valid_prompt_ids)
            response = tokenizer.decode(valid_response_ids)
            # remove bos and eos
            prompt = prompt.replace(tokenizer.sep_token, "")
            response = response.replace(tokenizer.eos_token, "")
            if i < num_examine:
                print(prompt, response)

            reward_output = compute_reward(prompt, response)
            dense_reward = reward_output[0].tolist()
            ground_truth_response = reward_output[1]["ground_truth_response"]
            last_reward = dense_reward[-1] if len(dense_reward) > 0 else 1 if len(ground_truth_response) == 0 else 0

            # pad to response_length
            for _ in range(reward_tensor.shape[-1] - len(dense_reward)):
                dense_reward.append(last_reward)

            dense_reward = torch.as_tensor(dense_reward, dtype=torch.float32, device=reward_tensor.device)
            reward_tensor[i] = dense_reward * response_mask

        if return_dict:
            return {"reward_tensor": reward_tensor}
        else:
            return reward_tensor

    return arithmetic_sequence_reward_function


@torchrpc_remote
@hydra.main(config_path="../../../../verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    # print initial config
    from pprint import pprint

    from omegaconf import OmegaConf

    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values

    # print the config
    # print initial config
    print("Config after normalizing batch_size")
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values

    # download the checkpoint from hdfs
    local_path = copy_to_local(config.actor_rollout_ref.model.path)
    local_path = os.path.expanduser(local_path)
    # instantiate tokenizern
    from transformers import LlamaConfig

    from tests.e2e.envs.digit_completion import CharTokenizer

    AutoTokenizer.register(LlamaConfig, CharTokenizer, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    print(f"Tokenizer vocab_size: {tokenizer.vocab_size}")

    # define worker classes
    from verl.trainer.ppo.torchrpc_trainer import ResourcePoolManager, Role
    from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

    role_worker_mapping = {
        Role.ActorRollout: ActorRolloutRefWorker,
        Role.Critic: CriticWorker,
    }

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
    }

    # use reward model
    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        role_worker_mapping[Role.RefPolicy] = ActorRolloutRefWorker
        mapping[Role.RefPolicy] = global_pool_id

    reward_fn = make_reward_function(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = TorchRPCPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        reward_fn=reward_fn,
        val_reward_fn=reward_fn,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    main()
