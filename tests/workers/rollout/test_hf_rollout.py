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

import torch
from omegaconf import OmegaConf
from torch.distributed.fsdp import CPUOffload, MixedPrecision
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardedStateDictConfig, ShardingStrategy, StateDictType
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl import DataProto
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.workers.rollout.hf_rollout import HFRollout

BASE_HF_ROLLOUT_CONFIG = {
    "temperature": 1.0,
    "top_k": -1,
    "top_p": 1,
    "prompt_length": 64,
    "response_length": 64,
    "do_sample": True,
    "n": 1,
    "val_kwargs": {
        "top_k": -1,
        "top_p": 1.0,
        "temperature": 0,
        "n": 1,
        "do_sample": False,
    },
}


def prepare_input_dataproto(tokenizer, config, validate):
    preencode_prompts = [
        [{"role": "user", "content": "Who won the Champions League in 2019?"}],
        [{"role": "user", "content": "The founder of Apple is"}],
        [{"role": "user", "content": "What's your name"}],
    ]
    formatted_prompts = [
        tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        for conversation in preencode_prompts
    ]
    prompts = tokenizer(formatted_prompts, return_tensors="pt", padding="max_length", max_length=config.prompt_length)
    input_dataproto = DataProto.from_dict(
        {
            "input_ids": prompts["input_ids"],
            "attention_mask": prompts["attention_mask"],
            "position_ids": compute_position_id_with_mask(prompts["attention_mask"]),
        },
        meta_info={
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "validate": validate,
        },
    )
    return input_dataproto


def prepare_fsdp_model(model, world_size):
    from torch.distributed.device_mesh import init_device_mesh

    device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"])

    mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)

    fsdp_model = FSDP(
        model,
        use_orig_params=True,
        auto_wrap_policy=None,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        cpu_offload=CPUOffload(offload_params=False),
        sync_module_states=False,
        device_mesh=device_mesh,
    )

    FSDP.set_state_dict_type(
        fsdp_model, state_dict_type=StateDictType.SHARDED_STATE_DICT, state_dict_config=ShardedStateDictConfig()
    )
    return fsdp_model


def test_hf_rollout(n: int = 1, do_sample: bool = True, validate: bool = False):
    config = OmegaConf.create(BASE_HF_ROLLOUT_CONFIG)
    config.update({"n": n, "do_sample": do_sample})

    assert torch.cuda.device_count() >= 2, "At least 2 GPUs is required to run tp+dp tests."
    local_rank, rank, world_size = initialize_global_process_group()

    # Initialize model and tokenizer
    local_cache_path = "~/.cache/verl/rlhf"
    local_cache_path = os.path.expanduser(local_cache_path)
    hdfs_path = "Qwen/Qwen2-7B-Instruct"
    local_model_path = copy_to_local(src=hdfs_path, cache_dir=local_cache_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, padding_side="left", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize FSDP model
    actor_model = AutoModelForCausalLM.from_pretrained(local_model_path, trust_remote_code=True)
    actor_model.to(torch.bfloat16)
    fsdp_model = prepare_fsdp_model(actor_model, world_size)

    # Initialize HFRollout and start generate
    hf_rollout = HFRollout(fsdp_model, OmegaConf.create(config))
    input = prepare_input_dataproto(tokenizer, config, validate).to(torch.cuda.current_device())
    outputs = hf_rollout.generate_sequences(input)

    # check generated batch size is expected
    generated_batch_size = outputs.batch.batch_size[0]
    assert generated_batch_size == input.batch.batch_size[0] * config.n

    for i in range(generated_batch_size):
        prompt_tokens = outputs.batch["prompts"][i]
        prompt_mask = prompt_tokens != tokenizer.pad_token_id
        prompt_tokens = prompt_tokens[prompt_mask]
        decoded_prompt = tokenizer.decode(prompt_tokens, skip_special_tokens=False)

        response_tokens = outputs.batch["responses"][i]
        response_mask = response_tokens != tokenizer.pad_token_id
        response_tokens = response_tokens[response_mask]
        decoded_response = tokenizer.decode(response_tokens, skip_special_tokens=False)

        attention_mask = outputs.batch["attention_mask"][i]
        position_ids = outputs.batch["position_ids"][i]
        prompt_length = outputs.batch["prompts"].size(1)
        response_length = outputs.batch["responses"].size(1)

        assert attention_mask.size(0) == prompt_length + response_length
        assert position_ids.size(0) == prompt_length + response_length

        # check response attention mask is expected
        response_attention = attention_mask[prompt_length:]
        eos_positions = (outputs.batch["responses"][i] == tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            first_eos_pos = eos_positions[0].item()
            assert response_attention[: first_eos_pos + 1].all(), "Response attention mask should be 1 until EOS"
            if first_eos_pos + 1 < response_length:
                assert not response_attention[first_eos_pos + 1 :].any(), (
                    "Response attention mask should be 0 after EOS"
                )
        else:
            assert response_attention.all(), "Response attention mask should be all 1 if no EOS token"

        # check response position ids is expected
        prompt_positions = position_ids[:prompt_length]
        response_positions = position_ids[prompt_length:]
        valid_response_length = min(len(response_tokens), response_length)
        if valid_response_length > 0:
            assert response_positions[0] == prompt_positions[-1] + 1
            for j in range(1, valid_response_length):
                assert response_positions[j] == response_positions[j - 1] + 1

        # print generated text for inspection
        if torch.distributed.get_rank() == 0:
            print(f"prompt: {decoded_prompt}")
            print(f"response: {decoded_response}")
            print("=" * 30)


if __name__ == "__main__":
    test_hf_rollout(n=2, do_sample=True, validate=False)
    # test_hf_rollout(n=1, do_sample=False, validate=True)
    # test_hf_rollout(n=1, do_sample=True, validate=False)
