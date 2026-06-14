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

import megatron.core.parallel_state as mpu
import torch
from megatron.core.transformer import MLATransformerConfig, TransformerConfig
from transformers import AutoConfig, PretrainedConfig

from verl.models.mcore import hf_to_mcore_config
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group

TEST_MODELS = [
    "Qwen/Qwen2.5-7B",  # Qwen2 dense
    "Qwen/Qwen3-8B",  # Qwen3 dense
    "deepseek-ai/deepseek-coder-1.3b-instruct",  # deepseek dense
    "Qwen/Qwen2-57B-A14B",  # Qwen2 moe
    "Qwen/Qwen3-30B-A3B",  # Qwen3 moe
    # "mistralai/Mixtral-8x7B-v0.1",  # Mixtral # require authentication
    "deepseek-ai/DeepSeek-V3-Base",  # Deepseek V3
]


def check_config_converter_results(tf_config: TransformerConfig | MLATransformerConfig, hf_config: PretrainedConfig):
    assert tf_config.num_layers == hf_config.num_hidden_layers, (
        f"Number of layers mismatch: {tf_config.num_layers} != {hf_config.num_hidden_layers}"
    )
    assert tf_config.hidden_size == hf_config.hidden_size, (
        f"Hidden size mismatch: {tf_config.hidden_size} != {hf_config.hidden_size}"
    )
    assert tf_config.num_attention_heads == hf_config.num_attention_heads, (
        f"Number of attention heads mismatch: {tf_config.num_attention_heads} != {hf_config.num_attention_heads}"
    )
    assert tf_config.num_query_groups == hf_config.num_key_value_heads, (
        f"Number of query groups mismatch: {tf_config.num_query_groups} != {hf_config.num_key_value_heads}"
    )
    assert tf_config.ffn_hidden_size == hf_config.intermediate_size, (
        f"FFN hidden size mismatch: {tf_config.ffn_hidden_size} != {hf_config.intermediate_size}"
    )
    assert tf_config.attention_dropout == hf_config.attention_dropout, (
        f"Attention dropout mismatch: {tf_config.attention_dropout} != {hf_config.attention_dropout}"
    )
    assert tf_config.hidden_dropout == getattr(hf_config, "hidden_dropout", 0.0), (
        f"Hidden dropout mismatch: {tf_config.hidden_dropout} != {getattr(hf_config, 'hidden_dropout', 0.0)}"
    )
    if getattr(hf_config, "head_dim", None) is not None:
        assert tf_config.kv_channels == getattr(hf_config, "head_dim", None), (
            f"Head dim mismatch: {tf_config.kv_channels} != {getattr(hf_config, 'head_dim', None)}"
        )
    assert tf_config.layernorm_epsilon == hf_config.rms_norm_eps, (
        f"Layernorm epsilon mismatch: {tf_config.layernorm_epsilon} != {hf_config.rms_norm_eps}"
    )


def modify_hf_config(name: str, hf_config: PretrainedConfig):
    if name == "deepseek-ai/DeepSeek-V3-Base":
        hf_config.num_nextn_predict_layers = 0
        hf_config.quantization_config = None
    return hf_config


def test_mcore_config_converter():
    """
    Test the conversion of Hugging Face model configurations to MCore configurations.
    """
    local_rank, rank, world_size = initialize_global_process_group()
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=None,
        use_sharp=False,
        context_parallel_size=2,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=None,
        nccl_communicator_config_path=None,
    )
    for model_name in TEST_MODELS:
        print(f"testing {model_name}")
        hf_config = AutoConfig.from_pretrained(os.path.expanduser(f"~/models/configs/{model_name}/config.json"))
        hf_config = modify_hf_config(model_name, hf_config)
        tf_config = hf_to_mcore_config(hf_config, torch.bfloat16)
        check_config_converter_results(tf_config, hf_config)

    destroy_global_process_group()


if __name__ == "__main__":
    test_mcore_config_converter()
