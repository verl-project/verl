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


def get_weight_loader(arch: str):
    from verl.models.mcore.loader import load_state_dict_to_megatron_gptmodel

    _MODEL_WEIGHT_MEGATRON_LOADER_REGISTRY = {
        "LlamaForCausalLM": load_state_dict_to_megatron_gptmodel,
        "Qwen2ForCausalLM": load_state_dict_to_megatron_gptmodel,
    }

    if arch in _MODEL_WEIGHT_MEGATRON_LOADER_REGISTRY:
        return _MODEL_WEIGHT_MEGATRON_LOADER_REGISTRY[arch]
    raise ValueError(
        f"Model architectures {arch} loader are not supported for now. Supported architectures: "
        f"{_MODEL_WEIGHT_MEGATRON_LOADER_REGISTRY.keys()}"
    )


def get_weight_saver(arch: str):
    from verl.models.mcore.saver import (
        merge_megatron_ckpt_gptmodel,
        merge_megatron_ckpt_gptmodel_dpskv3,
        merge_megatron_ckpt_gptmodel_mixtral,
        merge_megatron_ckpt_gptmodel_qwen2_5_vl,
        merge_megatron_ckpt_gptmodel_qwen_moe,
    )

    _MODEL_WEIGHT_MEGATRON_SAVER_REGISTRY = {
        "LlamaForCausalLM": merge_megatron_ckpt_gptmodel,
        "Qwen2ForCausalLM": merge_megatron_ckpt_gptmodel,
        "MixtralForCausalLM": merge_megatron_ckpt_gptmodel_mixtral,
        "Qwen2MoeForCausalLM": merge_megatron_ckpt_gptmodel_qwen_moe,
        "Qwen2_5_VLForConditionalGeneration": merge_megatron_ckpt_gptmodel_qwen2_5_vl,
        "DeepseekV3ForCausalLM": merge_megatron_ckpt_gptmodel_dpskv3,
        "Qwen3ForCausalLM": merge_megatron_ckpt_gptmodel,
        "Qwen3ForTokenClassification": merge_megatron_ckpt_gptmodel,
        "Qwen3MoeForCausalLM": merge_megatron_ckpt_gptmodel_qwen_moe,
        "LlamaForTokenClassification": merge_megatron_ckpt_gptmodel,
    }
    if arch in _MODEL_WEIGHT_MEGATRON_SAVER_REGISTRY:
        return _MODEL_WEIGHT_MEGATRON_SAVER_REGISTRY[arch]
    raise ValueError(
        f"Model architectures {arch} saver are not supported for now. Supported architectures: "
        f"{_MODEL_WEIGHT_MEGATRON_SAVER_REGISTRY.keys()}"
    )
