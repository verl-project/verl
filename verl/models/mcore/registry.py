# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
Registry module for model architecture components.
"""

from enum import Enum
from typing import Callable

import torch
import torch.nn as nn

from .model_forward import gptmodel_forward_no_padding, model_forward_gen
from .model_forward_fused import fused_forward_model_gen, fused_forward_no_padding_gen


class SupportedVLM(Enum):
    QWEN2_5_VL = "Qwen2_5_VLForConditionalGeneration"
    QWEN3_MOE_VL = "Qwen3VLMoeForConditionalGeneration"
    QWEN3_VL = "Qwen3VLForConditionalGeneration"


supported_vlm = [member.value for member in SupportedVLM]


def get_mcore_forward_fn(hf_config) -> Callable:
    """
    Get the forward function for given model architecture.
    """
    assert len(hf_config.architectures) == 1, "Only one architecture is supported for now"
    if hf_config.architectures[0] in supported_vlm:
        return model_forward_gen(True)
    else:
        # default to language model
        return model_forward_gen(False)


def get_mcore_forward_no_padding_fn(hf_config) -> Callable:
    """
    Get the forward function for given model architecture.
    """
    assert len(hf_config.architectures) == 1, "Only one architecture is supported for now"
    return gptmodel_forward_no_padding


def get_mcore_forward_fused_fn(hf_config) -> Callable:
    """
    Get the forward function for given model architecture.
    """
    assert len(hf_config.architectures) == 1, "Only one architecture is supported for now"
    if hf_config.architectures[0] in supported_vlm:
        return fused_forward_model_gen(True)
    else:
        # default to language model
        return fused_forward_model_gen(False)


def get_mcore_forward_fused_no_padding_fn(hf_config) -> Callable:
    """
    Get the fused forward function for no-padding inputs.
    """
    assert len(hf_config.architectures) == 1, "Only one architecture is supported for now"
    if hf_config.architectures[0] in supported_vlm:
        return fused_forward_no_padding_gen(True)
    else:
        # default to language model
        return fused_forward_no_padding_gen(False)


# ruff: noqa

########################################################
# below is the deprecated code
########################################################

from .config_converter import (
    PretrainedConfig,
    TransformerConfig,
    hf_to_mcore_config_dense,
    hf_to_mcore_config_dpskv3,
    hf_to_mcore_config_llama4,
    hf_to_mcore_config_mixtral,
    hf_to_mcore_config_qwen2_5_vl,
    hf_to_mcore_config_qwen2moe,
    hf_to_mcore_config_qwen3moe,
)
from .model_initializer import (
    BaseModelInitializer,
    DeepseekV3Model,
    DenseModel,
    MixtralModel,
    Qwen2MoEModel,
    Qwen3MoEModel,
    Qwen25VLModel,
)
from .weight_converter import (
    McoreToHFWeightConverterDense,
    McoreToHFWeightConverterDpskv3,
    McoreToHFWeightConverterMixtral,
    McoreToHFWeightConverterQwen2_5_VL,
    McoreToHFWeightConverterQwen2Moe,
    McoreToHFWeightConverterQwen3Moe,
)


class SupportedModel(Enum):
    LLAMA = "LlamaForCausalLM"  # tested
    QWEN2 = "Qwen2ForCausalLM"  # tested
    QWEN2_MOE = "Qwen2MoeForCausalLM"  # pending
    DEEPSEEK_V3 = "DeepseekV3ForCausalLM"  # not tested
    MIXTRAL = "MixtralForCausalLM"  # tested
    QWEN2_5_VL = "Qwen2_5_VLForConditionalGeneration"  # not supported
    LLAMA4 = "Llama4ForConditionalGeneration"  # not tested
    QWEN3 = "Qwen3ForCausalLM"  # tested
    QWEN3_MOE = "Qwen3MoeForCausalLM"  # tested
    GLM4_MOE = "Glm4MoeForCausalLM"
    QWEN3_TOKEN_CLASSIFICATION = "Qwen3ForTokenClassification"
    LLAMA_TOKEN_CLASSIFICATION = "LlamaForTokenClassification"
    QWEN3_MOE_VL = "Qwen3VLMoeForConditionalGeneration"
    QWEN3_VL = "Qwen3VLForConditionalGeneration"
    GPT_OSS = "GptOssForCausalLM"
    MiMO = "MiMoForCausalLM"


# Registry for model configuration converters
MODEL_CONFIG_CONVERTER_REGISTRY: dict[SupportedModel, Callable[[PretrainedConfig, torch.dtype], TransformerConfig]] = {
    SupportedModel.LLAMA: hf_to_mcore_config_dense,
    SupportedModel.QWEN2: hf_to_mcore_config_dense,
    SupportedModel.QWEN2_MOE: hf_to_mcore_config_qwen2moe,
    SupportedModel.DEEPSEEK_V3: hf_to_mcore_config_dpskv3,
    SupportedModel.MIXTRAL: hf_to_mcore_config_mixtral,
    SupportedModel.QWEN2_5_VL: hf_to_mcore_config_qwen2_5_vl,
    SupportedModel.LLAMA4: hf_to_mcore_config_llama4,
    SupportedModel.QWEN3: hf_to_mcore_config_dense,
    SupportedModel.QWEN3_MOE: hf_to_mcore_config_qwen3moe,
    SupportedModel.QWEN3_TOKEN_CLASSIFICATION: hf_to_mcore_config_dense,
    SupportedModel.LLAMA_TOKEN_CLASSIFICATION: hf_to_mcore_config_dense,
}

# Registry for model initializers
MODEL_INITIALIZER_REGISTRY: dict[SupportedModel, type[BaseModelInitializer]] = {
    SupportedModel.LLAMA: DenseModel,
    SupportedModel.QWEN2: DenseModel,
    SupportedModel.QWEN2_MOE: Qwen2MoEModel,
    SupportedModel.MIXTRAL: MixtralModel,
    SupportedModel.DEEPSEEK_V3: DeepseekV3Model,
    SupportedModel.QWEN2_5_VL: Qwen25VLModel,
    SupportedModel.LLAMA4: DenseModel,
    SupportedModel.QWEN3: DenseModel,
    SupportedModel.QWEN3_MOE: Qwen3MoEModel,
    SupportedModel.QWEN3_TOKEN_CLASSIFICATION: DenseModel,
    SupportedModel.LLAMA_TOKEN_CLASSIFICATION: DenseModel,
}

# Registry for model forward functions
MODEL_FORWARD_REGISTRY: dict[SupportedModel, Callable] = {
    SupportedModel.LLAMA: model_forward_gen(),
    SupportedModel.QWEN2: model_forward_gen(),
    SupportedModel.QWEN2_MOE: model_forward_gen(),
    SupportedModel.MIXTRAL: model_forward_gen(),
    SupportedModel.DEEPSEEK_V3: model_forward_gen(),
    SupportedModel.LLAMA4: model_forward_gen(),
    SupportedModel.QWEN3: model_forward_gen(),
    SupportedModel.QWEN3_MOE: model_forward_gen(),
    SupportedModel.QWEN2_5_VL: model_forward_gen(True),
    SupportedModel.QWEN3_MOE_VL: model_forward_gen(True),
    SupportedModel.QWEN3_VL: model_forward_gen(True),
    SupportedModel.GLM4_MOE: model_forward_gen(),
    SupportedModel.QWEN3_TOKEN_CLASSIFICATION: model_forward_gen(),
    SupportedModel.LLAMA_TOKEN_CLASSIFICATION: model_forward_gen(),
    SupportedModel.GPT_OSS: model_forward_gen(),
    SupportedModel.MiMO: model_forward_gen(),
}

# Registry for model forward functions
MODEL_FORWARD_NOPAD_REGISTRY: dict[SupportedModel, Callable] = {
    SupportedModel.LLAMA: gptmodel_forward_no_padding,
    SupportedModel.QWEN2: gptmodel_forward_no_padding,
    SupportedModel.QWEN2_MOE: gptmodel_forward_no_padding,
    SupportedModel.MIXTRAL: gptmodel_forward_no_padding,
    SupportedModel.DEEPSEEK_V3: gptmodel_forward_no_padding,
    SupportedModel.QWEN2_5_VL: gptmodel_forward_no_padding,
    SupportedModel.QWEN3_MOE_VL: gptmodel_forward_no_padding,
    SupportedModel.QWEN3_VL: gptmodel_forward_no_padding,
    SupportedModel.LLAMA4: gptmodel_forward_no_padding,
    SupportedModel.QWEN3: gptmodel_forward_no_padding,
    SupportedModel.QWEN3_MOE: gptmodel_forward_no_padding,
    SupportedModel.GLM4_MOE: gptmodel_forward_no_padding,
    SupportedModel.QWEN3_TOKEN_CLASSIFICATION: gptmodel_forward_no_padding,
    SupportedModel.LLAMA_TOKEN_CLASSIFICATION: gptmodel_forward_no_padding,
    SupportedModel.GPT_OSS: gptmodel_forward_no_padding,
    SupportedModel.MiMO: gptmodel_forward_no_padding,
}

# Registry for model forward functions
MODEL_FORWARD_FUSED_REGISTRY: dict[SupportedModel, Callable] = {
    SupportedModel.LLAMA: fused_forward_model_gen(),
    SupportedModel.QWEN2: fused_forward_model_gen(),
    SupportedModel.QWEN2_MOE: fused_forward_model_gen(),
    SupportedModel.MIXTRAL: fused_forward_model_gen(),
    SupportedModel.QWEN2_5_VL: fused_forward_model_gen(True),
    SupportedModel.QWEN3_MOE_VL: fused_forward_model_gen(True),
    SupportedModel.QWEN3_VL: fused_forward_model_gen(True),
    SupportedModel.LLAMA4: fused_forward_model_gen(),
    SupportedModel.QWEN3: fused_forward_model_gen(),
    SupportedModel.QWEN3_MOE: fused_forward_model_gen(),
    SupportedModel.DEEPSEEK_V3: fused_forward_model_gen(),
    SupportedModel.GLM4_MOE: fused_forward_model_gen(),
    SupportedModel.GPT_OSS: fused_forward_model_gen(),
    SupportedModel.MiMO: fused_forward_model_gen(),
}

# Registry for model weight converters
MODEL_WEIGHT_CONVERTER_REGISTRY: dict[SupportedModel, type] = {
    SupportedModel.LLAMA: McoreToHFWeightConverterDense,
    SupportedModel.QWEN2: McoreToHFWeightConverterDense,
    SupportedModel.QWEN2_MOE: McoreToHFWeightConverterQwen2Moe,
    SupportedModel.MIXTRAL: McoreToHFWeightConverterMixtral,
    SupportedModel.DEEPSEEK_V3: McoreToHFWeightConverterDpskv3,
    SupportedModel.QWEN3: McoreToHFWeightConverterDense,
    SupportedModel.QWEN3_MOE: McoreToHFWeightConverterQwen3Moe,
    SupportedModel.QWEN2_5_VL: McoreToHFWeightConverterQwen2_5_VL,
    SupportedModel.QWEN3_TOKEN_CLASSIFICATION: McoreToHFWeightConverterDense,
    SupportedModel.LLAMA_TOKEN_CLASSIFICATION: McoreToHFWeightConverterDense,
}


def get_supported_model(model_type: str) -> SupportedModel:
    try:
        return SupportedModel(model_type)
    except ValueError as err:
        supported_models = [e.value for e in SupportedModel]
        raise NotImplementedError(
            f"Model Type: {model_type} not supported. Supported models: {supported_models}"
        ) from err


def hf_to_mcore_config(
    hf_config: PretrainedConfig, dtype: torch.dtype, **override_transformer_config_kwargs
) -> TransformerConfig:
    """Convert huggingface PretrainedConfig to mcore TransformerConfig.

    Args:
        hf_config: The huggingface PretrainedConfig.
        dtype: The dtype of the model.
        **override_transformer_config_kwargs: The kwargs to override the transformer config.

    Returns:
        The mcore TransformerConfig.
    """
    assert len(hf_config.architectures) == 1, "Only one architecture is supported for now"
    model = get_supported_model(hf_config.architectures[0])
    return MODEL_CONFIG_CONVERTER_REGISTRY[model](hf_config, dtype, **override_transformer_config_kwargs)


def init_mcore_model(
    tfconfig: TransformerConfig,
    hf_config: PretrainedConfig,
    pre_process: bool = True,
    post_process: bool = None,
    *,
    share_embeddings_and_output_weights: bool = False,
    value: bool = False,
    **extra_kwargs,  # may be used for vlm and moe
) -> nn.Module:
    """
    Initialize a Mcore model.

    Args:
        tfconfig: The transformer config.
        hf_config: The HuggingFace config.
        pre_process: Optional pre-processing function.
        post_process: Optional post-processing function.
        share_embeddings_and_output_weights: Whether to share embeddings and output weights.
        value: Whether to use value.
        **extra_kwargs: Additional keyword arguments.

    Returns:
        The initialized model.
    """
    assert len(hf_config.architectures) == 1, "Only one architecture is supported for now"
    model = get_supported_model(hf_config.architectures[0])
    initializer_cls = MODEL_INITIALIZER_REGISTRY[model]
    initializer = initializer_cls(tfconfig, hf_config)
    return initializer.initialize(
        pre_process=pre_process,
        post_process=post_process,
        share_embeddings_and_output_weights=share_embeddings_and_output_weights,
        value=value,
        **extra_kwargs,
    )


def get_mcore_weight_converter(hf_config: PretrainedConfig, dtype: torch.dtype) -> Callable:
    """
    Get the weight converter for given model architecture.
    """
    assert len(hf_config.architectures) == 1, "Only one architecture is supported for now"
    model = get_supported_model(hf_config.architectures[0])
    tfconfig = hf_to_mcore_config(hf_config, dtype)
    return MODEL_WEIGHT_CONVERTER_REGISTRY[model](hf_config, tfconfig)
