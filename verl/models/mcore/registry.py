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

from .model_forward import gptmodel_forward_model_engine
from .model_forward_fused import fused_forward_model_engine


class SupportedVLM(Enum):
    QWEN2_5_VL = "Qwen2_5_VLForConditionalGeneration"
    QWEN3_MOE_VL = "Qwen3VLMoeForConditionalGeneration"
    QWEN3_VL = "Qwen3VLForConditionalGeneration"
    QWEN3_5_MOE_VL = "Qwen3_5MoeForConditionalGeneration"
    QWEN3_5_VL = "Qwen3_5ForConditionalGeneration"


supported_vlm = [member.value for member in SupportedVLM]


def get_mcore_engine_forward_fn(hf_config) -> Callable:
    """
    Get the forward function for given model architecture.
    """
    assert len(hf_config.architectures) == 1, "Only one architecture is supported for now"
    return gptmodel_forward_model_engine


def get_mcore_forward_fused_model_engine_fn(hf_config) -> Callable:
    """
    Get the fused forward function for no-padding inputs.
    """
    assert len(hf_config.architectures) == 1, "Only one architecture is supported for now"
    if hf_config.architectures[0] in supported_vlm:
        return fused_forward_model_engine(True)
    else:
        # default to language model
        return fused_forward_model_engine(False)
