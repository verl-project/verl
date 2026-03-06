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

import numbers

import torch
from megatron.core import ModelParallelConfig
from torch import nn
from transformers import LlamaConfig

from verl.utils.megatron import sequence_parallel as sp_utils


class ParallelLlamaRMSNorm(nn.Module):
    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        if isinstance(config.hidden_size, numbers.Integral):
            normalized_shape = (config.hidden_size,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.variance_epsilon = config.rms_norm_eps

        if megatron_config.sequence_parallel:
            sp_utils.mark_parameter_as_sequence_parallel(self.weight)

    def forward(self, hidden_states):
        from apex.normalization.fused_layer_norm import fused_rms_norm_affine

        return fused_rms_norm_affine(
            input=hidden_states,
            weight=self.weight,
            normalized_shape=self.normalized_shape,
            eps=self.variance_epsilon,
            memory_efficient=True,
        )
