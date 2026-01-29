# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from verl.trainer.speculators.config import SpeculatorConfigBase
from verl.models.speculator.speculator import LayerNormParameterized


@dataclass
class MLPSpeculatorConfig(SpeculatorConfigBase):
    base_model_name_or_path: str
    emb_dim: int
    inner_dim: int
    vocab_size: int
    n_predict: int
    tie_weights: bool = False
    scale_input: bool = False

    def __post_init__(self):
        self.architectures = "MLPSpeculatorPreTrainedModel"
        self.model_type = "mlp_speculator"
        self.n_candidates = self.n_predict
        self.top_k_tokens_per_head = [1 for _ in range(self.n_predict)]
        self.torch_dtype = "bfloat16"
        self.transformers_version = transformers.__version__



class MLPSpeculator(nn.Module):
    """
    MLP-based speculator that conditions on base model embeddings and prior tokens.
    """

    def __init__(self, config: MLPSpeculatorConfig):
        super().__init__()

        self.config = config
        self.n_predict = config.n_predict
        self.emb_dim = config.emb_dim
        inner_dim = config.inner_dim
        self.inner_dim = inner_dim if inner_dim != 0 else self.emb_dim
        self.vocab_size = config.vocab_size
        self.scale_input = config.scale_input
        self.tie_weights = config.tie_weights

        self.emb = nn.ModuleList([nn.Embedding(self.vocab_size, self.inner_dim) for _ in range(self.n_predict)])
        self.proj = nn.ModuleList(
            [
                nn.Linear(
                    (self.emb_dim if i == 0 else self.inner_dim),
                    self.inner_dim,
                    bias=False,
                )
                for i in range(self.n_predict)
            ]
        )
        self.head = nn.ModuleList(
            [nn.Linear(self.inner_dim, self.vocab_size, bias=False) for _ in range(self.n_predict)]
        )
        self.ln = nn.ModuleList(
            [
                LayerNormParameterized(self.inner_dim, elementwise_shift=True, elementwise_scale=True)
                for _ in range(self.n_predict)
            ]
        )
        if self.scale_input:
            self.ln0 = LayerNormParameterized(self.emb_dim, elementwise_shift=False, elementwise_scale=False)

        self.state_weight = 0.5 ** (0.5 / self.n_predict)
        self.emb_weight = math.sqrt((1 - self.state_weight**2) * (self.inner_dim / 2))
        self.activation = nn.GELU()

        if self.tie_weights:
            assert self.n_predict > 1, "You cannot tie weights between stages when only 1 exists"

            for emb in self.emb:
                emb.weight = self.emb[0].weight

            for head in self.head:
                head.weight = self.head[0].weight

            for ln in self.ln:
                ln.weight = self.ln[0].weight
                ln.bias = self.ln[0].bias

            for i in range(2, self.n_predict):
                self.proj[i].weight = self.proj[1].weight

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Embedding, nn.Linear)):
                nn.init.normal_(m.weight, 0, 1 / math.sqrt(self.inner_dim))
            elif isinstance(m, LayerNormParameterized) and hasattr(m, "weight"):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, state: torch.Tensor, inds: torch.Tensor) -> torch.Tensor:
        out = []
        if self.scale_input:
            state = self.ln0(state) / (2**0.5)

        for i in range(self.n_predict):
            z = self.emb[i](inds[:, i : i + state.size(1)])
            state = self.proj[i](state)
            state = torch.add(state, z, alpha=self.emb_weight / self.state_weight)
            state = self.activation(self.ln[i](state))
            out.append(self.head[i](state))

        return torch.stack(out, dim=0)
