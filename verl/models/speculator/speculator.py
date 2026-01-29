# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The model definition was built on-top of the model defined in:
# https://github.com/foundation-model-stack/fms-extras/blob/c5c294defa01459ff435e8ff6132c707eff9d22b/fms_extras/models/speculator.py.
# Modifications have been made by Snowflake.

import math

import torch
import torch.nn as nn

# from .configs import MLPSpeculatorConfig
from dataclasses import make_dataclass


class LayerNormParameterized(nn.Module):
    """
    A generalized LayerNorm implementation. With all optional arguments set to True, equivalent to nn.LayerNorm up to epsilon stabilization term
    (this class divides inputs by min(norm, eps), while nn.LayerNorm divides by norm + eps).
    ...
    Args
    ----
    normalized_shape : int
        Dimensionality of input data (size of final tensor axis)
    eps : float
        Safety term to prevent division by zero. Make sure the chosen value fits in the range of your encoding scheme (i.e. fp16 requires eps >= 6e-8).
    elementwise_scale : bool
        Include a learned scaling term after normalization?
    elementwise_shift : bool
        Include a learned bias term after normalization?
    use_mean : bool
        Recenter inputs around zero before normalizing, or just rescale?
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-06,
        elementwise_scale=True,
        elementwise_shift=False,
        use_mean=False,
        use_high_precision_pow=False,
    ):
        super(LayerNormParameterized, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_scale = elementwise_scale
        self.elementwise_shift = elementwise_shift
        self.use_mean = use_mean
        self.use_high_precision_pow = use_high_precision_pow

        if self.elementwise_scale:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape))
        # else:
        #     self.register_parameter("weight", None)
        if self.elementwise_shift:
            self.bias = nn.Parameter(torch.empty(self.normalized_shape))
        # else:
        #     self.register_parameter("bias", None)

    def reset_parameters(self):
        if self.elementwise_scale:
            self.weight.data.fill_(1)
        if self.elementwise_shift:
            self.bias.data.zero_()

    def forward(self, x):
        if self.use_mean:
            x = x - x.mean(-1, keepdim=True)
        # x = F.normalize(x, dim=-1)*math.sqrt(x.size(-1))
        xf = x
        if self.use_high_precision_pow:
            xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        x = xf.type_as(x)
        if self.elementwise_scale:
            x = self.weight * x
        if self.elementwise_shift:
            x = x + self.bias
        return x


class ArcticLSTMSpeculator(nn.Module):
    def __init__(self, config):  # noqa: C901
        super().__init__()

        self.config = config
        self.n_predict = config.n_predict
        self.input_hidden_dim = config.input_hidden_dim

        def parse_dim(s):
            if isinstance(s, int):
                return [s]
            elif isinstance(s, str):
                return [int(i) for i in s.split(".")]
            else:
                raise NotImplementedError

        self.inner_dim = parse_dim(config.inner_dim)
        self.emb_dim = parse_dim(config.emb_dim)
        self.proj_dim = parse_dim(config.proj_dim)

        self.vocab_size = config.vocab_size
        self.scale_input = config.scale_input
        self.tie_weights = config.tie_weights
        self.tie_lstm_embs = config.tie_lstm_embs
        self.method = config.method
        self.activation = nn.GELU()

        if self.method == "sum_rnn":
            embs = []
            for n_i in range(self.n_predict):
                if not self.tie_weights or n_i == 0:
                    seqs = [nn.Embedding(self.vocab_size, self.emb_dim[0])]
                    for i in range(1, len(self.emb_dim)):
                        print(f"ADDING ANOTHER EMB {i}")
                        seqs.append(
                            LayerNormParameterized(
                                self.emb_dim[i],
                                elementwise_shift=True,
                                elementwise_scale=True,
                            )
                        )
                        seqs.append(self.activation)
                        seqs.append(nn.Linear(self.emb_dim[i - 1], self.emb_dim[i], bias=False))
                    embs.append(nn.Sequential(*seqs))
            self.emb = nn.ModuleList(embs)

            projs = []
            for n_i in range(self.n_predict):
                if not self.tie_weights or n_i <= 1:
                    seqs = [
                        nn.Linear(
                            (self.input_hidden_dim if n_i == 0 else self.inner_dim[-1]),
                            self.proj_dim[0],
                            bias=False,
                        )
                    ]
                    for i in range(1, len(self.proj_dim)):
                        print(f"ADDING ANOTHER PROJ {i}")
                        seqs.append(
                            LayerNormParameterized(
                                self.proj_dim[i],
                                elementwise_shift=True,
                                elementwise_scale=True,
                            )
                        )
                        seqs.append(self.activation)
                        seqs.append(nn.Linear(self.proj_dim[i - 1], self.proj_dim[i], bias=False))
                    projs.append(nn.Sequential(*seqs))
            self.proj = nn.ModuleList(projs)

            lns = []
            for n_i in range(self.n_predict):
                if not self.tie_weights or n_i == 0:
                    seqs = [
                        LayerNormParameterized(
                            self.inner_dim[0],
                            elementwise_shift=True,
                            elementwise_scale=True,
                        )
                    ]
                    for i in range(1, len(self.inner_dim)):
                        print(f"ADDING ANOTHER LN {i}")
                        seqs.append(self.activation)
                        seqs.append(nn.Linear(self.inner_dim[i - 1], self.inner_dim[i], bias=False))
                        seqs.append(
                            LayerNormParameterized(
                                self.inner_dim[i],
                                elementwise_shift=True,
                                elementwise_scale=True,
                            )
                        )
                    lns.append(nn.Sequential(*seqs))
            self.ln = nn.ModuleList(lns)

        elif self.method == "sum_lstm":
            assert self.tie_weights
            self.forget_emb = nn.ModuleList([nn.Embedding(self.vocab_size, self.emb_dim[0])])
            if self.tie_lstm_embs:
                self.input_emb = self.cell_emb = self.output_emb = self.forget_emb
            else:
                self.input_emb = nn.ModuleList([nn.Embedding(self.vocab_size, self.emb_dim[0])])
                self.cell_emb = nn.ModuleList([nn.Embedding(self.vocab_size, self.emb_dim[0])])
                self.output_emb = nn.ModuleList([nn.Embedding(self.vocab_size, self.emb_dim[0])])
            self.forget_proj = nn.ModuleList(
                [
                    nn.Linear(self.input_hidden_dim, self.proj_dim[0], bias=False),
                    nn.Linear(self.inner_dim[-1], self.proj_dim[0], bias=False),
                ]
            )
            self.input_proj = nn.ModuleList(
                [
                    nn.Linear(self.input_hidden_dim, self.proj_dim[0], bias=False),
                    nn.Linear(self.inner_dim[-1], self.proj_dim[0], bias=False),
                ]
            )
            self.cell_proj = nn.ModuleList(
                [
                    nn.Linear(self.input_hidden_dim, self.proj_dim[0], bias=False),
                    nn.Linear(self.inner_dim[-1], self.proj_dim[0], bias=False),
                ]
            )
            self.output_proj = nn.ModuleList(
                [
                    nn.Linear(self.input_hidden_dim, self.proj_dim[0], bias=False),
                    nn.Linear(self.inner_dim[-1], self.proj_dim[0], bias=False),
                ]
            )
            self.cell_ln = nn.ModuleList(
                [
                    LayerNormParameterized(
                        self.inner_dim[0],
                        elementwise_shift=True,
                        elementwise_scale=True,
                    )
                ]
            )
            self.state_ln = nn.ModuleList(
                [
                    LayerNormParameterized(
                        self.inner_dim[0],
                        elementwise_shift=True,
                        elementwise_scale=True,
                    )
                ]
            )

        if self.scale_input:
            self.ln0 = LayerNormParameterized(self.input_hidden_dim, elementwise_shift=False, elementwise_scale=False)

        self.head = nn.ModuleList(
            [nn.Linear(self.inner_dim[-1], self.vocab_size, bias=False) for _ in range(self.n_predict)]
        )

        # Weights ensure that state_0 accounts for 50% of state magnitude by final head in expectation
        self.state_weight = 0.5 ** (0.5 / self.n_predict)
        self.emb_weight = math.sqrt((1 - self.state_weight**2) * (self.emb_dim[-1] / 2))

        # Handle weight tying as specified
        if self.tie_weights and self.n_predict > 1:
            for head in self.head:
                head.weight = self.head[0].weight

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 1 / math.sqrt(min(m.weight.shape)))
            elif isinstance(m, LayerNormParameterized) and hasattr(m, "weight"):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(
        self,
        state: torch.Tensor,
        inds: torch.Tensor,
    ) -> torch.Tensor:
        """
        FOR TRAINING
        A parallel forward pass on pre-existing ground-truth tokens in pretraining contexts.
        Produces self.n_predict predicted tokens for each token embedding in state.
        Inds requires self.n_predict extra tokens on the right to "simulate" recursive
        behavior for end positions.
        ...
        Args
        ----
        state : torch.Tensor
            Embedding vectors from the base model for a given sequence.
            Expects size [b n d] where b is batch size, n is seq len, and d is model width.
        inds : torch.Tensor
            Ground-truth token indices. inds[:,i] is the prediction coming from state[:,i]
            (or the legal fiction ground truth corresponding to that prediction).
            Expects size [b n+self.n_predict].
        ...
        Output : torch.Tensor
            Prediction logits at each position, for each head of the speculator.
            Has size [self.n_predict b n v] where v is vocab size.
        """
        out = []
        if self.scale_input:
            state = self.ln0(state) / (2**0.5)

        state_shapes = list(state.shape)
        state_shapes[-1] = self.inner_dim[-1]
        if self.method == "sum_lstm":
            cell_state = torch.zeros(state_shapes, device=state.device, dtype=state.dtype)
            for i in range(self.n_predict):
                prev_state = state
                actual_i = 0 if self.tie_weights else i
                actual_proj_i = 1 if self.tie_weights and i >= 2 else i

                # inds_index = torch.arange(i, i + state.size(1), device=inds.device)
           
                # inds_slice = inds.index_select(1, inds_index)
                inds_slice=inds[:, i : i + state.size(1)]
                z = self.forget_emb[actual_i](inds_slice)  # b n d
                state = self.forget_proj[actual_proj_i](prev_state)
                forget_gate = torch.sigmoid(torch.add(state, z, alpha=self.emb_weight / self.state_weight))

                z = self.input_emb[actual_i](inds_slice)  # b n d
                state = self.input_proj[actual_proj_i](prev_state)
                input_gate = torch.sigmoid(torch.add(state, z, alpha=self.emb_weight / self.state_weight))

                z = self.cell_emb[actual_i](inds_slice)  # b n d
                state = self.cell_proj[actual_proj_i](prev_state)
                cell_candidate = torch.add(state, z, alpha=self.emb_weight / self.state_weight)
                cell_candidate = self.activation(self.cell_ln[actual_i](cell_candidate))  # b n d
                cell_candidate = cell_candidate * input_gate

                z = self.output_emb[actual_i](inds_slice)  # b n d
                state = self.output_proj[actual_proj_i](prev_state)
                output_gate = torch.sigmoid(torch.add(state, z, alpha=self.emb_weight / self.state_weight))

                cell_state = cell_state * forget_gate
                cell_state = cell_state + cell_candidate

                state_candidate = self.activation(self.state_ln[actual_i](cell_state))
                state = state_candidate * output_gate

                # Weighted add of state_weight*state and emb_weight*z
                # Let subsequent LN take care of denominator
                # state_weight is close to 1, so shouldn't be any precision issues
                out.append(self.head[i](state))  # b n v

        else:
            assert self.method == "sum_rnn"
            for i in range(self.n_predict):
                actual_i = 0 if self.tie_weights else i
                actual_proj_i = 1 if self.tie_weights and i >= 2 else i

                inds_index = torch.arange(i, i + state.size(1), device=inds.device)
                inds_slice = inds.index_select(1, inds_index)
                z = self.emb[actual_i](inds_slice)  # b n d
                state = self.proj[actual_proj_i](state)
                # Weighted add of state_weight*state and emb_weight*z
                # Let subsequent LN take care of denominator
                # state_weight is close to 1, so shouldn't be any precision issues
                state = torch.add(state, z, alpha=self.emb_weight / self.state_weight)
                state = self.activation(self.ln[actual_i](state))  # b n d
                out.append(self.head[i](state))  # b n v

        return torch.stack(out, dim=0)  # h b n v


def create_speculator_from_config(speculator_config: dict):
    """
    Create an ArcticLSTMSpeculator instance from a configuration dictionary.
    """
    fields = [
        ("n_predict", int),
        ("input_hidden_dim", int),
        ("inner_dim", str),
        ("emb_dim", str),
        ("proj_dim", str),
        ("vocab_size", int),
        ("scale_input", bool),
        ("tie_weights", bool),
        ("tie_lstm_embs", bool),
        ("method", str),
    ]
    ConfigClass = make_dataclass("SpeculatorConfig", fields)
    config = ConfigClass(**speculator_config)
    return ArcticLSTMSpeculator(config)
