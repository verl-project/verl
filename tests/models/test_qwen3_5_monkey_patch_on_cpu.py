# Copyright contributors to the PEAR project
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

import sys
import types

import pytest
import torch
from torch import nn

from verl.models.transformers import qwen3_5

HIDDEN = 4
HEAD_DIM = 2  # channels of the one key head and of the one value head
CONV_WIDTH = 4


def _decoder_layer(**type_attribute):
    """A Qwen3.5 decoder layer stand-in: linear attention adds 1, full attention 2 and the MLP nothing."""
    return types.SimpleNamespace(
        **type_attribute,
        input_layernorm=nn.Identity(),
        post_attention_layernorm=nn.Identity(),
        linear_attn=lambda hidden_states, **kwargs: hidden_states + 1,
        self_attn=lambda hidden_states, **kwargs: (hidden_states + 2, None),
        mlp=torch.zeros_like,
    )


def _kernels(calls):
    """The convolution and the gated delta rule under their transformers module names, recording their calls."""

    # the signature of transformers' torch fallback, whose input is hidden_states where causal-conv1d's is x
    def causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
        calls.append("convolution")
        return hidden_states

    def torch_chunk_gated_delta_rule(query, key, value, **kwargs):
        calls.append("gated delta rule")
        return value, None

    return {
        "causal_conv1d_fn": causal_conv1d_fn,
        "causal_conv1d_update": None,
        "torch_chunk_gated_delta_rule": torch_chunk_gated_delta_rule,
        "torch_recurrent_gated_delta_rule": None,
    }


def _gated_delta_net(monkeypatch, kernels, kernels_on_layer=False):
    """A Qwen3.5 GatedDeltaNet stand-in whose kernels are its attributes (transformers 5.12) or its module's (5.16+)."""
    module = types.ModuleType("modeling_qwen3_5_stand_in")
    module.apply_mask_to_padding_states = lambda hidden_states, attention_mask: hidden_states
    monkeypatch.setitem(sys.modules, module.__name__, module)
    layer = type("GatedDeltaNet", (), {"__module__": module.__name__})()
    if kernels_on_layer:
        # the attributes of transformers 5.12 lack the torch_ prefix of the module functions
        vars(layer).update({name.removeprefix("torch_"): kernel for name, kernel in kernels.items()})
    else:
        vars(module).update(kernels)

    channels = 3 * HEAD_DIM  # query, key and value
    vars(layer).update(
        in_proj_qkv=nn.Linear(HIDDEN, channels, bias=False),
        in_proj_z=nn.Linear(HIDDEN, HEAD_DIM, bias=False),
        in_proj_b=nn.Linear(HIDDEN, 1, bias=False),
        in_proj_a=nn.Linear(HIDDEN, 1, bias=False),
        conv1d=nn.Conv1d(channels, channels, CONV_WIDTH, groups=channels, padding=CONV_WIDTH - 1, bias=False),
        conv_kernel_size=CONV_WIDTH,
        activation="silu",
        layer_idx=0,
        key_dim=HEAD_DIM,
        value_dim=HEAD_DIM,
        head_k_dim=HEAD_DIM,
        head_v_dim=HEAD_DIM,
        num_k_heads=1,
        num_v_heads=1,
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(1),
        norm=lambda core_attn_out, z: core_attn_out,
        out_proj=nn.Identity(),
    )
    return layer


@pytest.mark.parametrize("type_name", ["block_type", "layer_type"])  # transformers 5.16 and later, 5.12
@pytest.mark.parametrize(("layer_type", "mixer_output"), [("linear_attention", 1.0), ("full_attention", 2.0)])
def test_the_decoder_layer_runs_the_mixer_its_type_names(type_name, layer_type, mixer_output):
    layer = _decoder_layer(**{type_name: layer_type})

    output = qwen3_5.qwen3_5_decoder_layer_forward(layer, torch.zeros(1, 3, HIDDEN), position_embeddings=None)

    assert torch.equal(output, torch.full((1, 3, HIDDEN), mixer_output))


@pytest.mark.parametrize("kernels_on_layer", [True, False])  # transformers 5.12, 5.16 and later
def test_the_gated_delta_net_runs_the_kernels_where_its_transformers_keeps_them(monkeypatch, kernels_on_layer):
    calls = []
    monkeypatch.setattr(qwen3_5, "is_causal_conv1d_available", lambda: True)  # the causal-conv1d package installed
    layer = _gated_delta_net(monkeypatch, _kernels(calls), kernels_on_layer)

    output = qwen3_5.qwen3_5_gated_delta_net_forward(layer, torch.randn(1, 3, HIDDEN))

    assert output.shape == (1, 3, HEAD_DIM)
    assert calls == ["convolution", "gated delta rule"]


def test_packed_rows_take_the_fla_convolution_without_the_causal_conv1d_package(monkeypatch):
    calls = []

    def packed_causal_conv1d(layer, mixed_qkv, cu_seqlens, cu_seqlens_cpu):
        calls.append("FLA convolution")
        return mixed_qkv

    monkeypatch.setattr(qwen3_5, "is_causal_conv1d_available", lambda: False)
    monkeypatch.setattr(qwen3_5, "_packed_causal_conv1d", packed_causal_conv1d)
    layer = _gated_delta_net(monkeypatch, _kernels(calls))

    qwen3_5.qwen3_5_gated_delta_net_forward(layer, torch.randn(1, 3, HIDDEN), cu_seqlens=torch.tensor([0, 1, 3]))

    assert calls == ["FLA convolution", "gated delta rule"]
