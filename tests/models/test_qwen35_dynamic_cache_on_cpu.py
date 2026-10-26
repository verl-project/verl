# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Regression tests for the Transformers 5 Qwen3.5 cache API."""

import pytest
import torch

pytest.importorskip("transformers.models.qwen3_5")

from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

from verl.models.transformers.qwen3_5 import (
    _cache_has_previous_state,
    qwen3_5_gated_delta_net_forward,
)


class _LinearAttentionLayerCache:
    def __init__(self, indexed: bool):
        self.conv_states = {0: None} if indexed else None
        self.recurrent_states = {0: None} if indexed else None
        self.has_previous_state = {0: False} if indexed else False


class _DynamicCacheDouble:
    """Minimal cache double matching tensor and Transformers 5.16.1 shapes."""

    def __init__(self, indexed: bool):
        self.layers = [_LinearAttentionLayerCache(indexed)]
        self.indexed = indexed

    def has_previous_state(self, layer_idx: int, state_idx: int | None = None) -> bool:
        previous_state = self.layers[layer_idx].has_previous_state
        if self.indexed:
            return all(previous_state.values()) if state_idx is None else previous_state[state_idx]
        return previous_state

    def update_conv_state(self, conv_states: torch.Tensor, layer_idx: int, state_idx: int = 0, **kwargs):
        del kwargs
        layer = self.layers[layer_idx]
        if self.indexed:
            layer.conv_states[state_idx] = conv_states
            layer.has_previous_state[state_idx] = True
        else:
            layer.conv_states = conv_states
            layer.has_previous_state = True
        return conv_states

    def update_recurrent_state(self, recurrent_states: torch.Tensor, layer_idx: int, state_idx: int = 0, **kwargs):
        del kwargs
        layer = self.layers[layer_idx]
        if self.indexed:
            layer.recurrent_states[state_idx] = recurrent_states
            layer.has_previous_state[state_idx] = True
        else:
            layer.recurrent_states = recurrent_states
            layer.has_previous_state = True
        return recurrent_states


def test_qwen35_dynamic_cache_selects_state_zero_for_indexed_state_flags():
    cache = _DynamicCacheDouble(indexed=True)
    cache.layers[0].has_previous_state = {0: True, 1: False}

    assert _cache_has_previous_state(cache, layer_idx=0)


@pytest.mark.parametrize(
    "cache_kind",
    ["transformers", "tensor-valued", "state-indexed"],
    ids=["transformers", "tensor-valued", "state-indexed"],
)
def test_qwen35_gated_delta_net_accepts_both_dynamic_cache_state_shapes(cache_kind):
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention"],
        dtype=torch.float32,
    )
    torch.manual_seed(0)
    layer = Qwen3_5GatedDeltaNet(config, layer_idx=0).eval()
    reference_layer = Qwen3_5GatedDeltaNet(config, layer_idx=0).eval()
    sequential_layer = Qwen3_5GatedDeltaNet(config, layer_idx=0).eval()
    reference_layer.load_state_dict(layer.state_dict())
    sequential_layer.load_state_dict(layer.state_dict())
    cache = (
        DynamicCache(config=config)
        if cache_kind == "transformers"
        else _DynamicCacheDouble(indexed=cache_kind == "state-indexed")
    )
    reference_cache = DynamicCache(config=config)
    sequential_cache = (
        DynamicCache(config=config)
        if cache_kind == "transformers"
        else _DynamicCacheDouble(indexed=cache_kind == "state-indexed")
    )
    hidden_states = torch.randn(1, 4, config.hidden_size)
    continuation_states = torch.randn(1, 2, config.hidden_size)
    decode_states = torch.randn(1, 1, config.hidden_size)

    output = qwen3_5_gated_delta_net_forward(layer, hidden_states, cache_params=cache)
    reference_output = reference_layer(hidden_states, cache_params=reference_cache)
    continuation_output = qwen3_5_gated_delta_net_forward(layer, continuation_states, cache_params=cache)
    decode_output = qwen3_5_gated_delta_net_forward(layer, decode_states, cache_params=cache)

    qwen3_5_gated_delta_net_forward(sequential_layer, hidden_states, cache_params=sequential_cache)
    sequential_continuation = torch.cat(
        [
            qwen3_5_gated_delta_net_forward(
                sequential_layer, continuation_states[:, :1], cache_params=sequential_cache
            ),
            qwen3_5_gated_delta_net_forward(
                sequential_layer, continuation_states[:, 1:], cache_params=sequential_cache
            ),
        ],
        dim=1,
    )
    sequential_decode = qwen3_5_gated_delta_net_forward(sequential_layer, decode_states, cache_params=sequential_cache)

    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all()
    assert continuation_output.shape == continuation_states.shape
    assert torch.isfinite(continuation_output).all()
    assert decode_output.shape == decode_states.shape
    assert torch.isfinite(decode_output).all()
    torch.testing.assert_close(output, reference_output)
    torch.testing.assert_close(continuation_output, sequential_continuation, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(decode_output, sequential_decode, rtol=1e-4, atol=1e-5)
