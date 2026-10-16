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

"""verl's patched Qwen3.5 decoder-layer forward must work across transformers versions.

transformers renamed ``Qwen3_5DecoderLayer.layer_type`` to ``block_type`` in 5.13.0. The patched forward
used to read ``self.layer_type`` and raised
``AttributeError: 'Qwen3_5DecoderLayer' object has no attribute 'layer_type'`` on newer versions.
"""

import pytest
import torch

pytest.importorskip("transformers.models.qwen3_5")

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

from verl.models.transformers.qwen3_5 import qwen3_5_decoder_layer_forward


@pytest.mark.parametrize("attr_name", ["layer_type", "block_type"])
def test_full_attention_layer_forward(attr_name):
    config = Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        layer_types=["linear_attention", "full_attention"],
    )
    config._attn_implementation = "sdpa"
    layer = Qwen3_5DecoderLayer(config, layer_idx=1)

    # Force the layer to expose only `attr_name`, mirroring either side of the
    # transformers 5.13 rename regardless of which attribute the installed
    # transformers actually uses, so both fallback branches get exercised.
    value = getattr(layer, "layer_type", None) or layer.block_type
    for name in ("layer_type", "block_type"):
        if hasattr(layer, name):
            delattr(layer, name)
    setattr(layer, attr_name, value)

    x = torch.randn(1, 8, 64)
    position_ids = torch.arange(8)[None]
    position_embeddings = Qwen3_5TextRotaryEmbedding(config)(x, position_ids[None].expand(3, 1, 8))

    out = qwen3_5_decoder_layer_forward(layer, x, position_embeddings, attention_mask=None, position_ids=position_ids)
    assert out.shape == x.shape
