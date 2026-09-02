# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import asyncio

import torch

from verl.utils.sglang import sglang_mxfp8_utils as mxfp8


async def _collect(async_iterable):
    return [item async for item in async_iterable]


def test_get_mxfp8_quant_config_returns_expected_fields():
    config = mxfp8.get_mxfp8_quant_config()

    assert config["activation_scheme"] == "dynamic"
    assert config["fmt"] == "e4m3"
    assert config["quant_method"] == "mxfp8"
    assert config["weight_block_size"] == [1, 32]
    assert config["scale_fmt"] == "ue8m0"
    # Mirrors the FP8 hardcode: exactly these five fields, no module skip lists.
    assert set(config) == {"activation_scheme", "fmt", "quant_method", "weight_block_size", "scale_fmt"}


def test_should_quantize_param_matches_fp8_policy():
    # SGLangMXFP8QuantizerHelper reuses the base FP8 should_quantize_param, so the
    # selection policy here mirrors verl/utils/fp8_utils.py (name-based, no shape check).
    helper = mxfp8.SGLangMXFP8QuantizerHelper(mxfp8.get_mxfp8_quant_config())
    quantizable = torch.empty(8, 64, dtype=torch.bfloat16)

    assert helper.should_quantize_param("model.layers.0.self_attn.q_proj.weight", quantizable)
    assert helper.should_quantize_param("model.layers.0.mlp.gate_proj.weight", quantizable)
    assert not helper.should_quantize_param("model.layers.0.mlp.gate.weight", quantizable)
    assert not helper.should_quantize_param("model.layers.0.input_layernorm.weight", quantizable)
    assert not helper.should_quantize_param("model.embed_tokens.weight", quantizable)
    assert not helper.should_quantize_param("model.layers.0.self_attn.q_proj.bias", quantizable)


def test_quant_weights_by_name_emits_mxfp8_scale_tensor(monkeypatch):
    def fake_quantize(tensor_2d):
        qweight = torch.zeros(tensor_2d.shape, dtype=torch.uint8)
        scale = torch.ones((tensor_2d.shape[0], tensor_2d.shape[1] // 32), dtype=torch.uint8)
        return qweight, scale

    monkeypatch.setattr(mxfp8, "_quantize_with_sglang", fake_quantize)

    helper = mxfp8.SGLangMXFP8QuantizerHelper(mxfp8.get_mxfp8_quant_config())
    weights = [
        ("model.layers.0.self_attn.q_proj.weight", torch.randn(2, 64, dtype=torch.bfloat16)),
        ("model.layers.0.input_layernorm.weight", torch.randn(64, dtype=torch.bfloat16)),
    ]

    result = asyncio.run(_collect(helper.quant_weights_by_name(weights)))

    assert [name for name, _ in result] == [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.weight_scale_inv",
        "model.layers.0.input_layernorm.weight",
    ]
    assert result[0][1].shape == torch.Size([2, 64])
    assert result[1][1].shape == torch.Size([2, 2])
    assert result[1][1].dtype == torch.uint8
    assert result[2][1] is weights[1][1]
