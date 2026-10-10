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
"""MXFP8 rollout quantization helpers (config builder, layer selection, TE padding glue)."""

import asyncio
import math
import sys
import types

import pytest
import torch

from verl.utils.sglang.sglang_mxfp8_utils import (
    MXFP8_GROUP_SIZE,
    SGLangMXFP8QuantizerHelper,
    build_sglang_mxfp8_quant_config,
    mxfp8_quantize,
)


def test_build_sglang_mxfp8_quant_config():
    config = build_sglang_mxfp8_quant_config()
    assert config["quant_method"] == "mxfp8"
    assert config["weight_block_size"] == [1, 32]
    assert config["activation_scheme"] == "dynamic"


def test_build_sglang_mxfp8_quant_config_merges_ignored_layers():
    class FakeHFConfig:
        quantization_config = {"ignored_layers": ["re:model\\.layers\\.(0|35)\\..*"]}

    config = build_sglang_mxfp8_quant_config(FakeHFConfig(), ignored_layers=["linear_attn"])
    assert "re:model\\.layers\\.(0|35)\\..*" in config["ignored_layers"]
    assert "linear_attn" in config["ignored_layers"]


def test_should_quantize_param_respects_ignored_layers():
    helper = SGLangMXFP8QuantizerHelper({"ignored_layers": ["re:model\\.layers\\.(0|35)\\..*"]})
    # bf16 head/tail layers excluded via regex (pairs with first_last_layers_bf16 in training)
    assert not helper.should_quantize_param("model.layers.0.self_attn.q_proj.weight")
    assert not helper.should_quantize_param("model.layers.35.mlp.down_proj.weight")
    # middle layers quantized
    assert helper.should_quantize_param("model.layers.7.self_attn.q_proj.weight")
    assert helper.should_quantize_param("model.layers.7.mlp.gate_proj.weight")
    # structurally excluded regardless of layer index
    assert not helper.should_quantize_param("model.layers.7.input_layernorm.weight")
    assert not helper.should_quantize_param("model.layers.7.mlp.gate.weight")
    assert not helper.should_quantize_param("model.embed_tokens.weight")
    assert not helper.should_quantize_param("lm_head.weight")
    assert not helper.should_quantize_param("model.layers.7.self_attn.q_proj.bias")


def _install_fake_te(monkeypatch, recorded):
    """Stub transformer_engine with a quantizer that mimics TE's padded MXFP8 output."""

    class FakeQuantized:
        def __init__(self, data):
            rows, k = data.shape
            recorded["quantized_shape"] = (rows, k)
            self._rowwise_data = data.to(torch.uint8)
            scale_rows = math.ceil(rows / 128) * 128
            scale_cols = math.ceil((k // MXFP8_GROUP_SIZE) / 4) * 4
            self._rowwise_scale_inv = (
                torch.arange(scale_rows * scale_cols, dtype=torch.float32)
                .reshape(scale_rows, scale_cols)
                .to(torch.uint8)
            )

    class FakeMXFP8Quantizer:
        def __init__(self, fp8_dtype, rowwise, columnwise):
            recorded["rowwise"] = rowwise
            recorded["columnwise"] = columnwise

        def quantize(self, tensor):
            return FakeQuantized(tensor)

    te = types.ModuleType("transformer_engine")
    te_pytorch = types.ModuleType("transformer_engine.pytorch")
    te_constants = types.ModuleType("transformer_engine.pytorch.constants")
    te_pytorch.MXFP8Quantizer = FakeMXFP8Quantizer
    te_constants.TE_DType = {torch.float8_e4m3fn: "kFloat8E4M3"}
    te.pytorch = te_pytorch
    te_pytorch.constants = te_constants
    monkeypatch.setitem(sys.modules, "transformer_engine", te)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", te_pytorch)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.constants", te_constants)


def test_mxfp8_quantize_pads_rows_and_slices_back(monkeypatch):
    recorded = {}
    _install_fake_te(monkeypatch, recorded)

    # 17 rows: not 32-aligned, must be zero-padded to 32 before quantization
    weight = torch.randn(17, 64, dtype=torch.bfloat16)
    qweight, scale = mxfp8_quantize(weight)

    assert recorded["quantized_shape"] == (32, 64)
    assert recorded["rowwise"] is True and recorded["columnwise"] is False
    assert qweight.shape == (17, 64)
    assert qweight.dtype == torch.float8_e4m3fn
    assert scale.shape == (17, 64 // MXFP8_GROUP_SIZE)
    assert scale.dtype == torch.uint8


def test_mxfp8_quantize_rejects_unaligned_k(monkeypatch):
    recorded = {}
    _install_fake_te(monkeypatch, recorded)
    with pytest.raises(ValueError, match="divisible by 32"):
        mxfp8_quantize(torch.randn(4, 48, dtype=torch.bfloat16))


def test_quant_weights_by_name_rejects_unaligned_layer_with_remedy(monkeypatch):
    recorded = {}
    _install_fake_te(monkeypatch, recorded)

    helper = SGLangMXFP8QuantizerHelper({})
    # 3420 mirrors Qwen2.5-VL's vision intermediate_size (3420 % 32 == 28)
    weights = [("visual.blocks.0.mlp.down_proj.weight", torch.randn(16, 3420, dtype=torch.bfloat16))]

    async def _collect():
        return [(k, v) async for k, v in helper.quant_weights_by_name(iter(weights))]

    with pytest.raises(ValueError, match="ignored_layers"):
        asyncio.run(_collect())


def test_quant_weights_by_name_yields_weight_and_scale(monkeypatch):
    recorded = {}
    _install_fake_te(monkeypatch, recorded)

    helper = SGLangMXFP8QuantizerHelper({})
    weights = [
        ("model.layers.1.self_attn.q_proj.weight", torch.randn(32, 64, dtype=torch.bfloat16)),
        ("model.layers.1.input_layernorm.weight", torch.randn(64, dtype=torch.bfloat16)),
    ]

    async def _collect():
        return [(k, v) async for k, v in helper.quant_weights_by_name(iter(weights))]

    out = asyncio.run(_collect())

    names = [k for k, _ in out]
    assert names == [
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.1.self_attn.q_proj.weight_scale_inv",
        "model.layers.1.input_layernorm.weight",
    ]
    assert out[0][1].dtype == torch.float8_e4m3fn
    assert out[1][1].dtype == torch.uint8
    assert out[1][1].shape == (32, 2)
    assert out[2][1].dtype == torch.bfloat16
