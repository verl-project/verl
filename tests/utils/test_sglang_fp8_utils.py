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
from types import SimpleNamespace

from verl.utils.sglang.sglang_fp8_utils import SGLangFP8QuantizerHelper, build_sglang_fp8_quant_config


class MappingLikeConfig:
    def __init__(self, values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


def test_build_sglang_fp8_quant_config_preserves_defaults(monkeypatch):
    monkeypatch.delenv("SGLANG_FP8_IGNORED_LAYERS", raising=False)

    quant_config = build_sglang_fp8_quant_config()

    assert quant_config == {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    }


def test_sglang_fp8_quant_config_merges_hf_ignored_layers(monkeypatch):
    monkeypatch.delenv("SGLANG_FP8_IGNORED_LAYERS", raising=False)
    hf_config = SimpleNamespace(
        quantization_config={
            "ignored_layers": ["model.layers.0.self_attn.q_proj"],
            "modules_to_not_convert": ["model.layers.1.mlp.down_proj"],
        }
    )

    quant_config = build_sglang_fp8_quant_config(hf_config)
    helper = SGLangFP8QuantizerHelper(quant_config)

    assert quant_config["ignored_layers"] == [
        "model.layers.0.self_attn.q_proj",
        "model.layers.1.mlp.down_proj",
    ]
    assert not helper.should_quantize_param("model.layers.0.self_attn.q_proj.weight")
    assert not helper.should_quantize_param("model.layers.1.mlp.down_proj.weight")
    assert helper.should_quantize_param("model.layers.2.mlp.down_proj.weight")


def test_sglang_fp8_quant_config_accepts_mapping_like_config(monkeypatch):
    monkeypatch.delenv("SGLANG_FP8_IGNORED_LAYERS", raising=False)
    hf_config = MappingLikeConfig(
        {
            "quantization_config": MappingLikeConfig(
                {
                    "ignored_layers": ["model.layers.0.linear_attn"],
                }
            )
        }
    )

    quant_config = build_sglang_fp8_quant_config(hf_config)
    helper = SGLangFP8QuantizerHelper(quant_config)

    assert quant_config["ignored_layers"] == ["model.layers.0.linear_attn"]
    assert not helper.should_quantize_param("model.layers.0.linear_attn.in_proj_ba.weight")


def test_sglang_fp8_quantizer_matches_regex_ignored_layers(monkeypatch):
    monkeypatch.delenv("SGLANG_FP8_IGNORED_LAYERS", raising=False)
    hf_config = SimpleNamespace(
        quantization_config={
            "ignored_layers": ["re:.*linear_attn.*"],
        }
    )

    quant_config = build_sglang_fp8_quant_config(hf_config)
    helper = SGLangFP8QuantizerHelper(quant_config)

    assert quant_config["ignored_layers"] == ["re:.*linear_attn.*"]
    assert not helper.should_quantize_param("model.layers.0.linear_attn.in_proj_ba.weight")
    assert not helper.should_quantize_param("model.layers.0.linear_attn.g_proj.weight")
    assert helper.should_quantize_param("model.layers.0.mlp.experts.0.up_proj.weight")


def test_sglang_fp8_quantizer_reads_sglang_env_ignored_layers(monkeypatch):
    monkeypatch.setenv("SGLANG_FP8_IGNORED_LAYERS", "linear_attn")

    quant_config = build_sglang_fp8_quant_config()
    helper = SGLangFP8QuantizerHelper(quant_config)

    assert quant_config["ignored_layers"] == ["linear_attn"]
    assert not helper.should_quantize_param("model.layers.0.linear_attn.in_proj_ba.weight")
    assert not helper.should_quantize_param("model.layers.0.linear_attn.g_proj.weight")
    assert helper.should_quantize_param("model.layers.0.mlp.experts.0.up_proj.weight")
