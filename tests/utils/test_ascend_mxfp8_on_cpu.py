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

import json
import sys
from types import SimpleNamespace

import pytest
import torch


def _patch_torch_npu(monkeypatch):
    calls = []

    def npu_dynamic_mx_quant(weight, axis, dst_type):
        calls.append((weight, axis, dst_type))
        groups = (weight.shape[-1] + 31) // 32
        scale = torch.ones(*weight.shape[:-1], groups, 1, dtype=torch.uint8)
        return weight.to(dst_type), scale

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu_dynamic_mx_quant=npu_dynamic_mx_quant, float8_e4m3fn=torch.float8_e4m3fn),
    )
    return calls


def test_native_ascend_dispatch_reads_modelslim_description(tmp_path):
    from verl.utils.qat.ascend import AscendMxfp8WeightQuantizer
    from verl.utils.qat.compressed_tensors import dispatch_rollout_weight_quantizer

    description = {"quant_method": "ascend", "model.layers.0.self_attn.q_proj.weight": "W8A8_MXFP8"}
    (tmp_path / "quant_model_description.json").write_text(json.dumps(description))
    model_config = SimpleNamespace(
        local_path=str(tmp_path), hf_config=SimpleNamespace(model_type="qwen3_moe", text_config=None)
    )
    rollout_config = SimpleNamespace(
        name="vllm", quantization="ascend", checkpoint_engine=SimpleNamespace(backend="naive")
    )

    quantizer = dispatch_rollout_weight_quantizer(model_config=model_config, rollout_config=rollout_config)

    assert isinstance(quantizer, AscendMxfp8WeightQuantizer)
    assert quantizer.quant_description == description


def test_ascend_quantizer_emits_mxfp8_weights_and_scales(monkeypatch):
    from verl.utils.qat.ascend import AscendMxfp8WeightQuantizer

    calls = _patch_torch_npu(monkeypatch)
    name = "model.layers.0.self_attn.q_proj.weight"
    gate = "model.layers.0.mlp.gate.weight"
    quantizer = AscendMxfp8WeightQuantizer({name: "W8A8_MXFP8", gate: "FLOAT"})
    weight = torch.randn(2, 64, dtype=torch.bfloat16)
    gate_weight = torch.randn(4, 64, dtype=torch.bfloat16)

    output = dict(quantizer([(name, weight), (gate, gate_weight)]))

    assert set(output) == {name, name + "_scale", gate}
    assert output[name].dtype == torch.float8_e4m3fn
    assert output[name + "_scale"].shape == (2, 2)
    assert output[name + "_scale"].dtype == torch.uint8
    assert output[gate] is gate_weight
    assert calls == [(weight, -1, torch.float8_e4m3fn)]


def test_ascend_quantizer_expands_packed_experts(monkeypatch):
    from verl.utils.qat.ascend import AscendMxfp8WeightQuantizer

    _patch_torch_npu(monkeypatch)
    prefix = "model.layers.0.mlp.experts.0"
    description = {
        f"{prefix}.{projection}.weight": "W8A8_MXFP8" for projection in ("gate_proj", "up_proj", "down_proj")
    }
    quantizer = AscendMxfp8WeightQuantizer(description)

    output = dict(
        quantizer(
            [
                ("model.layers.0.mlp.experts.gate_up_proj", torch.randn(1, 64, 64, dtype=torch.bfloat16)),
                ("model.layers.0.mlp.experts.down_proj", torch.randn(1, 64, 32, dtype=torch.bfloat16)),
            ]
        )
    )

    assert len(output) == 6
    assert output[f"{prefix}.gate_proj.weight_scale"].shape == (32, 2)
    assert output[f"{prefix}.up_proj.weight_scale"].shape == (32, 2)
    assert output[f"{prefix}.down_proj.weight_scale"].shape == (64, 1)


def test_ascend_quantizer_rejects_unsupported_weight_types():
    from verl.utils.qat.ascend import AscendMxfp8WeightQuantizer

    with pytest.raises(NotImplementedError, match="W8A8_DYNAMIC"):
        AscendMxfp8WeightQuantizer({"model.layers.0.self_attn.q_proj.weight": "W8A8_DYNAMIC"})


def test_native_string_quantization_needs_no_hf_override():
    from verl.workers.rollout.vllm_rollout.vllm_async_server import build_cli_args_from_config, vLLMHttpServer

    server = SimpleNamespace(config=SimpleNamespace(quantization="ascend"))

    assert vLLMHttpServer._apply_quantization(server) == {}
    engine_kwargs = {}
    vLLMHttpServer._set_native_quantization(server, engine_kwargs)
    assert engine_kwargs == {"quantization": "ascend"}
    assert build_cli_args_from_config(engine_kwargs) == ["--quantization", "ascend"]


def test_structured_quantization_does_not_set_native_engine_arg():
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    server = SimpleNamespace(config=SimpleNamespace(quantization=SimpleNamespace(enabled=True)))
    engine_kwargs = {}

    vLLMHttpServer._set_native_quantization(server, engine_kwargs)

    assert engine_kwargs == {}


def test_ascend_mx_config_is_quantized(monkeypatch):
    from verl.utils.vllm.vllm_quant_utils import is_quantized_model

    class AscendModelSlimConfig:
        def __init__(self, description):
            self.quant_description = description

    quantization_module = SimpleNamespace(AscendModelSlimConfig=AscendModelSlimConfig)
    monkeypatch.setitem(sys.modules, "vllm_ascend.quantization", quantization_module)
    config = SimpleNamespace(quant_config=AscendModelSlimConfig({"layer.weight": "W8A8_MXFP8"}))

    assert is_quantized_model(config)


def test_prepare_and_process_brackets_ascend_layout_reload():
    from verl.utils.vllm.vllm_quant_utils import (
        prepare_quanted_weights_for_loading,
        process_quanted_weights_after_loading,
    )

    calls = []

    class Scheme:
        def restore_weights_for_rl_loading(self, layer):
            calls.append("restore")
            layer._mxfp8_transformed = False

        def process(self, layer):
            calls.append("process")
            layer._mxfp8_transformed = True

    class Method:
        def __init__(self, scheme):
            self.quant_method = scheme

        def process_weights_after_loading(self, layer):
            self.quant_method.process(layer)

    scheme = Scheme()
    layer = SimpleNamespace(_mxfp8_transformed=True, _mxfp8_original_shapes={}, quant_method=Method(scheme))

    class Model:
        def named_modules(self):
            return [("layer", layer)]

        def modules(self):
            return [layer]

    model = Model()
    state = prepare_quanted_weights_for_loading(model)
    assert not layer._mxfp8_transformed

    process_quanted_weights_after_loading(model, state)

    assert layer._mxfp8_transformed
    assert calls == ["restore", "process"]
