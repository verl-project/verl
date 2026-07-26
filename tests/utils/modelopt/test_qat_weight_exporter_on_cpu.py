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

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _install_exporter_dependencies(monkeypatch):
    quant_utils = types.ModuleType("modelopt.torch.export.quant_utils")
    quant_utils.QUANTIZATION_NONE = "none"
    quant_utils.QUANTIZATION_NVFP4 = "nvfp4"
    quant_utils.QUANTIZATION_MXFP4 = "mxfp4"
    quant_utils.get_quantization_format = lambda module: "none"
    quant_utils.get_weight_block_size = lambda module: 0
    quant_utils.to_quantized_weight = lambda *args, **kwargs: None

    class FakeNVFP4QTensor:
        @staticmethod
        def get_weights_scaling_factor(weight, block_size, weights_scaling_factor_2):
            return (torch.ones(weight.shape[-1] // block_size),)

    class FakeMXFP4QTensor:
        @classmethod
        def quantize(cls, weight, block_size):
            scale = torch.arange(weight.numel() // block_size, dtype=torch.uint8).reshape(-1, 1)
            return SimpleNamespace(_quantized_data=torch.empty(0, dtype=torch.uint8)), scale

    modules = {
        "modelopt": types.ModuleType("modelopt"),
        "modelopt.torch": types.ModuleType("modelopt.torch"),
        "modelopt.torch.export": types.ModuleType("modelopt.torch.export"),
        "modelopt.torch.export.quant_utils": quant_utils,
        "modelopt.torch.quantization": types.ModuleType("modelopt.torch.quantization"),
        "modelopt.torch.quantization.qtensor": types.ModuleType("modelopt.torch.quantization.qtensor"),
        "modelopt.torch.quantization.qtensor.nvfp4_tensor": types.ModuleType(
            "modelopt.torch.quantization.qtensor.nvfp4_tensor"
        ),
        "modelopt.torch.quantization.qtensor.mxfp4_tensor": types.ModuleType(
            "modelopt.torch.quantization.qtensor.mxfp4_tensor"
        ),
    }
    modules["modelopt.torch.quantization.qtensor.nvfp4_tensor"].NVFP4QTensor = FakeNVFP4QTensor
    modules["modelopt.torch.quantization.qtensor.mxfp4_tensor"].MXFP4QTensor = FakeMXFP4QTensor

    megatron_utils = types.ModuleType("verl.utils.megatron_utils")
    megatron_utils.unwrap_model = lambda model: model
    modules["verl.utils.megatron_utils"] = megatron_utils
    modelopt_package = types.ModuleType("verl.utils.modelopt")
    modelopt_package.__path__ = [str(Path(__file__).parents[3] / "verl" / "utils" / "modelopt")]
    modules["verl.utils.modelopt"] = modelopt_package

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    sys.modules.pop("verl.utils.modelopt.qat_weight_exporter", None)
    return importlib.import_module("verl.utils.modelopt.qat_weight_exporter")


def test_mxfp4_export_packs_last_dimension_and_reshapes_block_scales(monkeypatch):
    exporter_module = _install_exporter_dependencies(monkeypatch)
    weight = torch.arange(2 * 96, dtype=torch.float32).reshape(2, 96)
    expected_packed = torch.arange(2 * 48, dtype=torch.uint8).reshape(2, 48)
    calls = []

    def fake_to_quantized_weight(weight_arg, scale, qformat, scale_2=None, block_size=None):
        calls.append((weight_arg, scale.clone(), qformat, scale_2, block_size))
        return expected_packed

    monkeypatch.setattr(exporter_module, "to_quantized_weight", fake_to_quantized_weight)
    exporter = object.__new__(exporter_module.QATWeightExporter)
    meta = exporter_module._QuantMeta(qformat="mxfp4", block_size=32, weight_amax=None)

    result = list(exporter._quantize_mxfp4("model.layers.0.mlp.experts.0.gate_proj.weight", weight, meta))

    assert [name for name, _ in result] == [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight_scale",
    ]
    assert torch.equal(result[0][1], expected_packed)
    assert result[1][1].shape == (2, 3)
    assert result[1][1].dtype == torch.uint8
    assert calls[0][2:] == ("mxfp4", None, 32)
    assert torch.equal(calls[0][1], result[1][1])


def test_mxfp4_export_rejects_non_ocp_block_size(monkeypatch):
    exporter_module = _install_exporter_dependencies(monkeypatch)
    exporter = object.__new__(exporter_module.QATWeightExporter)
    meta = exporter_module._QuantMeta(qformat="mxfp4", block_size=16, weight_amax=None)

    with pytest.raises(ValueError, match="block size 32"):
        list(exporter._quantize_mxfp4("model.layers.0.mlp.experts.0.down_proj.weight", torch.ones(2, 32), meta))


def test_process_weights_iterator_dispatches_mxfp4(monkeypatch):
    exporter_module = _install_exporter_dependencies(monkeypatch)
    exporter = object.__new__(exporter_module.QATWeightExporter)
    meta = exporter_module._QuantMeta(qformat="mxfp4", block_size=32, weight_amax=None)
    monkeypatch.setattr(exporter, "_resolve_quant_metadata", lambda name: meta)
    monkeypatch.setattr(
        exporter,
        "_quantize_mxfp4",
        lambda name, weight, metadata: iter([(name + ".mxfp4", weight)]),
        raising=False,
    )

    result = list(
        exporter.process_weights_iterator(iter([("model.layers.0.mlp.experts.0.up_proj.weight", torch.ones(1))]))
    )

    assert result[0][0].endswith(".mxfp4")


def test_export_only_mode_synthesizes_mxfp4_metadata_and_honors_ignores(monkeypatch):
    exporter_module = _install_exporter_dependencies(monkeypatch)
    exporter = object.__new__(exporter_module.QATWeightExporter)
    exporter._metadata = {}
    exporter._registry = SimpleNamespace(_reverse_patterns=[])
    exporter._use_modelopt_fake_quant = False
    exporter.qat_mode = "mxfp4"
    exporter._block_size = 32
    exporter._ignore_patterns = ["lm_head", "embed_tokens", "re:.*mlp\\.gate$"]

    meta = exporter._resolve_quant_metadata("model.layers.0.mlp.experts.0.gate_proj.weight")

    assert meta.qformat == "mxfp4"
    assert meta.block_size == 32
    assert exporter._resolve_quant_metadata("model.layers.0.mlp.gate.weight") is None
    assert exporter._resolve_quant_metadata("model.embed_tokens.weight") is None


def test_nvfp4_export_only_mode_computes_current_weight_amax(monkeypatch):
    exporter_module = _install_exporter_dependencies(monkeypatch)
    calls = []

    def fake_to_quantized_weight(weight, scale, qformat, scale_2=None, block_size=None):
        calls.append((scale_2.clone(), qformat, block_size))
        return torch.zeros(weight.shape[0], weight.shape[1] // 2, dtype=torch.uint8)

    monkeypatch.setattr(exporter_module, "to_quantized_weight", fake_to_quantized_weight)
    exporter = object.__new__(exporter_module.QATWeightExporter)
    weight = torch.tensor([[1.0, -7.0] * 8])
    meta = exporter_module._QuantMeta(qformat="nvfp4", block_size=16, weight_amax=None)

    list(exporter._quantize_nvfp4("model.layers.0.self_attn.q_proj.weight", weight, meta))

    assert calls[0][0].item() == pytest.approx(7.0 / (6.0 * 448.0))
    assert calls[0][1:] == ("nvfp4", 16)
