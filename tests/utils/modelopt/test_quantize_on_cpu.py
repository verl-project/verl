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
import json
import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[3]


def _load_quantize(monkeypatch):
    mtq = types.ModuleType("modelopt.torch.quantization")
    mtq.normalize_quant_cfg_list = lambda config: [config]
    config_module = types.ModuleType("modelopt.torch.quantization.config")
    config_module._default_disabled_quantizer_cfg = {}

    modules = {
        "modelopt": types.ModuleType("modelopt"),
        "modelopt.torch": types.ModuleType("modelopt.torch"),
        "modelopt.torch.quantization": mtq,
        "modelopt.torch.quantization.config": config_module,
    }
    modelopt_package = types.ModuleType("verl.utils.modelopt")
    modelopt_package.__path__ = [str(_REPO_ROOT / "verl" / "utils" / "modelopt")]
    modules["verl.utils.modelopt"] = modelopt_package
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    sys.modules.pop("verl.utils.modelopt.quantize", None)
    return importlib.import_module("verl.utils.modelopt.quantize")


def test_build_mxfp4_weight_only_config_is_dynamic_block32(monkeypatch):
    quantize = _load_quantize(monkeypatch)

    config = quantize.build_quantize_config("mxfp4")
    quant_cfg = config["quant_cfg"][0]

    assert quant_cfg["*weight_quantizer"] == {
        "num_bits": (2, 1),
        "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
        "axis": None,
        "enable": True,
    }
    assert quant_cfg["*input_quantizer"] == {"enable": False}
    assert config["algorithm"] is None


def test_mxfp4_rollout_config_matches_exporter_contract():
    with (_REPO_ROOT / "examples" / "qat" / "mxfp4_w4a16.json").open() as stream:
        config = json.load(stream)

    group = config["config_groups"]["group_0"]
    assert config["quant_method"] == "compressed-tensors"
    assert config["format"] == "mxfp4-pack-quantized"
    assert group["weights"]["group_size"] == 32
    assert group["weights"]["scale_dtype"] == "torch.uint8"
    assert group["weights"]["type"] == "float"
    assert group["input_activations"] is None


def test_qat_engine_config_can_export_without_second_fake_quant_layer():
    from verl.workers.config.engine import QATEngineConfig

    config = QATEngineConfig(enable=True, apply_modelopt_fake_quant=False, mode="mxfp4", group_size=32)

    assert config.enable is True
    assert config.apply_modelopt_fake_quant is False
