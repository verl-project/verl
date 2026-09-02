# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock


def _load_npu_vllm_patch_module():
    module_path = Path(__file__).parents[2] / "verl" / "utils" / "vllm" / "npu_vllm_patch.py"
    spec = importlib.util.spec_from_file_location("test_npu_vllm_patch", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_apply_npu_vllm_patches_accepts_modular_fused_moe(monkeypatch):
    npu_vllm_patch = _load_npu_vllm_patch_module()
    monkeypatch.setattr(npu_vllm_patch, "is_torch_npu_available", lambda check_device=False: True)
    rotary_patch = Mock()
    monkeypatch.setattr(npu_vllm_patch, "patch_vllm013_rotary_emb", rotary_patch)

    vllm = ModuleType("vllm")
    model_executor = ModuleType("vllm.model_executor")
    layers = ModuleType("vllm.model_executor.layers")
    fused_moe = ModuleType("vllm.model_executor.layers.fused_moe")
    fused_moe.FusedMoEFactory = lambda *args, **kwargs: None
    layers.fused_moe = fused_moe

    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.model_executor", model_executor)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers", layers)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.fused_moe", fused_moe)

    npu_vllm_patch.apply_npu_vllm_patches()

    rotary_patch.assert_called_once_with()
