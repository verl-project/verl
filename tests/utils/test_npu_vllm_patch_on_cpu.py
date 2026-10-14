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
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import torch


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


def test_legacy_fused_moe_weight_loader_transposes_non_square_w13_once():
    npu_vllm_patch = _load_npu_vllm_patch_module()
    observed_shapes = []

    def weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id, return_success=False):
        observed_shapes.append(tuple(param.shape))

    wrapped_loader = npu_vllm_patch.vllm_v013_weight_loader_method_wrapper(weight_loader)
    fused_moe = SimpleNamespace(hidden_size=4)
    param = torch.nn.Parameter(torch.empty(2, 4, 6), requires_grad=False)

    wrapped_loader(fused_moe, param, None, "w13", "w1", 0)
    wrapped_loader(fused_moe, param, None, "w13", "w3", 0)

    assert observed_shapes == [(2, 6, 4), (2, 6, 4)]


def test_legacy_fused_moe_weight_loader_does_not_infer_square_w13_layout():
    npu_vllm_patch = _load_npu_vllm_patch_module()

    def weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id, return_success=False):
        return None

    wrapped_loader = npu_vllm_patch.vllm_v013_weight_loader_method_wrapper(weight_loader)
    fused_moe = SimpleNamespace(hidden_size=4)
    param = torch.nn.Parameter(torch.empty(2, 4, 4), requires_grad=False)
    original_stride = param.stride()

    wrapped_loader(fused_moe, param, None, "w13", "w1", 0)
    wrapped_loader(fused_moe, param, None, "w13", "w3", 0)

    assert param.stride() == original_stride


def test_patch_legacy_fused_moe_weight_loader_installs_guarded_wrapper():
    npu_vllm_patch = _load_npu_vllm_patch_module()

    class LegacyFusedMoE:
        hidden_size = 4

        def weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id, return_success=False):
            return tuple(param.shape)

    assert npu_vllm_patch._patch_legacy_fused_moe_weight_loader(LegacyFusedMoE)
    assert LegacyFusedMoE.weight_loader._verl_npu_weight_loader_patched

    fused_moe = LegacyFusedMoE()
    square_param = torch.nn.Parameter(torch.empty(2, 4, 4), requires_grad=False)
    assert fused_moe.weight_loader(square_param, None, "w13", "w1", 0) == (2, 4, 4)
