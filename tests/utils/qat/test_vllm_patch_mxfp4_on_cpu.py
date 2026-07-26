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

import torch

from verl.utils.qat import vllm_patch


class _FakeMxfp4Method:
    def get_fused_moe_quant_config(self, layer):
        return None


class _FakeMxfp4Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter("w13_weight_packed", self._param((2, 8, 16)))
        self.register_parameter("w2_weight_packed", self._param((2, 4, 16)))
        self.register_parameter("w13_weight_scale", self._param((2, 8, 1)))
        self.register_parameter("w2_weight_scale", self._param((2, 4, 1)))

    @staticmethod
    def _param(shape):
        param = torch.nn.Parameter(torch.zeros(shape, dtype=torch.uint8), requires_grad=False)
        param.weight_loader = lambda *args, **kwargs: None
        return param


class _FakeMxfp4DenseLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter("weight_packed", _FakeMxfp4Layer._param((8, 16)))
        self.register_parameter("weight_scale", _FakeMxfp4Layer._param((8, 1)))


def _fake_original_process(self, layer):
    layer.w13_weight = torch.nn.Parameter(layer.w13_weight_packed.detach().clone(), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(layer.w2_weight_packed.detach().clone(), requires_grad=False)
    delattr(layer, "w13_weight_packed")
    delattr(layer, "w2_weight_packed")


def _fake_original_dense_process(self, layer):
    layer.weight = torch.nn.Parameter(layer.weight_packed.detach().clone(), requires_grad=False)
    delattr(layer, "weight_packed")


def _fake_original_current_nvfp4_dense_process(self, layer):
    layer.weight = torch.nn.Parameter(layer.weight_packed.detach().clone(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(layer.weight_scale.detach().clone() + 1, requires_grad=False)
    delattr(layer, "weight_packed")


def _fake_original_current_nvfp4_moe_process(self, layer):
    _fake_original_process(self, layer)
    layer.w13_weight_scale = torch.nn.Parameter(layer.w13_weight_scale.detach().clone() + 1, requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(layer.w2_weight_scale.detach().clone() + 1, requires_grad=False)


def test_current_nvfp4_dense_patch_preserves_compute_addresses(monkeypatch):
    monkeypatch.setitem(
        vllm_patch._original_current_nvfp4_dense_process_weights_after_loading,
        _FakeMxfp4Method,
        _fake_original_current_nvfp4_dense_process,
    )
    layer = _FakeMxfp4DenseLayer()
    method = _FakeMxfp4Method()

    vllm_patch.patched_current_nvfp4_dense_process_weights_after_loading(method, layer)
    original_ptrs = {name: getattr(layer, name).data_ptr() for name in ("weight", "weight_scale")}

    model = torch.nn.Module()
    model.add_module("linear", layer)
    vllm_patch.prepare_qat_for_load_weights(model, device=torch.device("cpu"))
    layer.weight_packed.fill_(5)
    layer.weight_scale.fill_(7)
    vllm_patch.patched_current_nvfp4_dense_process_weights_after_loading(method, layer)

    assert torch.all(layer.weight == 5)
    assert torch.all(layer.weight_scale == 8)
    assert {name: getattr(layer, name).data_ptr() for name in ("weight", "weight_scale")} == original_ptrs


def test_current_nvfp4_moe_patch_preserves_compute_addresses(monkeypatch):
    monkeypatch.setattr(
        vllm_patch,
        "_original_current_nvfp4_moe_process_weights_after_loading",
        _fake_original_current_nvfp4_moe_process,
    )
    layer = _FakeMxfp4Layer()
    method = _FakeMxfp4Method()

    vllm_patch.patched_current_nvfp4_moe_process_weights_after_loading(method, layer)
    original_ptrs = {
        name: getattr(layer, name).data_ptr()
        for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")
    }

    model = torch.nn.Module()
    model.add_module("experts", layer)
    vllm_patch.prepare_qat_for_load_weights(model, device=torch.device("cpu"))
    layer.w13_weight_packed.fill_(3)
    layer.w2_weight_packed.fill_(4)
    layer.w13_weight_scale.fill_(5)
    layer.w2_weight_scale.fill_(6)
    vllm_patch.patched_current_nvfp4_moe_process_weights_after_loading(method, layer)

    assert torch.all(layer.w13_weight == 3)
    assert torch.all(layer.w2_weight == 4)
    assert torch.all(layer.w13_weight_scale == 6)
    assert torch.all(layer.w2_weight_scale == 7)
    assert {
        name: getattr(layer, name).data_ptr()
        for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")
    } == original_ptrs


def test_mxfp4_dense_patch_rebuilds_hf_params_and_preserves_compute_addresses(monkeypatch):
    monkeypatch.setattr(vllm_patch, "_original_mxfp4_dense_process_weights_after_loading", _fake_original_dense_process)
    layer = _FakeMxfp4DenseLayer()

    vllm_patch.patched_mxfp4_dense_process_weights_after_loading(object(), layer)
    original_ptrs = {name: getattr(layer, name).data_ptr() for name in ("weight", "weight_scale")}

    model = torch.nn.Module()
    model.add_module("linear", layer)
    vllm_patch.prepare_qat_for_load_weights(model, device=torch.device("cpu"))
    layer.weight_packed.fill_(5)
    layer.weight_scale.fill_(7)

    vllm_patch.patched_mxfp4_dense_process_weights_after_loading(object(), layer)

    assert torch.all(layer.weight == 5)
    assert torch.all(layer.weight_scale == 7)
    assert {name: getattr(layer, name).data_ptr() for name in ("weight", "weight_scale")} == original_ptrs


def test_mxfp4_patch_rebuilds_hf_params_and_preserves_compute_addresses(monkeypatch):
    monkeypatch.setattr(vllm_patch, "_original_mxfp4_moe_process_weights_after_loading", _fake_original_process)
    layer = _FakeMxfp4Layer()
    method = _FakeMxfp4Method()

    vllm_patch.patched_mxfp4_moe_process_weights_after_loading(method, layer)
    original_ptrs = {
        name: getattr(layer, name).data_ptr()
        for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")
    }

    model = torch.nn.Module()
    model.add_module("experts", layer)
    vllm_patch.prepare_qat_for_load_weights(model, device=torch.device("cpu"))
    layer.w13_weight_packed.fill_(7)
    layer.w2_weight_packed.fill_(9)
    layer.w13_weight_scale.fill_(11)
    layer.w2_weight_scale.fill_(13)

    vllm_patch.patched_mxfp4_moe_process_weights_after_loading(method, layer)

    assert torch.all(layer.w13_weight == 7)
    assert torch.all(layer.w2_weight == 9)
    assert torch.all(layer.w13_weight_scale == 11)
    assert torch.all(layer.w2_weight_scale == 13)
    assert {
        name: getattr(layer, name).data_ptr()
        for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")
    } == original_ptrs
