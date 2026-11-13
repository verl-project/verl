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

import pytest
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


normalize_deepseek_v4_weight_names = _load_npu_vllm_patch_module().normalize_deepseek_v4_weight_names


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("self_attn.q_a_proj.weight", "self_attn.wq_a.weight"),
        ("self_attn.kv_proj.weight", "self_attn.wkv.weight"),
        ("self_attn.kv_a_proj_with_mqa.weight", "self_attn.wkv.weight"),
        ("self_attn.q_b_proj.weight", "self_attn.wq_b.weight"),
        ("self_attn.o_a_proj.weight", "self_attn.wo_a.weight"),
        ("self_attn.o_b_proj.weight", "self_attn.wo_b.weight"),
        ("self_attn.q_a_norm.weight", "self_attn.q_norm.weight"),
        ("self_attn.sinks", "self_attn.attn_sink"),
        ("self_attn.compressor.kv_proj.weight", "self_attn.compressor.wkv.weight"),
        ("self_attn.compressor.gate_proj.weight", "self_attn.compressor.wgate.weight"),
        ("self_attn.compressor.kv_norm.weight", "self_attn.compressor.norm.weight"),
        ("self_attn.compressor.position_bias", "self_attn.compressor.ape"),
        ("self_attn.compressor.indexer.position_bias", "self_attn.indexer.compressor.ape"),
        ("self_attn.compressor.indexer.kv_proj.weight", "self_attn.indexer.compressor.wkv.weight"),
        ("self_attn.compressor.indexer.gate_proj.weight", "self_attn.indexer.compressor.wgate.weight"),
        ("self_attn.compressor.indexer.q_b_proj.weight", "self_attn.indexer.wq_b.weight"),
        ("self_attn.compressor.indexer.kv_norm.weight", "self_attn.indexer.compressor.norm.weight"),
        ("self_attn.compressor.indexer.extra.weight", "self_attn.indexer.extra.weight"),
        ("self_attn.indexer.scorer.weights_proj.weight", "self_attn.indexer.weights_proj.weight"),
        ("attn_hc.fn", "hc_attn_fn"),
        ("attn_hc.base", "hc_attn_base"),
        ("attn_hc.scale", "hc_attn_scale"),
        ("ffn_hc.fn", "hc_ffn_fn"),
        ("ffn_hc.base", "hc_ffn_base"),
        ("ffn_hc.scale", "hc_ffn_scale"),
        ("mlp.experts.0.gate_proj.weight", "mlp.experts.0.gate_proj.weight"),
    ],
)
def test_deepseek_v4_weight_name_mapping(source, target):
    model = SimpleNamespace(config=SimpleNamespace(model_type="deepseek_v4"))
    tensor = torch.ones(2)
    weights = [(f"model.layers.0.{source}", tensor)]

    normalized = list(normalize_deepseek_v4_weight_names(model, weights))

    assert normalized[0][0] == f"model.layers.0.{target}"
    assert normalized[0][1] is tensor
    assert weights[0][0] == f"model.layers.0.{source}"
    assert list(normalize_deepseek_v4_weight_names(model, normalized))[0][0] == normalized[0][0]


@pytest.mark.parametrize("prefix", ["", "model."])
@pytest.mark.parametrize("suffix", ["fn", "base", "scale"])
def test_deepseek_v4_hc_head_weight_name_mapping(prefix, suffix):
    model = SimpleNamespace(config=SimpleNamespace(model_type="deepseek_v4"))
    tensor = torch.ones(2)

    normalized = list(normalize_deepseek_v4_weight_names(model, [(f"{prefix}hc_head.hc_{suffix}", tensor)]))

    assert normalized[0][0] == f"{prefix}hc_head_{suffix}"
    assert normalized[0][1] is tensor


@pytest.mark.parametrize("model_type", [None, "deepseek_v3", "qwen3"])
def test_deepseek_v4_name_mapping_preserves_other_models(model_type):
    model = SimpleNamespace(config=SimpleNamespace(model_type=model_type))
    weights = [("model.layers.0.self_attn.q_a_proj.weight", torch.ones(2))]

    assert normalize_deepseek_v4_weight_names(model, weights) is weights
    assert normalize_deepseek_v4_weight_names(object(), weights) is weights


@pytest.mark.parametrize("weight_count", [0, 3])
def test_deepseek_v4_weight_name_mapping_consumes_weights_lazily(weight_count):
    model = SimpleNamespace(config=SimpleNamespace(model_type="deepseek_v4"))
    tensor = torch.ones(2)
    consumed = []

    def weights():
        for index in range(weight_count):
            consumed.append(index)
            yield f"model.layers.{index}.self_attn.q_a_proj.weight", tensor

    normalized = iter(normalize_deepseek_v4_weight_names(model, weights()))
    assert consumed == []
    for index in range(weight_count):
        name, normalized_tensor = next(normalized)
        assert consumed == list(range(index + 1))
        assert name == f"model.layers.{index}.self_attn.wq_a.weight"
        assert normalized_tensor is tensor
    with pytest.raises(StopIteration):
        next(normalized)


def test_deepseek_v4_load_weights_patch_is_idempotent_and_accepts_generators():
    patch_module = _load_npu_vllm_patch_module()
    observed = []

    class Model:
        config = SimpleNamespace(model_type="deepseek_v4")

        def load_weights(self, weights, **kwargs):
            observed.append((list(weights), kwargs))
            return {"loaded"}

    class OtherModel:
        def load_weights(self, weights):
            return weights

    other_loader = OtherModel.load_weights
    assert patch_module._patch_deepseek_v4_weight_loader(Model)
    assert not patch_module._patch_deepseek_v4_weight_loader(Model)
    tensor = torch.ones(2)
    weights = iter([("model.layers.0.self_attn.q_a_proj.weight", tensor)])
    assert Model().load_weights(weights, extra=True) == {"loaded"}
    assert len(observed) == 1
    assert observed[0][0][0][0] == "model.layers.0.self_attn.wq_a.weight"
    assert observed[0][0][0][1] is tensor
    assert observed[0][1] == {"extra": True}
    assert OtherModel.load_weights is other_loader


def test_deepseek_v4_load_weights_patch_preserves_lazy_consumption():
    patch_module = _load_npu_vllm_patch_module()
    consumed = []
    tensor = torch.ones(2)

    def weights():
        for index in range(3):
            consumed.append(index)
            yield f"model.layers.{index}.self_attn.q_a_proj.weight", tensor

    class Model:
        config = SimpleNamespace(model_type="deepseek_v4")

        def load_weights(self, weights):
            assert consumed == []
            name, loaded_tensor = next(iter(weights))
            assert consumed == [0]
            assert name == "model.layers.0.self_attn.wq_a.weight"
            assert loaded_tensor is tensor
            return {name}

    assert patch_module._patch_deepseek_v4_weight_loader(Model)
    assert Model().load_weights(weights()) == {"model.layers.0.self_attn.wq_a.weight"}
    assert consumed == [0]


def test_apply_npu_vllm_patches_installs_deepseek_v4_patch(monkeypatch):
    patch_module = _load_npu_vllm_patch_module()
    monkeypatch.setattr(patch_module, "is_torch_npu_available", lambda check_device=False: True)
    patches = {}
    for name in (
        "patch_vllm_ascend_v023_sched_yield",
        "patch_vllm_ascend_v023_fia_decode_mask",
        "patch_vllm_ascend_deepseek_v4_weight_loader",
        "patch_vllm013_rotary_emb",
    ):
        patches[name] = Mock()
        monkeypatch.setattr(patch_module, name, patches[name])
    monkeypatch.setattr(patch_module, "_glm52_patch_enabled", lambda: False)
    vllm = ModuleType("vllm")
    vllm.__version__ = "0.23.0"
    layers = ModuleType("vllm.model_executor.layers")
    layers.fused_moe = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.model_executor", ModuleType("vllm.model_executor"))
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers", layers)

    patch_module.apply_npu_vllm_patches()

    for installed_patch in patches.values():
        installed_patch.assert_called_once_with()
