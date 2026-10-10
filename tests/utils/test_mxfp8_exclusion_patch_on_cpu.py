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

"""CPU coverage of provenance, instance isolation and fused exclusion matching.

The lightweight vLLM doubles model the relevant v0.24 matching behavior; no GPU,
quantized weights, or optional vLLM installation is needed for this suite.
"""

import importlib.util
import pickle
import sys
from contextlib import ExitStack
from fnmatch import fnmatchcase
from pathlib import Path
from types import ModuleType

import pytest


def _load_patch():
    path = Path(__file__).parents[2] / "verl/utils/vllm/mxfp8_exclusion_patch.py"
    spec = importlib.util.spec_from_file_location("mxfp8_exclusion_patch_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _is_layer_skipped(prefix, ignored_layers, fused_mapping):
    """Model vLLM's exact-match helper, including fusion and expert-group handling."""
    leaf = prefix.rsplit(".", 1)[-1]
    if leaf in fused_mapping:
        if prefix in ignored_layers:
            return True
        parent, separator, _ = prefix.rpartition(".")
        skipped = [parent + separator + shard in ignored_layers for shard in fused_mapping[leaf]]
        if any(skipped) != all(skipped):
            raise ValueError("All shards of fused layers must have the same precision")
        return all(skipped)
    if "experts" in prefix:
        return any(prefix in entry for entry in ignored_layers if "experts" in entry)
    return prefix in ignored_layers


class _ModelOptBase:
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, excluded):
        self.exclude_modules = list(excluded)

    def is_layer_excluded(self, prefix):
        if _is_layer_skipped(prefix, self.exclude_modules, self.packed_modules_mapping):
            return True
        # The legacy ModelOpt fallback that causes the router/gate_up collision.
        return any(p in prefix or fnmatchcase(prefix, p) for p in self.exclude_modules)

    def get_quant_method(self, prefix):
        return "UnquantizedLinearMethod" if self.is_layer_excluded(prefix) else "ModelOptMxFp8LinearMethod"


class _MxFp8Config(_ModelOptBase):
    @classmethod
    def from_config(cls, config):
        # Mimic upstream normalization: unknown HF keys are not kept on the config.
        return cls(config.get("ignored_layers", config.get("exclude_modules", [])))


class _NvFp4Config(_ModelOptBase):
    pass


@pytest.fixture
def patch_module(monkeypatch):
    modelopt = ModuleType("vllm.model_executor.layers.quantization.modelopt")
    modelopt.ModelOptMxFp8Config = _MxFp8Config
    quant_utils = ModuleType("vllm.model_executor.layers.quantization.utils.quant_utils")
    quant_utils.is_layer_skipped = _is_layer_skipped
    monkeypatch.setitem(sys.modules, modelopt.__name__, modelopt)
    monkeypatch.setitem(sys.modules, quant_utils.__name__, quant_utils)
    module = _load_patch()
    with ExitStack() as stack:
        for patcher in module.build_mxfp8_exclusion_patchers():
            stack.enter_context(patcher)
        yield module


def _opt_in(patch_module, cfg):
    return patch_module._mark_generated_config(
        cfg,
        hf_quant_config={"quant_method": "mxfp8", patch_module.VERL_EXACT_MXFP8_EXCLUSIONS: True},
    )


def test_28_dense_layers_keep_gate_up_quantized_and_routers_excluded(patch_module):
    cfg = _MxFp8Config([f"model.layers.{i}.mlp.gate" for i in range(28)] + ["lm_head", "model.embed_tokens"])
    names = [f"model.layers.{i}.mlp.gate_up_proj" for i in range(28)]
    # Each merged projection represents two HF parameters: reproduce 56 before patching.
    assert sum(2 for name in names if cfg.is_layer_excluded(name)) == 56
    assert _opt_in(patch_module, cfg)
    assert sum(2 for name in names if cfg.is_layer_excluded(name)) == 0
    for i in range(28):
        base = f"model.layers.{i}"
        assert cfg.get_quant_method(f"{base}.mlp.gate") == "UnquantizedLinearMethod"
        for suffix in (
            "mlp.gate_up_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            "self_attn.qkv_proj",
            "self_attn.o_proj",
        ):
            assert cfg.get_quant_method(f"{base}.{suffix}") == "ModelOptMxFp8LinearMethod"
    assert cfg.is_layer_excluded("lm_head")
    assert cfg.is_layer_excluded("model.embed_tokens")


def test_instance_isolation_idempotence_and_checkpoint_compatibility(patch_module):
    generated = _MxFp8Config(["mlp.gate"])
    checkpoint = _MxFp8Config(["mlp.gate"])
    nvfp4 = _NvFp4Config(["mlp.gate"])
    base_method = _ModelOptBase.is_layer_excluded
    assert _opt_in(patch_module, generated)
    installed = generated.is_layer_excluded
    assert not _opt_in(patch_module, generated)
    assert generated.is_layer_excluded.__func__ is installed.__func__
    assert _ModelOptBase.is_layer_excluded is base_method
    assert _MxFp8Config.is_layer_excluded is not base_method
    assert checkpoint.is_layer_excluded("mlp.gate_up_proj")
    assert nvfp4.is_layer_excluded("mlp.gate_up_proj")
    assert _MxFp8Config(["mlp.gate"]).is_layer_excluded("mlp.gate_up_proj")
    assert not generated.is_layer_excluded("mlp.gate_up_proj")


@pytest.mark.parametrize("marker", [None, False, "true", 1])
def test_unmarked_configs_are_noops_without_importing_vllm(marker):
    module = _load_patch()
    # Passing a plain object proves no vLLM type/method inspection is needed.
    assert not module._mark_generated_config(
        object(), hf_quant_config={"quant_method": "mxfp8", module.VERL_EXACT_MXFP8_EXCLUSIONS: marker}
    )


@pytest.mark.parametrize("method", ["fp8", "modelopt", "modelopt_mxfp8", "nvfp4", None])
def test_marked_other_formats_are_rejected_without_mutation(patch_module, method):
    cfg = _MxFp8Config([])
    with pytest.raises(ValueError, match="quant_method"):
        patch_module._mark_generated_config(
            cfg, hf_quant_config={"quant_method": method, patch_module.VERL_EXACT_MXFP8_EXCLUSIONS: True}
        )
    assert "is_layer_excluded" not in vars(cfg)


def test_marked_other_config_class_is_rejected(patch_module):
    cfg = _NvFp4Config(["mlp.gate"])
    with pytest.raises(TypeError, match="ModelOptMxFp8Config"):
        _opt_in(patch_module, cfg)
    assert "is_layer_excluded" not in vars(cfg)


@pytest.mark.parametrize("excluded", [None, "mlp.gate", [1]])
def test_invalid_exclusions_fail_closed(patch_module, excluded):
    cfg = _MxFp8Config([])
    cfg.exclude_modules = excluded
    with pytest.raises(TypeError, match="exclude_modules"):
        _opt_in(patch_module, cfg)
    assert "is_layer_excluded" not in vars(cfg)


def test_missing_mapping_fails_closed(patch_module):
    cfg = _MxFp8Config([])
    cfg.packed_modules_mapping = None
    with pytest.raises(TypeError, match="packed_modules_mapping"):
        _opt_in(patch_module, cfg)


def test_does_not_overwrite_another_instance_patch(patch_module):
    cfg = _MxFp8Config([])
    matcher = lambda prefix: True
    cfg.is_layer_excluded = matcher
    with pytest.raises(RuntimeError, match="existing instance-specific"):
        _opt_in(patch_module, cfg)
    assert cfg.is_layer_excluded is matcher


@pytest.mark.parametrize(
    "fused,shards",
    [
        ("mlp.gate_up_proj", ["mlp.gate_proj", "mlp.up_proj"]),
        ("self_attn.qkv_proj", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]),
    ],
)
def test_fusion_full_partial_and_direct_exclusions(patch_module, fused, shards):
    cfg = _MxFp8Config(shards)
    _opt_in(patch_module, cfg)
    assert cfg.is_layer_excluded(fused)
    cfg.exclude_modules = shards[:1]
    with pytest.raises(ValueError, match="shards"):
        cfg.is_layer_excluded(fused)
    cfg.exclude_modules = [fused]
    assert cfg.is_layer_excluded(fused)


def test_globs_respect_fused_shards_including_mixed_exact_glob(patch_module):
    cfg = _MxFp8Config(["*.gate_proj", "*.up_proj"])
    _opt_in(patch_module, cfg)
    assert cfg.is_layer_excluded("model.layers.0.mlp.gate_up_proj")
    cfg.exclude_modules = ["model.layers.0.mlp.gate_proj", "*.up_proj"]
    assert cfg.is_layer_excluded("model.layers.0.mlp.gate_up_proj")
    cfg.exclude_modules = ["*.gate_proj"]
    with pytest.raises(ValueError, match="shards"):
        cfg.is_layer_excluded("model.layers.0.mlp.gate_up_proj")
    cfg.exclude_modules = ["*.mlp.*"]
    assert cfg.is_layer_excluded("model.layers.0.mlp.gate_up_proj")


def test_no_implicit_parent_or_suffix_matching(patch_module):
    cfg = _MxFp8Config(["model.layers.0.mlp", "lm_head"])
    _opt_in(patch_module, cfg)
    assert not cfg.is_layer_excluded("model.layers.0.mlp.down_proj")
    assert not cfg.is_layer_excluded("another.lm_head")
    cfg.exclude_modules = ["model.layers.0.mlp.*"]
    assert cfg.is_layer_excluded("model.layers.0.mlp.down_proj")


def test_explicit_router_names_and_expert_group_behavior(patch_module):
    cfg = _MxFp8Config(["model.layers.0.block_sparse_moe.gate", "model.layers.0.mlp.experts.0.gate_proj"])
    _opt_in(patch_module, cfg)
    assert cfg.is_layer_excluded("model.layers.0.block_sparse_moe.gate")
    assert not cfg.is_layer_excluded("model.layers.0.block_sparse_moe.gate_up_proj")
    # Preserve the upstream helper's treatment of exclusions inside fused experts.
    assert cfg.is_layer_excluded("model.layers.0.mlp.experts")


def test_empty_and_updated_config_are_read_live(patch_module):
    cfg = _MxFp8Config([])
    _opt_in(patch_module, cfg)
    assert not cfg.is_layer_excluded("mlp.gate")
    cfg.exclude_modules = ["mlp.gate"]
    assert cfg.is_layer_excluded("mlp.gate")
    assert not cfg.is_layer_excluded("mlp.gate_up_proj")


def test_from_config_carries_opt_in_and_preserves_unmarked_checkpoint(patch_module):
    raw = {
        "quant_method": "mxfp8",
        "ignored_layers": ["mlp.gate"],
        patch_module.VERL_EXACT_MXFP8_EXCLUSIONS: True,
    }
    generated = _MxFp8Config.from_config(raw)
    assert getattr(generated, patch_module.VERL_EXACT_MXFP8_EXCLUSIONS) is True
    assert generated.is_layer_excluded("mlp.gate")
    assert not generated.is_layer_excluded("mlp.gate_up_proj")
    assert "is_layer_excluded" not in vars(generated)
    checkpoint = _MxFp8Config.from_config({"quant_method": "mxfp8", "ignored_layers": ["mlp.gate"]})
    assert checkpoint.is_layer_excluded("mlp.gate_up_proj")
    assert not hasattr(checkpoint, patch_module.VERL_EXACT_MXFP8_EXCLUSIONS)


def test_opt_in_survives_serialization_without_bound_method(patch_module):
    cfg = _MxFp8Config.from_config(
        {
            "quant_method": "mxfp8",
            "ignored_layers": ["mlp.gate"],
            patch_module.VERL_EXACT_MXFP8_EXCLUSIONS: True,
        }
    )
    restored = pickle.loads(pickle.dumps(cfg))
    assert getattr(restored, patch_module.VERL_EXACT_MXFP8_EXCLUSIONS) is True
    assert "is_layer_excluded" not in vars(restored)
    assert not restored.is_layer_excluded("mlp.gate_up_proj")


def test_from_config_preserves_subclass_binding(patch_module):
    class ChildConfig(_MxFp8Config):
        pass

    parsed = ChildConfig.from_config({"quant_method": "mxfp8", "ignored_layers": []})
    assert type(parsed) is ChildConfig


def test_marked_wrong_schema_is_rejected_by_parser(patch_module):
    with pytest.raises(ValueError, match="quant_method"):
        _MxFp8Config.from_config({"quant_method": "modelopt", patch_module.VERL_EXACT_MXFP8_EXCLUSIONS: True})


def test_patch_stop_restores_original_class_descriptors(patch_module):
    parser = _MxFp8Config.__dict__["from_config"]
    matcher = _MxFp8Config.__dict__["is_layer_excluded"]
    with ExitStack() as stack:
        for patcher in patch_module.build_mxfp8_exclusion_patchers():
            stack.enter_context(patcher)
        assert _MxFp8Config.__dict__["from_config"] is not parser
    assert _MxFp8Config.__dict__["from_config"] is parser
    assert _MxFp8Config.__dict__["is_layer_excluded"] is matcher


def test_unmarked_configs_delegate_to_original_matcher(patch_module, monkeypatch):
    original = _MxFp8Config.is_layer_excluded
    calls = []

    def spy(self, prefix):
        calls.append(prefix)
        return original(self, prefix)

    monkeypatch.setattr(_MxFp8Config, "is_layer_excluded", spy)
    with ExitStack() as stack:
        for patcher in patch_module.build_mxfp8_exclusion_patchers():
            stack.enter_context(patcher)
        cfg = _MxFp8Config.from_config({"ignored_layers": ["mlp.gate"]})
        assert cfg.is_layer_excluded("mlp.gate_up_proj")
        assert calls == ["mlp.gate_up_proj"]


def test_common_registry_includes_parser_and_matcher(patch_module, monkeypatch):
    from packaging.version import Version

    fp8 = ModuleType("vllm.model_executor.layers.quantization.fp8")

    class Fp8Method:
        def process_weights_after_loading(self, layer):
            pass

    fp8.Fp8LinearMethod = Fp8Method
    fp8.Fp8MoEMethod = Fp8Method
    monkeypatch.setitem(sys.modules, fp8.__name__, fp8)
    monkeypatch.setitem(sys.modules, "verl.utils.vllm.mxfp8_exclusion_patch", patch_module)
    path = Path(__file__).parents[2] / "verl/utils/vllm/vllm_fp8_utils.py"
    spec = importlib.util.spec_from_file_location("fp8_patch_registry_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    patchers = module.build_fp8_method_patchers(Version("0.24.0"))
    assert len(patchers) == 4
    assert [p.attribute for p in patchers[-2:]] == ["from_config", "is_layer_excluded"]
    assert all(p.getter() is _MxFp8Config for p in patchers[-2:])
    assert len(module.build_fp8_method_patchers(Version("0.19.0"))) == 2
    # An older vLLM without ModelOpt MXFP8 still gets its normal two FP8 patches.
    monkeypatch.delattr(sys.modules["vllm.model_executor.layers.quantization.modelopt"], "ModelOptMxFp8Config")
    assert len(module.build_fp8_method_patchers(Version("0.24.0"))) == 2
