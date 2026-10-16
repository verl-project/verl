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

"""CPU tests for fused-MoE handling in ``verl/utils/vllm/vllm_quant_utils.py``.

vLLM 0.24.0 (MoE refactor, vllm-project/vllm#41184) turned ``FusedMoE`` from an
``nn.Module`` class into a factory *function* that returns a ``MoERunner``, and
moved the fused expert weights onto a ``RoutedExperts`` submodule
(``experts`` -> ``experts.routed_experts``). The old code called
``isinstance(module, FusedMoE)`` which raised
``TypeError: isinstance() arg 2 must be a type`` once ``FusedMoE`` became a
function. These tests exercise both the pre-0.24 and post-0.24 module layouts
with lightweight fakes (no real vLLM required).
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VLLM_UTILS_DIR = _REPO_ROOT / "verl/utils/vllm"
_MODULE_PATH = _VLLM_UTILS_DIR / "vllm_quant_utils.py"


def _make_module(name: str) -> types.ModuleType:
    return types.ModuleType(name)


def _load_quant_utils(fused_moe_is_function: bool):
    """Load ``vllm_quant_utils`` with vLLM/verl heavyweight deps stubbed.

    Args:
        fused_moe_is_function: when True, emulate vLLM >= 0.24 where ``FusedMoE``
            is a factory function and the expert weights live on ``RoutedExperts``
            owned by a ``MoERunner``. When False, emulate vLLM < 0.24 where
            ``FusedMoE`` is the ``nn.Module`` class holding the weights.

    Returns:
        (module, namespace) where ``namespace`` exposes the fake classes used to
        build test models so callers can assert against the exact class objects
        the loaded module resolved.
    """

    class _FakeLinearBase(torch.nn.Module):
        def __init__(self, dtype=torch.float8_e4m3fn):
            super().__init__()
            self.weight = torch.empty(2, 2, dtype=dtype)

    ns: dict = {"LinearBase": _FakeLinearBase}

    fused_moe_pkg = _make_module("vllm.model_executor.layers.fused_moe")
    fused_moe_layer = _make_module("vllm.model_executor.layers.fused_moe.layer")
    linear_mod = _make_module("vllm.model_executor.layers.linear")
    linear_mod.LinearBase = _FakeLinearBase

    if fused_moe_is_function:

        class _FakeRoutedExperts(torch.nn.Module):
            def __init__(self, dtype=torch.float8_e4m3fn):
                super().__init__()
                self.w13_weight = torch.empty(2, 2, dtype=dtype)
                self.w2_weight = torch.empty(2, 2, dtype=dtype)

        class _FakeMoERunner(torch.nn.Module):
            def __init__(self, dtype=torch.float8_e4m3fn):
                super().__init__()
                self.routed_experts = _FakeRoutedExperts(dtype)

        def _fused_moe_factory(*args, **kwargs):  # noqa: N802 - mirrors vLLM name
            return _FakeMoERunner()

        fused_moe_layer.FusedMoE = _fused_moe_factory
        fused_moe_pkg.FusedMoE = _fused_moe_factory
        fused_moe_pkg.RoutedExperts = _FakeRoutedExperts
        fused_moe_pkg.MoERunner = _FakeMoERunner
        ns.update(RoutedExperts=_FakeRoutedExperts, MoERunner=_FakeMoERunner)
    else:

        class _FakeFusedMoE(torch.nn.Module):
            def __init__(self, dtype=torch.float8_e4m3fn):
                super().__init__()
                self.w13_weight = torch.empty(2, 2, dtype=dtype)
                self.w2_weight = torch.empty(2, 2, dtype=dtype)

        fused_moe_layer.FusedMoE = _FakeFusedMoE
        fused_moe_pkg.FusedMoE = _FakeFusedMoE
        ns.update(FusedMoE=_FakeFusedMoE)

    # verl leaf modules that pull in torch/triton or vLLM at import time.
    fake_kernel = _make_module("verl.utils.kernel.fp8_kernel")
    fake_kernel.scaled_fp8_blockwise = lambda *a, **k: (None, None)

    # The vllm_fp8_utils / vllm_fp4_utils siblings only touch vLLM inside
    # function bodies, so they can be imported for real. Stub the package
    # ``__init__`` (which does pull heavyweight deps) but point its search path
    # at the real directory so the submodule imports still resolve.
    fake_vllm_pkg = _make_module("verl.utils.vllm")
    fake_vllm_pkg.__path__ = [str(_VLLM_UTILS_DIR)]

    fakes = {
        "vllm": _make_module("vllm"),
        "vllm.model_executor": _make_module("vllm.model_executor"),
        "vllm.model_executor.layers": _make_module("vllm.model_executor.layers"),
        "vllm.model_executor.layers.fused_moe": fused_moe_pkg,
        "vllm.model_executor.layers.fused_moe.layer": fused_moe_layer,
        "vllm.model_executor.layers.linear": linear_mod,
        "verl.utils.kernel": _make_module("verl.utils.kernel"),
        "verl.utils.kernel.fp8_kernel": fake_kernel,
        "verl.utils.vllm": fake_vllm_pkg,
    }

    saved = {name: sys.modules.get(name) for name in fakes}
    preexisting = set(sys.modules)
    try:
        sys.modules.update(fakes)
        spec = importlib.util.spec_from_file_location("verl_vllm_quant_utils_under_test", _MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        # The sibling modules imported above are bound to the stubbed package,
        # so drop them rather than leave them for the next importer.
        for name in set(sys.modules) - preexisting:
            if name.startswith("verl.utils.vllm."):
                sys.modules.pop(name, None)
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
    return module, ns


PACKED_MODULES_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


def test_vlm_checkpoint_names_resolve_to_live_module():
    """Verify the checkpoint mapper resolves VLM parameter names to the live projection module."""
    mod, ns = _load_quant_utils(fused_moe_is_function=False)
    model = torch.nn.Module()
    model.packed_modules_mapping = {}
    model.language_model = torch.nn.Module()
    model.language_model.model = torch.nn.Module()
    projection = ns["LinearBase"](dtype=torch.bfloat16)
    model.language_model.model.proj = projection
    model.hf_to_vllm_mapper = types.SimpleNamespace(
        apply_list=lambda names: [name.replace("model.language_model.", "language_model.model.", 1) for name in names]
    )
    assert mod.get_module_from_param_name(model, "model.language_model.proj.weight") is projection


def test_mtp_filter_reads_multimodal_text_config():
    """MTP weights in a multimodal config must not be traversed during refit."""
    mod, _ = _load_quant_utils(fused_moe_is_function=False)
    model = torch.nn.Module()
    model.config = types.SimpleNamespace(text_config=types.SimpleNamespace(num_hidden_layers=45))

    assert mod._is_absent_mtp_layer_weight("language_model.model.layers.44.norm.weight", model) is False
    assert mod._is_absent_mtp_layer_weight("language_model.model.layers.45.norm.weight", model) is True


def _build_model(ns: dict, moe_dtype=torch.float8_e4m3fn):
    """Build ``model.layers.0`` with an attention linear and a fused-MoE block."""
    linear_cls = ns["LinearBase"]
    moe_cls = ns.get("MoERunner") or ns.get("FusedMoE")

    attn = torch.nn.Module()
    attn.qkv_proj = linear_cls(torch.float8_e4m3fn)

    mlp = torch.nn.Module()
    mlp.experts = moe_cls(moe_dtype)
    mlp.gate = linear_cls(torch.bfloat16)  # router: must NOT be treated as fp8

    layer0 = torch.nn.Module()
    layer0.self_attn = attn
    layer0.mlp = mlp

    inner = torch.nn.Module()
    inner.layers = torch.nn.ModuleList([layer0])

    model = torch.nn.Module()
    model.model = inner
    model.packed_modules_mapping = PACKED_MODULES_MAPPING
    return model


@pytest.mark.parametrize("fused_moe_is_function", [False, True])
@pytest.mark.parametrize("mapped_names", [False, True])
@pytest.mark.parametrize("layer_container", ["dict", "list"])
def test_mtp_refit_quantizes_only_live_target_layers(
    monkeypatch, capsys, fused_moe_is_function, mapped_names, layer_container
):
    """Main-model refit skips absent MTP layers; drafter refit emits FP8 weights and scales."""
    mod, ns = _load_quant_utils(fused_moe_is_function)
    main_model = _build_model(ns)
    drafter = _build_model(ns)
    predictors = []
    for _ in range(2):
        predictor = torch.nn.Module()
        predictor.mtp_block = _build_model(ns).model.layers[0]
        predictors.append(predictor)
    if layer_container == "dict":
        drafter.model.layers = torch.nn.ModuleDict({"1": predictors[0], "2": predictors[1]})
    else:
        drafter.model.layers = torch.nn.ModuleList([torch.nn.Module(), *predictors])

    prefix = "model."
    for model in (main_model, drafter):
        model.config = types.SimpleNamespace(text_config=types.SimpleNamespace(num_hidden_layers=1))
        if mapped_names:
            inner = model.model
            del model.model
            model.language_model = torch.nn.Module()
            model.language_model.model = inner
            model.hf_to_vllm_mapper = types.SimpleNamespace(
                apply_list=lambda names: [
                    name.replace("model.language_model.", "language_model.model.", 1) for name in names
                ]
            )
            prefix = "model.language_model."
        model.load_weights = lambda weights: dict(weights)

    quant_utils = _make_module("vllm.model_executor.layers.quantization.utils.quant_utils")
    quant_utils._normalize_quant_group_shape = lambda weight, block: block
    quant_utils.scaled_quantize = lambda weight, *args, **kwargs: (
        (weight / 2).to(torch.float8_e4m3fn),
        torch.full((1, 1), 2.0),
    )
    monkeypatch.setitem(sys.modules, quant_utils.__name__, quant_utils)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    runner = types.SimpleNamespace(
        model=main_model,
        drafter=types.SimpleNamespace(model=drafter),
        vllm_config=types.SimpleNamespace(
            quant_config=types.SimpleNamespace(weight_block_size=[2, 2]),
            model_config=types.SimpleNamespace(dtype=torch.bfloat16),
        ),
    )
    quantized_names = [
        f"{prefix}layers.1.mtp_block.self_attn.q_proj.weight",
        f"{prefix}layers.1.mtp_block.mlp.experts.0.gate_proj.weight",
        f"{prefix}layers.2.mtp_block.mlp.experts.0.down_proj.weight",
    ]
    unquantized_names = [
        f"{prefix}layers.1.mtp_block.mlp.gate.weight",
        f"{prefix}layers.3.mtp_block.self_attn.q_proj.weight",
    ]
    weight = torch.arange(4, dtype=torch.bfloat16).reshape(2, 2)
    weights = [(name, weight) for name in quantized_names + unquantized_names]

    # Load the same checkpoint into both targets, as the rollout worker does.
    main_loaded = mod.load_quanted_weights(weights, runner)
    assert main_loaded.keys() == dict(weights).keys()
    assert all(value is weight for value in main_loaded.values())

    draft_loaded = mod.load_quanted_weights(weights, runner, is_drafter=True)
    assert set(draft_loaded) == set(main_loaded) | {name + "_scale_inv" for name in quantized_names}
    for name in quantized_names:
        assert draft_loaded[name].dtype == torch.float8_e4m3fn
        torch.testing.assert_close(draft_loaded[name].float(), weight.float() / 2)
        torch.testing.assert_close(draft_loaded[name + "_scale_inv"], torch.tensor([2.0]))
    assert all(draft_loaded[name] is weight for name in unquantized_names)
    assert "Could not find module" not in capsys.readouterr().out


def test_new_vllm_resolves_routed_experts_and_does_not_crash():
    """vLLM >= 0.24: per-expert names must resolve to RoutedExperts, no TypeError."""
    mod, ns = _load_quant_utils(fused_moe_is_function=True)
    model = _build_model(ns)

    routed_experts = model.model.layers[0].mlp.experts.routed_experts
    assert isinstance(routed_experts, ns["RoutedExperts"])

    # Class resolution landed on the concrete post-refactor classes.
    assert ns["RoutedExperts"] in mod._EXPERT_WEIGHT_CLASSES
    assert ns["MoERunner"] in mod._MOE_STOP_CLASSES

    # The exact name that crashed before the fix: a per-expert HF weight that
    # walks into the fused-MoE block (index + proj remapped to gate_up_proj).
    name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    resolved = mod.get_module_from_param_name(model, name)
    assert resolved is routed_experts

    # Already-fused expert weight name resolves to the same holder.
    resolved_fused = mod.get_module_from_param_name(model, "model.layers.0.mlp.experts.w13_weight")
    assert resolved_fused is routed_experts


def test_new_vllm_is_fp8_weight_detects_moe_and_linear():
    mod, ns = _load_quant_utils(fused_moe_is_function=True)
    model = _build_model(ns, moe_dtype=torch.float8_e4m3fn)

    mod.fp8_state.seen_params.clear()
    mod.fp8_state.fp8_param_names.clear()

    # fp8 expert weight (per-expert HF name) -> detected as fp8
    assert mod.is_fp8_weight("model.layers.0.mlp.experts.0.gate_proj.weight", model) is True
    # fp8 attention linear -> detected as fp8
    assert mod.is_fp8_weight("model.layers.0.self_attn.q_proj.weight", model) is True
    # bf16 router gate -> NOT fp8
    assert mod.is_fp8_weight("model.layers.0.mlp.gate.weight", model) is False


def test_new_vllm_bf16_experts_not_flagged_fp8():
    mod, ns = _load_quant_utils(fused_moe_is_function=True)
    model = _build_model(ns, moe_dtype=torch.bfloat16)

    mod.fp8_state.seen_params.clear()
    mod.fp8_state.fp8_param_names.clear()

    assert mod.is_fp8_weight("model.layers.0.mlp.experts.0.gate_proj.weight", model) is False


def test_old_vllm_fusedmoe_class_still_supported():
    """vLLM < 0.24: FusedMoE is an nn.Module class holding the weights."""
    mod, ns = _load_quant_utils(fused_moe_is_function=False)
    model = _build_model(ns, moe_dtype=torch.float8_e4m3fn)

    assert ns["FusedMoE"] in mod._EXPERT_WEIGHT_CLASSES

    experts = model.model.layers[0].mlp.experts
    resolved = mod.get_module_from_param_name(model, "model.layers.0.mlp.experts.0.gate_proj.weight")
    assert resolved is experts

    mod.fp8_state.seen_params.clear()
    mod.fp8_state.fp8_param_names.clear()
    assert mod.is_fp8_weight("model.layers.0.mlp.experts.0.gate_proj.weight", model) is True
