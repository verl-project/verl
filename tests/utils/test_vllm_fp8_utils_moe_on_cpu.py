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

"""CPU tests for fused-MoE handling in ``verl/utils/vllm/vllm_fp8_utils.py``.

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

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO_ROOT / "verl/utils/vllm/vllm_fp8_utils.py"


def _make_module(name: str) -> types.ModuleType:
    return types.ModuleType(name)


def _load_fp8_utils(fused_moe_is_function: bool):
    """Load ``vllm_fp8_utils`` with vLLM/verl heavyweight deps stubbed.

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

    fake_dsv4 = _make_module("verl.utils.vllm.vllm_dsv4_fp8_utils")
    for fn in (
        "cache_deepseek_v4_dense_fp8_scales",
        "is_deepseek_v4_model",
        "iter_deepseek_v4_weights",
        "prepare_deepseek_v4_weights_for_loading",
        "process_deepseek_v4_weights_after_loading",
        "reload_deepseek_v4_dense_fp8_scales",
    ):
        setattr(fake_dsv4, fn, lambda *a, **k: None)

    fakes = {
        "vllm": _make_module("vllm"),
        "vllm.model_executor": _make_module("vllm.model_executor"),
        "vllm.model_executor.layers": _make_module("vllm.model_executor.layers"),
        "vllm.model_executor.layers.fused_moe": fused_moe_pkg,
        "vllm.model_executor.layers.fused_moe.layer": fused_moe_layer,
        "vllm.model_executor.layers.linear": linear_mod,
        "verl.utils.kernel": _make_module("verl.utils.kernel"),
        "verl.utils.kernel.fp8_kernel": fake_kernel,
        "verl.utils.vllm": _make_module("verl.utils.vllm"),
        "verl.utils.vllm.vllm_dsv4_fp8_utils": fake_dsv4,
    }

    saved = {name: sys.modules.get(name) for name in fakes}
    try:
        sys.modules.update(fakes)
        spec = importlib.util.spec_from_file_location("verl_vllm_fp8_utils_under_test", _MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
    finally:
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


def test_new_vllm_resolves_routed_experts_and_does_not_crash():
    """vLLM >= 0.24: per-expert names must resolve to RoutedExperts, no TypeError."""
    mod, ns = _load_fp8_utils(fused_moe_is_function=True)
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
    mod, ns = _load_fp8_utils(fused_moe_is_function=True)
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
    mod, ns = _load_fp8_utils(fused_moe_is_function=True)
    model = _build_model(ns, moe_dtype=torch.bfloat16)

    mod.fp8_state.seen_params.clear()
    mod.fp8_state.fp8_param_names.clear()

    assert mod.is_fp8_weight("model.layers.0.mlp.experts.0.gate_proj.weight", model) is False


def test_old_vllm_fusedmoe_class_still_supported():
    """vLLM < 0.24: FusedMoE is an nn.Module class holding the weights."""
    mod, ns = _load_fp8_utils(fused_moe_is_function=False)
    model = _build_model(ns, moe_dtype=torch.float8_e4m3fn)

    assert ns["FusedMoE"] in mod._EXPERT_WEIGHT_CLASSES

    experts = model.model.layers[0].mlp.experts
    resolved = mod.get_module_from_param_name(model, "model.layers.0.mlp.experts.0.gate_proj.weight")
    assert resolved is experts

    mod.fp8_state.seen_params.clear()
    mod.fp8_state.fp8_param_names.clear()
    assert mod.is_fp8_weight("model.layers.0.mlp.experts.0.gate_proj.weight", model) is True
