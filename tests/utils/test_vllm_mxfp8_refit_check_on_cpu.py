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
"""The vLLM MXFP8 refit self-check probes compressed-tensors MXFP8 layers and catches a stale kernel copy.

vLLM's compressed-tensors MXFP8 kernels read a copy derived from the canonical weight
scales when a layer is processed, and leave the parameter itself in a kernel layout. The
stand-ins below do the same: processing stores the derived copy the kernel reads and
flattens the parameter, and staging hands the loader a checkpoint-shaped parameter again,
so a sync whose layers are not re-processed leaves the kernel on the previous scales.
The modules are loaded by path, so these run without vLLM.
"""

import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from verl.utils.mxfp8_refit_check import mxfp8_dequantize, mxfp8_moe_expert_reference

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CORE_LOGGER = "verl.utils.mxfp8_refit_check"
N, K = 16, 128
E, INTER, HID = 2, 32, 64


def _load_by_path(name, relpath):
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / relpath)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = _load_by_path("vllm_mxfp8_refit_check_under_test", "verl/utils/vllm/vllm_mxfp8_refit_check.py")


@pytest.fixture(autouse=True)
def _check_raises(monkeypatch):
    # These tests exercise the failure path; the default mode (warn) is tested on its own.
    monkeypatch.setenv("VERL_MXFP8_REFIT_CHECK", "raise")
    monkeypatch.setattr(check, "_WARNED", set())


def _mxfp8(shape, seed):
    g = torch.Generator().manual_seed(seed)
    # fp8-representable values: small integers scaled by a per-block power of two
    q = torch.randint(-6, 7, shape, generator=g).to(torch.float8_e4m3fn)
    scale = torch.randint(124, 130, (*shape[:-1], shape[-1] // 32), generator=g).to(torch.uint8)
    return q, scale


def _param(t):
    return torch.nn.Parameter(t, requires_grad=False)


# The stand-ins carry vLLM's class names: the check selects layers by them.
class CompressedTensorsW8A8Mxfp8:
    def process_weights_after_loading(self, layer):
        layer._kernel_scale = layer.weight_scale.data.clone()
        layer.weight_scale = _param(layer.weight_scale.data.reshape(-1).clone())

    def apply_weights(self, layer, x, bias=None):
        return (x.float() @ mxfp8_dequantize(layer.weight.data, layer._kernel_scale).t()).to(torch.bfloat16)


class CompressedTensorsW8A8Fp8(CompressedTensorsW8A8Mxfp8):
    pass


class CompressedTensorsLinearMethod:
    def process_weights_after_loading(self, layer):
        layer.scheme.process_weights_after_loading(layer)

    def apply(self, layer, x, bias=None):
        return layer.scheme.apply_weights(layer, x, bias=bias)


class CompressedTensorsW8A8Mxfp8MoEMethod:
    is_monolithic = False
    topk_indices_dtype = None

    def process_weights_after_loading(self, layer):
        layer._kernel_scales = (layer.w13_weight_scale.data.clone(), layer.w2_weight_scale.data.clone())
        for name in ("w13_weight_scale", "w2_weight_scale"):
            setattr(layer, name, _param(getattr(layer, name).data.reshape(E, -1).clone()))


def _linear(n=N, scheme=CompressedTensorsW8A8Mxfp8, seed=0):
    layer = torch.nn.Module()
    q, scale = _mxfp8((n, K), seed)
    layer.weight = _param(q)
    layer.weight_scale = _param(scale)
    layer.scheme = scheme()
    layer.quant_method = CompressedTensorsLinearMethod()
    layer.quant_method.process_weights_after_loading(layer)  # initial load
    return layer


class _RoutedExperts(torch.nn.Module):
    """vLLM RoutedExperts stand-in: fused expert weights and the modular forward the probe calls."""

    def __init__(self, ep=1, dp=1, activation="silu"):
        super().__init__()
        w13, s13 = _mxfp8((E, 2 * INTER, HID), 2)
        w2, s2 = _mxfp8((E, HID, INTER), 3)
        self.w13_weight, self.w13_weight_scale = _param(w13), _param(s13)
        self.w2_weight, self.w2_weight_scale = _param(w2), _param(s2)
        parallel = SimpleNamespace(tp_size=1, ep_size=ep, dp_size=dp, use_ep=ep > 1)
        self.moe_config = SimpleNamespace(moe_parallel_config=parallel)
        self.top_k = 2
        self.activation = activation
        self.quant_method = CompressedTensorsW8A8Mxfp8MoEMethod()
        self.quant_method.process_weights_after_loading(self)  # initial load

    def forward_modular(self, x, topk_weights, topk_ids):
        s13, s2 = self._kernel_scales
        out = torch.zeros(x.shape[0], HID)
        for r in range(x.shape[0]):
            for j in range(topk_ids.shape[1]):
                weight, e = float(topk_weights[r, j]), int(topk_ids[r, j])
                if weight:
                    w13, w2 = self.w13_weight.data[e], self.w2_weight.data[e]
                    out[r] += weight * mxfp8_moe_expert_reference(x[r : r + 1], w13, s13[e], w2, s2[e])[0]
        return out.to(torch.bfloat16)


def _stage(model):
    """The refit staging: give the loader checkpoint-shaped parameters again; returns the staged layers."""
    staged = []
    for module in model.modules():
        if isinstance(getattr(module, "quant_method", None), CompressedTensorsLinearMethod):
            n, k = module.weight.shape
            module.weight_scale = _param(module.weight_scale.data.reshape(n, k // 32).clone())
        elif isinstance(getattr(module, "quant_method", None), CompressedTensorsW8A8Mxfp8MoEMethod):
            module.w13_weight_scale = _param(module.w13_weight_scale.data.reshape(E, 2 * INTER, HID // 32).clone())
            module.w2_weight_scale = _param(module.w2_weight_scale.data.reshape(E, HID, INTER // 32).clone())
        else:
            continue
        staged.append(module)
    return staged


def _reprocess(layers):
    for layer in layers:
        layer.quant_method.process_weights_after_loading(layer)


def _sync(model, delta):
    """A weight sync writing new canonical scales (each step of ``delta`` doubles the weights)."""
    for module in model.modules():
        for name in ("weight_scale", "w13_weight_scale", "w2_weight_scale"):
            if isinstance(getattr(module, name, None), torch.Tensor):
                getattr(module, name).data.add_(delta)


def test_linear_check_passes_on_a_re_derived_layout_and_fails_on_a_stale_one():
    model = torch.nn.Module()
    model.big = _linear(n=64, seed=1)
    model.small = _linear()

    staged = _stage(model)
    _sync(model, 2)
    snapshot = check.snapshot_mxfp8_linear_for_check(model)
    assert snapshot[0] == "small"
    _reprocess(staged)
    check.self_check_mxfp8_linear(snapshot)

    _stage(model)
    _sync(model, 1)
    snapshot = check.snapshot_mxfp8_linear_for_check(model)
    # Not re-processed: the kernel still reads the previous sync's scales, half the new weights.
    with pytest.raises(RuntimeError, match="refit self-check failed on vllm layer 'small'"):
        check.self_check_mxfp8_linear(snapshot)


def test_only_compressed_tensors_mxfp8_linears_are_probed():
    model = torch.nn.Module()
    model.fp8 = _linear(scheme=CompressedTensorsW8A8Fp8)
    model.plain = torch.nn.Linear(K, N)
    _stage(model)
    assert check.snapshot_mxfp8_linear_for_check(model) is None


def test_linear_check_warns_when_no_mxfp8_layer_holds_a_canonical_scale(caplog):
    model = torch.nn.Module()
    model.proj = _linear()  # processed and never staged: its scale is still in the kernel layout
    with caplog.at_level(logging.WARNING, logger=check.logger.name):
        assert check.snapshot_mxfp8_linear_for_check(model) is None
    assert "none of the 1 compressed-tensors MXFP8 linear layers" in caplog.text


def test_moe_check_passes_on_a_re_derived_layout_and_fails_on_a_stale_one():
    model = torch.nn.Module()
    model.experts = _RoutedExperts()

    staged = _stage(model)
    _sync(model, 1)
    snapshot = check.snapshot_mxfp8_moe_for_check(model)
    assert snapshot[0] == "experts" and snapshot[2] == 0 and tuple(snapshot[3].shape) == (2 * INTER, HID)
    _reprocess(staged)
    check.self_check_mxfp8_moe(snapshot)

    _stage(model)
    _sync(model, 1)
    snapshot = check.snapshot_mxfp8_moe_for_check(model)
    with pytest.raises(RuntimeError, match="vllm MoE layer 'experts', expert 0"):
        check.self_check_mxfp8_moe(snapshot)


def test_moe_check_is_skipped_where_the_probe_cannot_reproduce_the_layer(caplog):
    for kwargs, reason in (
        ({"ep": 2}, "expert or data parallelism (ep_size=2, dp_size=1)"),
        ({"dp": 2}, "expert or data parallelism (ep_size=1, dp_size=2)"),
        ({"activation": "swigluoai"}, "activation swigluoai"),
    ):
        check._WARNED.clear()
        caplog.clear()
        model = torch.nn.Module()
        model.experts = _RoutedExperts(**kwargs)
        _stage(model)
        with caplog.at_level(logging.WARNING, logger=check.logger.name):
            assert check.snapshot_mxfp8_moe_for_check(model) is None, kwargs
            assert check.snapshot_mxfp8_moe_for_check(model) is None, kwargs  # the next sync
        msgs = [r.getMessage() for r in caplog.records if "MoE refit check skipped on experts" in r.getMessage()]
        assert len(msgs) == 1 and reason in msgs[0], (kwargs, msgs)  # a warning, once per layer
    caplog.clear()

    model = torch.nn.Module()
    model.experts = _RoutedExperts()  # never staged: the expert scales are still in the kernel layout
    with caplog.at_level(logging.WARNING, logger=check.logger.name):
        assert check.snapshot_mxfp8_moe_for_check(model) is None
    assert "not the canonical uint8" in caplog.text


def test_the_check_can_be_disabled(monkeypatch):
    model = torch.nn.Module()
    model.proj = _linear()
    model.experts = _RoutedExperts()
    _stage(model)
    monkeypatch.setenv("VERL_MXFP8_REFIT_CHECK", "0")
    assert check.snapshot_mxfp8_linear_for_check(model) is None
    assert check.snapshot_mxfp8_moe_for_check(model) is None


def _load_quant_utils(monkeypatch):
    """``vllm_quant_utils`` with its FP8/FP4 siblings stubbed by the staging above."""
    fp4 = types.ModuleType("verl.utils.vllm.vllm_fp4_utils")
    for name in (
        "install_mxfp4_mxfp8_compressed_tensors_patch",
        "install_nvfp4_cutlass_compressed_tensors_patch",
        "is_deepseek_v4_model",
        "process_mxfp4_moe_weights_after_loading",
        "process_nvfp4_moe_weights_after_loading",
        "stage_mxfp4_moe_params_for_loading",
        "stage_nvfp4_moe_params_for_loading",
    ):
        setattr(fp4, name, lambda *args: [])
    fp8 = types.ModuleType("verl.utils.vllm.vllm_fp8_utils")
    fp8.build_fp8_method_patchers = lambda vllm_version: []
    fp8.stage_fp8_params_for_loading = _stage
    fp8.process_fp8_weights_after_loading = _reprocess
    unquant = types.ModuleType("verl.utils.vllm.vllm_unquant_utils")
    unquant.stage_unquantized_moe_params = lambda model: []
    unquant.process_unquantized_moe_weights_after_loading = lambda model, staged: 0
    monkeypatch.setitem(sys.modules, "verl.utils.vllm.vllm_fp4_utils", fp4)
    monkeypatch.setitem(sys.modules, "verl.utils.vllm.vllm_fp8_utils", fp8)
    monkeypatch.setitem(sys.modules, "verl.utils.vllm.vllm_unquant_utils", unquant)
    monkeypatch.setitem(sys.modules, "verl.utils.vllm.vllm_mxfp8_refit_check", check)
    return _load_by_path("vllm_quant_utils_under_test", "verl/utils/vllm/vllm_quant_utils.py")


def test_refit_finalize_checks_the_layers_against_the_weights_the_sync_wrote(monkeypatch, caplog):
    quant_utils = _load_quant_utils(monkeypatch)
    model = torch.nn.Module()
    model.proj = _linear()
    model.experts = _RoutedExperts()

    state = quant_utils.prepare_quanted_weights_for_loading(model)
    _sync(model, 1)
    with caplog.at_level(logging.DEBUG, logger=_CORE_LOGGER):
        quant_utils.process_quanted_weights_after_loading(model, state)
    # Both probes ran, so both snapshots were taken while the scales were canonical, before re-processing.
    assert "refit self-check passed on proj (vllm)" in caplog.text
    assert "MoE refit self-check passed on experts expert 0 (vllm)" in caplog.text

    monkeypatch.setattr(quant_utils, "process_fp8_weights_after_loading", lambda layers: None)
    state = quant_utils.prepare_quanted_weights_for_loading(model)
    _sync(model, 1)
    with pytest.raises(RuntimeError, match="refit self-check failed on vllm layer 'proj'"):
        quant_utils.process_quanted_weights_after_loading(model, state)


def _monolithic_experts(**routing):
    """RoutedExperts on a monolithic kernel that, like vLLM's, routes inside and applies routed_scaling_factor."""
    model = torch.nn.Module()
    model.experts = layer = _RoutedExperts()
    layer.quant_method.is_monolithic = True
    layer.routed_scaling_factor = 1.0
    for key, value in routing.items():
        setattr(layer, key, value)

    def forward_monolithic(x, router_logits):
        s13, s2 = layer._kernel_scales
        top = torch.softmax(router_logits.float(), dim=-1).topk(layer.top_k, dim=-1)
        out = torch.zeros(x.shape[0], HID)
        for r in range(x.shape[0]):
            for weight, e in zip(top.values[r].tolist(), top.indices[r].tolist(), strict=True):
                if weight:
                    w13, w2 = layer.w13_weight.data[e], layer.w2_weight.data[e]
                    out[r] += weight * mxfp8_moe_expert_reference(x[r : r + 1], w13, s13[e], w2, s2[e])[0]
        return (out * layer.routed_scaling_factor).to(torch.bfloat16)

    layer.forward_monolithic = forward_monolithic
    return model


def test_monolithic_probe_runs_where_the_routing_parameters_are_neutral():
    model = _monolithic_experts()
    staged = _stage(model)
    _sync(model, 1)
    snapshot = check.snapshot_mxfp8_moe_for_check(model)
    _reprocess(staged)
    check.self_check_mxfp8_moe(snapshot)  # the kernel's own routing gives the probed expert weight 1


@pytest.mark.parametrize(
    ("routing", "reason"),
    [
        ({"routed_scaling_factor": 2.5}, "routed scaling factor 2.5"),  # DeepSeek-V3: the kernel scales the output
        ({"use_grouped_topk": True, "num_expert_group": 8}, "grouped top-k routing"),
        ({"e_score_correction_bias": torch.zeros(E)}, "a score correction bias"),
        ({"custom_routing_function": lambda *args: None}, "a custom routing function"),
    ],
)
def test_monolithic_probe_is_skipped_once_where_the_kernel_routing_cannot_pin_one_expert(routing, reason, caplog):
    model = _monolithic_experts(**routing)
    _stage(model)
    with caplog.at_level(logging.WARNING, logger=check.logger.name):
        assert check.snapshot_mxfp8_moe_for_check(model) is None
        assert check.snapshot_mxfp8_moe_for_check(model) is None
    msgs = [r.getMessage() for r in caplog.records if "MoE refit check skipped on experts" in r.getMessage()]
    assert len(msgs) == 1 and reason in msgs[0]


def test_the_modular_probe_ignores_the_routing_parameters():
    model = torch.nn.Module()
    model.experts = _RoutedExperts()
    model.experts.routed_scaling_factor = 2.5  # bypassed: forward_modular takes the hand-built top-k
    staged = _stage(model)
    _sync(model, 1)
    snapshot = check.snapshot_mxfp8_moe_for_check(model)
    _reprocess(staged)
    check.self_check_mxfp8_moe(snapshot)


def test_by_default_a_stale_expert_is_logged_and_the_sync_goes_on(monkeypatch, caplog):
    monkeypatch.delenv("VERL_MXFP8_REFIT_CHECK")
    model = torch.nn.Module()
    model.experts = _RoutedExperts()
    _stage(model)
    _sync(model, 1)
    snapshot = check.snapshot_mxfp8_moe_for_check(model)  # nothing re-derives the kernel copy afterwards
    with caplog.at_level(logging.WARNING):
        check.self_check_mxfp8_moe(snapshot)
    assert "vllm MoE layer 'experts', expert 0" in caplog.text and "The sync goes on" in caplog.text
