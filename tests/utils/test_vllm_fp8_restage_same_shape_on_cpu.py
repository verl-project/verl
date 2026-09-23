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

"""The FP8 refit staging predicate must see kernel repacks that keep the checkpoint shape.

vLLM's FlashInfer MoE preps hand ``replace_parameter`` a rewritten copy of ``w13`` (and its block
scale) with the checkpoint's shape and dtype: CUTLASS swaps the gate/up halves, TRT-LLM's MXFP8
path interleaves and tile-shuffles the rows. ``_layer_needs_fp8_staging`` compared shape, dtype
and ``is_shuffled`` only, so such a layer was never staged on a refit: the sync wrote canonical
weights straight into the repacked live buffer and ``process_weights_after_loading`` never re-ran
(first seen on 1xB200 with vLLM 0.24: the MoE refit probe read rel err 1.739 at the first sync).

``vllm_fp8_utils`` is loaded by path against a stub ``vllm...quantization.fp8`` module, so these
run without vLLM.
"""

import importlib.util
import sys
import types
from pathlib import Path

import torch
from packaging import version

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FP8_MODULE = "vllm.model_executor.layers.quantization.fp8"
E, INTER, HID = 2, 32, 64


def _load_fp8_utils():
    path = _REPO_ROOT / "verl/utils/vllm/vllm_fp8_utils.py"
    spec = importlib.util.spec_from_file_location("vllm_fp8_utils_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fp8():
    return sys.modules[_FP8_MODULE]


def _swap_halves(t: torch.Tensor) -> torch.Tensor:
    """FlashInfer's W13 -> W31 swap: a new tensor with the same shape and dtype."""
    return t.reshape(t.shape[0], 2, t.shape[1] // 2, *t.shape[2:]).flip(1).reshape(t.shape).contiguous()


def _cutlass_like_process(self, layer):
    """FlashInfer CUTLASS block-FP8 prep: gate/up halves of w13 and its scale swapped, shapes kept."""
    layer._process_calls = getattr(layer, "_process_calls", 0) + 1
    _fp8().replace_parameter(layer, "w13_weight", _swap_halves(layer.w13_weight.data))
    _fp8().replace_parameter(layer, "w13_weight_scale_inv", _swap_halves(layer.w13_weight_scale_inv.data))
    # what vLLM's _setup_kernel captures right after the replace calls
    layer._kernel_refs = (layer.w13_weight, layer.w13_weight_scale_inv)


def _identity_process(self, layer):
    """A backend that keeps the checkpoint layout (Triton / vLLM-CUTLASS): replace with the same tensor."""
    for name in ("w13_weight", "w2_weight", "w13_weight_scale_inv", "w2_weight_scale_inv"):
        _fp8().replace_parameter(layer, name, getattr(layer, name))


def _repacking_process(self, layer):
    """A backend that changes the scale's shape (DeepGEMM-like): the pre-existing detection path."""
    _fp8().replace_parameter(layer, "w13_weight_scale_inv", layer.w13_weight_scale_inv.data.reshape(E, -1).clone())


class _StubVllmFp8:
    NAMES = (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.quantization",
        _FP8_MODULE,
    )

    def __init__(self, moe_process):
        self._moe_process = moe_process

    def __enter__(self):
        self._saved = {n: sys.modules.get(n) for n in self.NAMES}
        mods = {n: types.ModuleType(n) for n in self.NAMES}
        fp8 = mods[_FP8_MODULE]

        def replace_parameter(layer, name, new_data, prefer_copy=False):
            setattr(layer, name, torch.nn.Parameter(new_data, requires_grad=False))

        fp8.replace_parameter = replace_parameter
        fp8.Fp8LinearMethod = type("Fp8LinearMethod", (), {"process_weights_after_loading": lambda self, layer: None})
        fp8.Fp8MoEMethod = type("Fp8MoEMethod", (), {"process_weights_after_loading": self._moe_process})
        sys.modules.update(mods)
        self.fp8 = fp8
        return self

    def __exit__(self, *exc):
        for name, prev in self._saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


class _FakeRoutedExperts(torch.nn.Module):
    """vLLM RoutedExperts stand-in under block FP8: fused expert weights plus ``*_scale_inv`` block scales."""

    def __init__(self, quant_method):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.w13_weight = torch.nn.Parameter(
            torch.randint(-6, 7, (E, 2 * INTER, HID), generator=g).to(torch.float8_e4m3fn), requires_grad=False
        )
        self.w2_weight = torch.nn.Parameter(
            torch.randint(-6, 7, (E, HID, INTER), generator=g).to(torch.float8_e4m3fn), requires_grad=False
        )
        self.w13_weight_scale_inv = torch.nn.Parameter(torch.rand(E, 2, 1, generator=g), requires_grad=False)
        self.w2_weight_scale_inv = torch.nn.Parameter(torch.rand(E, 1, 1, generator=g), requires_grad=False)
        self.quant_method = quant_method


def _loaded_layer(module, stub):
    """Build the layer and run vLLM's post-load hook once, as the initial model load does."""
    layer = _FakeRoutedExperts(stub.fp8.Fp8MoEMethod())
    layer.quant_method.process_weights_after_loading(layer)
    model = torch.nn.Module()
    model.experts = layer
    return model, layer


def test_same_shape_repack_is_restaged_and_reprocessed_into_the_live_storage():
    module = _load_fp8_utils()
    with _StubVllmFp8(_cutlass_like_process) as stub:
        patchers = module.build_fp8_method_patchers(version.parse("0.24.0"))
        assert len(patchers) == 2
        for p in patchers:
            p.start()
        try:
            layer = _FakeRoutedExperts(stub.fp8.Fp8MoEMethod())
            canonical = layer.w13_weight.data.clone()
            layer.quant_method.process_weights_after_loading(layer)  # initial load
            assert tuple(layer.w13_weight.shape) == (E, 2 * INTER, HID)  # shape and dtype unchanged: the blind spot
            assert torch.equal(layer.w13_weight.data.view(torch.uint8), _swap_halves(canonical).view(torch.uint8))
            assert layer._verl_fp8_repacked == {"w13_weight", "w13_weight_scale_inv"}
            live_w, live_s = layer.w13_weight, layer.w13_weight_scale_inv
            model = torch.nn.Module()
            model.experts = layer

            staged = module.stage_fp8_params_for_loading(model)
            assert staged == [layer], "a same-shape repack must still be staged"
            fresh = torch.randint(-6, 7, canonical.shape).to(torch.float8_e4m3fn)
            fresh_scale = torch.rand(E, 2, 1)
            layer.w13_weight.data.copy_(fresh)  # what load_weights writes: checkpoint layout
            layer.w13_weight_scale_inv.data.copy_(fresh_scale)
            module.process_fp8_weights_after_loading(staged)

            assert layer._process_calls == 2, "the refit must re-run the kernel prep"
            assert layer.w13_weight is live_w and layer.w13_weight.data_ptr() == live_w.data_ptr()
            assert torch.equal(layer.w13_weight.data.view(torch.uint8), _swap_halves(fresh).view(torch.uint8))
            assert torch.equal(layer.w13_weight_scale_inv.data, _swap_halves(fresh_scale))
            # the kernel captured right after the replace calls points at the live params, not staging buffers
            assert layer._kernel_refs[0] is live_w and layer._kernel_refs[1] is live_s
        finally:
            for p in patchers:
                p.stop()


def test_layout_keeping_backend_is_not_staged():
    module = _load_fp8_utils()
    with _StubVllmFp8(_identity_process) as stub:
        patchers = module.build_fp8_method_patchers(version.parse("0.24.0"))
        for p in patchers:
            p.start()
        try:
            model, layer = _loaded_layer(module, stub)
            assert not getattr(layer, "_verl_fp8_repacked", None)
            assert module.stage_fp8_params_for_loading(model) == []
        finally:
            for p in patchers:
                p.stop()


def test_shape_changing_repack_is_still_staged():
    module = _load_fp8_utils()
    with _StubVllmFp8(_repacking_process) as stub:
        patchers = module.build_fp8_method_patchers(version.parse("0.24.0"))
        for p in patchers:
            p.start()
        try:
            model, layer = _loaded_layer(module, stub)
            assert tuple(layer.w13_weight_scale_inv.shape) == (E, 2)
            assert module.stage_fp8_params_for_loading(model) == [layer]
        finally:
            for p in patchers:
                p.stop()
