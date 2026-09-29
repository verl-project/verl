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

"""A refit must reach unquantized MoE layers whose kernel prep reshaped the expert weights.

FlashInfer TRT-LLM -- vLLM's default bf16 MoE backend on SM100 -- keeps ``w13_weight`` /
``w2_weight`` in a 4-D block layout, so the refit's per-expert checkpoint-layout loads do not
fit the live parameters (verl-project/verl#7978). ``vllm_unquant_utils`` stages the checkpoint
layout as a view over the live storage and folds the re-derived kernel layout back into it.

The kernel prep here is a toy block transpose with the same property that matters: a new shape
over the same bytes, reordered so that converting in place would corrupt it. The module is loaded
by path against stub vLLM modules, so these run without vLLM.
"""

import importlib.util
import sys
import types
import weakref
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FUSED_MOE_PKG = "vllm.model_executor.layers.fused_moe"
_UNQUANTIZED_MODULE = f"{_FUSED_MOE_PKG}.unquantized_fused_moe_method"
_LAYERWISE_MODULE = "vllm.model_executor.model_loader.reload.layerwise"
E, INTER, HID, BLOCK = 2, 4, 8, 4


def _load_by_path(name, relpath):
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_unquant_utils():
    fp8_utils = _load_by_path("vllm_fp8_utils_for_unquant_utils", "verl/utils/vllm/vllm_fp8_utils.py")
    saved = sys.modules.get("verl.utils.vllm.vllm_fp8_utils")
    sys.modules["verl.utils.vllm.vllm_fp8_utils"] = fp8_utils
    try:
        return _load_by_path("vllm_unquant_utils_under_test", "verl/utils/vllm/vllm_unquant_utils.py")
    finally:
        if saved is None:
            sys.modules.pop("verl.utils.vllm.vllm_fp8_utils", None)
        else:
            sys.modules["verl.utils.vllm.vllm_fp8_utils"] = saved


unquant_utils = _load_unquant_utils()


def _block_layout(raw: torch.Tensor) -> torch.Tensor:
    """[E, R, C] -> [E, C/B, R, B]: TRT-LLM's BlockMajorK in miniature (new shape, same bytes)."""
    e, r, c = raw.shape
    return raw.reshape(e, r, c // BLOCK, BLOCK).permute(0, 2, 1, 3).contiguous()


def _padded_block_layout(raw: torch.Tensor) -> torch.Tensor:
    """Pad before the block transpose, as vLLM 0.29 pads TRT-LLM's intermediate dim to a multiple of 128."""
    return _block_layout(torch.nn.functional.pad(raw, (0, BLOCK)))


def _halved_layout(raw: torch.Tensor) -> torch.Tensor:
    """A kernel layout with fewer bytes than the checkpoint layout: no room to stage in place."""
    return raw[..., ::2].contiguous()


def _replace_parameter(layer, param_name, new_data, prefer_copy=False):
    """vLLM's ``replace_parameter``: copy into the old parameter only when shape and dtype match."""
    old = getattr(layer, param_name, None)
    if prefer_copy and old is not None and old.shape == new_data.shape and old.dtype == new_data.dtype:
        old.copy_(new_data)
        return
    new_param = torch.nn.Parameter(new_data, requires_grad=False)
    if old is not None and hasattr(old, "weight_loader"):
        new_param.weight_loader = old.weight_loader
    setattr(layer, param_name, new_param)


@pytest.fixture
def vllm(monkeypatch):
    """Stub vLLM's unquantized MoE method module and the per-layer metadata it records at model init."""
    unquantized = types.ModuleType(_UNQUANTIZED_MODULE)
    unquantized.replace_parameter = _replace_parameter

    class UnquantizedFusedMoEMethod:
        """``process_weights_after_loading`` as vLLM's ``_setup_kernel`` does it: kernel prep, then
        ``replace_parameter`` through the module-level name, copying into place on a weight update."""

        def __init__(self, kernel_layout=_block_layout, names=("w13_weight", "w2_weight")):
            self.kernel_layout = kernel_layout
            self.names = names
            self.moe_kernel = None

        def process_weights_after_loading(self, layer):
            is_weight_update = self.moe_kernel is not None
            for name in self.names:
                new_data = self.kernel_layout(getattr(layer, name).data)
                unquantized.replace_parameter(layer, name, new_data, prefer_copy=is_weight_update)
            if not is_weight_update:
                self.moe_kernel = (layer.w13_weight, layer.w2_weight)

    unquantized.UnquantizedFusedMoEMethod = UnquantizedFusedMoEMethod
    fused_moe = types.ModuleType(_FUSED_MOE_PKG)
    fused_moe.unquantized_fused_moe_method = unquantized
    layerwise = types.ModuleType(_LAYERWISE_MODULE)
    layerwise.LAYERWISE_INFO = weakref.WeakKeyDictionary()

    monkeypatch.setitem(sys.modules, _FUSED_MOE_PKG, fused_moe)
    monkeypatch.setitem(sys.modules, _UNQUANTIZED_MODULE, unquantized)
    monkeypatch.setitem(sys.modules, _LAYERWISE_MODULE, layerwise)
    return types.SimpleNamespace(unquantized=unquantized, layerwise_info=layerwise.LAYERWISE_INFO)


class _FakeRoutedExperts(torch.nn.Module):
    """vLLM ``RoutedExperts`` stand-in: fused bf16 expert weights and a per-expert weight loader."""

    def __init__(self, quant_method, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        w13 = torch.randn(E, 2 * INTER, HID, generator=g).bfloat16()
        w2 = torch.randn(E, HID, INTER, generator=g).bfloat16()
        self.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
        self.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
        for param in (self.w13_weight, self.w2_weight):
            param.weight_loader = self.weight_loader
        self.quant_method = quant_method

    def weight_loader(self, param, loaded_weight, expert_id):
        param.data[expert_id].copy_(loaded_weight)


def _engine_init(vllm, layers):
    """Model init as vLLM runs it: record the checkpoint layout, then load and run the kernel prep."""
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList(layers)
    for layer in layers:
        params = {name: getattr(layer, name).data.to("meta") for name in ("w13_weight", "w2_weight")}
        vllm.layerwise_info[layer] = types.SimpleNamespace(restore_metadata=(params, {}))
        layer.quant_method.process_weights_after_loading(layer)
    return model


def _new_weights(seed):
    g = torch.Generator().manual_seed(seed)
    return {
        "w13_weight": torch.randn(E, 2 * INTER, HID, generator=g).bfloat16(),
        "w2_weight": torch.randn(E, HID, INTER, generator=g).bfloat16(),
    }


def _load(layer, weights):
    """The refit's per-expert loads, through whatever parameter is on the layer (``params_dict``)."""
    for name, tensor in weights.items():
        param = getattr(layer, name)
        for expert_id in range(E):
            param.weight_loader(param, tensor[expert_id], expert_id)


def test_block_layout_is_staged_as_a_view_and_folded_back_into_the_same_storage(vllm):
    layer = _FakeRoutedExperts(vllm.unquantized.UnquantizedFusedMoEMethod())
    model = _engine_init(vllm, [layer])
    live = {name: getattr(layer, name) for name in ("w13_weight", "w2_weight")}
    assert live["w13_weight"].shape == (E, HID // BLOCK, 2 * INTER, BLOCK)

    staged = unquant_utils.stage_unquantized_moe_params(model)

    assert staged == [layer]
    checkpoint, _ = vllm.layerwise_info[layer].restore_metadata
    for name, param in live.items():
        view = getattr(layer, name)
        assert view is not param
        assert view.shape == checkpoint[name].shape
        assert view.data_ptr() == param.data_ptr(), "the staging view must reuse the live storage"
        assert view.weight_loader == param.weight_loader

    weights = _new_weights(seed=1)
    _load(layer, weights)
    # The loads went straight into the live storage, in checkpoint layout.
    for name, param in live.items():
        assert torch.equal(param.data.reshape(-1), weights[name].reshape(-1))

    with unquant_utils.fold_unquantized_moe_params(staged):
        layer.quant_method.process_weights_after_loading(layer)

    assert vllm.unquantized.replace_parameter is _replace_parameter
    assert not hasattr(layer, unquant_utils._LIVE_ATTR)
    for name, param in live.items():
        assert getattr(layer, name) is param, "the live parameter object must be back on the layer"
        assert torch.equal(param.data, _block_layout(weights[name]))


def test_each_layer_is_folded_before_the_next_is_processed(vllm):
    """The re-derived weights land in the live storage inside the layer's own post-load hook, so its
    temporary has no owner left by the time the next layer allocates one: one layer's worth at a time."""
    layers = [_FakeRoutedExperts(vllm.unquantized.UnquantizedFusedMoEMethod(), seed=i) for i in range(3)]
    model = _engine_init(vllm, layers)
    live = [{name: getattr(layer, name) for name in ("w13_weight", "w2_weight")} for layer in layers]
    staged = unquant_utils.stage_unquantized_moe_params(model)
    weights = [_new_weights(seed=10 + i) for i in range(len(layers))]
    for layer, new in zip(layers, weights, strict=True):
        _load(layer, new)

    # vLLM's loop runs every module's post-load hook inside the one fold.
    with unquant_utils.fold_unquantized_moe_params(staged):
        for i, layer in enumerate(layers):
            layer.quant_method.process_weights_after_loading(layer)
            for name, param in live[i].items():
                assert getattr(layer, name) is param, f"layer {i} was not folded before the next layer"
                assert torch.equal(param.data, _block_layout(weights[i][name]))


def test_layout_preserving_backends_are_not_staged(vllm):
    """Triton keeps the checkpoint layout; FlashInfer CUTLASS swaps gate/up in place of the same shape."""
    swap = lambda t: t.reshape(t.shape[0], 2, t.shape[1] // 2, t.shape[2]).flip(1).reshape(t.shape)  # noqa: E731
    layers = [
        _FakeRoutedExperts(vllm.unquantized.UnquantizedFusedMoEMethod(kernel_layout=lambda t: t.clone())),
        _FakeRoutedExperts(vllm.unquantized.UnquantizedFusedMoEMethod(kernel_layout=swap)),
    ]
    model = _engine_init(vllm, layers)
    before = [(layer.w13_weight, layer.w2_weight) for layer in layers]

    assert unquant_utils.stage_unquantized_moe_params(model) == []
    assert [(layer.w13_weight, layer.w2_weight) for layer in layers] == before


def test_layers_without_recorded_metadata_or_another_quant_method_are_not_staged(vllm):
    unrecorded = _FakeRoutedExperts(vllm.unquantized.UnquantizedFusedMoEMethod())
    unrecorded.quant_method.process_weights_after_loading(unrecorded)  # no metadata recorded
    other = _FakeRoutedExperts(vllm.unquantized.UnquantizedFusedMoEMethod())
    model = _engine_init(vllm, [other])
    other.quant_method = object()
    model.unrecorded = unrecorded

    assert unquant_utils.stage_unquantized_moe_params(model) == []


def test_a_padded_kernel_layout_is_staged_in_the_live_storage(vllm):
    layer = _FakeRoutedExperts(vllm.unquantized.UnquantizedFusedMoEMethod(kernel_layout=_padded_block_layout))
    model = _engine_init(vllm, [layer])
    live = {name: getattr(layer, name) for name in ("w13_weight", "w2_weight")}
    assert live["w13_weight"].numel() > 2 * INTER * HID * E

    staged = unquant_utils.stage_unquantized_moe_params(model)

    for name, param in live.items():
        view = getattr(layer, name)
        assert view.dim() == 3 and view.data_ptr() == param.data_ptr(), "the padded live storage has room"
    weights = _new_weights(seed=2)
    _load(layer, weights)
    with unquant_utils.fold_unquantized_moe_params(staged):
        layer.quant_method.process_weights_after_loading(layer)
    for name, param in live.items():
        assert getattr(layer, name) is param
        assert torch.equal(param.data, _padded_block_layout(weights[name]))


def test_a_smaller_kernel_layout_gets_a_fresh_staging_buffer(vllm):
    layer = _FakeRoutedExperts(vllm.unquantized.UnquantizedFusedMoEMethod(kernel_layout=_halved_layout))
    model = _engine_init(vllm, [layer])
    live = layer.w13_weight

    staged = unquant_utils.stage_unquantized_moe_params(model)

    view = layer.w13_weight
    assert view.shape == (E, 2 * INTER, HID) and view.data_ptr() != live.data_ptr()
    assert torch.isnan(view.data).all(), "an unwritten staging buffer must not look like real weights"
    weights = _new_weights(seed=3)
    _load(layer, weights)
    with unquant_utils.fold_unquantized_moe_params(staged):
        layer.quant_method.process_weights_after_loading(layer)
    assert layer.w13_weight is live
    assert torch.equal(live.data, _halved_layout(weights["w13_weight"]))


def test_a_staged_weight_that_was_not_re_derived_fails_loudly(vllm):
    layer = _FakeRoutedExperts(vllm.unquantized.UnquantizedFusedMoEMethod())
    model = _engine_init(vllm, [layer])
    staged = unquant_utils.stage_unquantized_moe_params(model)
    _load(layer, _new_weights(seed=3))
    layer.quant_method.names = ("w13_weight",)  # the update path re-derives w13 only

    with pytest.raises(RuntimeError, match="w2_weight was not re-derived"):
        with unquant_utils.fold_unquantized_moe_params(staged):
            layer.quant_method.process_weights_after_loading(layer)


def _assert_record_is(layer, live):
    stash = getattr(layer, unquant_utils._LIVE_ATTR)
    assert list(stash) == list(live) and all(stash[name] is param for name, param in live.items())


def _refit_restores(layer, model, live, seed):
    """The next refit restages from the record and lands the new weights in the live storage."""
    staged = unquant_utils.stage_unquantized_moe_params(model)
    weights = _new_weights(seed=seed)
    _load(layer, weights)
    with unquant_utils.fold_unquantized_moe_params(staged):
        layer.quant_method.process_weights_after_loading(layer)
    for name, param in live.items():
        assert getattr(layer, name) is param
        assert torch.equal(param.data, _block_layout(weights[name]))


def test_a_failed_fold_keeps_the_record_the_next_refit_restores_from(vllm):
    layer = _FakeRoutedExperts(vllm.unquantized.UnquantizedFusedMoEMethod())
    model = _engine_init(vllm, [layer])
    live = {name: getattr(layer, name) for name in ("w13_weight", "w2_weight")}
    staged = unquant_utils.stage_unquantized_moe_params(model)
    _load(layer, _new_weights(seed=6))
    layer.quant_method.kernel_layout = lambda raw: _block_layout(raw).flatten(1)  # re-derived in the wrong shape

    with pytest.raises(RuntimeError):
        with unquant_utils.fold_unquantized_moe_params(staged):
            layer.quant_method.process_weights_after_loading(layer)

    _assert_record_is(layer, live)
    layer.quant_method.kernel_layout = _block_layout
    _refit_restores(layer, model, live, seed=7)


def test_a_staging_failure_part_way_keeps_the_record_the_next_refit_restores_from(vllm, monkeypatch):
    layer = _FakeRoutedExperts(vllm.unquantized.UnquantizedFusedMoEMethod())
    model = _engine_init(vllm, [layer])
    live = {name: getattr(layer, name) for name in ("w13_weight", "w2_weight")}
    real, calls = unquant_utils._staging_data, []

    def fails_on_the_second_weight(param, shape, dtype):
        calls.append(shape)
        if len(calls) == 2:
            raise RuntimeError("CUDA out of memory")
        return real(param, shape, dtype)

    monkeypatch.setattr(unquant_utils, "_staging_data", fails_on_the_second_weight)
    with pytest.raises(RuntimeError):
        unquant_utils.stage_unquantized_moe_params(model)

    assert layer.w13_weight is not live["w13_weight"]  # already swapped for its view
    _assert_record_is(layer, {"w13_weight": live["w13_weight"]})
    monkeypatch.setattr(unquant_utils, "_staging_data", real)
    _refit_restores(layer, model, live, seed=8)


def test_a_layer_left_staged_by_a_failed_refit_is_restaged_from_its_live_params(vllm):
    layer = _FakeRoutedExperts(vllm.unquantized.UnquantizedFusedMoEMethod())
    model = _engine_init(vllm, [layer])
    live = {name: getattr(layer, name) for name in ("w13_weight", "w2_weight")}
    unquant_utils.stage_unquantized_moe_params(model)
    _load(layer, _new_weights(seed=4))
    # The fold got as far as w13 before the refit failed.
    layer.w13_weight = live["w13_weight"]
    del getattr(layer, unquant_utils._LIVE_ATTR)["w13_weight"]

    staged = unquant_utils.stage_unquantized_moe_params(model)  # the next refit

    assert staged == [layer]
    for name, param in live.items():
        assert getattr(layer, name).shape != param.shape and getattr(layer, name).data_ptr() == param.data_ptr()
    weights = _new_weights(seed=5)
    _load(layer, weights)
    with unquant_utils.fold_unquantized_moe_params(staged):
        layer.quant_method.process_weights_after_loading(layer)
    for name, param in live.items():
        assert getattr(layer, name) is param
        assert torch.equal(param.data, _block_layout(weights[name]))
