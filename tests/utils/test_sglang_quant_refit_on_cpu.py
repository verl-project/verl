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
"""CPU tests for the SGLang quantized refit loader on MXFP8 layers (sglang stubbed)."""

import contextlib
import enum
import logging
import sys
import types
from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional

import pytest
import torch

from verl.utils.mxfp8_refit_check import mxfp8_dequantize
from verl.utils.sglang import sglang_fp8_utils as fp8_utils
from verl.utils.sglang import sglang_mxfp8_refit_check as refit_check
from verl.utils.sglang import sglang_quant_utils as quant_utils
from verl.utils.sglang.sglang_sync_format_check import resolve_sglang_param, sglang_sync_format_check


class _Backend:
    def __init__(self, name):
        self.name = name

    def is_flashinfer_trtllm(self):
        return self.name == "flashinfer_trtllm"

    def is_flashinfer_cutlass(self):
        return self.name == "flashinfer_cutlass"

    def is_flashinfer_cutedsl(self):
        return self.name == "flashinfer_cutedsl"

    def is_deep_gemm(self):
        return self.name == "deep_gemm"


class _StandardTopKOutput(NamedTuple):  # sglang.srt.layers.moe.topk.StandardTopKOutput
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor


@dataclass
class _TopKConfig:  # sglang.srt.layers.moe.topk.TopKConfig (the fields the probe reads)
    top_k: int
    use_grouped_topk: bool = False
    renormalize: bool = True
    num_fused_shared_experts: int = 0
    custom_routing_function: Optional[Callable] = None
    correction_bias: Optional[torch.Tensor] = None
    routed_scaling_factor: Optional[float] = None
    scoring_func: str = "softmax"


class _BypassedTopKOutput(NamedTuple):  # sglang.srt.layers.moe.topk.BypassedTopKOutput
    hidden_states: torch.Tensor
    router_logits: torch.Tensor
    topk_config: _TopKConfig


class _RoutingMethodType(enum.IntEnum):  # sglang.srt.layers.moe.utils.RoutingMethodType
    Default = 0
    Renormalize = 1
    DeepSeekV3 = 2
    Llama4 = 3
    RenormalizeNaive = 4
    TopK = 5


class _TopK(torch.nn.Module):
    """sglang ``TopK.forward_cuda``: the router logits themselves for FlashInfer TRT-LLM, a softmax top-k otherwise."""

    def __init__(self, top_k=2, runner="standard", **cfg):
        super().__init__()
        self.topk_config = _TopKConfig(top_k=top_k, **cfg)
        self.runner = runner

    def forward(self, hidden_states, router_logits):
        if self.runner == "trtllm":
            return _BypassedTopKOutput(hidden_states, router_logits, self.topk_config)
        top = torch.softmax(router_logits.float(), dim=-1).topk(self.topk_config.top_k, dim=-1)
        weights = top.values / top.values.sum(-1, keepdim=True) if self.topk_config.renormalize else top.values
        return _StandardTopKOutput(weights, top.indices.to(torch.int32), router_logits)


# The sglang stubs are installed through the running test's monkeypatch, so they are removed again when the
# test ends instead of leaking into later tests in the same process.
_STUB_PATCHES: list = []


@pytest.fixture(autouse=True)
def _undo_sglang_stubs(monkeypatch):
    _STUB_PATCHES.append(monkeypatch)
    # The self-check tests exercise the failure path; the default mode (warn) has its own tests.
    monkeypatch.setenv("VERL_MXFP8_REFIT_CHECK", "raise")
    yield
    _STUB_PATCHES.pop()


def _set_module(name, module):
    _STUB_PATCHES[-1].setitem(sys.modules, name, module)


def _install_stub(backend_name):
    mod = types.ModuleType("sglang.srt.layers.quantization.fp8_utils")
    mod.get_fp8_gemm_runner_backend = lambda: _Backend(backend_name)
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.layers",
        "sglang.srt.layers.quantization",
        "sglang.srt.layers.moe",
    ):
        if name not in sys.modules:
            _set_module(name, types.ModuleType(name))
    _set_module("sglang.srt.layers.quantization.fp8_utils", mod)
    topk = types.ModuleType("sglang.srt.layers.moe.topk")
    topk.StandardTopKOutput = _StandardTopKOutput
    topk.BypassedTopKOutput = _BypassedTopKOutput
    topk.TopK = _TopK
    _set_module("sglang.srt.layers.moe.topk", topk)
    utils = types.ModuleType("sglang.srt.layers.moe.utils")
    utils.RoutingMethodType = _RoutingMethodType
    _set_module("sglang.srt.layers.moe.utils", utils)


class _QuantMethod:
    use_mxfp8 = True
    is_checkpoint_fp8_serialized = True

    def __init__(self, resolved_backend=None):
        self.calls = 0
        # sglang >= 0.5.18 stores the resolved MXFP8 dense backend on the quant method
        if resolved_backend is not None:
            self.mxfp8_dense_backend = _Backend(resolved_backend)

    def process_weights_after_loading(self, layer):
        self.calls += 1
        layer.weight_scale_inv_swizzled = layer.weight_scale_inv.clone() + 1

    def apply(self, layer, x, bias=None):
        # A FlashInfer-style kernel reads only the derived copy; "+1" is the fake swizzle.
        scale = layer.weight_scale_inv_swizzled - 1
        return (x.float() @ mxfp8_dequantize(layer.weight.data, scale).t()).to(torch.bfloat16)


class _Linear(torch.nn.Module):
    def __init__(self, resolved_backend=None):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.weight = torch.nn.Parameter(
            torch.randint(-6, 7, (4, 64), generator=g).to(torch.float8_e4m3fn), requires_grad=False
        )
        self.weight_scale_inv = torch.nn.Parameter(torch.full((4, 2), 127, dtype=torch.uint8), requires_grad=False)
        self.quant_method = _QuantMethod(resolved_backend)
        self.weight_scale_inv_swizzled = self.weight_scale_inv.clone() + 1  # as after the initial load


class _MoEQuantMethod:
    """sglang ``Fp8MoEMethod`` with an fp8-serialized MXFP8 checkpoint on the CUTLASS MoE runner of sglang >= 0.5.16:
    post-load processing swizzles the expert scales *in place* into a different shape."""

    use_mxfp8 = True
    is_checkpoint_fp8_serialized = True

    def __init__(self):
        self.calls = 0

    def process_weights_after_loading(self, layer):
        self.calls += 1
        # the copy the kernel reads is derived from the canonical scales present at this moment
        layer._kernel_scales = (layer.w13_weight_scale_inv.data.clone(), layer.w2_weight_scale_inv.data.clone())
        for name in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
            p = getattr(layer, name)
            swizzled = p.data.reshape(p.data.shape[0], -1).clone() + 1
            if p.data.shape == swizzled.shape:
                p.data.copy_(swizzled)
            else:
                p.data = swizzled  # sglang's _copy_or_rebind on shape change


class _MoE(torch.nn.Module):
    """sglang ``FusedMoE`` stand-in. ``runner="trtllm"`` behaves like the FlashInfer TRT-LLM runner: w13 is stored as
    ``[up; gate]`` and the layer only accepts the router logits (it routes inside the kernel: top-k, then softmax)."""

    def __init__(self, experts=2, inter=64, hidden=64, random=False, runner="standard"):
        super().__init__()
        self.runner = runner
        self.use_flashinfer_trtllm_moe = runner == "trtllm"
        self.routing_method_type = _RoutingMethodType.Renormalize if runner == "trtllm" else None
        self.moe_runner_config = types.SimpleNamespace(routed_scaling_factor=None)
        self.formats_seen = []
        g = torch.Generator().manual_seed(0)
        w13 = (
            torch.randint(-6, 7, (experts, 2 * inter, hidden), generator=g)
            if random
            else torch.zeros(experts, 2 * inter, hidden)
        )
        w2 = (
            torch.randint(-6, 7, (experts, hidden, inter), generator=g)
            if random
            else torch.zeros(experts, hidden, inter)
        )
        self.w13_weight = torch.nn.Parameter(w13.to(torch.float8_e4m3fn), requires_grad=False)
        self.w2_weight = torch.nn.Parameter(w2.to(torch.float8_e4m3fn), requires_grad=False)
        self.moe_ep_size = 1
        self.moe_tp_size = 1
        self.reduce_results = False
        self.num_experts = experts
        self.w13_weight_scale_inv = torch.nn.Parameter(
            torch.full((experts, 2 * inter, hidden // 32), 127, dtype=torch.uint8), requires_grad=False
        )
        self.w2_weight_scale_inv = torch.nn.Parameter(
            torch.full((experts, hidden, inter // 32), 127, dtype=torch.uint8), requires_grad=False
        )
        self.quant_method = _MoEQuantMethod()

    def forward(self, x, topk_output):
        """sglang FusedMoE.forward stand-in: routed sum of gated-MLP experts on the scales the kernel reads."""
        from verl.utils.mxfp8_refit_check import mxfp8_moe_expert_reference

        if self.runner == "trtllm":
            assert isinstance(topk_output, _BypassedTopKOutput)  # flashinfer_trtllm.py: a bare assert in sglang
            top = topk_output.router_logits.float().topk(topk_output.topk_config.top_k, dim=-1)
            ids, weights = top.indices, torch.softmax(top.values, dim=-1)
        else:
            assert isinstance(topk_output, _StandardTopKOutput)
            ids, weights = topk_output.topk_ids, topk_output.topk_weights
        self.formats_seen.append(type(topk_output).__name__)
        s13, s2 = self._kernel_scales
        out = torch.zeros(x.shape[0], self.w2_weight.shape[1])
        for j in range(ids.shape[1]):
            for e in ids[:, j].unique().tolist():
                rows = ids[:, j] == e
                y = mxfp8_moe_expert_reference(
                    x[rows],
                    self.w13_weight.data[e],
                    s13[e],
                    self.w2_weight.data[e],
                    s2[e],
                    up_first=self.runner == "trtllm",
                )
                out[rows] += weights[rows, j : j + 1].float() * y
        return out.to(torch.bfloat16)


class _Model(torch.nn.Module):
    def __init__(self, resolved_backend=None):
        super().__init__()
        self.a = _Linear(resolved_backend)
        self.b = _Linear(resolved_backend)
        self.plain = torch.nn.Linear(2, 2)  # no quant_method → ignored
        self.loaded = []

    def load_weights(self, named_tensors):
        for name, t in named_tensors:
            self.loaded.append(name)
            module_name, param = name.rsplit(".", 1)
            getattr(getattr(self, module_name), param).data.copy_(t)


def test_cutlass_backend_reprocesses_every_mxfp8_layer_after_load():
    _install_stub("flashinfer_cutlass")
    m = _Model()
    new_scale = torch.full((4, 2), 7, dtype=torch.uint8)
    quant_utils.load_and_reprocess(
        m, [("a.weight_scale_inv", new_scale), ("b.weight_scale_inv", new_scale)] + quant_utils.finalize_tensors()
    )
    assert m.loaded == ["a.weight_scale_inv", "b.weight_scale_inv"]
    assert m.a.quant_method.calls == 1 and m.b.quant_method.calls == 1
    # swizzled copy derived from the *new* canonical scales
    assert torch.equal(m.a.weight_scale_inv_swizzled, new_scale + 1)


def test_triton_backend_is_a_noop_after_load():
    _install_stub("triton")
    m = _Model()
    n = fp8_utils.reprocess_mxfp8_layers(m)
    assert n == 0 and m.a.quant_method.calls == 0


def test_trtllm_backend_is_rejected():
    _install_stub("flashinfer_trtllm")
    m = _Model()
    try:
        fp8_utils.reprocess_mxfp8_layers(m)
    except NotImplementedError as e:
        assert "flashinfer_trtllm" in str(e)
    else:
        raise AssertionError("expected NotImplementedError")


def test_resolved_backend_on_layer_wins_over_launch_flag():
    # sglang >= 0.5.18: --fp8-gemm-backend left at auto, but the quant method resolved the
    # MXFP8 dense backend to FlashInfer CuTe-DSL (the Blackwell default). The derived copy
    # must be rebuilt even though the requested backend is not "flashinfer_cutlass".
    _install_stub("auto")
    m = _Model(resolved_backend="flashinfer_cutedsl")
    new_scale = torch.full((4, 2), 3, dtype=torch.uint8)
    quant_utils.load_and_reprocess(m, [("a.weight_scale_inv", new_scale)] + quant_utils.finalize_tensors())
    assert m.a.quant_method.calls == 1 and m.b.quant_method.calls == 1
    assert torch.equal(m.a.weight_scale_inv_swizzled, new_scale + 1)


def test_resolved_deep_gemm_backend_is_reprocessed():
    _install_stub("auto")
    m = _Model(resolved_backend="deep_gemm")
    assert fp8_utils.reprocess_mxfp8_layers(m) == 2


def test_resolved_trtllm_backend_is_rejected():
    _install_stub("auto")
    m = _Model(resolved_backend="flashinfer_trtllm")
    try:
        fp8_utils.reprocess_mxfp8_layers(m)
    except NotImplementedError as e:
        assert "flashinfer_trtllm" in str(e)
    else:
        raise AssertionError("expected NotImplementedError")


def test_moe_expert_scales_are_staged_reprocessed_and_folded_into_live_storage():
    _install_stub("auto")
    m = _Model(resolved_backend="flashinfer_cutlass")
    m.moe = _MoE()
    m.moe.quant_method.process_weights_after_loading(m.moe)  # initial load: scales now swizzled [E, 256]
    assert tuple(m.moe.w13_weight_scale_inv.shape) == (2, 256)
    live_ptr = m.moe.w13_weight_scale_inv.data_ptr()

    new_w13 = torch.full((2, 128, 2), 7, dtype=torch.uint8)  # canonical [E, N, K/32] as the sync sends it
    new_w2 = torch.full((2, 64, 2), 5, dtype=torch.uint8)
    quant_utils.load_and_reprocess(
        m, [("moe.w13_weight_scale_inv", new_w13), ("moe.w2_weight_scale_inv", new_w2)] + quant_utils.finalize_tensors()
    )

    # load_weights could only write canonical tensors because the scales were staged back to that layout
    assert m.moe.quant_method.calls == 2
    p = m.moe.w13_weight_scale_inv
    assert tuple(p.shape) == (2, 256) and p.data_ptr() == live_ptr  # CUDA-graph storage kept
    assert torch.equal(p.data, torch.full((2, 256), 8, dtype=torch.uint8))  # re-swizzled from the NEW scales
    assert torch.equal(m.moe.w2_weight_scale_inv.data, torch.full((2, 128), 6, dtype=torch.uint8))


def test_moe_scales_whose_shape_the_runner_keeps_are_not_staged():
    class _CanonicalMoEQuantMethod(_MoEQuantMethod):
        # keeps the scales' shape and dtype, as the in-place runners do (FlashInfer TRT-LLM; CUTLASS on
        # sglang <= 0.5.15); staging is decided by shape and dtype, so the bytes left here do not matter
        def process_weights_after_loading(self, layer):
            self.calls += 1

    moe = _MoE()
    moe.quant_method = _CanonicalMoEQuantMethod()
    m = torch.nn.Module()
    m.moe = moe
    assert fp8_utils.stage_mxfp8_moe_scales(m) == []
    assert fp8_utils.reprocess_mxfp8_moe_layers(m, []) == 1


def test_moe_scales_the_sync_never_wrote_are_reported_instead_of_swizzled():
    # The Mixtral case: expert names (block_sparse_moe.experts.N.w1/w2/w3) miss the sync-side rule, so the
    # sync ships bf16 weights and no scales; the model's load_weights casts them into the fp8 buffer
    # silently. The staged scale buffers still hold 0xFF afterwards and the loader must say so by name.
    _install_stub("auto")
    m = _Model(resolved_backend="flashinfer_cutlass")
    m.moe = _MoE()
    m.moe.quant_method.process_weights_after_loading(m.moe)
    live_storage = m.moe.w13_weight_scale_inv.data  # what the CUDA graph reads
    live = live_storage.clone()
    try:
        quant_utils.load_and_reprocess(
            m, [("moe.w13_weight", torch.zeros(2, 128, 64, dtype=torch.bfloat16))] + quant_utils.finalize_tensors()
        )
    except RuntimeError as e:
        assert "moe.w13_weight_scale_inv: the weight sync did not write 512 of 512" in str(e)
        assert "block_sparse_moe.experts" in str(e)
    else:
        raise AssertionError("expected the loader to report expert scales the sync never wrote")
    assert m.moe.quant_method.calls == 1  # nothing was re-processed on top of sentinel scales
    assert torch.equal(live_storage, live)  # the storage the CUDA graph reads still holds the pre-sync scales

    # a sync that writes every scale is untouched by the check (covered end-to-end by the staging test)
    m2 = _Model(resolved_backend="flashinfer_cutlass")
    m2.moe = _MoE()
    m2.moe.quant_method.process_weights_after_loading(m2.moe)
    quant_utils.load_and_reprocess(
        m2,
        [
            ("moe.w13_weight_scale_inv", torch.full((2, 128, 2), 7, dtype=torch.uint8)),
            ("moe.w2_weight_scale_inv", torch.full((2, 64, 2), 5, dtype=torch.uint8)),
        ]
        + quant_utils.finalize_tensors(),
    )
    assert m2.moe.quant_method.calls == 2


def _two_moe_model():
    """Two MoE layers on the swizzling runner after their initial load, next to the two dense layers."""
    _install_stub("auto")
    m = _Model(resolved_backend="flashinfer_cutlass")
    m.moe1, m.moe2 = _MoE(), _MoE()
    for moe in (m.moe1, m.moe2):
        moe.quant_method.process_weights_after_loading(moe)  # initial load: scales swizzled in place
    return m


def _moe_scales(name, w13=7, w2=5):
    return [
        (f"{name}.w13_weight_scale_inv", torch.full((2, 128, 2), w13, dtype=torch.uint8)),
        (f"{name}.w2_weight_scale_inv", torch.full((2, 64, 2), w2, dtype=torch.uint8)),
    ]


def test_a_sync_split_across_requests_re_derives_every_layer_once_after_the_last_bucket():
    # One sync arrives as one request per bucket. Re-deriving per request rebuilt moe2 from its still-0xFF
    # staging buffer during request 1: rejected with the check on, folded into the live buffer with it off.
    m = _two_moe_model()
    live = {name: getattr(m, name).w13_weight_scale_inv.data for name in ("moe1", "moe2")}
    dense = [("a.weight_scale_inv", torch.full((4, 2), 7, dtype=torch.uint8))]
    quant_utils.load_and_reprocess(m, _moe_scales("moe1") + dense)
    quant_utils.load_and_reprocess(m, _moe_scales("moe2"))
    # nothing is re-derived before the closing request, dense layers included
    assert m.moe1.quant_method.calls == m.moe2.quant_method.calls == 1 and m.a.quant_method.calls == 0

    quant_utils.load_and_reprocess(m, quant_utils.finalize_tensors())
    for name in ("moe1", "moe2"):
        moe = getattr(m, name)
        assert moe.quant_method.calls == 2
        assert moe.w13_weight_scale_inv.data_ptr() == live[name].data_ptr()  # CUDA-graph storage kept
        assert torch.equal(moe.w13_weight_scale_inv.data, torch.full((2, 256), 8, dtype=torch.uint8))
        assert torch.equal(moe.w2_weight_scale_inv.data, torch.full((2, 128), 6, dtype=torch.uint8))
    assert m.a.quant_method.calls == 1
    assert torch.equal(m.a.weight_scale_inv_swizzled, torch.full((4, 2), 8, dtype=torch.uint8))


def test_a_moe_layer_whose_scales_span_two_requests_is_re_derived_from_both():
    m = _two_moe_model()
    w13, w2 = _moe_scales("moe1")
    quant_utils.load_and_reprocess(m, [w13] + _moe_scales("moe2"))
    quant_utils.load_and_reprocess(m, [w2] + quant_utils.finalize_tensors())
    assert torch.equal(m.moe1.w13_weight_scale_inv.data, torch.full((2, 256), 8, dtype=torch.uint8))
    assert torch.equal(m.moe1.w2_weight_scale_inv.data, torch.full((2, 128), 6, dtype=torch.uint8))


def test_scales_a_whole_sync_never_wrote_are_reported_by_the_closing_request():
    m = _two_moe_model()
    quant_utils.load_and_reprocess(m, _moe_scales("moe1"))  # moe2's scales are not part of this sync
    try:
        quant_utils.load_and_reprocess(m, quant_utils.finalize_tensors())
    except RuntimeError as e:
        assert "moe2.w13_weight_scale_inv: the weight sync did not write 512 of 512" in str(e)
    else:
        raise AssertionError("expected the closing request to report moe2's unwritten scales")


def test_closing_request_without_a_loaded_bucket_re_derives_nothing():
    m = _two_moe_model()
    quant_utils.load_and_reprocess(m, quant_utils.finalize_tensors())
    # re-running the processing on already-derived layouts would apply it twice
    assert m.moe1.quant_method.calls == 1 and m.a.quant_method.calls == 0


def test_each_sync_stages_once_and_re_derives_once():
    m = _two_moe_model()
    for sync, value in enumerate((7, 9), start=1):
        quant_utils.load_and_reprocess(m, _moe_scales("moe1", w13=value) + _moe_scales("moe2", w13=value))
        quant_utils.load_and_reprocess(m, quant_utils.finalize_tensors())
        assert m.moe1.quant_method.calls == 1 + sync
        assert torch.equal(m.moe1.w13_weight_scale_inv.data, torch.full((2, 256), value + 1, dtype=torch.uint8))


def _live_scales(m):
    """The storage each expert scale had after the initial load: what the CUDA graph captured."""
    return {
        (name, sname): getattr(getattr(m, name), sname).data
        for name in ("moe1", "moe2")
        for sname in ("w13_weight_scale_inv", "w2_weight_scale_inv")
    }


def _assert_scales_on(m, live):
    for (name, sname), storage in live.items():
        assert getattr(getattr(m, name), sname).data_ptr() == storage.data_ptr(), f"{name}.{sname}"


def _assert_next_sync_folds_into(m, live, value=9):
    quant_utils.load_and_reprocess(m, _moe_scales("moe1", w13=value) + _moe_scales("moe2", w13=value))
    quant_utils.load_and_reprocess(m, quant_utils.finalize_tensors())
    _assert_scales_on(m, live)
    for name in ("moe1", "moe2"):
        expected = torch.full((2, 256), value + 1, dtype=torch.uint8)
        assert torch.equal(getattr(m, name).w13_weight_scale_inv.data, expected)


def test_a_bucket_that_fails_to_load_ends_the_sync_with_every_scale_on_its_live_storage():
    m = _two_moe_model()
    live = _live_scales(m)
    quant_utils.load_and_reprocess(m, _moe_scales("moe1"))  # the first request stages every expert scale
    with pytest.raises(RuntimeError):  # load_weights cannot copy a [3, 3] tensor into the scale
        quant_utils.load_and_reprocess(m, [("moe2.w13_weight_scale_inv", torch.zeros(3, 3, dtype=torch.uint8))])
    _assert_scales_on(m, live)
    assert quant_utils._SYNC_FP8_ATTR not in m.__dict__
    _assert_next_sync_folds_into(m, live)


def test_a_closing_request_that_fails_leaves_no_scale_on_its_staging_buffer():
    # The closing request used to drop the sync marker first. When the re-derivation then raised, moe2 stayed
    # on its staging buffer, whose canonical shape made the next sync skip staging it and re-derive it into a
    # new tensor, while the CUDA graph kept reading the live one.
    m = _two_moe_model()
    live = _live_scales(m)
    quant_utils.load_and_reprocess(m, _moe_scales("moe1") + _moe_scales("moe2"))

    def fail(layer):
        raise RuntimeError("post-load processing failed")

    m.moe2.quant_method.process_weights_after_loading = fail
    with pytest.raises(RuntimeError, match="post-load processing failed"):
        quant_utils.load_and_reprocess(m, quant_utils.finalize_tensors())
    del m.moe2.quant_method.process_weights_after_loading
    _assert_scales_on(m, live)
    assert quant_utils._SYNC_FP8_ATTR not in m.__dict__
    _assert_next_sync_folds_into(m, live)


def test_staging_rebinds_nothing_when_a_staging_buffer_cannot_be_allocated(monkeypatch):
    m = _two_moe_model()
    live = _live_scales(m)
    empty, allocated = torch.empty, []

    def empty_or_oom(*args, **kwargs):
        if len(allocated) == 2:  # the third staging buffer, moe2's w13 scale
            raise RuntimeError("CUDA out of memory")
        allocated.append(args)
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty_or_oom)
    with pytest.raises(RuntimeError, match="out of memory"):
        quant_utils.load_and_reprocess(m, _moe_scales("moe1"))
    monkeypatch.setattr(torch, "empty", empty)
    _assert_scales_on(m, live)  # moe1's two scales were not left on buffers no list records
    assert quant_utils._SYNC_FP8_ATTR not in m.__dict__
    _assert_next_sync_folds_into(m, live)


def _canonical_moe_model(runner="standard"):
    """A model whose MoE runner keeps scales canonical (no staging) but derives a kernel copy at load."""

    class _CanonicalRunner(_MoEQuantMethod):
        def process_weights_after_loading(self, layer):
            self.calls += 1
            layer._kernel_scales = (layer.w13_weight_scale_inv.data.clone(), layer.w2_weight_scale_inv.data.clone())

    _install_stub("auto")
    m = _Model(resolved_backend="flashinfer_cutlass")
    m.moe = _MoE(random=True, runner=runner)
    m.topk = _TopK(top_k=2, runner=runner)  # the MoE block's router, a sibling of the experts as in sglang
    m.moe.quant_method = _CanonicalRunner()
    m.moe.quant_method.process_weights_after_loading(m.moe)  # initial load
    return m


def test_moe_self_check_passes_after_a_full_refit_cycle_on_the_swizzling_runner():
    _install_stub("auto")
    m = _Model(resolved_backend="flashinfer_cutlass")
    m.moe = _MoE(random=True)
    m.topk = _TopK(top_k=2)
    m.moe.quant_method.process_weights_after_loading(m.moe)  # initial load: scales swizzled, kernel copy = 127s
    new_w13 = torch.full((2, 128, 2), 129, dtype=torch.uint8)  # a sync ships 4x scales
    new_w2 = torch.full((2, 64, 2), 129, dtype=torch.uint8)
    # stage -> load -> snapshot (canonical) -> reprocess (kernel copy re-derived from the NEW scales) -> probe
    quant_utils.load_and_reprocess(
        m, [("moe.w13_weight_scale_inv", new_w13), ("moe.w2_weight_scale_inv", new_w2)] + quant_utils.finalize_tensors()
    )
    assert torch.equal(m.moe._kernel_scales[0], new_w13)
    assert m.moe.formats_seen == ["_StandardTopKOutput"]  # the probe ran, through the model's TopK


def test_moe_self_check_catches_expert_scales_the_kernel_never_re_derived():
    m = _canonical_moe_model()
    new_w13 = torch.full((2, 128, 2), 129, dtype=torch.uint8)
    new_w2 = torch.full((2, 64, 2), 129, dtype=torch.uint8)
    # the sync writes new canonical scales in place, but nothing re-derives the kernel copy: stale by 4x
    m.load_weights([("moe.w13_weight_scale_inv", new_w13), ("moe.w2_weight_scale_inv", new_w2)])
    snap = refit_check.snapshot_mxfp8_moe_for_check(m)
    assert snap is not None and snap[0] == "moe" and snap[2] == 0
    try:
        refit_check.self_check_mxfp8_moe(snap)
    except RuntimeError as e:
        assert "MoE layer 'moe', expert 0" in str(e) and "w13_/w2_weight_scale" in str(e)
    else:
        raise AssertionError("expected the MoE self-check to fail on stale expert scales")
    fp8_utils.reprocess_mxfp8_moe_layers(m, [])  # the loader's re-processing is exactly what fixes it
    refit_check.self_check_mxfp8_moe(snap)


def test_moe_self_check_is_skipped_under_expert_parallelism_and_when_disabled(monkeypatch, caplog):
    m = _canonical_moe_model()
    refit_check._WARNED.clear()
    with caplog.at_level(logging.WARNING, logger=refit_check.logger.name):
        m.moe.moe_ep_size = 2
        assert refit_check.snapshot_mxfp8_moe_for_check(m) is None  # cannot route to a global expert without EP
        assert refit_check.snapshot_mxfp8_moe_for_check(m) is None  # the next sync
        m.moe.moe_ep_size = 1
        m.moe.moe_tp_size = 2  # TP without reduce_results: the layer returns this rank's partial sum
        assert refit_check.snapshot_mxfp8_moe_for_check(m) is None
        m.moe.moe_tp_size = 1
    msgs = [r.getMessage() for r in caplog.records if "MoE refit check skipped on moe (sglang)" in r.getMessage()]
    assert len(msgs) == 2  # a warning per reason, once
    assert "expert parallelism (moe_ep_size=2)" in msgs[0] and "TP 2 without reduce_results" in msgs[1]
    monkeypatch.setenv("VERL_MXFP8_REFIT_CHECK", "0")
    assert refit_check.snapshot_mxfp8_moe_for_check(m) is None
    monkeypatch.delenv("VERL_MXFP8_REFIT_CHECK")
    assert refit_check.snapshot_mxfp8_moe_for_check(m) is not None


def test_moe_probe_is_skipped_with_a_warning_when_the_scales_are_not_canonical(caplog):
    m = _canonical_moe_model()
    s = m.moe.w13_weight_scale_inv
    # the scales left in a kernel layout instead of the canonical [E, N, K/32]
    m.moe.w13_weight_scale_inv = torch.nn.Parameter(s.data.reshape(s.shape[0], -1), requires_grad=False)
    refit_check._WARNED.clear()
    with caplog.at_level(logging.WARNING, logger=refit_check.logger.name):
        assert refit_check.snapshot_mxfp8_moe_for_check(m) is None
        assert refit_check.snapshot_mxfp8_moe_for_check(m) is None  # the next sync
    msgs = [r.getMessage() for r in caplog.records if "MoE refit check skipped on moe (sglang)" in r.getMessage()]
    assert len(msgs) == 1 and "not the canonical uint8" in msgs[0]


def test_moe_probe_feeds_the_trtllm_runner_router_logits_and_catches_stale_scales():
    # FlashInfer TRT-LLM (sglang's default MXFP8 MoE runner) routes inside the kernel and rejects a precomputed
    # top-k; the probe goes through the model's TopK, so the layer gets logits and the probe still sees stale scales
    m = _canonical_moe_model("trtllm")
    new_w13 = torch.full((2, 128, 2), 129, dtype=torch.uint8)
    new_w2 = torch.full((2, 64, 2), 129, dtype=torch.uint8)
    m.load_weights([("moe.w13_weight_scale_inv", new_w13), ("moe.w2_weight_scale_inv", new_w2)])
    snap = refit_check.snapshot_mxfp8_moe_for_check(m)
    with pytest.raises(RuntimeError, match="MoE layer 'moe', expert 0"):
        refit_check.self_check_mxfp8_moe(snap)
    fp8_utils.reprocess_mxfp8_moe_layers(m, [])
    refit_check.self_check_mxfp8_moe(snap)
    assert set(m.moe.formats_seen) == {"_BypassedTopKOutput"}


def test_moe_probe_reference_follows_the_engines_gate_up_order():
    m = _canonical_moe_model("trtllm")  # w13 stored as [up; gate], as sglang loads it for FlashInfer TRT-LLM
    snap = refit_check.snapshot_mxfp8_moe_for_check(m)
    assert snap[-1] is True
    refit_check.self_check_mxfp8_moe(snap)
    with pytest.raises(RuntimeError, match="MoE layer 'moe', expert 0"):
        refit_check.self_check_mxfp8_moe(snap[:-1] + (False,))  # reading [up; gate] as [gate; up]


@pytest.mark.parametrize(
    "tweak, reason",
    [
        (lambda m: delattr(m, "topk"), "no single sglang TopK module next to the experts"),
        (lambda m: setattr(m.topk.topk_config, "scoring_func", "sigmoid"), "sigmoid scoring"),
        (lambda m: setattr(m.topk.topk_config, "use_grouped_topk", True), "grouped, biased or custom top-k routing"),
        (lambda m: setattr(m.topk.topk_config, "correction_bias", torch.zeros(2)), "grouped, biased or custom"),
        (lambda m: setattr(m.topk.topk_config, "num_fused_shared_experts", 1), "shared experts fused"),
        (lambda m: setattr(m.moe.moe_runner_config, "routed_scaling_factor", 2.5), "routed scaling factor 2.5"),
        (lambda m: setattr(m.moe, "routing_method_type", _RoutingMethodType.Llama4), "routing method Llama4"),
        # Fp8MoEMethod passes an unset or Default (== 0) routing method to the kernel as DeepSeekV3
        (lambda m: setattr(m.moe, "routing_method_type", None), "routing method DeepSeekV3"),
        (lambda m: setattr(m.moe, "routing_method_type", _RoutingMethodType.Default), "routing method DeepSeekV3"),
    ],
)
def test_moe_probe_is_skipped_with_a_reason_where_logits_cannot_pin_one_expert(tweak, reason, caplog):
    m = _canonical_moe_model("trtllm")
    tweak(m)
    refit_check._WARNED.clear()
    with caplog.at_level(logging.WARNING, logger=refit_check.logger.name):
        assert refit_check.snapshot_mxfp8_moe_for_check(m) is None
        assert refit_check.snapshot_mxfp8_moe_for_check(m) is None
    msgs = [r.getMessage() for r in caplog.records if "MoE refit check skipped on moe" in r.getMessage()]
    assert len(msgs) == 1 and reason in msgs[0]  # once, not every sync


@pytest.mark.parametrize("method", [_RoutingMethodType.Renormalize, _RoutingMethodType.RenormalizeNaive])
def test_moe_probe_runs_on_the_softmax_trtllm_routing_methods(method):
    m = _canonical_moe_model("trtllm")
    m.moe.routing_method_type = method
    assert refit_check.snapshot_mxfp8_moe_for_check(m) is not None


def test_reprocess_reports_a_staged_scale_that_was_not_folded_back():
    _install_stub("auto")
    m = _Model(resolved_backend="flashinfer_cutlass")
    m.moe = _MoE()
    m.moe.quant_method.process_weights_after_loading(m.moe)
    staged = fp8_utils.stage_mxfp8_moe_scales(m)
    assert len(staged) == 2  # two scale params of one module: the count is per parameter, not per module
    orphan = _MoE()  # staged entry whose module is not in the model -> never visited by reprocess
    orphan.quant_method.process_weights_after_loading(orphan)
    staged += fp8_utils.stage_mxfp8_moe_scales(torch.nn.ModuleDict({"x": orphan}))
    for module, sname, _ in staged:
        getattr(module, sname).data.fill_(7)  # "the sync wrote every scale", so only the fold-back check can fire
    try:
        fp8_utils.reprocess_mxfp8_moe_layers(m, staged)
    except RuntimeError as e:
        assert "not folded back into its live storage" in str(e)
    else:
        raise AssertionError("expected the post-condition to fire for the orphaned staged scale")


def test_self_check_catches_a_stale_swizzled_copy():
    _install_stub("flashinfer_cutlass")
    m = _Model()
    refit_check.self_check_mxfp8_linear(m)  # healthy: derived copy matches canonical scale
    m.a.weight_scale_inv.data.fill_(130)  # a sync wrote new scales ...
    m.b.weight_scale_inv.data.fill_(130)
    try:  # ... but nothing rebuilt the copy the kernel reads
        refit_check.self_check_mxfp8_linear(m)
    except RuntimeError as e:
        assert "refit self-check failed" in str(e)
    else:
        raise AssertionError("expected the self-check to fail on stale swizzled scales")
    fp8_utils.reprocess_mxfp8_layers(m)  # the loader's re-processing is exactly what fixes it
    refit_check.self_check_mxfp8_linear(m)


def test_loader_fqn_matches_function():
    mod, fn = quant_utils.REFIT_LOADER_FQN.rsplit(".", 1)
    assert mod == quant_utils.__name__ and getattr(quant_utils, fn) is quant_utils.load_and_reprocess


class UnquantizedFusedMoEMethod:  # matched by class name, as sglang's
    def __init__(self, trtllm=False):
        self.use_flashinfer_trtllm_moe = trtllm


class _BF16MoE(torch.nn.Module):
    """sglang ``FusedMoE`` under ``UnquantizedFusedMoEMethod``: a MoE layer ``ignored_layers`` keeps in BF16."""

    def __init__(self, experts=2, inter=64, hidden=64, trtllm=False, w13_shape=None):
        super().__init__()
        self.num_local_experts, self.intermediate_size_per_partition, self.hidden_size = experts, inter, hidden
        self.moe_runner_config = types.SimpleNamespace(is_gated=True, routed_scaling_factor=None)
        self.quant_method = UnquantizedFusedMoEMethod(trtllm)
        shape = w13_shape or (experts, 2 * inter, hidden)
        self.w13_weight = torch.nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16), requires_grad=False)
        w2 = torch.zeros(experts, hidden, inter, dtype=torch.bfloat16)
        self.w2_weight = torch.nn.Parameter(w2, requires_grad=False)


def test_a_bf16_moe_layer_on_the_trtllm_runner_is_refused_before_anything_is_written():
    # B200, sglang 0.5.20, MoE runner left to sglang: ignore=["model.layers.0"] on a Qwen3-MoE put layer 0's BF16
    # experts on the TRT-LLM runner, which sglang picks for every MoE layer of an MXFP8 model; the first sync then
    # failed inside FusedMoE._load_w13 on the block layout ("size of tensor a (64) must match ... (512)").
    m = _two_moe_model()
    m.moe0 = _BF16MoE(trtllm=True)
    with pytest.raises(NotImplementedError, match=r"moe0 \(w13 \(2, 128, 64\) bfloat16, FlashInfer TRT-LLM runner\)"):
        quant_utils.load_and_reprocess(m, _moe_scales("moe1"))
    assert m.loaded == [] and quant_utils._SYNC_FP8_ATTR not in m.__dict__  # nothing written, the sync abandoned


def test_a_bf16_moe_layer_the_runner_re_laid_out_is_refused_on_any_runner():
    m = _two_moe_model()
    m.moe0 = _BF16MoE(w13_shape=(2, 128, 8, 8))
    with pytest.raises(NotImplementedError, match=r"checkpoint layout \(2, 128, 64\)"):
        quant_utils.load_and_reprocess(m, _moe_scales("moe1"))


def test_a_bf16_moe_layer_in_checkpoint_layout_is_synced():
    m = _two_moe_model()
    m.moe0 = _BF16MoE()
    assert fp8_utils.relaid_bf16_moe_layers(m) == []
    quant_utils.load_and_reprocess(m, _moe_scales("moe1"))
    assert m.loaded == ["moe1.w13_weight_scale_inv", "moe1.w2_weight_scale_inv"]


def test_quantized_moe_layers_are_not_taken_for_bf16_ones():
    class Mxfp4MoEMethod:  # packed FP4 experts do not have the BF16 checkpoint shape
        pass

    m = _two_moe_model()
    m.fp4 = _BF16MoE(w13_shape=(2, 128, 32))
    m.fp4.quant_method = Mxfp4MoEMethod()
    assert fp8_utils.relaid_bf16_moe_layers(m) == []


# ---------------------------------------------------------------------------
# The delta checkpoint engine
# ---------------------------------------------------------------------------


def _rollout(override=None):
    quantization = types.SimpleNamespace(
        enabled=override is not None, build_quantization_config=lambda path: dict(override or {})
    )
    return types.SimpleNamespace(quantization=quantization)


def _model(native=None, text_native=None):
    hf_config = types.SimpleNamespace(quantization_config=native)
    if text_native is not None:
        hf_config.text_config = types.SimpleNamespace(quantization_config=text_native)
    return types.SimpleNamespace(hf_config=hf_config, local_path="/unused")


_NATIVE_FP8 = {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [128, 128]}


@pytest.mark.parametrize(
    "rollout, model",
    [
        (_rollout({"quant_method": "compressed-tensors"}), _model()),  # rollout.quantization
        (_rollout(), _model(native=_NATIVE_FP8)),  # the checkpoint's own config, not on the refit loader
        (_rollout(), _model(text_native=_NATIVE_FP8)),  # under text_config
    ],
)
def test_the_delta_engine_is_refused_for_any_quantized_rollout(rollout, model):
    with pytest.raises(ValueError, match="delta checkpoint engine"):
        quant_utils.refuse_delta_sync_if_quantized(rollout, model)


def test_the_delta_engine_serves_a_bf16_rollout():
    quant_utils.refuse_delta_sync_if_quantized(_rollout(), _model())


def test_by_default_unwritten_expert_scales_are_logged_and_the_sync_goes_on(monkeypatch, caplog):
    monkeypatch.delenv("VERL_MXFP8_REFIT_CHECK")
    m = _two_moe_model()
    quant_utils.load_and_reprocess(m, _moe_scales("moe1"))  # moe2's scales are not part of this sync
    with caplog.at_level(logging.WARNING):
        quant_utils.load_and_reprocess(m, quant_utils.finalize_tensors())
    assert "moe2.w13_weight_scale_inv: the weight sync did not write 512 of 512" in caplog.text
    assert m.moe2.quant_method.calls == 2 and quant_utils._SYNC_FP8_ATTR not in m.__dict__  # the sync finished


# ---------------------------------------------------------------------------
# Export vs engine: the format check of the first weight sync
# ---------------------------------------------------------------------------


def _engine_model():
    """An SGLang-shaped Qwen-MoE-ish model: fused qkv / gate_up, fp8 where the engine quantized, bf16 elsewhere."""

    def lin(n, k, fp8):
        m = torch.nn.Module()
        m.weight = torch.nn.Parameter(
            torch.zeros(n, k, dtype=torch.float8_e4m3fn if fp8 else torch.bfloat16), requires_grad=False
        )
        return m

    layer = torch.nn.Module()
    layer.self_attn = torch.nn.Module()
    layer.self_attn.qkv_proj = lin(192, 64, True)
    layer.self_attn.o_proj = lin(64, 64, True)
    layer.input_layernorm = lin(64, 1, False)
    layer.mlp = torch.nn.Module()
    layer.mlp.gate = lin(8, 64, False)  # router: ReplicatedLinear(quant_config=None)
    layer.mlp.shared_expert_gate = lin(1, 64, False)
    layer.mlp.experts = _MoE(random=False)  # fp8 fused experts (w13_weight / w2_weight)
    layer.block_sparse_moe = torch.nn.Module()  # Mixtral naming, same fused experts
    layer.block_sparse_moe.experts = _MoE(random=False)
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.embed_tokens = lin(1000, 64, False)
    model.model.layers = torch.nn.ModuleList([layer])
    model.lm_head = lin(1000, 64, False)
    return model


def test_resolve_sglang_param_follows_sglang_fusion_and_stops_at_fused_experts():
    m = _engine_model()
    L = m.model.layers[0]
    resolve = resolve_sglang_param
    assert resolve(m, "model.layers.0.self_attn.q_proj.weight") is L.self_attn.qkv_proj.weight
    assert resolve(m, "model.layers.0.self_attn.k_proj.weight") is L.self_attn.qkv_proj.weight
    assert resolve(m, "model.layers.0.mlp.experts.3.gate_proj.weight") is L.mlp.experts.w13_weight
    assert resolve(m, "model.layers.0.mlp.experts.3.down_proj.weight") is L.mlp.experts.w2_weight
    mixtral = L.block_sparse_moe.experts
    assert resolve(m, "model.layers.0.block_sparse_moe.experts.5.w1.weight") is mixtral.w13_weight
    assert resolve(m, "model.layers.0.block_sparse_moe.experts.5.w2.weight") is mixtral.w2_weight
    # transformers >= 5 fused expert names
    assert resolve(m, "model.layers.0.mlp.experts.gate_up_proj") is L.mlp.experts.w13_weight
    assert resolve(m, "model.layers.0.mlp.gate.weight") is L.mlp.gate.weight
    assert resolve(m, "lm_head.weight") is m.lm_head.weight
    assert resolve(m, "model.visual.blocks.0.attn.qkv.weight") is None  # unknown: not judged
    # compressed-tensors' packed FP4 weight lands in weight_packed, or in weight on a layer the engine keeps in BF16
    L.mlp.packed = torch.nn.Module()
    L.mlp.packed.weight_packed = torch.nn.Parameter(torch.zeros(8, 32, dtype=torch.uint8), requires_grad=False)
    assert resolve(m, "model.layers.0.mlp.packed.weight_packed") is L.mlp.packed.weight_packed
    assert resolve(m, "model.layers.0.mlp.gate.weight_packed") is L.mlp.gate.weight


def _fp8(*shape):
    return torch.zeros(*shape, dtype=torch.float8_e4m3fn)


def _bf16(*shape):
    return torch.zeros(*shape, dtype=torch.bfloat16)


_CONSISTENT = [
    ("model.embed_tokens.weight", _bf16(1000, 64)),
    ("model.layers.0.self_attn.q_proj.weight", _fp8(64, 64)),
    ("model.layers.0.self_attn.q_proj.weight_scale_inv", torch.zeros(64, 2, dtype=torch.uint8)),
    ("model.layers.0.input_layernorm.weight", _bf16(64)),
    ("model.layers.0.mlp.gate.weight", _bf16(8, 64)),
    ("model.layers.0.mlp.shared_expert_gate.weight", _bf16(1, 64)),
    ("model.layers.0.mlp.experts.0.gate_proj.weight", _fp8(64, 64)),
    ("model.layers.0.mlp.experts.0.down_proj.weight", _fp8(64, 64)),
    ("model.visual.blocks.0.attn.qkv.weight", _bf16(4, 4)),  # unresolved name: not judged
    ("lm_head.weight", _bf16(1000, 64)),
]
_BROKEN = _CONSISTENT + [
    ("model.layers.0.block_sparse_moe.experts.0.w1.weight", _bf16(64, 64)),  # the exporter skipped them
    ("model.layers.0.block_sparse_moe.experts.0.w2.weight", _bf16(64, 64)),
    ("model.layers.0.mlp.shared_expert_gate.weight", _fp8(1, 64)),  # quantized a layer the engine keeps bf16
]


@contextlib.contextmanager
def _audit_log(caplog, level=logging.INFO):
    """The audit logger writes to stderr on its own (no propagation); let caplog see it."""
    log = logging.getLogger("verl.utils.quant_layer_audit")
    old = log.level
    log.addHandler(caplog.handler)
    log.setLevel(level)
    try:
        yield
    finally:
        log.setLevel(old)
        log.removeHandler(caplog.handler)


def test_sync_format_check_names_every_format_mismatch(monkeypatch):
    monkeypatch.setenv("VERL_QUANT_LAYER_AUDIT", "raise")
    m = _engine_model()
    check = sglang_sync_format_check(m)
    check.check_bucket(m, _CONSISTENT)
    assert (check.checked, check.disagreeing, check.unmapped) == (8, 0, 1)
    with pytest.raises(RuntimeError, match="disagree on the format of 3 weight") as err:
        check.check_bucket(m, _BROKEN)
    msg = str(err.value)
    assert "block_sparse_moe.experts.0.w1.weight: the sync sends bfloat16, the engine parameter is float8_e4m3fn" in msg
    assert "mlp.shared_expert_gate.weight: the sync sends float8_e4m3fn, the engine parameter is bfloat16" in msg
    assert "stopped before this bucket is written" in msg
    monkeypatch.setenv("VERL_QUANT_LAYER_AUDIT", "0")
    fresh = _engine_model()
    off = sglang_sync_format_check(fresh)
    off.check_bucket(fresh, _BROKEN)
    assert off.checked == 0


def test_by_default_a_format_mismatch_is_logged_and_the_sync_goes_on(monkeypatch, caplog):
    monkeypatch.delenv("VERL_QUANT_LAYER_AUDIT", raising=False)
    m = _engine_model()
    with _audit_log(caplog, logging.WARNING):
        sglang_sync_format_check(m).check_bucket(m, _BROKEN)
    assert "disagree on the format of 3 weight" in caplog.text and "The sync goes on" in caplog.text


def test_loader_runs_the_format_check_before_writing_anything(monkeypatch):
    monkeypatch.setenv("VERL_QUANT_LAYER_AUDIT", "raise")
    _install_stub("auto")
    m = _Model(resolved_backend="flashinfer_cutlass")
    m.a.weight_scale_inv.data.fill_(3)
    before = m.a.weight_scale_inv.data.clone()
    # a bf16 tensor for a.weight, which the engine holds in fp8: refused before load_weights runs
    request = [
        ("a.weight_scale_inv", torch.full_like(before, 9)),
        ("a.weight", torch.zeros(4, 64, dtype=torch.bfloat16)),
    ]
    with pytest.raises(RuntimeError, match="a.weight: the sync sends bfloat16, the engine parameter is float8_e4m3fn"):
        quant_utils.load_and_reprocess(m, request)
    assert torch.equal(m.a.weight_scale_inv.data, before) and m.loaded == []  # nothing was written


def _bucket(m, layer):
    # the layer's own fp8 weight (what the engine holds) and a new canonical scale
    return [
        (f"{layer}.weight", getattr(m, layer).weight.data.clone()),
        (f"{layer}.weight_scale_inv", torch.full((4, 2), 7, dtype=torch.uint8)),
    ]


def test_loader_checks_formats_on_every_bucket_of_the_first_sync_only(caplog):
    # What a sync sends follows the quantization config and the engine's parameters keep their formats, so a later
    # sync cannot disagree where the first one agreed: the loader checks the first sync only.
    _install_stub("flashinfer_cutlass")
    m = _Model()
    check = sglang_sync_format_check(m)
    with _audit_log(caplog):
        quant_utils.load_and_reprocess(m, _bucket(m, "a"))
        assert check.checked == 1 and not check.finished  # the first bucket, read before it was written
        quant_utils.load_and_reprocess(m, _bucket(m, "b") + quant_utils.finalize_tensors())
    assert check.finished and check.checked == 2  # a.weight and b.weight, one from each bucket
    assert "export vs sglang engine, first weight sync: 2 weights checked, 0 in a format other" in caplog.text
    quant_utils.load_and_reprocess(m, _bucket(m, "a"))
    quant_utils.load_and_reprocess(m, _bucket(m, "b") + quant_utils.finalize_tensors())
    assert check.checked == 2  # the second sync is not checked
    assert m.a.quant_method.calls == 2 and m.b.quant_method.calls == 2  # it still re-derives the kernel layouts


def test_a_first_sync_that_fails_is_checked_again_in_full():
    _install_stub("flashinfer_cutlass")
    m = _Model()
    check = sglang_sync_format_check(m)

    def lose_the_bucket(named_tensors):
        raise RuntimeError("bucket lost")

    m.load_weights = lose_the_bucket
    with pytest.raises(RuntimeError, match="bucket lost"):
        quant_utils.load_and_reprocess(m, _bucket(m, "a"))
    assert check.checked == 1 and not check.finished
    del m.load_weights  # back to the model's own loader
    quant_utils.load_and_reprocess(m, _bucket(m, "a") + quant_utils.finalize_tensors())
    assert check.finished and check.checked == 1  # the failed sync's count was dropped, not added to
