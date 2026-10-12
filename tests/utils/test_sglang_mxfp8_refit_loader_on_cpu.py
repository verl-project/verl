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
"""CPU tests for the SGLang MXFP8 refit loader (sglang stubbed)."""

import sys
import types
from typing import NamedTuple

import torch

from verl.workers.rollout.sglang_rollout import mxfp8_refit_loader as refit


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
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["sglang.srt.layers.quantization.fp8_utils"] = mod
    topk = types.ModuleType("sglang.srt.layers.moe.topk")
    topk.StandardTopKOutput = _StandardTopKOutput
    sys.modules["sglang.srt.layers.moe.topk"] = topk


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
    def __init__(self, experts=2, inter=64, hidden=64, random=False):
        super().__init__()
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
    refit.load_and_reprocess(
        m, [("a.weight_scale_inv", new_scale), ("b.weight_scale_inv", new_scale)] + refit.finalize_tensors()
    )
    assert m.loaded == ["a.weight_scale_inv", "b.weight_scale_inv"]
    assert m.a.quant_method.calls == 1 and m.b.quant_method.calls == 1
    # swizzled copy derived from the *new* canonical scales
    assert torch.equal(m.a.weight_scale_inv_swizzled, new_scale + 1)


def test_triton_backend_is_a_noop_after_load():
    _install_stub("triton")
    m = _Model()
    n = refit.reprocess_mxfp8_layers(m)
    assert n == 0 and m.a.quant_method.calls == 0


def test_trtllm_backend_is_rejected():
    _install_stub("flashinfer_trtllm")
    m = _Model()
    try:
        refit.reprocess_mxfp8_layers(m)
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
    refit.load_and_reprocess(m, [("a.weight_scale_inv", new_scale)] + refit.finalize_tensors())
    assert m.a.quant_method.calls == 1 and m.b.quant_method.calls == 1
    assert torch.equal(m.a.weight_scale_inv_swizzled, new_scale + 1)


def test_resolved_deep_gemm_backend_is_reprocessed():
    _install_stub("auto")
    m = _Model(resolved_backend="deep_gemm")
    assert refit.reprocess_mxfp8_layers(m) == 2


def test_resolved_trtllm_backend_is_rejected():
    _install_stub("auto")
    m = _Model(resolved_backend="flashinfer_trtllm")
    try:
        refit.reprocess_mxfp8_layers(m)
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
    refit.load_and_reprocess(
        m, [("moe.w13_weight_scale_inv", new_w13), ("moe.w2_weight_scale_inv", new_w2)] + refit.finalize_tensors()
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
    assert refit.stage_mxfp8_moe_scales(m) == []
    assert refit.reprocess_mxfp8_moe_layers(m, []) == 1


def _canonical_moe_model():
    """A model whose MoE runner keeps scales canonical (no staging) but derives a kernel copy at load."""

    class _CanonicalRunner(_MoEQuantMethod):
        def process_weights_after_loading(self, layer):
            self.calls += 1
            layer._kernel_scales = (layer.w13_weight_scale_inv.data.clone(), layer.w2_weight_scale_inv.data.clone())

    _install_stub("auto")
    m = _Model(resolved_backend="flashinfer_cutlass")
    m.moe = _MoE(random=True)
    m.moe.quant_method = _CanonicalRunner()
    m.moe.quant_method.process_weights_after_loading(m.moe)  # initial load
    return m


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
    # staging buffer during request 1 and folded that into its live buffer.
    m = _two_moe_model()
    live = {name: getattr(m, name).w13_weight_scale_inv.data for name in ("moe1", "moe2")}
    dense = [("a.weight_scale_inv", torch.full((4, 2), 7, dtype=torch.uint8))]
    refit.load_and_reprocess(m, _moe_scales("moe1") + dense)
    refit.load_and_reprocess(m, _moe_scales("moe2"))
    # nothing is re-derived before the closing request, dense layers included
    assert m.moe1.quant_method.calls == m.moe2.quant_method.calls == 1 and m.a.quant_method.calls == 0

    refit.load_and_reprocess(m, refit.finalize_tensors())
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
    refit.load_and_reprocess(m, [w13] + _moe_scales("moe2"))
    refit.load_and_reprocess(m, [w2] + refit.finalize_tensors())
    assert torch.equal(m.moe1.w13_weight_scale_inv.data, torch.full((2, 256), 8, dtype=torch.uint8))
    assert torch.equal(m.moe1.w2_weight_scale_inv.data, torch.full((2, 128), 6, dtype=torch.uint8))


def test_closing_request_without_a_loaded_bucket_re_derives_nothing():
    m = _two_moe_model()
    refit.load_and_reprocess(m, refit.finalize_tensors())
    # re-running the processing on already-derived layouts would apply it twice
    assert m.moe1.quant_method.calls == 1 and m.a.quant_method.calls == 0


def test_each_sync_stages_once_and_re_derives_once():
    m = _two_moe_model()
    for sync, value in enumerate((7, 9), start=1):
        refit.load_and_reprocess(m, _moe_scales("moe1", w13=value) + _moe_scales("moe2", w13=value))
        refit.load_and_reprocess(m, refit.finalize_tensors())
        assert m.moe1.quant_method.calls == 1 + sync
        assert torch.equal(m.moe1.w13_weight_scale_inv.data, torch.full((2, 256), value + 1, dtype=torch.uint8))


def test_reprocess_reports_a_staged_scale_that_was_not_folded_back():
    _install_stub("auto")
    m = _Model(resolved_backend="flashinfer_cutlass")
    m.moe = _MoE()
    m.moe.quant_method.process_weights_after_loading(m.moe)
    staged = refit.stage_mxfp8_moe_scales(m)
    assert len(staged) == 2  # two scale params of one module: the count is per parameter, not per module
    orphan = _MoE()  # staged entry whose module is not in the model -> never visited by reprocess
    orphan.quant_method.process_weights_after_loading(orphan)
    staged += refit.stage_mxfp8_moe_scales(torch.nn.ModuleDict({"x": orphan}))
    for module, sname, _ in staged:
        getattr(module, sname).data.fill_(7)  # "the sync wrote every scale", so only the fold-back check can fire
    try:
        refit.reprocess_mxfp8_moe_layers(m, staged)
    except RuntimeError as e:
        assert "not folded back into its live storage" in str(e)
    else:
        raise AssertionError("expected the post-condition to fire for the orphaned staged scale")


def test_loader_fqn_matches_function():
    mod, fn = refit.LOADER_FQN.rsplit(".", 1)
    assert mod == refit.__name__ and getattr(refit, fn) is refit.load_and_reprocess


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
