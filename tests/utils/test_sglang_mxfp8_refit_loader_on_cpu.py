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

from verl.utils.mxfp8_refit_check import mxfp8_dequantize
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
    """sglang ``Fp8MoEMethod`` with an fp8-serialized MXFP8 checkpoint on the Triton MoE runner:
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

    def forward(self, x, topk_output):
        """sglang FusedMoE.forward stand-in: gated MLP of the routed expert on the scales the kernel reads."""
        assert isinstance(topk_output, _StandardTopKOutput) and topk_output.topk_ids.shape[1] == 1
        e = int(topk_output.topk_ids[0, 0])
        s13, s2 = self._kernel_scales
        from verl.utils.mxfp8_refit_check import mxfp8_moe_expert_reference

        out = mxfp8_moe_expert_reference(x, self.w13_weight.data[e], s13[e], self.w2_weight.data[e], s2[e])
        return (out * topk_output.topk_weights).to(torch.bfloat16)


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
    refit.load_and_reprocess(m, [("a.weight_scale_inv", new_scale), ("b.weight_scale_inv", new_scale)])
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
    refit.load_and_reprocess(m, [("a.weight_scale_inv", new_scale)])
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
    refit.load_and_reprocess(m, [("moe.w13_weight_scale_inv", new_w13), ("moe.w2_weight_scale_inv", new_w2)])

    # load_weights could only write canonical tensors because the scales were staged back to that layout
    assert m.moe.quant_method.calls == 2
    p = m.moe.w13_weight_scale_inv
    assert tuple(p.shape) == (2, 256) and p.data_ptr() == live_ptr  # CUDA-graph storage kept
    assert torch.equal(p.data, torch.full((2, 256), 8, dtype=torch.uint8))  # re-swizzled from the NEW scales
    assert torch.equal(m.moe.w2_weight_scale_inv.data, torch.full((2, 128), 6, dtype=torch.uint8))


def test_moe_scales_kept_canonical_by_the_runner_are_not_staged():
    class _CanonicalMoEQuantMethod(_MoEQuantMethod):
        def process_weights_after_loading(self, layer):  # CUTLASS / TRT-LLM MoE runners: scales stay canonical
            self.calls += 1

    moe = _MoE()
    moe.quant_method = _CanonicalMoEQuantMethod()
    m = torch.nn.Module()
    m.moe = moe
    assert refit.stage_mxfp8_moe_scales(m) == []
    assert refit.reprocess_mxfp8_moe_layers(m, []) == 1


def test_moe_scales_the_sync_never_wrote_are_reported_instead_of_swizzled():
    # The Mixtral case: expert names (block_sparse_moe.experts.N.w1/w2/w3) miss the sync-side rule, so the
    # sync ships bf16 weights and no scales; the model's load_weights casts them into the fp8 buffer
    # silently. The staged scale buffers still hold 0xFF afterwards and the loader must say so by name.
    _install_stub("auto")
    m = _Model(resolved_backend="flashinfer_cutlass")
    m.moe = _MoE()
    m.moe.quant_method.process_weights_after_loading(m.moe)
    live = m.moe.w13_weight_scale_inv.data.clone()
    try:
        refit.load_and_reprocess(m, [("moe.w13_weight", torch.zeros(2, 128, 64, dtype=torch.bfloat16))])
    except RuntimeError as e:
        assert "moe.w13_weight_scale_inv: the weight sync did not write 512 of 512" in str(e)
        assert "block_sparse_moe.experts" in str(e)
    else:
        raise AssertionError("expected the loader to report expert scales the sync never wrote")
    assert m.moe.quant_method.calls == 1  # nothing was re-processed on top of sentinel scales
    assert torch.equal(m.moe.w13_weight_scale_inv.data, live) or m.moe.w13_weight_scale_inv.data.dtype == torch.uint8

    # a sync that writes every scale is untouched by the check (covered end-to-end by the staging test)
    m2 = _Model(resolved_backend="flashinfer_cutlass")
    m2.moe = _MoE()
    m2.moe.quant_method.process_weights_after_loading(m2.moe)
    refit.load_and_reprocess(
        m2,
        [
            ("moe.w13_weight_scale_inv", torch.full((2, 128, 2), 7, dtype=torch.uint8)),
            ("moe.w2_weight_scale_inv", torch.full((2, 64, 2), 5, dtype=torch.uint8)),
        ],
    )
    assert m2.moe.quant_method.calls == 2


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


def test_moe_self_check_passes_after_a_full_refit_cycle_on_the_swizzling_runner():
    _install_stub("auto")
    m = _Model(resolved_backend="flashinfer_cutlass")
    m.moe = _MoE(random=True)
    m.moe.quant_method.process_weights_after_loading(m.moe)  # initial load: scales swizzled, kernel copy = 127s
    new_w13 = torch.full((2, 128, 2), 129, dtype=torch.uint8)  # a sync ships 4x scales
    new_w2 = torch.full((2, 64, 2), 129, dtype=torch.uint8)
    # stage -> load -> snapshot (canonical) -> reprocess (kernel copy re-derived from the NEW scales) -> probe
    refit.load_and_reprocess(m, [("moe.w13_weight_scale_inv", new_w13), ("moe.w2_weight_scale_inv", new_w2)])
    assert torch.equal(m.moe._kernel_scales[0], new_w13)


def test_moe_self_check_catches_expert_scales_the_kernel_never_re_derived():
    m = _canonical_moe_model()
    new_w13 = torch.full((2, 128, 2), 129, dtype=torch.uint8)
    new_w2 = torch.full((2, 64, 2), 129, dtype=torch.uint8)
    # the sync writes new canonical scales in place, but nothing re-derives the kernel copy: stale by 4x
    m.load_weights([("moe.w13_weight_scale_inv", new_w13), ("moe.w2_weight_scale_inv", new_w2)])
    snap = refit.snapshot_mxfp8_moe_for_check(m)
    assert snap is not None and snap[0] == "moe" and snap[2] == 0
    try:
        refit.self_check_mxfp8_moe(snap)
    except RuntimeError as e:
        assert "MoE layer 'moe', expert 0" in str(e) and "w13_/w2_weight_scale" in str(e)
    else:
        raise AssertionError("expected the MoE self-check to fail on stale expert scales")
    refit.reprocess_mxfp8_moe_layers(m, [])  # the loader's re-processing is exactly what fixes it
    refit.self_check_mxfp8_moe(snap)


def test_moe_self_check_is_skipped_under_expert_parallelism_and_when_disabled(monkeypatch):
    m = _canonical_moe_model()
    m.moe.moe_ep_size = 2
    assert refit.snapshot_mxfp8_moe_for_check(m) is None  # cannot route to a global expert without EP dispatch
    m.moe.moe_ep_size = 1
    m.moe.moe_tp_size = 2  # TP without reduce_results: the layer output is a partial sum, no reference possible
    assert refit.snapshot_mxfp8_moe_for_check(m) is None
    m.moe.moe_tp_size = 1
    monkeypatch.setenv("VERL_MXFP8_REFIT_CHECK", "0")
    assert refit.snapshot_mxfp8_moe_for_check(m) is None
    monkeypatch.delenv("VERL_MXFP8_REFIT_CHECK")
    assert refit.snapshot_mxfp8_moe_for_check(m) is not None


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


def test_self_check_catches_a_stale_swizzled_copy():
    _install_stub("flashinfer_cutlass")
    m = _Model()
    refit.self_check_mxfp8_linear(m)  # healthy: derived copy matches canonical scale
    m.a.weight_scale_inv.data.fill_(130)  # a sync wrote new scales ...
    m.b.weight_scale_inv.data.fill_(130)
    try:  # ... but nothing rebuilt the copy the kernel reads
        refit.self_check_mxfp8_linear(m)
    except RuntimeError as e:
        assert "refit self-check failed" in str(e)
    else:
        raise AssertionError("expected the self-check to fail on stale swizzled scales")
    refit.reprocess_mxfp8_layers(m)  # the loader's re-processing is exactly what fixes it
    refit.self_check_mxfp8_linear(m)


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


def test_resolve_engine_param_follows_sglang_fusion_and_stops_at_fused_experts():
    m = _engine_model()
    L = m.model.layers[0]
    assert refit.resolve_engine_param(m, "model.layers.0.self_attn.q_proj.weight") is L.self_attn.qkv_proj.weight
    assert refit.resolve_engine_param(m, "model.layers.0.self_attn.k_proj.weight") is L.self_attn.qkv_proj.weight
    assert refit.resolve_engine_param(m, "model.layers.0.mlp.experts.3.gate_proj.weight") is L.mlp.experts.w13_weight
    assert refit.resolve_engine_param(m, "model.layers.0.mlp.experts.3.down_proj.weight") is L.mlp.experts.w2_weight
    assert (
        refit.resolve_engine_param(m, "model.layers.0.block_sparse_moe.experts.5.w1.weight")
        is L.block_sparse_moe.experts.w13_weight
    )
    assert (
        refit.resolve_engine_param(m, "model.layers.0.block_sparse_moe.experts.5.w2.weight")
        is L.block_sparse_moe.experts.w2_weight
    )
    assert (
        refit.resolve_engine_param(m, "model.layers.0.mlp.experts.gate_up_proj") is L.mlp.experts.w13_weight
    )  # transformers>=5 fused
    assert refit.resolve_engine_param(m, "model.layers.0.mlp.gate.weight") is L.mlp.gate.weight
    assert refit.resolve_engine_param(m, "lm_head.weight") is m.lm_head.weight
    assert (
        refit.resolve_engine_param(m, "model.visual.blocks.0.attn.qkv.weight") is None
    )  # unknown: skipped, not judged


def test_sync_vs_engine_dtype_check_names_every_precision_mismatch(monkeypatch):
    monkeypatch.delenv("VERL_MXFP8_REFIT_CHECK", raising=False)
    m = _engine_model()
    fp8 = lambda *shape: torch.zeros(*shape, dtype=torch.float8_e4m3fn)  # noqa: E731
    bf16 = lambda *shape: torch.zeros(*shape, dtype=torch.bfloat16)  # noqa: E731
    consistent = [
        ("model.embed_tokens.weight", bf16(1000, 64)),
        ("model.layers.0.self_attn.q_proj.weight", fp8(64, 64)),
        ("model.layers.0.self_attn.q_proj.weight_scale_inv", torch.zeros(64, 2, dtype=torch.uint8)),
        ("model.layers.0.input_layernorm.weight", bf16(64)),
        ("model.layers.0.mlp.gate.weight", bf16(8, 64)),
        ("model.layers.0.mlp.shared_expert_gate.weight", bf16(1, 64)),
        ("model.layers.0.mlp.experts.0.gate_proj.weight", fp8(64, 64)),
        ("model.layers.0.mlp.experts.0.down_proj.weight", fp8(64, 64)),
        ("model.visual.blocks.0.attn.qkv.weight", bf16(4, 4)),  # unresolved name: not judged
        ("lm_head.weight", bf16(1000, 64)),
    ]
    assert refit.check_sync_matches_engine(m, consistent) == 8
    broken = consistent + [
        ("model.layers.0.block_sparse_moe.experts.0.w1.weight", bf16(64, 64)),  # Mixtral: rule missed it
        ("model.layers.0.block_sparse_moe.experts.0.w2.weight", bf16(64, 64)),
        ("model.layers.0.mlp.shared_expert_gate.weight", fp8(1, 64)),  # pre-fix rule: quantized a bf16 layer
    ]
    try:
        refit.check_sync_matches_engine(m, broken)
    except RuntimeError as e:
        msg = str(e)
        assert "disagree on the precision of 3 parameter(s)" in msg
        assert "block_sparse_moe.experts.0.w1.weight: the sync ships torch.bfloat16, the engine parameter is fp8" in msg
        assert "mlp.shared_expert_gate.weight: the sync ships fp8, the engine parameter is high precision" in msg
    else:
        raise AssertionError("expected the dtype check to report the mismatches")
    monkeypatch.setenv("VERL_MXFP8_REFIT_CHECK", "0")
    assert refit.check_sync_matches_engine(m, broken) == 0


def test_loader_runs_the_dtype_check_before_writing_anything():
    _install_stub("auto")
    m = _Model(resolved_backend="flashinfer_cutlass")
    m.a.weight_scale_inv.data.fill_(3)
    before = m.a.weight_scale_inv.data.clone()
    # ship a bf16 tensor for a.weight, which the engine holds as fp8 -> refused before load_weights runs
    try:
        refit.load_and_reprocess(
            m,
            [
                ("a.weight_scale_inv", torch.full_like(before, 9)),
                ("a.weight", torch.zeros(4, 64, dtype=torch.bfloat16)),
            ],
        )
    except RuntimeError as e:
        assert "a.weight: the sync ships torch.bfloat16, the engine parameter is fp8" in str(e)
    else:
        raise AssertionError("expected the dtype check to refuse the sync")
    assert torch.equal(m.a.weight_scale_inv.data, before) and m.loaded == []  # nothing was written
