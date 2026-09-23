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
"""CPU tests for the vLLM MXFP8 (ModelOpt) rollout path in ``verl/utils/vllm`` (vLLM stubbed).

Three things the MXFP8 additions to ``vllm_quant_utils.py`` / ``vllm_fp8_utils.py`` have to get
right, each checked here without a GPU or a real vLLM:

1. ``is_fp8_model`` / ``is_mxfp8_vllm_cuda`` recognise vLLM's ``ModelOptMxFp8Config`` so the
   weight sync takes the quantize + stage/reprocess path instead of the plain bf16 path.
2. ``quant_weights`` under that config goes through ``mxfp8_quantize`` and yields the scale under
   the ModelOpt name ``<weight>_scale`` (blockwise fp8 uses ``_scale_inv``).
3. ``build_fp8_method_patchers`` (vLLM >= 0.20) wraps the two ModelOpt MXFP8 quant methods, so a
   layer whose kernel rewrites ``weight_scale`` at load records its checkpoint layout and can be
   staged / re-processed on refit exactly like the blockwise fp8 layers. It also registers the two
   opt-in exclusion patchers (``ModelOptMxFp8Config.from_config`` / ``is_layer_excluded``), so six
   patchers in total.
"""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
from packaging import version

_HERE = Path(__file__).resolve().parent
_sibling_spec = importlib.util.spec_from_file_location(
    "_vllm_quant_utils_moe_test_helpers", _HERE / "test_vllm_quant_utils_moe_on_cpu.py"
)
_helpers = importlib.util.module_from_spec(_sibling_spec)
assert _sibling_spec is not None and _sibling_spec.loader is not None
_sibling_spec.loader.exec_module(_helpers)


class _FakeQuantConfig:
    weight_block_size = [128, 128]


class _StubVllm:
    """Install stub ``vllm.model_executor.layers.quantization.{fp8,modelopt}`` modules."""

    NAMES = (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.quantization",
        "vllm.model_executor.layers.quantization.fp8",
        "vllm.model_executor.layers.quantization.modelopt",
    )

    def __enter__(self):
        self.saved = {n: sys.modules.get(n) for n in self.NAMES}
        mods = {n: types.ModuleType(n) for n in self.NAMES}
        fp8 = mods["vllm.model_executor.layers.quantization.fp8"]
        modelopt = mods["vllm.model_executor.layers.quantization.modelopt"]

        class Fp8Config(_FakeQuantConfig):
            pass

        class ModelOptMxFp8Config(_FakeQuantConfig):
            # The exclusion patchers wrap these two methods on the class, so the stub must carry them.
            exclude_modules: list = []
            packed_modules_mapping: dict = {}

            @classmethod
            def from_config(cls, config):
                return cls()

            def is_layer_excluded(self, prefix):
                return False

        def _noop_process(self, layer):
            pass

        def _swizzle_process(self, layer):
            # Mimic the CUDA kernel post-processing: the checkpoint-layout [n, k/32] uint8 scale
            # is rewritten in place into a differently shaped inference layout.
            scale = layer.weight_scale
            layer.weight_scale = torch.nn.Parameter(scale.data.reshape(-1).clone() + 1, requires_grad=False)

        def _apply(self, layer, x, bias=None):
            # The kernel reads the swizzled copy; "+1" is the fake swizzle, so undo it for the math.
            from verl.utils.mxfp8_refit_check import mxfp8_dequantize

            n, k = layer.weight.shape
            scale = layer.weight_scale.data.reshape(n, k // 32) - 1
            return (x.float() @ mxfp8_dequantize(layer.weight.data, scale).t()).to(torch.bfloat16)

        def replace_parameter(layer, name, new):
            setattr(layer, name, torch.nn.Parameter(new, requires_grad=False))

        fp8.Fp8Config = Fp8Config
        fp8.Fp8LinearMethod = type("Fp8LinearMethod", (), {"process_weights_after_loading": _noop_process})
        fp8.Fp8MoEMethod = type("Fp8MoEMethod", (), {"process_weights_after_loading": _noop_process})
        fp8.replace_parameter = replace_parameter
        modelopt.replace_parameter = replace_parameter  # ModelOpt binds its own copy at import, like vLLM
        modelopt.ModelOptMxFp8Config = ModelOptMxFp8Config
        modelopt.ModelOptMxFp8LinearMethod = type(
            "ModelOptMxFp8LinearMethod", (), {"process_weights_after_loading": _swizzle_process, "apply": _apply}
        )

        def _moe_process(self, layer):
            # vLLM 0.24: processes once, then returns early forever (the flag verl must clear on refit).
            if getattr(layer, "_already_called_process_weights_after_loading", False):
                return
            layer._already_called_process_weights_after_loading = True
            # the copy the kernel reads is derived from the canonical scales present now ...
            layer._kernel_scales = (layer.w13_weight_scale.data.clone(), layer.w2_weight_scale.data.clone())
            # ... and the parameters are rewritten into a differently shaped inference layout
            for n in ("w13_weight_scale", "w2_weight_scale"):
                p = getattr(layer, n)
                setattr(layer, n, torch.nn.Parameter(p.data.reshape(p.shape[0], -1).clone() + 1, requires_grad=False))

        modelopt.ModelOptMxFp8FusedMoE = type(
            "ModelOptMxFp8FusedMoE",
            (),
            {
                "process_weights_after_loading": _moe_process,
                "is_monolithic": False,
                "topk_indices_dtype": None,
                "moe_kernel": None,
            },
        )
        sys.modules.update(mods)
        self.fp8, self.modelopt = fp8, modelopt
        return self

    def __exit__(self, *exc):
        for n, prev in self.saved.items():
            if prev is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = prev


def test_is_fp8_model_recognises_modelopt_mxfp8_config():
    module, _ = _helpers._load_quant_utils(fused_moe_is_function=True)
    with _StubVllm() as stub:
        mx = stub.modelopt.ModelOptMxFp8Config()
        assert module.is_mxfp8_vllm_cuda(mx)
        assert module.is_fp8_model(SimpleNamespace(quant_config=mx))
        assert module.is_fp8_model(SimpleNamespace(quant_config=stub.fp8.Fp8Config()))
        assert not module.is_mxfp8_vllm_cuda(stub.fp8.Fp8Config())
        assert not module.is_fp8_model(SimpleNamespace(quant_config=object()))


def test_quant_weights_mxfp8_cuda_uses_te_quantizer_and_modelopt_scale_name(monkeypatch):
    module, ns = _helpers._load_quant_utils(fused_moe_is_function=True)
    model = _helpers._build_model(ns)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    calls = []

    def fake_mxfp8_quantize(w):
        calls.append(tuple(w.shape))
        return w.to(torch.float8_e4m3fn), torch.zeros(w.shape[0], w.shape[1] // 32, dtype=torch.uint8)

    fake_mod = types.ModuleType("verl.utils.mxfp8_quant")
    fake_mod.mxfp8_quantize = fake_mxfp8_quantize
    monkeypatch.setitem(sys.modules, "verl.utils.mxfp8_quant", fake_mod)

    weights = [
        ("model.layers.0.self_attn.q_proj.weight", torch.randn(8, 64, dtype=torch.bfloat16)),
        ("model.layers.0.mlp.gate.weight", torch.randn(8, 64, dtype=torch.bfloat16)),  # router: bf16, untouched
    ]
    with _StubVllm() as stub:
        out = list(module.quant_weights(iter(weights), model, stub.modelopt.ModelOptMxFp8Config()))

    names = [n for n, _ in out]
    assert names == [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.weight_scale",  # ModelOpt name, not "_scale_inv"
        "model.layers.0.mlp.gate.weight",
    ]
    assert calls == [(8, 64)]
    assert out[0][1].dtype == torch.float8_e4m3fn
    assert out[1][1].dtype == torch.uint8 and tuple(out[1][1].shape) == (8, 2)
    assert out[2][1].dtype == torch.bfloat16


def test_modelopt_mxfp8_methods_are_patched_and_survive_a_refit():
    module, _ = _helpers._load_quant_utils(fused_moe_is_function=True)
    with _StubVllm() as stub:
        patchers = module.build_fp8_method_patchers(version.parse("0.24.0"))
        targets = {p.attribute for p in patchers}
        assert len(patchers) == 6 and targets == {"process_weights_after_loading", "from_config", "is_layer_excluded"}
        for p in patchers:
            p.start()
        try:
            layer = torch.nn.Module()
            layer.weight = torch.nn.Parameter(torch.zeros(4, 64, dtype=torch.float8_e4m3fn), requires_grad=False)
            layer.weight_scale = torch.nn.Parameter(torch.zeros(4, 2, dtype=torch.uint8), requires_grad=False)
            layer.quant_method = stub.modelopt.ModelOptMxFp8LinearMethod()

            # Initial load: the wrapped hook records the checkpoint layout before the kernel
            # rewrites weight_scale into its inference layout.
            layer.quant_method.process_weights_after_loading(layer)
            assert layer._verl_fp8_pristine["weight_scale"] == ((4, 2), torch.uint8)
            assert tuple(layer.weight_scale.shape) == (8,)
            live_ptr = layer.weight_scale.data_ptr()

            # Refit: stage exposes a [4, 2] buffer for load_weights, reprocess re-derives the
            # inference layout from the new scales into the storage the CUDA graph captured.
            model = torch.nn.Module()
            model.proj = layer
            staged = module.stage_fp8_params_for_loading(model)
            assert staged == [layer] and tuple(layer.weight_scale.shape) == (4, 2)
            layer.weight_scale.data.copy_(torch.full((4, 2), 5, dtype=torch.uint8))
            module.process_fp8_weights_after_loading(staged)
            assert tuple(layer.weight_scale.shape) == (8,)
            assert layer.weight_scale.data_ptr() == live_ptr
            assert torch.equal(layer.weight_scale.data, torch.full((8,), 6, dtype=torch.uint8))
        finally:
            for p in patchers:
                p.stop()


def test_full_refit_cycle_runs_the_self_check(monkeypatch):
    module, _ = _helpers._load_quant_utils(fused_moe_is_function=True)
    monkeypatch.delenv("VERL_MXFP8_REFIT_CHECK", raising=False)
    # The MXFP4 MoE hooks import vLLM's fused-MoE oracle, which the stub cannot provide; they are
    # exercised by their own tests and are unrelated to the fp8 stage/reprocess cycle under test.
    monkeypatch.setattr(module, "stage_mxfp4_moe_params_for_loading", lambda model: [])
    monkeypatch.setattr(module, "process_mxfp4_moe_weights_after_loading", lambda modules: None)
    with _StubVllm() as stub:
        patchers = module.build_fp8_method_patchers(version.parse("0.24.0"))
        for p in patchers:
            p.start()
        try:
            layer = torch.nn.Module()
            g = torch.Generator().manual_seed(0)
            layer.weight = torch.nn.Parameter(
                torch.randint(-6, 7, (4, 64), generator=g).to(torch.float8_e4m3fn), requires_grad=False
            )
            layer.weight_scale = torch.nn.Parameter(torch.full((4, 2), 127, dtype=torch.uint8), requires_grad=False)
            layer.quant_method = stub.modelopt.ModelOptMxFp8LinearMethod()
            layer.quant_method.process_weights_after_loading(layer)  # initial load
            model = torch.nn.Module()
            model.proj = layer

            # Healthy refit: stage -> write new canonical scale -> reprocess -> self-check passes.
            state = module.prepare_quanted_weights_for_loading(model)
            layer.weight_scale.data.copy_(torch.full((4, 2), 129, dtype=torch.uint8))
            module.process_quanted_weights_after_loading(model, state)
            assert torch.equal(layer.weight_scale.data, torch.full((8,), 130, dtype=torch.uint8))

            # Broken refit: the kernel keeps an old layout -> the self-check must raise.
            state = module.prepare_quanted_weights_for_loading(model)
            layer.weight_scale.data.copy_(torch.full((4, 2), 125, dtype=torch.uint8))

            def _stale_process(self, lyr):  # post-processing that derives the kernel layout from OLD scales
                lyr.weight_scale = torch.nn.Parameter(torch.full((8,), 130, dtype=torch.uint8), requires_grad=False)

            monkeypatch.setattr(type(layer.quant_method), "process_weights_after_loading", _stale_process)
            try:
                module.process_quanted_weights_after_loading(model, state)
            except RuntimeError as e:
                assert "refit self-check failed" in str(e)
            else:
                raise AssertionError("expected the self-check to fail")
        finally:
            for p in patchers:
                p.stop()


E, INTER, HID = 2, 32, 64


class _FakeRoutedExperts(torch.nn.Module):
    """vLLM 0.24 RoutedExperts stand-in: fused expert weights + the modular forward the runner calls."""

    def __init__(self, quant_method, tp=1, ep=1, dp=1, activation="silu"):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.w13_weight = torch.nn.Parameter(
            torch.randint(-6, 7, (E, 2 * INTER, HID), generator=g).to(torch.float8_e4m3fn), requires_grad=False
        )
        self.w2_weight = torch.nn.Parameter(
            torch.randint(-6, 7, (E, HID, INTER), generator=g).to(torch.float8_e4m3fn), requires_grad=False
        )
        self.w13_weight_scale = torch.nn.Parameter(
            torch.full((E, 2 * INTER, HID // 32), 127, dtype=torch.uint8), requires_grad=False
        )
        self.w2_weight_scale = torch.nn.Parameter(
            torch.full((E, HID, INTER // 32), 127, dtype=torch.uint8), requires_grad=False
        )
        self.quant_method = quant_method
        self.moe_config = SimpleNamespace(
            moe_parallel_config=SimpleNamespace(tp_size=tp, ep_size=ep, dp_size=dp, use_ep=ep > 1)
        )
        self.top_k = 2
        self.activation = activation

    def forward_modular(self, x, topk_weights, topk_ids, shared_experts=None, shared_experts_input=None):
        from verl.utils.mxfp8_refit_check import mxfp8_moe_expert_reference

        s13, s2 = self._kernel_scales
        out = torch.zeros(x.shape[0], HID)
        for j in range(topk_ids.shape[1]):
            for r in range(x.shape[0]):
                e = int(topk_ids[r, j])
                w = topk_weights[r, j]
                if w != 0:
                    out[r] += (
                        w
                        * mxfp8_moe_expert_reference(
                            x[r : r + 1], self.w13_weight.data[e], s13[e], self.w2_weight.data[e], s2[e]
                        )[0]
                    )
        return out.to(torch.bfloat16)


def _moe_model(module, stub, **kw):
    layer = _FakeRoutedExperts(stub.modelopt.ModelOptMxFp8FusedMoE(), **kw)
    layer.quant_method.process_weights_after_loading(layer)  # initial load: records pristine, swizzles, sets the flag
    model = torch.nn.Module()
    model.experts = layer
    return model, layer


def test_moe_refit_cycle_clears_the_process_once_flag_and_passes_the_expert_probe(monkeypatch):
    module, _ = _helpers._load_quant_utils(fused_moe_is_function=True)
    monkeypatch.delenv("VERL_MXFP8_REFIT_CHECK", raising=False)
    monkeypatch.setattr(module, "stage_mxfp4_moe_params_for_loading", lambda model: [])
    monkeypatch.setattr(module, "process_mxfp4_moe_weights_after_loading", lambda modules: None)
    with _StubVllm() as stub:
        patchers = module.build_fp8_method_patchers(version.parse("0.24.0"))
        for p in patchers:
            p.start()
        try:
            model, layer = _moe_model(module, stub)
            assert layer._already_called_process_weights_after_loading  # vLLM's guard is armed after the initial load
            state = module.prepare_quanted_weights_for_loading(model)
            assert tuple(layer.w13_weight_scale.shape) == (E, 2 * INTER, HID // 32)  # staged back to canonical
            layer.w13_weight_scale.data.fill_(129)  # the sync ships 4x scales
            layer.w2_weight_scale.data.fill_(129)
            module.process_quanted_weights_after_loading(model, state)
            # Without clearing the flag vLLM's hook would have returned early and the kernel copy would still
            # hold the 127s (and the probe would have failed); with it the copy follows the new scales.
            assert torch.equal(layer._kernel_scales[0], torch.full((E, 2 * INTER, HID // 32), 129, dtype=torch.uint8))
        finally:
            for p in patchers:
                p.stop()


def test_moe_expert_probe_catches_a_kernel_copy_that_was_not_re_derived(monkeypatch):
    module, _ = _helpers._load_quant_utils(fused_moe_is_function=True)
    monkeypatch.delenv("VERL_MXFP8_REFIT_CHECK", raising=False)
    monkeypatch.setattr(module, "stage_mxfp4_moe_params_for_loading", lambda model: [])
    monkeypatch.setattr(module, "process_mxfp4_moe_weights_after_loading", lambda modules: None)
    with _StubVllm() as stub:
        patchers = module.build_fp8_method_patchers(version.parse("0.24.0"))
        for p in patchers:
            p.start()
        try:
            model, layer = _moe_model(module, stub)
            state = module.prepare_quanted_weights_for_loading(model)
            layer.w13_weight_scale.data.fill_(129)
            layer.w2_weight_scale.data.fill_(129)

            def _stale(self, lyr):  # re-derives the layout but from the OLD kernel copy: scales stay 127
                for n in ("w13_weight_scale", "w2_weight_scale"):
                    p = getattr(lyr, n)
                    setattr(lyr, n, torch.nn.Parameter(p.data.reshape(p.shape[0], -1).clone() + 1, requires_grad=False))

            monkeypatch.setattr(type(layer.quant_method), "process_weights_after_loading", _stale)
            try:
                module.process_quanted_weights_after_loading(model, state)
            except RuntimeError as e:
                assert "vllm MoE layer 'experts', expert 0" in str(e)
            else:
                raise AssertionError("expected the MoE expert probe to fail on a stale kernel copy")
        finally:
            for p in patchers:
                p.stop()


def test_moe_expert_probe_is_skipped_where_it_cannot_reproduce_the_layer(monkeypatch):
    module, _ = _helpers._load_quant_utils(fused_moe_is_function=True)
    monkeypatch.delenv("VERL_MXFP8_REFIT_CHECK", raising=False)
    with _StubVllm() as stub:
        for kw in ({"ep": 2}, {"dp": 2}, {"activation": "swigluoai"}):
            layer = _FakeRoutedExperts(stub.modelopt.ModelOptMxFp8FusedMoE(), **kw)
            model = torch.nn.Module()
            model.experts = layer
            assert module.snapshot_mxfp8_moe_for_check(model) is None, kw
        layer = _FakeRoutedExperts(stub.modelopt.ModelOptMxFp8FusedMoE())
        model = torch.nn.Module()
        model.experts = layer
        snap = module.snapshot_mxfp8_moe_for_check(model)
        assert snap is not None and snap[0] == "experts" and snap[2] == 0 and tuple(snap[3].shape) == (2 * INTER, HID)
        monkeypatch.setenv("VERL_MXFP8_REFIT_CHECK", "0")
        assert module.snapshot_mxfp8_moe_for_check(model) is None


def _trtllm_like_process(self, layer):
    """FlashInfer TRT-LLM MXFP8 MoE prep as vLLM 0.24 does it: W13->W31 swap + shuffle returned through
    ``replace_parameter`` as NEW tensors with the checkpoint's shape and dtype (no ``is_shuffled`` mark)."""
    from vllm.model_executor.layers.quantization import modelopt

    if getattr(layer, "_already_called_process_weights_after_loading", False):
        return
    layer._already_called_process_weights_after_loading = True
    layer._process_calls = getattr(layer, "_process_calls", 0) + 1
    for n in ("w13_weight", "w13_weight_scale"):
        modelopt.replace_parameter(layer, n, getattr(layer, n).data.flip(1).clone())
    # the kernel snapshot vLLM rebuilds right after the replace calls (moe_quant_config / moe_kernel)
    layer._kernel_refs = (layer.w13_weight, layer.w13_weight_scale)


def _identity_process(self, layer):
    """A backend that leaves the checkpoint layout alone (Triton / vLLM-CUTLASS): replace with the same tensor."""
    from vllm.model_executor.layers.quantization import modelopt

    for n in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
        modelopt.replace_parameter(layer, n, getattr(layer, n))


def test_layout_preserving_moe_repack_is_restaged_and_reprocessed():
    """B200 2026-09-21: the TRT-LLM MXFP8 MoE layer kept its shape/dtype through post-processing, was never
    staged, and the kernel read the synced canonical expert weights as shuffled (MoE probe rel err 1.739)."""
    module, _ = _helpers._load_quant_utils(fused_moe_is_function=True)
    with _StubVllm() as stub:
        stub.modelopt.ModelOptMxFp8FusedMoE.process_weights_after_loading = _trtllm_like_process
        patchers = module.build_fp8_method_patchers(version.parse("0.24.0"))
        for p in patchers:
            p.start()
        try:
            layer = _FakeRoutedExperts(stub.modelopt.ModelOptMxFp8FusedMoE())
            canonical = layer.w13_weight.data.clone()
            layer.quant_method.process_weights_after_loading(layer)  # initial load
            assert torch.equal(layer.w13_weight.data, canonical.flip(1))
            assert tuple(layer.w13_weight.shape) == (E, 2 * INTER, HID)  # shape/dtype unchanged: the blind spot
            assert layer._verl_fp8_repacked == {"w13_weight", "w13_weight_scale"}
            live_w, live_s = layer.w13_weight, layer.w13_weight_scale
            model = torch.nn.Module()
            model.experts = layer

            staged = module.stage_fp8_params_for_loading(model)
            assert staged == [layer], "a layout-preserving repack must still be staged"
            fresh = torch.randint(-6, 7, canonical.shape).to(torch.float8_e4m3fn)
            layer.w13_weight.data.copy_(fresh)  # what load_weights writes: checkpoint layout
            layer.w13_weight_scale.data.fill_(129)
            module.process_fp8_weights_after_loading(staged)

            assert layer._process_calls == 2, "the once-flag must not block the refit's reprocess"
            assert layer.w13_weight is live_w and layer.w13_weight.data_ptr() == live_w.data_ptr()
            assert torch.equal(layer.w13_weight.data, fresh.flip(1)), "kernel layout re-derived from the new weights"
            assert torch.equal(layer.w13_weight_scale.data, torch.full_like(live_s.data, 129))
            # the kernel rebuilt after the replace calls points at the live params, not staging buffers
            assert layer._kernel_refs[0] is live_w and layer._kernel_refs[1] is live_s
        finally:
            for p in patchers:
                p.stop()


def test_identity_moe_backend_is_not_restaged():
    module, _ = _helpers._load_quant_utils(fused_moe_is_function=True)
    with _StubVllm() as stub:
        stub.modelopt.ModelOptMxFp8FusedMoE.process_weights_after_loading = _identity_process
        patchers = module.build_fp8_method_patchers(version.parse("0.24.0"))
        for p in patchers:
            p.start()
        try:
            layer = _FakeRoutedExperts(stub.modelopt.ModelOptMxFp8FusedMoE())
            layer.quant_method.process_weights_after_loading(layer)
            assert not getattr(layer, "_verl_fp8_repacked", None)
            model = torch.nn.Module()
            model.experts = layer
            assert module.stage_fp8_params_for_loading(model) == []
        finally:
            for p in patchers:
                p.stop()
