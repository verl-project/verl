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
   staged / re-processed on refit exactly like the blockwise fp8 layers.
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
            pass

        def _noop_process(self, layer):
            pass

        def _swizzle_process(self, layer):
            # Mimic the CUDA kernel post-processing: the checkpoint-layout [n, k/32] uint8 scale
            # is rewritten in place into a differently shaped inference layout.
            scale = layer.weight_scale
            layer.weight_scale = torch.nn.Parameter(scale.data.reshape(-1).clone() + 1, requires_grad=False)

        def replace_parameter(layer, name, new):
            setattr(layer, name, torch.nn.Parameter(new, requires_grad=False))

        fp8.Fp8Config = Fp8Config
        fp8.Fp8LinearMethod = type("Fp8LinearMethod", (), {"process_weights_after_loading": _noop_process})
        fp8.Fp8MoEMethod = type("Fp8MoEMethod", (), {"process_weights_after_loading": _noop_process})
        fp8.replace_parameter = replace_parameter
        modelopt.ModelOptMxFp8Config = ModelOptMxFp8Config
        modelopt.ModelOptMxFp8LinearMethod = type(
            "ModelOptMxFp8LinearMethod", (), {"process_weights_after_loading": _swizzle_process}
        )
        modelopt.ModelOptMxFp8FusedMoE = type(
            "ModelOptMxFp8FusedMoE", (), {"process_weights_after_loading": _noop_process}
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
        assert len(patchers) == 4 and targets == {"process_weights_after_loading"}
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
