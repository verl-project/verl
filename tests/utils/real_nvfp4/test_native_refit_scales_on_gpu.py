# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""GPU lifecycle regression using installed vLLM methods, not a full-model test."""

from types import MethodType, SimpleNamespace

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph regression requires a GPU")
def test_actual_setup_kernel_keeps_current_scales_for_eager_and_graph(monkeypatch):
    from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import TrtLlmNvFp4ExpertsMonolithic
    from vllm.model_executor.layers.quantization.online import nvfp4
    from vllm.model_executor.model_loader.reload.layerwise import _copy_and_restore_kernel_tensors

    from verl.utils.real_nvfp4.vllm_runtime import _attest_native_scale_references

    names = (
        "w13_weight",
        "w13_weight_scale",
        "w13_weight_scale_2",
        "w13_input_scale",
        "w2_weight",
        "w2_weight_scale",
        "w2_weight_scale_2",
        "w2_input_scale",
    )
    layer = torch.nn.Module()
    layer._expert_routing_tables = lambda: None

    def replace(name, value):
        layer.register_parameter(name, torch.nn.Parameter(value, requires_grad=False))

    for name in names:
        replace(name, torch.ones(128, device="cuda"))

    def config(current):
        return SimpleNamespace(
            g1_alphas=current.w13_weight_scale_2,
            g2_alphas=current.w2_weight_scale_2,
            a2_gscale=1.0 / current.w2_input_scale,
            w1_scale=current.w13_weight_scale,
            w2_scale=current.w2_weight_scale,
        )

    processors = []

    def make_kernel(**kwargs):
        expert = SimpleNamespace(
            quant_config=kwargs["moe_quant_config"],
            moe_config=kwargs["moe_config"],
            gemm1_clamp_limit=None,
            gemm1_beta=None,
            gemm1_alpha=None,
        )
        expert.process_weights_after_loading = MethodType(
            TrtLlmNvFp4ExpertsMonolithic.process_weights_after_loading, expert
        )
        processors.append(expert)
        return SimpleNamespace(fused_experts=expert)

    monkeypatch.setattr(nvfp4, "make_nvfp4_moe_kernel", make_kernel)
    monkeypatch.setattr(nvfp4, "replace_parameter", lambda current, name, value: replace(name, value))
    monkeypatch.setattr(
        nvfp4,
        "convert_to_nvfp4_moe_kernel_format",
        lambda **kwargs: tuple(getattr(kwargs["layer"], name) for name in names),
    )
    method = SimpleNamespace(
        nvfp4_backend=object(),
        moe=SimpleNamespace(is_act_and_mul=True),
        experts_cls=object(),
        moe_kernel=None,
        get_fused_moe_quant_config=config,
    )
    nvfp4.Nvfp4OnlineMoEMethod._setup_kernel(method, layer)
    retained_kernel, retained_config = method.moe_kernel, method.moe_quant_config
    captured_scale = retained_kernel.fused_experts.g1_scale_c
    graph = torch.cuda.CUDAGraph()
    x = torch.ones(128, device="cuda")
    with torch.cuda.graph(graph):
        graph_result = captured_scale * x
    for value in (0.01, 0.003, 0.02):
        info = SimpleNamespace(
            kernel_tensors=(dict(layer.named_parameters()), dict(layer.named_buffers())),
            kernel_non_persistent_buffers=set(),
            loaded_weights=[],
        )
        # Mimic native layerwise materialization, with unchanged packed dimensions.
        for name in names:
            replace(name, getattr(layer, name).clone())
        layer.w13_weight_scale_2.data.fill_(value)
        layer.w2_weight_scale_2.data.fill_(value * 2)
        nvfp4.Nvfp4OnlineMoEMethod._setup_kernel(method, layer)
        assert method.moe_kernel is retained_kernel and method.moe_quant_config is retained_config
        assert processors[-1] is not retained_kernel.fused_experts
        _copy_and_restore_kernel_tensors(layer, info)
        _attest_native_scale_references(layer, retained_kernel.fused_experts)
        assert retained_kernel.fused_experts.g1_scale_c is captured_scale
        graph.replay()
        torch.cuda.synchronize()
        expected = torch.full_like(graph_result, value)
        torch.testing.assert_close(graph_result, expected, rtol=0, atol=0)
        torch.testing.assert_close(retained_kernel.fused_experts.g1_scale_c, expected, rtol=0, atol=0)
    assert len(processors) == 4
