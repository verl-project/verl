# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Real CuMem sleep/copyback regression; not a full-model generation test.

The kernel factory is a metadata stand-in; the real-layout case also exercises
the installed FlashInfer weight/scale conversion (including expanded scales). Quant
config construction, scale registration, expert postprocessing, memory discard,
native copyback and CUDA graph replay use the installed implementations.
"""

from types import MethodType, SimpleNamespace

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CuMem sleep regression requires a GPU")
@pytest.mark.parametrize("real_layout", [False, True], ids=["metadata_layout", "real_layout"])
def test_actual_sleep_wake_refit_restores_retained_activation_scales(monkeypatch, real_layout):
    from vllm.device_allocator.cumem import CuMemAllocator
    from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import TrtLlmNvFp4ExpertsMonolithic
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
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
    layer.activation = SimpleNamespace(is_gated=True)
    layer.moe_config = SimpleNamespace(hidden_dim=128, hidden_dim_unpadded=None, intermediate_size_per_partition=128)
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
    if not real_layout:
        monkeypatch.setattr(
            nvfp4,
            "convert_to_nvfp4_moe_kernel_format",
            lambda **kwargs: tuple(getattr(kwargs["layer"], name) for name in names),
        )
    method = SimpleNamespace(
        nvfp4_backend=NvFp4MoeBackend.FLASHINFER_TRTLLM,
        moe=SimpleNamespace(is_act_and_mul=True),
        experts_cls=object(),
        moe_kernel=None,
    )
    method.get_fused_moe_quant_config = MethodType(nvfp4.Nvfp4OnlineMoEMethod.get_fused_moe_quant_config, method)

    def prepare(value):
        # Native reload materializes new tensors; never read discarded storage.
        for name in names:
            tensor = torch.ones(128, device="cuda")
            if real_layout and name in ("w13_weight", "w2_weight"):
                rows = 256 if name == "w13_weight" else 128
                tensor = torch.zeros((128, rows, 64), dtype=torch.uint8, device="cuda")
            elif real_layout and name in ("w13_weight_scale", "w2_weight_scale"):
                rows = 256 if name == "w13_weight_scale" else 128
                tensor = torch.ones((128, rows, 8), device="cuda").to(torch.float8_e4m3fn)
            if name.endswith("weight_scale_2"):
                tensor.fill_(value)
            layer.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))
        # Crucially, the test does NOT register reciprocal scales itself.
        nvfp4.Nvfp4OnlineMoEMethod._setup_kernel(method, layer)

    allocator = CuMemAllocator.get_instance()
    with allocator.use_memory_pool(tag="nvfp4_sleep_regression"):
        prepare(0.001)
    retained_kernel, retained_config = method.moe_kernel, method.moe_quant_config
    retained = retained_kernel.fused_experts
    _attest_native_scale_references(layer, retained)
    original_pointers = {name: tensor.data_ptr() for name, tensor in layer.named_parameters()}
    graph = torch.cuda.CUDAGraph()
    x = torch.ones(128, device="cuda")
    with torch.cuda.graph(graph):
        graph_result = retained.g1_scale_c * x
    try:
        for value in (0.01, 0.003, 0.02):
            torch.cuda.synchronize()
            # Real level-2 discard: there is no CPU backup for these allocations.
            allocator.sleep(offload_tags=())
            allocator.wake_up()
            info = SimpleNamespace(
                kernel_tensors=(dict(layer.named_parameters()), dict(layer.named_buffers())),
                kernel_non_persistent_buffers=set(),
                loaded_weights=[],
            )
            prepare(value)
            assert method.moe_kernel is retained_kernel and method.moe_quant_config is retained_config
            assert processors[-1] is not retained
            _copy_and_restore_kernel_tensors(layer, info)
            _attest_native_scale_references(layer, retained)
            assert {name: tensor.data_ptr() for name, tensor in layer.named_parameters()} == original_pointers
            for config_name, input_name in (("a1_gscale", "w13_input_scale"), ("a2_gscale", "w2_input_scale")):
                torch.testing.assert_close(
                    getattr(retained_config, config_name), 1.0 / getattr(layer, input_name), rtol=0, atol=0
                )
            graph.replay()
            torch.cuda.synchronize()
            expected = torch.full_like(graph_result, value)
            torch.testing.assert_close(graph_result, expected, rtol=0, atol=0)
            torch.testing.assert_close(retained.g1_scale_c, expected, rtol=0, atol=0)
        assert len(processors) == 4
    finally:
        graph.reset()
