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

import asyncio
import os
from types import SimpleNamespace

import pytest
import torch

from verl.utils.fp8_utils import FP8QuantizerHelper
from verl.utils.kernel.fp8_kernel import scaled_fp8_blockwise
from verl.utils.modelopt.qat_weight_exporter import QATWeightExporter, _QuantMeta
from verl.utils.modelopt.quantize import build_quantize_config
from verl.utils.qat import QATConfig, is_fp8_qat_mode, load_quantization_config
from verl.utils.qat.linear import QATLinear, QATMode, fp8_fake_quant_blockwise
from verl.utils.qat.quantizer import QATQuantizer


class _FakeModel:
    def load_weights(self, weights):
        return [name for name, _ in weights]


def _make_worker(model):
    from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension

    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.model_runner = SimpleNamespace(model=model)
    return worker


def test_load_quantization_config_builds_default_fp8_config_without_file():
    config = load_quantization_config(QATConfig(enable=True, mode="fp8"))

    assert config["quant_method"] == "fp8"
    assert config["activation_scheme"] == "dynamic"
    assert config["weight_block_size"] == [128, 128]
    assert is_fp8_qat_mode("fp8")


def test_contains_serialized_fp8_weights_detects_qat_exported_tensors():
    from verl.workers.rollout.vllm_rollout.utils import _contains_serialized_fp8_weights

    assert _contains_serialized_fp8_weights(
        [
            ("model.layers.0.self_attn.q_proj.weight", torch.empty(1, dtype=torch.float8_e4m3fn)),
        ]
    )
    assert _contains_serialized_fp8_weights(
        [
            ("model.layers.0.self_attn.q_proj.weight_scale_inv", torch.empty(1)),
        ]
    )
    assert not _contains_serialized_fp8_weights(
        [
            ("model.layers.0.self_attn.q_proj.weight", torch.empty(1, dtype=torch.bfloat16)),
        ]
    )


def test_update_weights_chooses_serialized_loader_for_fp8_qat_bucket(monkeypatch):
    import verl.workers.rollout.vllm_rollout.utils as worker_utils

    calls = []

    monkeypatch.setattr(worker_utils, "is_fp8_model", lambda vllm_config: True)
    monkeypatch.setattr(
        worker_utils,
        "load_serialized_fp8_weights",
        lambda weights, model_runner: calls.append(("serialized", weights)) or ["weight"],
    )
    monkeypatch.setattr(
        worker_utils,
        "load_quanted_weights",
        lambda weights, model_runner: calls.append(("quantize", weights)) or ["weight"],
    )

    worker = _make_worker(_FakeModel())
    worker.model_runner.vllm_config = SimpleNamespace(quant_config=SimpleNamespace())

    worker._update_weights(
        [("model.layers.0.self_attn.q_proj.weight_scale_inv", torch.empty(1))],
        peft_config=None,
        base_sync_done=False,
    )
    worker._update_weights(
        [("model.layers.0.self_attn.q_proj.weight", torch.empty(1, dtype=torch.bfloat16))],
        peft_config=None,
        base_sync_done=False,
    )

    assert [loader for loader, _ in calls] == ["serialized", "quantize"]


def test_qat_linear_fp8_forward_backward_on_cpu():
    layer = QATLinear(8, 4, mode=QATMode.W8A8, weight_block_size=[4, 4])
    x = torch.randn(2, 3, 8, requires_grad=True)

    loss = layer(x).sum()
    loss.backward()

    assert x.grad is not None
    assert layer.weight.grad is not None


def test_scaled_fp8_blockwise_supports_rectangular_cpu_blocks():
    x = torch.randn(3, 5)

    q, descale = scaled_fp8_blockwise(x, weight_block_size=(1, 4))
    y = fp8_fake_quant_blockwise(x, (1, 4))

    assert q.shape == x.shape
    assert descale.shape == (3, 2, 1)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_scaled_fp8_blockwise_cpu_does_not_mutate_input():
    x = torch.randn(6, 8)
    original = x.clone()

    scaled_fp8_blockwise(x, weight_block_size=(1, 4))

    torch.testing.assert_close(x, original)


def test_fp8_quantizer_helper_preserves_2d_scale_with_single_column(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    async def collect():
        helper = FP8QuantizerHelper({"weight_block_size": [128, 128]})
        weights = [("model.layers.0.self_attn.q_proj.weight", torch.randn(4, 4))]
        return [item async for item in helper.quant_weights_by_name(weights, dtype=torch.float32)]

    output = dict(asyncio.run(collect()))

    assert output["model.layers.0.self_attn.q_proj.weight_scale_inv"].shape == (1, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for Triton FP8 path")
def test_fp8_fake_quant_uses_cuda_path_backward():
    x = torch.randn(9, 17, device="cuda", dtype=torch.bfloat16)

    y = fp8_fake_quant_blockwise(x, (4, 8))

    assert y.shape == x.shape
    assert y.dtype == x.dtype

    layer = QATLinear(17, 4, mode=QATMode.W8A8, weight_block_size=[4, 8]).to(device="cuda", dtype=torch.bfloat16)
    x = x.detach().requires_grad_(True)
    layer(x).sum().backward()
    assert x.grad is not None
    assert layer.weight.grad is not None


def test_modelopt_fp8_config_uses_weight_block_size():
    config = build_quantize_config("fp8", weight_block_size=[64, 128])
    weight_quantizer = config["quant_cfg"]["*weight_quantizer"]

    assert weight_quantizer["num_bits"] == (4, 3)
    assert weight_quantizer["block_sizes"] == {-1: 128, -2: 64}
    assert config["quant_cfg"]["*input_quantizer"]["enable"] is True


def test_modelopt_fp8_qat_disables_lora_adapter_quantizers():
    import modelopt.torch.quantization as mtq
    import torch.nn as nn

    class AdapterModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Linear(2048, 64, bias=False)
            self.adapter = nn.Module()
            self.adapter.linear_out = nn.Linear(2048, 8, bias=False)
            nn.init.zeros_(self.adapter.linear_out.weight)

    model = AdapterModel()
    config = build_quantize_config("w8a16", weight_block_size=[64, 64])

    mtq.quantize(model, config)

    assert model.base.weight_quantizer.is_enabled
    assert not model.adapter.linear_out.weight_quantizer.is_enabled
    output = model.adapter.linear_out(torch.randn(2, 2048))
    assert torch.isfinite(output).all()


def test_qat_local_name_patch_skips_modelopt_quantizer_params(monkeypatch):
    import megatron.bridge.models.conversion.model_bridge as bridge_module

    from verl.utils.modelopt.megatron_qat_patch import (
        apply_local_name_to_global_patch,
        revert_local_name_to_global_patch,
    )

    if getattr(bridge_module, "_local_name_to_global_patched", False):
        revert_local_name_to_global_patch()

    def _raise_for_quantizer_params(*args, **kwargs):
        del args, kwargs
        raise AssertionError("quantizer params must not enter Bridge local-to-global conversion")

    param_name = "decoder.layers.0.mlp.experts.linear_fc1.weight_quantizer._amax"
    monkeypatch.setattr(bridge_module, "_megatron_local_name_to_global", _raise_for_quantizer_params)

    try:
        apply_local_name_to_global_patch()
        assert bridge_module._megatron_local_name_to_global(None, None, param_name) == param_name
    finally:
        revert_local_name_to_global_patch()


def test_qat_weight_exporter_serializes_fp8_from_qat_metadata():
    exporter = QATWeightExporter.__new__(QATWeightExporter)
    exporter.qat_mode = "w8a8"
    weight = torch.randn(3, 5)
    meta = _QuantMeta(
        qformat="fp8_pb_wo",
        block_size=4,
        block_sizes={-1: 4, -2: 2},
        weight_amax=None,
    )

    output = dict(exporter._quantize_fp8_blockwise("linear.weight", weight, meta))

    assert output["linear.weight"].dtype == torch.float8_e4m3fn
    assert output["linear.weight"].shape == weight.shape
    assert output["linear.weight_scale_inv"].shape == (2, 2)


def test_qat_weight_exporter_dequantizes_fp8_for_w8a16_rollout():
    exporter = QATWeightExporter.__new__(QATWeightExporter)
    exporter.qat_mode = "w8a16"
    weight = torch.randn(3, 5, dtype=torch.bfloat16)
    meta = _QuantMeta(
        qformat="fp8_pb_wo",
        block_size=4,
        block_sizes={-1: 4, -2: 2},
        weight_amax=None,
    )

    output = dict(exporter._quantize_fp8_blockwise("linear.weight", weight, meta))

    assert set(output) == {"linear.weight"}
    assert output["linear.weight"].dtype == weight.dtype
    assert output["linear.weight"].shape == weight.shape
    assert torch.isfinite(output["linear.weight"]).all()


def test_qat_quantizer_skips_lora_adapter_weights():
    quantizer = QATQuantizer(mode="w4a16", device=torch.device("cpu"))

    assert not quantizer._should_quantize(
        "model.layers.0.self_attn.q_proj.lora_A.default.weight",
        torch.empty(4, 16),
    )


def test_serialized_fp8_loader_restores_param_class_on_error():
    from verl.utils.vllm.vllm_fp8_utils import load_serialized_fp8_weights

    class _ParamSubclass(torch.nn.Parameter):
        pass

    class _FailingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.weight.subclass_type = _ParamSubclass

        def load_weights(self, weights):
            del weights
            raise RuntimeError("load failed")

    model = _FailingModel()

    with pytest.raises(RuntimeError, match="load failed"):
        load_serialized_fp8_weights([], SimpleNamespace(model=model))

    assert model.weight.__class__ is torch.nn.Parameter


def test_vllm19_fp8_linear_processing_preserves_refit_param_metadata():
    from verl.utils.vllm.vllm_fp8_utils import process_weights_after_loading_for_vllm14

    class _WeightParam(torch.nn.Parameter):
        pass

    class _ScaleParam(torch.nn.Parameter):
        pass

    class _Fp8Linear:
        def process_weights_after_loading(self, layer):
            layer.weight = torch.nn.Parameter(layer.weight.detach().clone(), requires_grad=False)
            layer.weight_scale_inv = torch.nn.Parameter(
                layer.weight_scale_inv.detach().clone(),
                requires_grad=False,
            )

    layer = torch.nn.Module()
    layer.weight = _WeightParam(torch.ones(1, 1), requires_grad=False)
    layer.weight.weight_loader = lambda *args, **kwargs: None
    layer.weight.output_dim = 0
    layer.weight.input_dim = 1
    layer.weight_scale_inv = _ScaleParam(torch.ones(1, 1), requires_grad=False)
    layer.weight_scale_inv.weight_loader = lambda *args, **kwargs: None
    layer.weight_scale_inv.output_dim = 0
    layer.weight_scale_inv.input_dim = 1

    quant_config = SimpleNamespace(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic")
    method = SimpleNamespace(block_quant=True, quant_config=quant_config, fp8_linear=_Fp8Linear())

    process_weights_after_loading_for_vllm14(method, layer)

    assert layer.input_scale is None
    assert layer.weight.subclass_type is _WeightParam
    assert layer.weight.output_dim == 0
    assert layer.weight.input_dim == 1
    assert hasattr(layer.weight, "weight_loader")
    assert layer.weight_scale_inv.subclass_type is _ScaleParam
    assert layer.weight_scale_inv.output_dim == 0
    assert layer.weight_scale_inv.input_dim == 1
    assert hasattr(layer.weight_scale_inv, "weight_loader")


def test_te_grouped_weight_calibration_skips_original_plain_weight_lookup(monkeypatch):
    import contextlib

    import modelopt.torch.quantization.model_calib as model_calib

    from verl.utils.modelopt.megatron_qat_patch import (
        apply_te_grouped_weight_calibration_patch,
        revert_te_grouped_weight_calibration_patch,
    )

    class _GroupedLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_gemms = 2
            self.weight0 = torch.nn.Parameter(torch.ones(1))
            self.weight1 = torch.nn.Parameter(torch.ones(1) * 2)
            self.quantized_weights = []

        def weight_quantizer(self, weight):
            self.quantized_weights.append(weight)

    grouped = _GroupedLinear()

    def _bad_weight_attr_names(module):
        if module is grouped:
            yield "weight"

    monkeypatch.setattr(model_calib, "weight_attr_names", _bad_weight_attr_names)
    monkeypatch.setattr(
        model_calib,
        "enable_weight_access_and_writeback",
        lambda module, model: contextlib.nullcontext(),
    )

    revert_te_grouped_weight_calibration_patch()
    try:
        apply_te_grouped_weight_calibration_patch()
        model_calib.weight_only_quantize(grouped)
    finally:
        revert_te_grouped_weight_calibration_patch()

    assert [id(weight) for weight in grouped.quantized_weights] == [id(grouped.weight0), id(grouped.weight1)]
    assert model_calib.weight_attr_names is _bad_weight_attr_names


def test_fp8_module_lookup_unwraps_lora_fused_moe(monkeypatch):
    import verl.utils.vllm.vllm_fp8_utils as fp8_utils

    class _FakeFusedMoE:
        pass

    class FusedMoEWithLoRA:
        def __init__(self, base_layer):
            self.base_layer = base_layer

    monkeypatch.setattr(fp8_utils, "FusedMoE", _FakeFusedMoE)

    fused_moe = _FakeFusedMoE()
    root = torch.nn.Module()
    root.packed_modules_mapping = {}
    root.model = torch.nn.Module()
    root.model.layers = torch.nn.ModuleList(torch.nn.Module() for _ in range(27))
    root.model.layers[26].mlp = torch.nn.Module()
    root.model.layers[26].mlp.experts = FusedMoEWithLoRA(fused_moe)

    assert (
        fp8_utils.get_module_from_param_name(
            root,
            "model.layers.26.mlp.experts.3.down_proj.base_layer.weight",
        )
        is fused_moe
    )


def test_fp8_module_lookup_applies_packed_mapping_before_base_layer():
    import verl.utils.vllm.vllm_fp8_utils as fp8_utils

    base_layer = torch.nn.Linear(4, 4)
    gate_up_proj = torch.nn.Module()
    gate_up_proj.base_layer = base_layer

    root = torch.nn.Module()
    root.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    root.model = torch.nn.Module()
    root.model.layers = torch.nn.ModuleList(torch.nn.Module() for _ in range(19))
    root.model.layers[18].mlp = torch.nn.Module()
    root.model.layers[18].mlp.shared_experts = torch.nn.Module()
    root.model.layers[18].mlp.shared_experts.gate_up_proj = gate_up_proj

    assert (
        fp8_utils.get_module_from_param_name(
            root,
            "model.layers.18.mlp.shared_experts.gate_proj.base_layer.weight",
        )
        is base_layer
    )
    assert (
        fp8_utils.get_module_from_param_name(
            root,
            "model.layers.18.mlp.shared_experts.up_proj.base_layer.weight",
        )
        is base_layer
    )


def test_update_weights_from_ipc_skips_qat_base_processing_for_lora_only_sync(monkeypatch):
    import verl.utils.qat as qat_utils
    import verl.workers.rollout.vllm_rollout.bucketed_weight_transfer as bucketed_weight_transfer

    class _FakeBucketReceiver:
        def __init__(self, zmq_handle, device, use_shm):
            del zmq_handle, device, use_shm

        def receive_weights(self, on_bucket_received):
            on_bucket_received(
                [("layers.0.self_attn.q_proj.lora_A.weight", torch.ones(1))],
            )

    def _raise_qat_base_processing(*args, **kwargs):
        del args, kwargs
        raise AssertionError("adapter-only LoRA sync must not process QAT base weights")

    monkeypatch.setattr(bucketed_weight_transfer, "BucketedWeightReceiver", _FakeBucketReceiver)
    monkeypatch.setattr(qat_utils, "prepare_qat_for_load_weights", _raise_qat_base_processing)
    monkeypatch.setattr(qat_utils, "manual_process_weights_after_loading", _raise_qat_base_processing)

    worker = _make_worker(_FakeModel())
    worker.model_runner.vllm_config = SimpleNamespace()
    worker.device = torch.device("cpu")
    worker.local_rank = 0
    worker._is_qat_model = True
    worker._is_modelopt_qat = False
    worker._get_zmq_handle = lambda: "ipc:///tmp/test-bucketed-lora-qAT.sock"
    worker.remove_lora = lambda lora_id: None
    worker.add_lora = lambda lora_request: True

    worker.update_weights_from_ipc(peft_config={"r": 1}, base_sync_done=True)


def _make_vllm_http_server_for_fp8_validation(moe_intermediate_size=1408, tp_size=2):
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    server = vLLMHttpServer.__new__(vLLMHttpServer)
    server.config = SimpleNamespace(tensor_model_parallel_size=tp_size)
    server.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            moe_intermediate_size=moe_intermediate_size,
        )
    )
    return server


def test_fp8_weight_block_validation_rejects_unaligned_moonlight_moe_shard():
    server = _make_vllm_http_server_for_fp8_validation()

    with pytest.raises(ValueError, match="intermediate_size_per_partition=704.*block_n=128"):
        server._validate_fp8_weight_block_size([128, 128])


def test_fp8_weight_block_validation_accepts_moonlight_tp2_64_blocks():
    server = _make_vllm_http_server_for_fp8_validation()

    server._validate_fp8_weight_block_size([64, 64])


def test_w8a16_qat_rollout_uses_bf16_quantization_path(monkeypatch):
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    monkeypatch.setenv("VERL_VLLM_FP8_QUANT_ENABLED", "1")

    server = _make_vllm_http_server_for_fp8_validation()
    server.config = SimpleNamespace(
        quantization=None,
        quantization_config_file=None,
        tensor_model_parallel_size=2,
        qat={
            "enable": True,
            "mode": "w8a16",
            "weight_block_size": [64, 64],
            "quantization_config_path": None,
        },
    )

    quantization, hf_overrides = vLLMHttpServer._apply_quantization(server)

    assert quantization is None
    assert "quantization_config" not in hf_overrides
    assert "VERL_VLLM_FP8_QUANT_ENABLED" not in os.environ
