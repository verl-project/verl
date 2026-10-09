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

"""Rollout-side layer partition, server configuration and refit-state checks."""

from types import SimpleNamespace

import pytest
import torch

from verl.utils.real_nvfp4.config import real_nvfp4_moe_layer_partition, real_nvfp4_vllm_ignore_layers
from verl.utils.real_nvfp4.vllm_runtime import attest_vllm_native_nvfp4_runtime
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer


class Nvfp4OnlineMoEMethod:
    def __init__(self, per_token_activation: bool = True) -> None:
        self.nvfp4_backend = SimpleNamespace(name="FLASHINFER_TRTLLM")
        self.moe_kernel = SimpleNamespace(fused_experts=SimpleNamespace(per_token_activation=per_token_activation))


class UnquantizedFusedMoEMethod:
    pass


class _FakeNativeMoE(torch.nn.Module):
    def __init__(self, value: int = 1, per_token_activation: bool = True) -> None:
        super().__init__()
        self.quant_method = Nvfp4OnlineMoEMethod(per_token_activation)
        self._already_called_process_weights_after_loading = True
        self.w13_weight = torch.full((2, 4), value, dtype=torch.uint8)
        self.w2_weight = torch.full((2, 4), value + 1, dtype=torch.uint8)
        self.w13_weight_scale = torch.ones((2, 2), dtype=torch.float8_e4m3fn)
        self.w2_weight_scale = torch.ones((2, 2), dtype=torch.float8_e4m3fn)
        self.w13_weight_scale_2 = torch.ones(2, dtype=torch.float32)
        self.w2_weight_scale_2 = torch.ones(2, dtype=torch.float32)
        self.w13_input_scale = torch.ones(2)
        self.w2_input_scale = torch.ones(2)
        self.nvfp4_a1_gscale = torch.ones(2)
        self.nvfp4_a2_gscale = torch.ones(2)
        self.g1_scale_c = self.w13_weight_scale_2.clone()
        experts = self.quant_method.moe_kernel.fused_experts
        experts.quant_config = SimpleNamespace(
            g1_alphas=self.w13_weight_scale_2,
            g2_alphas=self.w2_weight_scale_2,
            w1_scale=self.w13_weight_scale,
            w2_scale=self.w2_weight_scale,
            a1_gscale=self.nvfp4_a1_gscale,
            a2_gscale=self.nvfp4_a2_gscale,
        )
        experts.moe_config = SimpleNamespace(is_act_and_mul=True)
        experts.g1_scale_c = self.g1_scale_c


class _FakeUnquantizedMoE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.quant_method = UnquantizedFusedMoEMethod()


def _attest_one(model):
    attest_vllm_native_nvfp4_runtime(model, quantized_layer_indices=[0], bf16_layer_indices=[])


def _single_layer(module):
    root = torch.nn.Module()
    root.layers = torch.nn.ModuleList([torch.nn.Module()])
    root.layers[0].mlp = torch.nn.Module()
    root.layers[0].mlp.experts = module
    return root


def test_native_vllm_runtime_attestation():
    _attest_one(_single_layer(_FakeNativeMoE()))
    with pytest.raises(RuntimeError, match="per-token activation"):
        _attest_one(_single_layer(_FakeNativeMoE(per_token_activation=False)))


def test_native_vllm_runtime_attests_exact_carveout_layers():
    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.model.layers = torch.nn.ModuleList()
    for index in range(8):
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        layer.mlp.experts = torch.nn.Module()
        layer.mlp.experts.routed_experts = _FakeNativeMoE() if 2 <= index < 6 else _FakeUnquantizedMoE()
        root.model.layers.append(layer)

    attest_vllm_native_nvfp4_runtime(root, quantized_layer_indices=[2, 3, 4, 5], bf16_layer_indices=[0, 1, 6, 7])
    with pytest.raises(RuntimeError, match="wrong MoE layers"):
        attest_vllm_native_nvfp4_runtime(root, quantized_layer_indices=[1, 2, 3, 4], bf16_layer_indices=[0, 5, 6, 7])


def test_native_vllm_rejects_one_refit_stale_derived_scale():
    model = _single_layer(_FakeNativeMoE())
    model.layers[0].mlp.experts.w13_weight_scale_2.mul_(2)
    with pytest.raises(RuntimeError, match="does not match current"):
        _attest_one(model)


def test_native_vllm_rejects_rebound_eager_scale_even_if_values_match():
    model = _single_layer(_FakeNativeMoE())
    model.layers[0].mlp.experts.quant_method.moe_kernel.fused_experts.g1_scale_c = model.layers[
        0
    ].mlp.experts.g1_scale_c.clone()
    with pytest.raises(RuntimeError, match="stale eager/CUDA-graph"):
        _attest_one(model)


@pytest.mark.parametrize("field", ["g1_alphas", "g2_alphas", "w1_scale", "w2_scale", "a1_gscale", "a2_gscale"])
def test_native_vllm_rejects_detached_quant_config_scale(field):
    model = _single_layer(_FakeNativeMoE())
    config = model.layers[0].mlp.experts.quant_method.moe_kernel.fused_experts.quant_config
    setattr(config, field, getattr(config, field).clone())
    with pytest.raises(RuntimeError, match="stale scale reference"):
        _attest_one(model)


@pytest.mark.parametrize("field", ["a1_gscale", "a2_gscale"])
def test_native_vllm_rejects_discarded_activation_scale_after_sleep(field):
    model = _single_layer(_FakeNativeMoE())
    getattr(model.layers[0].mlp.experts.quant_method.moe_kernel.fused_experts.quant_config, field).zero_()
    with pytest.raises(RuntimeError, match="current activation scale after sleep/refit"):
        _attest_one(model)


def test_online_nvfp4_ignore_is_a_model_config_argument(monkeypatch):
    # _apply_quantization intentionally exports these for vLLM worker
    # subprocesses. Track their original state so this unit test cannot leak
    # its 2/4 carve-out into tests that use a smaller fake model.
    monkeypatch.setenv("VERL_REAL_NVFP4_BF16_LAYERS_AT_START", "0")
    monkeypatch.setenv("VERL_REAL_NVFP4_BF16_LAYERS_AT_END", "0")
    server = object.__new__(vLLMHttpServer)
    server.config = SimpleNamespace(
        real_nvfp4={
            "enable": True,
            "num_layers_at_start_in_bf16": 2,
            "num_layers_at_end_in_bf16": 4,
        },
        qat={},
        quantization=None,
        quantization_config_file=None,
        dtype="bfloat16",
        load_format="dummy",
        expert_parallel_size=1,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        enforce_eager=False,
        mtp=None,
        checkpoint_engine=SimpleNamespace(backend="naive"),
    )
    server.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            num_hidden_layers=48,
            num_experts=128,
            decoder_sparse_step=1,
            mlp_only_layers=[],
        )
    )
    engine_kwargs = {}

    quantization, hf_overrides = server._apply_quantization(engine_kwargs)

    assert quantization == "nvfp4_per_token"
    assert hf_overrides == {}
    assert engine_kwargs["quantization_config"] == {
        "ignore": [
            "model.layers.0.mlp.experts",
            "model.layers.1.mlp.experts",
            "model.layers.44.mlp.experts",
            "model.layers.45.mlp.experts",
            "model.layers.46.mlp.experts",
            "model.layers.47.mlp.experts",
        ]
    }


def test_rollout_partition_and_vllm_ignore_list_follow_the_carveout():
    model = SimpleNamespace(num_hidden_layers=48, num_experts=128, decoder_sparse_step=1, mlp_only_layers=[])
    carveout = {"num_layers_at_start_in_bf16": 2, "num_layers_at_end_in_bf16": 4}
    assert real_nvfp4_moe_layer_partition(model, **carveout) == (list(range(2, 44)), [0, 1, 44, 45, 46, 47])
    assert real_nvfp4_vllm_ignore_layers(model, **carveout) == [
        f"model.layers.{index}.mlp.experts" for index in (0, 1, 44, 45, 46, 47)
    ]
