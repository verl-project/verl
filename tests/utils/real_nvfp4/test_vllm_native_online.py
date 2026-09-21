# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.utils.real_nvfp4.bf16_transport import attest_real_nvfp4_bf16_transport
from verl.utils.real_nvfp4.config import (
    real_nvfp4_expected_counts,
    real_nvfp4_moe_layer_indices,
    real_nvfp4_rollout_layer_partition,
    real_nvfp4_vllm_ignore_layers,
    validate_real_nvfp4_model_contract,
)
from verl.utils.real_nvfp4.r3_monolithic_capture import (
    _patch_moe_runner_class,
    attest_r3_rollout_routes,
)
from verl.utils.real_nvfp4.vllm_runtime import (
    attest_vllm_native_nvfp4_runtime,
    require_vllm_native_reload_contract,
    vllm_native_nvfp4_fingerprint,
)
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer


def _expert(projection: str, value: float):
    return (
        f"model.layers.0.mlp.experts.7.{projection}.weight",
        torch.full((2, 16), value, dtype=torch.bfloat16),
    )


def test_bf16_transport_rejects_actor_side_packing():
    plain = list(
        attest_real_nvfp4_bf16_transport(
            iter([_expert("down_proj", 1.0)]),
            expected_expert_weights=1,
        )
    )
    assert plain[0][1].dtype == torch.bfloat16

    packed = [("model.layers.0.mlp.experts.0.down_proj.weight_scale", torch.ones(1))]
    with pytest.raises(RuntimeError, match="packed tensor"):
        list(
            attest_real_nvfp4_bf16_transport(
                iter(packed),
                expected_expert_weights=1,
            )
        )


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


def test_native_vllm_runtime_attestation_and_fingerprint():
    model = torch.nn.Sequential(_FakeNativeMoE())
    assert attest_vllm_native_nvfp4_runtime(model, expected_moe_layers=1) == {
        "dense_layers": 0,
        "moe_layers": 1,
    }
    first = vllm_native_nvfp4_fingerprint(model)
    model[0].w13_weight[0, 0] += 1
    assert vllm_native_nvfp4_fingerprint(model) != first

    scale_fingerprint = vllm_native_nvfp4_fingerprint(model)
    model[0].w2_weight_scale_2[0] = 1.00001
    assert vllm_native_nvfp4_fingerprint(model) != scale_fingerprint

    bad = torch.nn.Sequential(_FakeNativeMoE(per_token_activation=False))
    with pytest.raises(RuntimeError, match="per-token activation"):
        attest_vllm_native_nvfp4_runtime(bad, expected_moe_layers=1)


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

    assert attest_vllm_native_nvfp4_runtime(
        root,
        expected_quantized_layer_indices=[2, 3, 4, 5],
        expected_bf16_layer_indices=[0, 1, 6, 7],
    ) == {"dense_layers": 0, "moe_layers": 4}

    with pytest.raises(RuntimeError, match="wrong MoE layers"):
        attest_vllm_native_nvfp4_runtime(
            root,
            expected_quantized_layer_indices=[1, 2, 3, 4],
            expected_bf16_layer_indices=[0, 5, 6, 7],
        )


def test_native_vllm_rejects_one_refit_stale_derived_scale():
    model = torch.nn.Sequential(_FakeNativeMoE())
    model[0].w13_weight_scale_2.mul_(2)
    with pytest.raises(RuntimeError, match="does not match current"):
        attest_vllm_native_nvfp4_runtime(model, expected_moe_layers=1)


def test_native_vllm_rejects_rebound_eager_scale_even_if_values_match():
    model = torch.nn.Sequential(_FakeNativeMoE())
    model[0].quant_method.moe_kernel.fused_experts.g1_scale_c = model[0].g1_scale_c.clone()
    with pytest.raises(RuntimeError, match="stale eager/CUDA-graph"):
        attest_vllm_native_nvfp4_runtime(model, expected_moe_layers=1)


@pytest.mark.parametrize("field", ["g1_alphas", "g2_alphas", "w1_scale", "w2_scale", "a1_gscale", "a2_gscale"])
def test_native_vllm_rejects_detached_quant_config_scale(field):
    model = torch.nn.Sequential(_FakeNativeMoE())
    config = model[0].quant_method.moe_kernel.fused_experts.quant_config
    setattr(config, field, getattr(config, field).clone())
    with pytest.raises(RuntimeError, match="stale scale reference"):
        attest_vllm_native_nvfp4_runtime(model, expected_moe_layers=1)


@pytest.mark.parametrize("field", ["a1_gscale", "a2_gscale"])
def test_native_vllm_rejects_discarded_activation_scale_after_sleep(field):
    model = torch.nn.Sequential(_FakeNativeMoE())
    getattr(model[0].quant_method.moe_kernel.fused_experts.quant_config, field).zero_()
    with pytest.raises(RuntimeError, match="current activation scale after sleep/refit"):
        attest_vllm_native_nvfp4_runtime(model, expected_moe_layers=1)


def test_native_vllm_fingerprint_covers_derived_scale():
    model = torch.nn.Sequential(_FakeNativeMoE())
    before = vllm_native_nvfp4_fingerprint(model)
    model[0].g1_scale_c[0] = 1.00001
    assert before != vllm_native_nvfp4_fingerprint(model)


def test_native_vllm_reload_contract_accepts_compatible_signatures():
    for reload_weights in (
        lambda weights_iterator=None, weights_path=None, is_checkpoint_format=True: None,
        lambda weights_iterator=None, is_checkpoint_format=False, *, optional_new_argument=None: None,
    ):
        require_vllm_native_reload_contract(SimpleNamespace(reload_weights=reload_weights))

    for reload_weights in (
        lambda weights_path=None: None,
        lambda required_new_argument, weights_iterator=None, is_checkpoint_format=True: None,
    ):
        with pytest.raises(RuntimeError, match="must accept"):
            require_vllm_native_reload_contract(SimpleNamespace(reload_weights=reload_weights))


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
        enable_rollout_routing_replay=True,
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


class _FakeRouter:
    def __init__(self, capture: bool) -> None:
        self.capture_fn = (lambda routes: None) if capture else None
        self.select_calls = 0

    def select_experts(self, **kwargs):
        self.select_calls += 1
        if self.capture_fn is not None:
            self.capture_fn(torch.tensor([[1, 2]]))
        return torch.ones(1, 2), torch.tensor([[1, 2]])


class _FakeRoutedExperts:
    def __init__(self, monolithic: bool) -> None:
        self.quant_method = SimpleNamespace(
            is_monolithic=monolithic,
            topk_indices_dtype=torch.int32,
        )

    def forward_monolithic(self, **kwargs):
        return "monolithic"

    def forward_modular(self, **kwargs):
        return "modular"


class _FakeMoERunner:
    def __init__(self, monolithic: bool, capture: bool) -> None:
        self.routed_experts = _FakeRoutedExperts(monolithic)
        self.router = _FakeRouter(capture)

    @property
    def _quant_method(self):
        return self.routed_experts.quant_method

    def _apply_quant_method(
        self,
        hidden_states,
        router_logits,
        shared_experts_input,
        input_ids=None,
    ):
        if self.routed_experts.quant_method.is_monolithic:
            result = (
                None,
                self.routed_experts.forward_monolithic(
                    x=hidden_states,
                    router_logits=router_logits,
                    input_ids=input_ids,
                ),
            )
        else:
            topk_weights, topk_ids = self.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_indices_dtype=self._quant_method.topk_indices_dtype,
                input_ids=input_ids,
            )
            result = (
                None,
                self.routed_experts.forward_modular(
                    x=hidden_states,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                ),
            )
        return result


def test_monolithic_r3_capture_patch_fires_only_when_needed():
    assert _patch_moe_runner_class(_FakeMoERunner) == "verl_wrapper"
    assert _patch_moe_runner_class(_FakeMoERunner) == "already_patched"

    monolithic_capture = _FakeMoERunner(monolithic=True, capture=True)
    assert monolithic_capture._apply_quant_method(None, None, None) == (
        None,
        "monolithic",
    )
    assert monolithic_capture.router.select_calls == 1

    monolithic_no_capture = _FakeMoERunner(monolithic=True, capture=False)
    monolithic_no_capture._apply_quant_method(None, None, None)
    assert monolithic_no_capture.router.select_calls == 0

    modular_capture = _FakeMoERunner(monolithic=False, capture=True)
    modular_capture._apply_quant_method(None, None, None)
    assert modular_capture.router.select_calls == 1


def test_r3_route_attestation_rejects_missed_capture():
    with pytest.raises(RuntimeError, match="all zero"):
        attest_r3_rollout_routes(np.zeros((4, 2, 2), dtype=np.int16))
    with pytest.raises(RuntimeError, match="empty"):
        attest_r3_rollout_routes(np.zeros((0, 2, 2), dtype=np.int16))

    routes = np.zeros((4, 2, 2), dtype=np.int16)
    routes[-1, 0] = [3, 7]
    assert attest_r3_rollout_routes(routes) is routes


def test_model_contract_accepts_all_moe_but_refuses_mixed_precision_scope():
    """Correct sparse-layer counts alone do not prove train/rollout precision parity."""

    all_moe = SimpleNamespace(
        architectures=["Qwen3MoeForCausalLM"],
        num_hidden_layers=48,
        num_experts=128,
        decoder_sparse_step=1,
        mlp_only_layers=[],
    )
    validate_real_nvfp4_model_contract(all_moe)
    assert real_nvfp4_moe_layer_indices(all_moe) == list(range(48))
    assert real_nvfp4_expected_counts(all_moe) == (48 * 128 * 3, 48 * 128 * 2)
    assert real_nvfp4_rollout_layer_partition(
        all_moe,
        num_layers_at_start_in_bf16=2,
        num_layers_at_end_in_bf16=4,
    ) == (list(range(2, 44)), [0, 1, 44, 45, 46, 47])
    assert real_nvfp4_vllm_ignore_layers(
        all_moe,
        num_layers_at_start_in_bf16=2,
        num_layers_at_end_in_bf16=4,
    ) == [
        "model.layers.0.mlp.experts",
        "model.layers.1.mlp.experts",
        "model.layers.44.mlp.experts",
        "model.layers.45.mlp.experts",
        "model.layers.46.mlp.experts",
        "model.layers.47.mlp.experts",
    ]

    # The layout helper can count mixed layers, but the training precision
    # recipe does not yet keep non-expert MLPs BF16 like rollout does.
    interleaved = SimpleNamespace(
        architectures=["SomeOtherMoeForCausalLM"],
        num_hidden_layers=48,
        num_experts=64,
        decoder_sparse_step=2,
        mlp_only_layers=[],
    )
    with pytest.raises(ValueError, match="mixed dense/MoE"):
        validate_real_nvfp4_model_contract(interleaved)
    assert real_nvfp4_moe_layer_indices(interleaved) == [i for i in range(48) if (i + 1) % 2 == 0]
    with pytest.raises(ValueError, match="mixed dense/MoE"):
        real_nvfp4_expected_counts(interleaved)

    # Explicitly dense prefixes are likewise unsupported by the current recipe.
    dense_prefix = SimpleNamespace(
        architectures=["SomeOtherMoeForCausalLM"],
        num_hidden_layers=48,
        num_experts=128,
        decoder_sparse_step=1,
        mlp_only_layers=[0, 1],
    )
    with pytest.raises(ValueError, match="mixed dense/MoE"):
        validate_real_nvfp4_model_contract(dense_prefix)
    assert real_nvfp4_moe_layer_indices(dense_prefix) == list(range(2, 48))
    assert real_nvfp4_rollout_layer_partition(
        dense_prefix,
        num_layers_at_start_in_bf16=2,
        num_layers_at_end_in_bf16=4,
    ) == (list(range(2, 44)), [44, 45, 46, 47])


def test_model_contract_still_refuses_layouts_it_cannot_count():
    no_experts = SimpleNamespace(architectures=["LlamaForCausalLM"], num_hidden_layers=32, num_experts=0)
    with pytest.raises(ValueError, match="routed-expert MoE model"):
        validate_real_nvfp4_model_contract(no_experts)

    # Shared experts add per-layer weights the expert-count arithmetic does not
    # model, so the refit attestation would be wrong rather than conservative.
    shared = SimpleNamespace(
        architectures=["SomeMoeForCausalLM"],
        num_hidden_layers=48,
        num_experts=128,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        n_shared_experts=1,
    )
    with pytest.raises(ValueError, match="shared experts"):
        validate_real_nvfp4_model_contract(shared)

    # A sparse step past the depth leaves nothing to quantize.
    no_sparse_layer = SimpleNamespace(
        architectures=["SomeMoeForCausalLM"],
        num_hidden_layers=4,
        num_experts=8,
        decoder_sparse_step=99,
        mlp_only_layers=[],
    )
    with pytest.raises(ValueError, match="no sparse decoder layer"):
        validate_real_nvfp4_model_contract(no_sparse_layer)


@pytest.mark.parametrize("release", ["0.26.0", "0.27.1"])
def test_monolithic_r3_public_entry_accepts_audited_releases(monkeypatch, release):
    from vllm.model_executor.layers.fused_moe.runner import moe_runner

    from verl.utils.real_nvfp4 import r3_monolithic_capture as capture

    class Runner(_FakeMoERunner):
        _apply_quant_method = getattr(
            _FakeMoERunner._apply_quant_method, "__wrapped__", _FakeMoERunner._apply_quant_method
        )

    monkeypatch.setattr(capture, "version", lambda name: release)
    monkeypatch.setattr(moe_runner, "MoERunner", Runner)
    assert capture.patch_vllm_monolithic_moe_r3_capture() == "verl_wrapper"
    runner = Runner(monolithic=True, capture=True)
    assert runner._apply_quant_method(None, None, None) == (None, "monolithic")
    assert runner.router.select_calls == 1


def test_monolithic_r3_public_entry_rejects_unaudited_release(monkeypatch):
    from verl.utils.real_nvfp4 import r3_monolithic_capture as capture

    monkeypatch.setattr(capture, "version", lambda name: "0.28.0")
    with pytest.raises(RuntimeError, match="audited vLLM"):
        capture.patch_vllm_monolithic_moe_r3_capture()
