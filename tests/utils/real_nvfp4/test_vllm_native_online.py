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
from verl.utils.real_nvfp4.config import validate_real_nvfp4_model_contract
from verl.utils.real_nvfp4.r3_monolithic_capture import (
    _patch_moe_runner_class,
    attest_r3_rollout_routes,
)
from verl.utils.real_nvfp4.vllm_runtime import (
    attest_vllm_native_nvfp4_runtime,
    require_vllm_native_reload_contract,
    vllm_native_nvfp4_fingerprint,
)


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


def test_native_vllm_reload_contract_is_exact():
    class _ValidRunner:
        @staticmethod
        def reload_weights(
            weights_iterator=None,
            weights_path=None,
            is_checkpoint_format=True,
        ):
            pass

    require_vllm_native_reload_contract(_ValidRunner())

    class _DriftedRunner:
        @staticmethod
        def reload_weights(weights_iterator=None, is_checkpoint_format=True):
            pass

    with pytest.raises(RuntimeError, match="API drifted"):
        require_vllm_native_reload_contract(_DriftedRunner())


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


def test_model_contract_accepts_only_qwen3_all_moe():
    valid = SimpleNamespace(
        architectures=["Qwen3MoeForCausalLM"],
        decoder_sparse_step=1,
        mlp_only_layers=[],
    )
    validate_real_nvfp4_model_contract(valid)

    invalid = SimpleNamespace(
        architectures=["Qwen3MoeForCausalLM"],
        decoder_sparse_step=2,
        mlp_only_layers=[],
    )
    with pytest.raises(ValueError, match="Qwen3 all-MoE"):
        validate_real_nvfp4_model_contract(invalid)
