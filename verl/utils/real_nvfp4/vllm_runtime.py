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

"""Validate the BF16 refit stream and the native NVFP4 state of the vLLM worker."""

import re
from collections.abc import Collection
from typing import Any

import torch

from .config import _hf_get, real_nvfp4_moe_layer_indices, validate_real_nvfp4_model_contract

NVFP4_PER_TOKEN_METHOD = "nvfp4_per_token"
REAL_NVFP4_MOE_BACKEND = "flashinfer_trtllm"

_EXPERT_WEIGHT_RE = re.compile(r"^.*\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)\.weight$")
_PACKED_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")


class RealNVFP4BF16TransportCheck:
    """Check one refit round: every routed-expert projection arrives exactly once, unpacked.

    Quantized weight and scale tensors are produced only inside the vLLM worker.
    Call :meth:`check_bucket` for each received bucket and :meth:`finish` at the end.
    """

    def __init__(self, hf_config: Any):
        validate_real_nvfp4_model_contract(hf_config)
        num_experts = int(_hf_get(hf_config, "num_experts") or _hf_get(hf_config, "n_routed_experts"))
        self.expected_names = {
            f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
            for layer in real_nvfp4_moe_layer_indices(hf_config)
            for expert in range(num_experts)
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        self.seen_names = set()

    def check_bucket(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        for name, tensor in weights:
            if name.endswith(_PACKED_SUFFIXES):
                raise RuntimeError(f"real NVFP4 BF16 refit unexpectedly contained packed tensor {name}")
            if not _EXPERT_WEIGHT_RE.match(name):
                continue
            if name not in self.expected_names:
                raise RuntimeError(f"real NVFP4 refit contains unexpected expert weight {name}")
            if name in self.seen_names:
                raise RuntimeError(f"real NVFP4 refit contains duplicate expert weight {name}")
            if tensor.dtype not in {torch.bfloat16, torch.float16, torch.float32}:
                raise RuntimeError(f"real NVFP4 expert refit tensor must be floating point, got {name}: {tensor.dtype}")
            self.seen_names.add(name)

    def finish(self) -> None:
        missing = sorted(self.expected_names - self.seen_names)
        if missing:
            raise RuntimeError(f"real NVFP4 refit is missing {len(missing)} expert weights: {missing[:8]}")


@torch.no_grad()
def _attest_native_scale_references(module, experts) -> None:
    """Check both original-storage references and derived scale values after refit."""
    quant_config = getattr(experts, "quant_config", None)
    for config_name, parameter_name in (
        ("g1_alphas", "w13_weight_scale_2"),
        ("g2_alphas", "w2_weight_scale_2"),
        ("w1_scale", "w13_weight_scale"),
        ("w2_scale", "w2_weight_scale"),
        ("a1_gscale", "nvfp4_a1_gscale"),
        ("a2_gscale", "nvfp4_a2_gscale"),
    ):
        actual = getattr(quant_config, config_name, None)
        registered = getattr(module, parameter_name, None)
        if not isinstance(actual, torch.Tensor) or not isinstance(registered, torch.Tensor):
            raise RuntimeError(f"native NVFP4 scale reference missing: {config_name}/{parameter_name}")
        if actual.device != registered.device or actual.data_ptr() != registered.data_ptr():
            raise RuntimeError(f"native NVFP4 stale scale reference: {config_name}/{parameter_name}")
    for config_name, input_name in (("a1_gscale", "w13_input_scale"), ("a2_gscale", "w2_input_scale")):
        actual = getattr(quant_config, config_name)
        input_scale = getattr(module, input_name, None)
        if (
            not isinstance(input_scale, torch.Tensor)
            or not torch.isfinite(input_scale).all()
            or not (input_scale > 0).all()
            or not torch.equal(actual, 1.0 / input_scale)
        ):
            raise RuntimeError(f"native NVFP4 {config_name} does not match current activation scale after sleep/refit")
    actual = getattr(experts, "g1_scale_c", None)
    registered = getattr(module, "g1_scale_c", None)
    if not isinstance(actual, torch.Tensor) or not isinstance(registered, torch.Tensor):
        raise RuntimeError("native NVFP4 derived scale g1_scale_c is missing")
    if actual.device != registered.device or actual.data_ptr() != registered.data_ptr():
        raise RuntimeError("native NVFP4 stale eager/CUDA-graph g1_scale_c reference")
    a2_gscale = getattr(quant_config, "a2_gscale", None)
    if not isinstance(a2_gscale, torch.Tensor):
        raise RuntimeError("native NVFP4 activation scale a2_gscale is missing")
    expected = a2_gscale
    if experts.moe_config.is_act_and_mul:
        expected = quant_config.g1_alphas * a2_gscale
    if not torch.equal(registered, expected):
        raise RuntimeError("native NVFP4 g1_scale_c does not match current weight/activation scales")


def _attest_native_moe_module(module) -> None:
    """Check one routed-expert module is packed, per-token and refit-consistent."""
    if not getattr(module, "_already_called_process_weights_after_loading", False):
        raise RuntimeError("native NVFP4 MoE weights were not processed after loading")
    for name in ("w13_weight", "w2_weight"):
        tensor = getattr(module, name, None)
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.uint8:
            raise RuntimeError(f"native NVFP4 MoE {name} is not packed uint8: {getattr(tensor, 'dtype', None)}")
    for name in ("w13_weight_scale", "w2_weight_scale"):
        tensor = getattr(module, name, None)
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float8_e4m3fn:
            raise RuntimeError(f"native NVFP4 MoE {name} is not FP8 block scale: {getattr(tensor, 'dtype', None)}")
    quant_method = module.quant_method
    backend = getattr(quant_method, "nvfp4_backend", None)
    backend_name = getattr(backend, "name", str(backend))
    if backend_name != "FLASHINFER_TRTLLM":
        raise RuntimeError(
            f"native per-token NVFP4 rollout requires the FlashInfer TRT-LLM backend, got {backend_name}"
        )
    experts = getattr(getattr(quant_method, "moe_kernel", None), "fused_experts", None)
    if not getattr(experts, "per_token_activation", False):
        raise RuntimeError("native NVFP4 MoE kernel is not executing per-token activation quantization")
    _attest_native_scale_references(module, experts)


def attest_vllm_native_nvfp4_runtime(
    model: torch.nn.Module,
    *,
    quantized_layer_indices: Collection[int],
    bf16_layer_indices: Collection[int],
) -> None:
    """Prove the exact routed-expert layer partition used by vLLM rollout."""

    expected_quantized = set(quantized_layer_indices)
    expected_bf16 = set(bf16_layer_indices)
    if expected_quantized & expected_bf16:
        raise ValueError("expected quantized and BF16 MoE layer sets overlap")

    quantized = set()
    unquantized = set()
    # vLLM's FusedMoE factory receives the quantization prefix ending in
    # `.mlp.experts`, then returns an MoERunner whose actual RoutedExperts
    # submodule is usually named `.mlp.experts.routed_experts`.
    layer_pattern = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.experts(?:\.routed_experts)?$")
    for module_name, module in model.named_modules():
        method_name = type(getattr(module, "quant_method", None)).__name__
        layer_match = layer_pattern.search(module_name)
        if layer_match and method_name == "UnquantizedFusedMoEMethod":
            unquantized.add(int(layer_match.group(1)))
        if method_name != "Nvfp4OnlineMoEMethod":
            continue
        if layer_match is None:
            raise RuntimeError(f"native NVFP4 MoE has an unrecognized module path: {module_name!r}")
        layer_index = int(layer_match.group(1))
        if layer_index in quantized:
            raise RuntimeError(f"native NVFP4 MoE layer {layer_index} was found twice")
        quantized.add(layer_index)
        _attest_native_moe_module(module)

    if quantized != expected_quantized:
        raise RuntimeError(
            "native NVFP4 rollout quantized the wrong MoE layers: "
            f"expected={sorted(expected_quantized)}, got={sorted(quantized)}"
        )
    missing_bf16 = expected_bf16 - unquantized
    if missing_bf16:
        raise RuntimeError(
            "native NVFP4 rollout did not leave the requested MoE layers unquantized: "
            f"missing_bf16={sorted(missing_bf16)}, observed_unquantized={sorted(unquantized)}"
        )
