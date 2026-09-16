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

"""Runtime proofs for vLLM 0.26's native online NVFP4 MoE path."""

import ast
import hashlib
import inspect
import logging
import re
import textwrap
from collections.abc import Collection
from functools import lru_cache
from importlib.metadata import version

import torch
from packaging.version import Version

logger = logging.getLogger(__name__)

NVFP4_PER_TOKEN_METHOD = "nvfp4_per_token"
REAL_NVFP4_MOE_BACKEND = "flashinfer_trtllm"

# Canonical function ASTs from vLLM v0.26.0 with upstream fixes #50029
# (9c22668436a4d94aab87ea74a220e060415cf1d8) and #50074
# (3ac9525507b2d0de5c1b08cbca96cc94850c7c7a). These are the exact
# implementations installed by runtime_backports/apply_vllm_online_nvfp4_50029_50074.py,
# including the local fresh-postprocess/retained-kernel fix: #50074 alone leaves
# TRTLLM's derived g1_scale_c one refit behind and rebinds eager references.
# Reciprocal activation scales are also registered for level-2 sleep/refit.
# Converted activation scales have writable storage, not stride-zero views.
# Unlike a version/marker check, this also rejects a partially patched wheel.
_NVFP4_BACKPORT_AST_SHA256 = {
    "_quantize_moe_weight_to_nvfp4": "11c7d914f31e74d425151fd3ea54b1a6e8aa92ea001aa7ddd6b9eafa30ab187a",
    "_setup_kernel": "b5a3160ff41a3eefc246e1b80ee7d51808cd8c7e0f6524cbf4f232b60a9410a7",
}


def _function_ast_sha256(function) -> str:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    # Python 3.12 adds this empty field; normalize for the supported 3.10–3.12
    # interpreters. ASTs already ignore whitespace, line numbers and comments.
    canonical = ast.dump(tree, include_attributes=False).replace(", type_params=[]", "")
    return hashlib.sha256(canonical.encode()).hexdigest()


@lru_cache(maxsize=1)
def require_vllm_nvfp4_backports() -> None:
    """Reject unaudited packing/reload implementations before model creation.

    This is an exact audited-implementation guard, not a general claim that
    every semantically equivalent implementation can be recognized. A future
    vLLM upgrade must review and update the contract together with its tests.
    """
    from vllm.model_executor.layers.quantization.online.nvfp4 import (
        Nvfp4OnlineMoEMethod,
        _quantize_moe_weight_to_nvfp4,
    )

    implementations = {
        "_quantize_moe_weight_to_nvfp4": _quantize_moe_weight_to_nvfp4,
        "_setup_kernel": Nvfp4OnlineMoEMethod._setup_kernel,
    }
    for name, function in implementations.items():
        try:
            actual = _function_ast_sha256(function)
        except (OSError, TypeError, SyntaxError) as exc:
            raise RuntimeError(f"cannot verify required vLLM NVFP4 backports for {name}") from exc
        if actual != _NVFP4_BACKPORT_AST_SHA256[name]:
            raise RuntimeError(
                f"real_nvfp4 requires audited vLLM #50029/#50074 and derived-scale backports: {name} "
                f"has unrecognized implementation {actual}. Use the validated runtime build; "
                "the unmodified vLLM 0.26 wheel is not sufficient."
            )


def require_vllm_native_nvfp4_per_token(vllm_config) -> None:
    """Fail unless this worker was built for vLLM's native online method."""

    current = Version(version("vllm"))
    if current != Version("0.26.0"):
        raise RuntimeError(f"real_nvfp4 requires the audited vLLM 0.26.0 native path, got {current}")
    model_config = getattr(vllm_config, "model_config", None)
    quantization = getattr(model_config, "quantization", None)
    if quantization != NVFP4_PER_TOKEN_METHOD:
        raise RuntimeError(
            f"real_nvfp4 worker quantization drifted: expected {NVFP4_PER_TOKEN_METHOD!r}, got {quantization!r}"
        )
    require_vllm_nvfp4_backports()


def require_vllm_native_reload_contract(model_runner) -> None:
    """Fail before refit unless the exact native layerwise API is present."""

    reload_weights = getattr(model_runner, "reload_weights", None)
    if reload_weights is None:
        raise RuntimeError("vLLM model runner has no native reload_weights API")
    parameters = inspect.signature(reload_weights).parameters
    expected = ("weights_iterator", "weights_path", "is_checkpoint_format")
    if tuple(parameters) != expected:
        raise RuntimeError(
            f"vLLM native reload_weights API drifted: expected parameters {expected}, got {tuple(parameters)}"
        )
    checkpoint_parameter = parameters["is_checkpoint_format"]
    if checkpoint_parameter.default is not True:
        raise RuntimeError("vLLM native reload_weights no longer defaults is_checkpoint_format=True")


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


def attest_vllm_native_nvfp4_runtime(
    model: torch.nn.Module,
    *,
    expected_moe_layers: int | None = None,
    expected_quantized_layer_indices: Collection[int] | None = None,
    expected_bf16_layer_indices: Collection[int] = (),
) -> dict[str, int]:
    """Prove the exact routed-expert layer partition used by vLLM rollout."""

    moe_count = 0
    quantized_layer_indices = set()
    unquantized_layer_indices = set()
    # vLLM 0.26's FusedMoE factory receives the quantization prefix ending in
    # `.mlp.experts`, then returns an MoERunner whose actual RoutedExperts
    # submodule is usually named `.mlp.experts.routed_experts`.
    layer_pattern = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.experts(?:\.routed_experts)?$")
    for module_name, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        layer_match = layer_pattern.search(module_name)
        if layer_match and type(quant_method).__name__ == "UnquantizedFusedMoEMethod":
            unquantized_layer_indices.add(int(layer_match.group(1)))
        if type(quant_method).__name__ != "Nvfp4OnlineMoEMethod":
            continue
        moe_count += 1
        if expected_quantized_layer_indices is not None:
            if layer_match is None:
                raise RuntimeError(f"native NVFP4 MoE has an unrecognized module path: {module_name!r}")
            quantized_layer_indices.add(int(layer_match.group(1)))
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
        experts = getattr(getattr(quant_method, "moe_kernel", None), "fused_experts", None)
        backend = getattr(quant_method, "nvfp4_backend", None)
        backend_name = getattr(backend, "name", str(backend))
        if backend_name != "FLASHINFER_TRTLLM":
            raise RuntimeError(
                f"native per-token NVFP4 rollout requires the FlashInfer TRT-LLM backend, got {backend_name}"
            )
        if not getattr(experts, "per_token_activation", False):
            raise RuntimeError("native NVFP4 MoE kernel is not executing per-token activation quantization")
        _attest_native_scale_references(module, experts)

    if expected_quantized_layer_indices is not None:
        expected_quantized = set(expected_quantized_layer_indices)
        expected_bf16 = set(expected_bf16_layer_indices)
        if expected_quantized & expected_bf16:
            raise ValueError("expected quantized and BF16 MoE layer sets overlap")
        if quantized_layer_indices != expected_quantized:
            raise RuntimeError(
                "native NVFP4 rollout quantized the wrong MoE layers: "
                f"expected={sorted(expected_quantized)}, got={sorted(quantized_layer_indices)}"
            )
        missing_bf16 = expected_bf16 - unquantized_layer_indices
        if missing_bf16:
            raise RuntimeError(
                "native NVFP4 rollout did not leave the requested MoE layers unquantized: "
                f"missing_bf16={sorted(missing_bf16)}, observed_unquantized={sorted(unquantized_layer_indices)}"
            )
        expected_moe_layers = len(expected_quantized)
    if expected_moe_layers is None:
        raise ValueError("an expected MoE layer count or exact quantized layer set is required")
    if moe_count != expected_moe_layers:
        raise RuntimeError(
            f"native NVFP4 rollout MoE-layer count mismatch: expected {expected_moe_layers}, got {moe_count}"
        )
    logger.warning(
        "VERL_REAL_NVFP4_ROLLOUT_ATTESTATION PASS dense_layers=0 moe_layers=%d expected=%d bf16_moe_layers=%s "
        "method=vllm_native_nvfp4_per_token backend=FLASHINFER_TRTLLM "
        "scope=routed_expert_mlp attention=bf16 activation=per_token scale_references=current derived_scale=current",
        moe_count,
        expected_moe_layers,
        sorted(expected_bf16_layer_indices),
    )
    return {"dense_layers": 0, "moe_layers": moe_count}


@torch.no_grad()
def vllm_native_nvfp4_fingerprint(model: torch.nn.Module) -> int:
    """Return a cheap fingerprint sampled from every packed expert layer."""

    fingerprint: torch.Tensor | None = None
    layer_index = 0
    for module in model.modules():
        quant_method = getattr(module, "quant_method", None)
        if type(quant_method).__name__ != "Nvfp4OnlineMoEMethod":
            continue
        layer_index += 1
        fingerprint_tensors = (
            "w13_weight",
            "w2_weight",
            "w13_weight_scale",
            "w2_weight_scale",
            "w13_weight_scale_2",
            "w2_weight_scale_2",
            "g1_scale_c",
        )
        for tensor_index, name in enumerate(fingerprint_tensors, start=1):
            flat = getattr(module, name).view(torch.uint8).flatten()
            # Packed weights and FP8 block scales are sampled; the small FP32
            # global-scale tensors are read in full. This makes refit-change
            # detection sensitive to scale changes even when a 1e-6 optimizer
            # step does not cross many 4-bit bins.
            sample_size = flat.numel() if name.endswith("_scale_2") or name == "g1_scale_c" else 2048
            stride = max(flat.numel() // sample_size, 1)
            sample = flat[::stride][:sample_size].to(torch.int64)
            coefficients = torch.arange(
                1,
                sample.numel() + 1,
                device=sample.device,
                dtype=torch.int64,
            )
            value = (sample * coefficients).sum() * (2 * layer_index + tensor_index)
            fingerprint = value if fingerprint is None else fingerprint + value
    if layer_index == 0 or fingerprint is None:
        raise RuntimeError("native NVFP4 fingerprint found no quantized MoE layers")
    return int(fingerprint.item())
