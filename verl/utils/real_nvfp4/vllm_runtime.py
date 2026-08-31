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

import inspect
import logging
from importlib.metadata import version

import torch
from packaging.version import Version

logger = logging.getLogger(__name__)

NVFP4_PER_TOKEN_METHOD = "nvfp4_per_token"
REAL_NVFP4_MOE_BACKEND = "flashinfer_trtllm"


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


def attest_vllm_native_nvfp4_runtime(
    model: torch.nn.Module,
    *,
    expected_moe_layers: int,
) -> dict[str, int]:
    """Prove every routed-expert layer executes native per-token W4A4."""

    moe_count = 0
    for module in model.modules():
        quant_method = getattr(module, "quant_method", None)
        if type(quant_method).__name__ != "Nvfp4OnlineMoEMethod":
            continue
        moe_count += 1
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

    if moe_count != expected_moe_layers:
        raise RuntimeError(
            f"native NVFP4 rollout MoE-layer count mismatch: expected {expected_moe_layers}, got {moe_count}"
        )
    logger.warning(
        "VERL_REAL_NVFP4_ROLLOUT_ATTESTATION PASS dense_layers=0 moe_layers=%d expected=%d "
        "method=vllm_native_nvfp4_per_token backend=FLASHINFER_TRTLLM "
        "scope=routed_expert_mlp attention=bf16 activation=per_token",
        moe_count,
        expected_moe_layers,
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
        )
        for tensor_index, name in enumerate(fingerprint_tensors, start=1):
            flat = getattr(module, name).view(torch.uint8).flatten()
            # Packed weights and FP8 block scales are sampled; the small FP32
            # global-scale tensors are read in full. This makes refit-change
            # detection sensitive to scale changes even when a 1e-6 optimizer
            # step does not cross many 4-bit bins.
            sample_size = flat.numel() if name.endswith("_scale_2") else 2048
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
