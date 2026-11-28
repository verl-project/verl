# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Quantized vLLM rollout support: startup patches, backend defaults and refit helpers.

vLLM's reload (``start_reload`` / ``finish_reload``) restores the checkpoint
layout of every quantization scheme for a refit; this module only keeps the
quantization features verl adds on top of vLLM.
"""

import logging
import os
from collections.abc import Mapping
from functools import wraps
from typing import Any
from unittest.mock import patch

import torch

from verl.utils.vllm.vllm_fp4_utils import (
    install_mxfp4_mxfp8_compressed_tensors_patch,
    is_deepseek_v4_model,
)
from verl.utils.vllm.vllm_nvfp4_per_token_utils import install_nvfp4_per_token_compressed_tensors_patch

logger = logging.getLogger(__name__)


def _is_nvfp4_quantization(quantization_config: Mapping[str, Any]) -> bool:
    """Whether a quantization config contains the NVFP4 format."""
    values = []

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            for nested in value.values():
                collect(nested)
        elif isinstance(value, list | tuple):
            for nested in value:
                collect(nested)
        elif isinstance(value, str):
            values.append(value.lower())

    collect(quantization_config)
    return any("nvfp4" in value for value in values)


def _uses_fp32_block_scales(quantization_config: Mapping[str, Any]) -> bool:
    """Whether FP8 block weights use checkpoint-layout FP32 scales."""
    for group in quantization_config.get("config_groups", {}).values():
        weights = group.get("weights", {})
        if weights.get("strategy") == "block" and weights.get("scale_dtype") in {None, "torch.float32"}:
            return True
    block_size = quantization_config.get("weight_block_size")
    return block_size == [128, 128] and quantization_config.get("scale_fmt") not in {"ue8m0", "e8m0"}


def _is_hopper() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_device_capability(90)


def _should_auto_disable_deep_gemm(hf_config: Any) -> bool:
    if hf_config is None:
        return False
    text_config = getattr(hf_config, "text_config", None)
    model_type = getattr(text_config, "model_type", None) or getattr(hf_config, "model_type", None)
    if model_type is None:
        return False
    try:
        from vllm.utils.deep_gemm import should_auto_disable_deep_gemm
    except ImportError:
        return False
    return bool(should_auto_disable_deep_gemm(model_type))


def set_default_vllm_quantization_backends(
    quantization_config: Mapping[str, Any] | None,
    engine_kwargs: dict[str, Any],
    hf_config: Any = None,
) -> None:
    """Default non-NVFP4 quantization backends to DeepGEMM."""
    if hasattr(quantization_config, "to_dict"):
        quantization_config = quantization_config.to_dict()
    if not isinstance(quantization_config, Mapping) or _is_nvfp4_quantization(quantization_config):
        return
    # DeepGEMM is NVIDIA-only; on ROCm, vLLM's automatic selection picks the AITER / Triton kernels.
    if torch.version.hip is not None:
        return
    # vLLM auto-disables DeepGEMM on certain Blackwell model types (e.g. qwen3_5_moe_text);
    # pinning deep_gemm there leaves ScaledMM / FusedMoE with no eligible kernel.
    if _should_auto_disable_deep_gemm(hf_config):
        return
    for option in ("linear_backend", "moe_backend"):
        engine_kwargs.setdefault(option, "deep_gemm")
    if (
        "deep_gemm" in (engine_kwargs["linear_backend"], engine_kwargs["moe_backend"])
        and _is_hopper()
        and _uses_fp32_block_scales(quantization_config)
    ):
        os.environ["VLLM_USE_DEEP_GEMM_E8M0"] = "0"


_vllm_patches: list = []


def is_quantized_model(vllm_config):
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    if hasattr(vllm_config, "quant_config"):
        if isinstance(vllm_config.quant_config, Fp8Config):
            return True
        elif is_compressed_tensors_model(vllm_config):
            return True

    return False


def is_compressed_tensors_model(vllm_config) -> bool:
    """Compressed-tensors rollout, whose refit weights the trainer already quantized."""
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

    return isinstance(getattr(vllm_config, "quant_config", None), CompressedTensorsConfig)


def configure_fp8_ignored_layers_match_mode(vllm_config) -> None:
    """Apply verl's optional native-FP8 ignore matching extension."""
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    quant_config = getattr(vllm_config, "quant_config", None)
    if not isinstance(quant_config, Fp8Config):
        return
    hf_config = getattr(getattr(vllm_config, "model_config", None), "hf_config", None)
    serialized = getattr(hf_config, "quantization_config", None)
    if hasattr(serialized, "to_dict"):
        serialized = serialized.to_dict()
    if not isinstance(serialized, dict):
        return
    match_mode = serialized.get("ignored_layers_match_mode", "exact")
    if match_mode not in ("exact", "substring", "suffix"):
        raise ValueError(f"Unsupported ignored_layers_match_mode={match_mode!r}")
    quant_config.ignored_layers_match_mode = match_mode


def refresh_rocm_attention_weight_caches(model):
    """Rebuild the bf16 ``wo_a`` copy vLLM derives from the loaded weight.

    ``rocm_aiter_mla_sparse._get_cached_wo_a_bf16`` caches a dequantized ``wo_a``
    on the module, assuming the weight is static. A refit updates the weight but
    not the cache, so attention would keep using the previous step's copy.

    Dropping the cache and letting the lazy rebuild handle it is only correct in
    eager mode. ``_o_proj`` runs during graph capture, so a captured graph holds
    the buffer's address and replays the einsum without re-entering the builder:
    a fresh allocation would leave it reading freed memory. Rebuild into the
    original storage instead.

    Must run after ``finish_reload``, which re-applies the inference layout.

    Only ROCm DeepSeek-V4 builds this cache, hence the guard below.
    """
    if torch.version.hip is None or not is_deepseek_v4_model(model):
        return

    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _get_cached_wo_a_bf16

    # The cache was built inside the forward pass, so it is an inference tensor
    # and PyTorch refuses to mutate it outside InferenceMode.
    with torch.inference_mode():
        for module in model.modules():
            live = getattr(module, "_dsv4_wo_a_bf16", None)
            if live is None:
                continue
            n_local_groups, o_lora_rank, hidden_dim = live.shape
            del module._dsv4_wo_a_bf16
            rebuilt = _get_cached_wo_a_bf16(module, n_local_groups, o_lora_rank, hidden_dim)
            if rebuilt.data_ptr() != live.data_ptr():
                live.copy_(rebuilt)
            module._dsv4_wo_a_bf16 = live


def reset_uncalibrated_kv_cache_scales(model):
    """Reset the q/k/v/prob attention scales to 1.0 when the quant config has no KV cache scheme.

    With a compressed-tensors config, vLLM attaches a ``CompressedTensorsKVCacheMethod`` to every
    attention layer even when ``kv_cache_scheme`` is None. ``load_format="dummy"`` fills its
    q/k/v_scale placeholders with random values, and its ``process_weights_after_loading`` adopts
    them as the live scales: unlike ``BaseKVCacheMethod``, it has no fallback to 1.0 for scales
    the checkpoint did not provide. A refit sends no attention scales and re-processes only the
    linear/MoE layers, so with ``kv_cache_dtype="fp8"`` the rollout keeps quantizing Q/K/V with
    those random scales and produces garbage.

    Configs that do carry a ``kv_cache_scheme`` are left untouched.
    """
    try:
        from vllm.model_executor.layers.attention.attention import set_default_quant_scales
        from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
    except ImportError:
        return
    for module in model.modules():
        quant_method = getattr(module, "quant_method", None)
        if not isinstance(quant_method, BaseKVCacheMethod):
            continue
        if getattr(quant_method.quant_config, "kv_cache_scheme", None) is not None:
            continue
        # Older vLLM keeps the scales as the placeholder Parameters, which require grad.
        with torch.no_grad():
            set_default_quant_scales(module, register_buffer=False)


def _reset_uncalibrated_kv_cache_scales_after_loading(process_weights_after_loading):
    """Reset dummy-loaded compressed-tensors KV scales before engine warmup.

    Blackwell's FlashInfer TRT-LLM backend copies q/k/v scales into host-side
    ``bmm1_scale`` / ``bmm2_scale`` values during warmup. Resetting only after
    the first live refit is too late because CUDA graphs have already captured
    those values. Patch the initial compressed-tensors post-load hook so an
    absent KV cache scheme means 1.0 from the start.
    """

    @wraps(process_weights_after_loading)
    def wrapped(quant_method, layer, *args, **kwargs):
        result = process_weights_after_loading(quant_method, layer, *args, **kwargs)
        if getattr(quant_method.quant_config, "kv_cache_scheme", None) is not None:
            return result

        from vllm.model_executor.layers.attention.attention import set_default_quant_scales

        with torch.no_grad():
            set_default_quant_scales(layer, register_buffer=False)
        return result

    return wrapped


def _build_compressed_tensors_kv_cache_scale_patcher():
    """Patch compressed-tensors KV scale initialization when vLLM provides it."""
    try:
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsKVCacheMethod,
        )
    except ImportError:
        logger.info("compressed-tensors KV cache method not found; skipping its startup scale patch")
        return None

    method_path = (
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors."
        "CompressedTensorsKVCacheMethod.process_weights_after_loading"
    )
    return patch(
        method_path,
        _reset_uncalibrated_kv_cache_scales_after_loading(CompressedTensorsKVCacheMethod.process_weights_after_loading),
    )


def apply_vllm_quant_patches():
    """Install verl's quantization patches. Must run before the engine builds."""
    install_mxfp4_mxfp8_compressed_tensors_patch()
    install_nvfp4_per_token_compressed_tensors_patch()
    if _vllm_patches:
        logger.debug("vLLM quantization patches already applied")
        return
    kv_cache_scale_patcher = _build_compressed_tensors_kv_cache_scale_patcher()
    if kv_cache_scale_patcher is not None:
        kv_cache_scale_patcher.start()
        _vllm_patches.append(kv_cache_scale_patcher)
