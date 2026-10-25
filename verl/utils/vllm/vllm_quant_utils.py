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

"""Entry point for refitting a quantized vLLM engine during RL rollout.

Everything the rollout worker calls lives here: startup patching
(``apply_vllm_quant_patches``), the load itself (``load_quanted_weights``), and
the prepare/finalize pair that brackets a bucketed transfer
(``prepare_quanted_weights_for_loading`` / ``process_quanted_weights_after_loading``).

The layout surgery each quantization scheme needs is delegated to a sibling
module: ``vllm_fp8_utils`` for ``Fp8LinearMethod`` / ``Fp8MoEMethod``, and
``vllm_fp4_utils`` for ``Mxfp4MoEMethod``. Each exposes a stage/process pair
that returns an empty result when the model has no layer it owns, so the
dispatch below stays unconditional.
"""

import importlib.metadata
import logging
import os
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch
from packaging import version

from verl.utils.vllm.vllm_fp4_utils import (
    install_mxfp4_mxfp8_compressed_tensors_patch,
    install_nvfp4_cutlass_compressed_tensors_patch,
    is_deepseek_v4_model,
    process_mxfp4_moe_weights_after_loading,
    process_nvfp4_moe_weights_after_loading,
    stage_mxfp4_moe_params_for_loading,
    stage_nvfp4_moe_params_for_loading,
)
from verl.utils.vllm.vllm_fp8_utils import (
    build_fp8_method_patchers,
    process_fp8_weights_after_loading,
    stage_fp8_params_for_loading,
)

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


def set_default_vllm_quantization_backends(
    quantization_config: Mapping[str, Any] | None,
    engine_kwargs: dict[str, Any],
) -> None:
    """Default non-NVFP4 quantization backends to DeepGEMM."""
    if hasattr(quantization_config, "to_dict"):
        quantization_config = quantization_config.to_dict()
    if not isinstance(quantization_config, Mapping) or _is_nvfp4_quantization(quantization_config):
        return
    # DeepGEMM is NVIDIA-only; on ROCm, vLLM's automatic selection picks the AITER / Triton kernels.
    if torch.version.hip is not None:
        return
    for option in ("linear_backend", "moe_backend"):
        engine_kwargs.setdefault(option, "deep_gemm")
    if (
        "deep_gemm" in (engine_kwargs["linear_backend"], engine_kwargs["moe_backend"])
        and _is_hopper()
        and _uses_fp32_block_scales(quantization_config)
    ):
        os.environ["VLLM_USE_DEEP_GEMM_E8M0"] = "0"


def _get_vllm_version():
    return version.parse(importlib.metadata.version("vllm"))


# Ref: https://github.com/NVIDIA-NeMo/RL/commit/bc24887c72a6e1b2699a228bc87c588546dfe6b7
@dataclass()
class FP8State:
    vllm_patches: list = field(default_factory=lambda: [])


fp8_state: FP8State = FP8State()


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

    Must run after the live FP8 parameters have been reinstated, otherwise the
    dequantization reads the staging buffers.

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


def prepare_quanted_weights_for_loading(model):
    """Restore quantized params to the layout their ``weight_loader`` expects.

    Must run once before the first bucket, and pairs with
    ``process_quanted_weights_after_loading``, which re-applies the inference
    layout once the last bucket has landed. Each backend selects its own work
    by inspecting the model and returns an empty result when it finds nothing,
    so this is safe to call for any quantization scheme. The returned value is
    opaque reload state for the paired call.
    """
    reload_state: dict[str, Any] = {
        "fp8_layers": stage_fp8_params_for_loading(model),
        "mxfp4_moe_modules": stage_mxfp4_moe_params_for_loading(model),
        "nvfp4_moe_modules": stage_nvfp4_moe_params_for_loading(model),
    }
    return reload_state


def process_quanted_weights_after_loading(model, reload_state):
    """Re-apply the inference layout undone by ``prepare_quanted_weights_for_loading``."""
    reload_state = reload_state or {}
    process_fp8_weights_after_loading(reload_state.get("fp8_layers") or [])
    process_mxfp4_moe_weights_after_loading(reload_state.get("mxfp4_moe_modules") or [])
    process_nvfp4_moe_weights_after_loading(reload_state.get("nvfp4_moe_modules") or [])
    # Last: the rebuild reads ``wo_a``, which is only back in its inference
    # layout once the staged FP8 params above have been reinstated.
    refresh_rocm_attention_weight_caches(model)


def load_quanted_weights(weights, model_runner, is_drafter=False):
    if is_drafter:
        drafter = getattr(model_runner, "drafter", None)
        model = drafter.model if drafter is not None and hasattr(drafter, "model") else None
        assert model is not None, (
            "load_quanted_weights(is_drafter=True) requires model_runner.drafter.model "
            "to be present and non-None for FP8 weight loading."
        )
    else:
        model = model_runner.model
    weights = list(weights)

    # Monkey patch the param class to their subclass, as certain models
    # will check the param type to call the proper weightloader
    for name, param in model.named_parameters():
        # Stable handle on the qualified name so the loaders below can name the
        # offending parameter when a shape check fails.
        param._verl_qualname = name
        if hasattr(param, "subclass_type"):
            param.orig_type = param.__class__
            param.__class__ = param.subclass_type
    # Finally load the weights into vllm.
    # MTP completeness check assumes a single full-checkpoint load; in RL refit
    # weights arrive bucketed, so disable it (like vLLM's own NCCL/IPC engines).
    # ``nullcontext`` covers older vLLM without the guard.
    try:
        from vllm.model_executor.model_loader.mtp_validation import disable_mtp_completeness_check
    except ImportError:
        disable_mtp_completeness_check = nullcontext
    try:
        with disable_mtp_completeness_check():
            loaded_params = model.load_weights(weights)
    finally:
        # Undo the type change above to the original type
        for name, param in model.named_parameters():
            if hasattr(param, "orig_type"):
                param.__class__ = param.orig_type
                del param.orig_type

    return loaded_params


def apply_vllm_quant_patches():
    """Install the vLLM patches the refit path depends on. Must run pre-engine.

    These have to be in place before the engine builds, because recording each
    layer's checkpoint layout happens inside the patched hook during the
    initial load. The MXFP4 refit path derives the checkpoint layout from the
    quant method; its startup patch only makes compressed-tensors honor MXFP8
    activations.
    """
    install_mxfp4_mxfp8_compressed_tensors_patch()
    install_nvfp4_cutlass_compressed_tensors_patch()
    if fp8_state.vllm_patches:
        logger.debug("vLLM quantization patches already applied")
        return

    for patcher in build_fp8_method_patchers(_get_vllm_version()):
        patcher.start()
        fp8_state.vllm_patches.append(patcher)
