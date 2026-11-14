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
import re
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import wraps
from typing import Any
from unittest.mock import patch

import torch
from packaging import version

from verl.utils.vllm.vllm_fp4_utils import (
    install_mxfp4_mxfp8_compressed_tensors_patch,
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
from verl.utils.vllm.vllm_nvfp4_per_token_utils import install_nvfp4_per_token_compressed_tensors_patch
from verl.utils.vllm.vllm_unquant_utils import (
    fold_unquantized_moe_params,
    stage_unquantized_moe_params,
    unquantized_moe_modules,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _collect_strings(value: Any, values: list[str] | None = None) -> list[str]:
    """Every string nested anywhere in a quantization config, lower-cased."""
    if values is None:
        values = []
    if isinstance(value, Mapping):
        for nested in value.values():
            _collect_strings(nested, values)
    elif isinstance(value, list | tuple):
        for nested in value:
            _collect_strings(nested, values)
    elif isinstance(value, str):
        values.append(value.lower())
    return values


def _is_nvfp4_quantization(quantization_config: Mapping[str, Any]) -> bool:
    """Whether a quantization config contains the NVFP4 format."""
    return any("nvfp4" in value for value in _collect_strings(quantization_config))


def _expert_linear_names(hf_config: Any):
    """The unfused expert Linear names (``model.layers.<l>.mlp.experts.<e>.<proj>``) compressed-tensors resolves.

    Layer and expert counts come from the model config (``text_config`` for multimodal wrappers); without one, a
    single probe name is used, which is enough for layer-agnostic targets and ``ignore`` patterns.
    """
    config = getattr(hf_config, "text_config", None) or hf_config
    layers = int(getattr(config, "num_hidden_layers", 0) or 0) or 1
    experts = 1
    for key in ("n_routed_experts", "num_experts", "num_local_experts"):
        value = getattr(config, key, None)
        if value:
            experts = int(value)
            break
    for layer in range(layers):
        for expert in range(experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                yield f"model.layers.{layer}.mlp.experts.{expert}.{proj}"


def _matches_target(name: str, targets: Any, *, class_names: bool) -> bool:
    """Whether a compressed-tensors ``targets`` / ``ignore`` entry reaches ``name`` (``re:`` regex or exact name,
    plus the ``Linear`` class when ``class_names`` is set)."""
    if isinstance(targets, str):
        targets = [targets]
    for target in targets or ():
        if not isinstance(target, str):
            continue
        if target.startswith("re:"):
            if re.match(target[3:], name):
                return True
        elif target == name or (class_names and target == "Linear"):
            return True
    return False


def _is_mxfp4_group(group: Mapping[str, Any]) -> bool:
    weights = group.get("weights") or {}
    if isinstance(weights, Mapping) and weights.get("num_bits") == 4:
        if str(weights.get("type", "")).lower() == "float":
            return True
    return any("mxfp4" in value for value in _collect_strings(group))


def _has_mxfp4_experts(
    quantization_config: Mapping[str, Any], hf_config: Any = None, checkpoint_quantization: bool = True
) -> bool:
    """Whether any routed expert is stored in MXFP4.

    Native DeepSeek-V4 checkpoints keep only the FP8 block config under
    ``quantization_config`` and declare the expert format as ``expert_dtype``
    (top-level in V4, inside ``quantization_config`` in V4.1); the ``-Base``
    checkpoints say ``fp8`` there and keep DeepGEMM on every architecture. That
    declaration only counts while the rollout keeps the checkpoint's
    quantization (``checkpoint_quantization``); a rollout override requantizes
    the experts too. Compressed-tensors configs are resolved per expert module
    the way compressed-tensors does it: an ``ignore`` entry wins, then a group
    that targets the module by name or regex, then a plain ``Linear`` group,
    and one MXFP4 expert anywhere in the model is enough (forcing DeepGEMM
    would fail to load it on Hopper).
    """
    if checkpoint_quantization:
        expert_dtypes = (
            getattr(hf_config, "expert_dtype", None),
            quantization_config.get("expert_dtype"),  # DeepSeek-V4.1 nests it under quantization_config
        )
        if any(str(dtype or "").lower() in {"fp4", "mxfp4"} for dtype in expert_dtypes):
            return True
    groups = quantization_config.get("config_groups")
    if not isinstance(groups, Mapping) or not groups:
        return any("mxfp4" in value for value in _collect_strings(quantization_config))
    groups = [group for group in groups.values() if isinstance(group, Mapping)]
    mxfp4 = [_is_mxfp4_group(group) for group in groups]
    if not any(mxfp4):
        return False
    ignore = quantization_config.get("ignore")
    group_targets = [group.get("targets") or ["Linear"] for group in groups]
    for name in _expert_linear_names(hf_config):
        if _matches_target(name, ignore, class_names=False):
            continue
        # A name / regex target wins over the ``Linear`` class name (a group without targets means ``Linear``).
        by_name = [i for i, targets in enumerate(group_targets) if _matches_target(name, targets, class_names=False)]
        by_class = [i for i, targets in enumerate(group_targets) if _matches_target(name, targets, class_names=True)]
        governing = (by_name or by_class or [None])[0]
        if governing is not None and mxfp4[governing]:
            return True
    return False


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
    hf_config: Any = None,
    checkpoint_quantization: bool = True,
) -> None:
    """Default non-NVFP4 quantization backends to DeepGEMM.

    ``VERL_ROLLOUT_USE_DEEP_GEMM=0`` leaves every backend to vLLM's automatic
    selection. MXFP4 routed experts keep vLLM's selection on Hopper, where vLLM
    has no DeepGEMM MXFP4 kernel and picks Marlin, the refittable backend there;
    forcing ``moe_backend=deep_gemm`` would fail at ``load_model``. ``hf_config``
    is the model config (layer / expert counts, native ``expert_dtype``);
    ``checkpoint_quantization`` is False when a rollout quantization override
    replaced the checkpoint's own, which requantizes the experts as well.

    On Hopper, FP8 block checkpoints with FP32 scales always disable DeepGEMM's
    E8M0 mode when DeepGEMM may run, whether verl set it, the user set it or
    vLLM's own selection may pick it.
    """
    if hasattr(quantization_config, "to_dict"):
        quantization_config = quantization_config.to_dict()
    if not isinstance(quantization_config, Mapping) or _is_nvfp4_quantization(quantization_config):
        return
    # DeepGEMM is NVIDIA-only; on ROCm, vLLM's automatic selection picks the AITER / Triton kernels.
    if torch.version.hip is not None:
        return
    opt_out = os.getenv("VERL_ROLLOUT_USE_DEEP_GEMM", "1") == "0"
    if opt_out:
        logger.info("VERL_ROLLOUT_USE_DEEP_GEMM=0: linear_backend and moe_backend are left to vLLM's selection")
    else:
        engine_kwargs.setdefault("linear_backend", "deep_gemm")
        if "moe_backend" not in engine_kwargs:
            if _is_hopper() and _has_mxfp4_experts(quantization_config, hf_config, checkpoint_quantization):
                logger.info(
                    "MXFP4 routed experts on Hopper: moe_backend is left to vLLM's automatic selection "
                    "(vLLM's DeepGEMM MXFP4 kernel needs SM100; Marlin is the refittable backend here)"
                )
            else:
                engine_kwargs["moe_backend"] = "deep_gemm"
    # DeepGEMM may run when verl or the user named it, or when a backend is unset / "auto" and vLLM picks it.
    deep_gemm_possible = any(
        engine_kwargs.get(option) in (None, "auto", "deep_gemm") for option in ("linear_backend", "moe_backend")
    )
    if deep_gemm_possible and _is_hopper() and _uses_fp32_block_scales(quantization_config):
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
        # Routed experts a quantized model keeps in BF16 (e.g. its first and last layers).
        "unquantized_moe_modules": unquantized_moe_modules(model),
        "staged_unquantized_moe_modules": stage_unquantized_moe_params(model),
    }
    return reload_state


def process_quanted_weights_after_loading(model, reload_state):
    """Re-apply the inference layout undone by ``prepare_quanted_weights_for_loading``."""
    reload_state = reload_state or {}
    process_fp8_weights_after_loading(reload_state.get("fp8_layers") or [])
    process_mxfp4_moe_weights_after_loading(reload_state.get("mxfp4_moe_modules") or [])
    process_nvfp4_moe_weights_after_loading(reload_state.get("nvfp4_moe_modules") or [])
    # Every BF16 expert layer, staged or not: a kernel prep that keeps the shape can still
    # reorder the weights (FlashInfer CUTLASS swaps gate and up).
    with fold_unquantized_moe_params(reload_state.get("staged_unquantized_moe_modules") or []):
        for module in reload_state.get("unquantized_moe_modules") or []:
            module.quant_method.process_weights_after_loading(module)
    reset_uncalibrated_kv_cache_scales(model)
    # Last: the rebuild reads ``wo_a``, which is only back in its inference
    # layout once the staged FP8 params above have been reinstated.
    refresh_rocm_attention_weight_caches(model)


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
    install_nvfp4_per_token_compressed_tensors_patch()
    if fp8_state.vllm_patches:
        logger.debug("vLLM quantization patches already applied")
        return

    patchers = build_fp8_method_patchers(_get_vllm_version())
    kv_cache_scale_patcher = _build_compressed_tensors_kv_cache_scale_patcher()
    if kv_cache_scale_patcher is not None:
        patchers.append(kv_cache_scale_patcher)

    for patcher in patchers:
        patcher.start()
        fp8_state.vllm_patches.append(patcher)
