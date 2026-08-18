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
(``apply_vllm_quant_patches``), the BF16 -> FP8 quantization of the outgoing
weight stream (``quant_weights``), the load itself (``load_quanted_weights``),
and the prepare/finalize pair that brackets a bucketed transfer
(``prepare_quanted_weights_for_loading`` / ``process_quanted_weights_after_loading``).

The layout surgery each quantization scheme needs is delegated to a sibling
module: ``vllm_fp8_utils`` for ``Fp8LinearMethod`` / ``Fp8MoEMethod``, and
``vllm_fp4_utils`` for ``Mxfp4MoEMethod``. Each exposes a stage/process pair
that returns an empty result when the model has no layer it owns, so the
dispatch below stays unconditional.
"""

import importlib.metadata
import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch
from packaging import version

try:
    from vllm.model_executor.layers.linear import LinearBase
except ImportError as e:
    raise ImportError("FP8 quantization not available") from e

try:
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
except ImportError:
    FusedMoE = None

from verl.utils.kernel.fp8_kernel import scaled_fp8_blockwise
from verl.utils.vllm.vllm_fp4_utils import (
    is_deepseek_v4_model,
    iter_deepseek_v4_weights,
    process_mxfp4_moe_weights_after_loading,
    stage_mxfp4_moe_params_for_loading,
)
from verl.utils.vllm.vllm_fp8_utils import (
    build_fp8_method_patchers,
    process_fp8_weights_after_loading,
    stage_fp8_params_for_loading,
)

logger = logging.getLogger(__name__)


def _get_vllm_version():
    return version.parse(importlib.metadata.version("vllm"))


# Ref: https://github.com/NVIDIA-NeMo/RL/commit/bc24887c72a6e1b2699a228bc87c588546dfe6b7
@dataclass()
class FP8State:
    # A cache of fp8 parameter names, we can check this cache to see if a
    # param name corresponds to a fp8 weight
    seen_params: set = field(default_factory=lambda: set())
    fp8_param_names: set = field(default_factory=lambda: set())
    vllm_patches: list = field(default_factory=lambda: [])


fp8_state: FP8State = FP8State()


def is_fp8_model(vllm_config):
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    if hasattr(vllm_config, "quant_config"):
        if isinstance(vllm_config.quant_config, Fp8Config):
            return True
        elif is_mxfp8_vllm_ascend(vllm_config.quant_config):
            return True

    return False


# vLLM 0.24.0 (MoE refactor, vllm-project/vllm#41184) removed the ``FusedMoE``
# ``nn.Module`` class. ``FusedMoE`` is now a factory *function* that builds a
# ``MoERunner``, and the fused expert weight tensors (``w13_weight`` /
# ``w2_weight``) moved onto a ``RoutedExperts`` submodule owned by the runner
# (i.e. ``experts`` -> ``experts.routed_experts``). vLLM 0.26.1 removed the
# ``FusedMoE`` name entirely. Resolve the concrete module classes once so
# ``isinstance`` checks keep working across vLLM versions -- calling
# ``isinstance(x, FusedMoE)`` when ``FusedMoE`` is a function raises
# ``TypeError: isinstance() arg 2 must be a type``, and ``FusedMoE`` may be
# ``None`` when the name no longer exists.
if isinstance(FusedMoE, type):
    # vLLM < 0.24: ``FusedMoE`` is itself the expert-weight-holding module.
    _MOE_STOP_CLASSES = (FusedMoE,)
    _EXPERT_WEIGHT_CLASSES = (FusedMoE,)
else:
    _stop_classes: list[type] = []
    _weight_classes: list[type] = []
    try:
        from vllm.model_executor.layers.fused_moe import RoutedExperts

        _stop_classes.append(RoutedExperts)
        _weight_classes.append(RoutedExperts)
    except ImportError:
        pass
    try:
        from vllm.model_executor.layers.fused_moe import MoERunner

        _stop_classes.append(MoERunner)
    except ImportError:
        pass
    _MOE_STOP_CLASSES = tuple(_stop_classes)
    _EXPERT_WEIGHT_CLASSES = tuple(_weight_classes)


def _is_expert_weight_module(module) -> bool:
    """Return whether ``module`` directly owns fused expert weights.

    True for the pre-0.24 ``FusedMoE`` module and the post-0.24 ``RoutedExperts``
    module (both expose ``w13_weight`` / ``w2_weight``). Falls back to duck typing
    so the check keeps working even if the concrete classes cannot be imported.
    """
    if _EXPERT_WEIGHT_CLASSES and isinstance(module, _EXPERT_WEIGHT_CLASSES):
        return True
    return hasattr(module, "w13_weight") and hasattr(module, "w2_weight")


def _is_fused_moe_block(module) -> bool:
    """Return whether ``module`` is the fused-MoE block reached while walking the
    module tree, so traversal must stop before descending into per-expert name
    parts (e.g. the expert index) that are not real submodules."""
    if _MOE_STOP_CLASSES and isinstance(module, _MOE_STOP_CLASSES):
        return True
    # ``MoERunner`` owns a ``routed_experts`` child; the weight holder owns the
    # expert weight tensors directly.
    return hasattr(module, "routed_experts") or _is_expert_weight_module(module)


def _resolve_expert_weight_module(module):
    """Return the submodule that actually owns the fused expert weights.

    On vLLM >= 0.24 the walked module is a ``MoERunner`` whose ``routed_experts``
    child holds the weights; on older vLLM the module already holds them.
    """
    routed_experts = getattr(module, "routed_experts", None)
    if routed_experts is not None:
        return routed_experts
    return module


def get_module_from_param_name(model, name: str):
    # Split the name into parts (e.g., 'layers', '0', 'self_attn', 'q_proj', 'weight')
    # The module path is all but the last part (the parameter's own name)
    path_parts = name.split(".")
    module_path = path_parts[:-1]
    # Replace with the fused model name
    packed_modules_mapping = model.packed_modules_mapping
    reversed_mapping = {
        original_name: fused_name
        for fused_name, original_names_list in packed_modules_mapping.items()
        for original_name in original_names_list
    }
    if module_path[-1] in reversed_mapping.keys():
        module_path[-1] = reversed_mapping[module_path[-1]]

    current_module = model
    try:
        # Traverse the model hierarchy
        for part in module_path:
            if _is_fused_moe_block(current_module):
                # Reached the fused-MoE block: the remaining name parts (expert
                # index, per-expert proj name, ...) are handled by vLLM's fused
                # weight loader, so return the module owning the expert weights.
                return _resolve_expert_weight_module(current_module)
            elif isinstance(current_module, torch.nn.ModuleList):
                current_module = current_module[int(part)]
            else:
                current_module = getattr(current_module, part)
    except (AttributeError, IndexError, ValueError) as e:
        print(f"Warning: Could not find module for parameter '{name}'. Error: {e}")
    return _resolve_expert_weight_module(current_module)


def is_fp8_weight(name, model):
    if name not in fp8_state.seen_params:
        fp8_state.seen_params.add(name)
        # Filter out bias params
        if name.endswith("weight"):
            module = get_module_from_param_name(model, name)
            # We currently only quantize linear and fused-MoE expert layers.
            is_fp8_linear = isinstance(module, LinearBase) and module.weight.dtype == torch.float8_e4m3fn
            is_fp8_moe = (
                _is_expert_weight_module(module)
                and module.w13_weight.dtype == torch.float8_e4m3fn
                and module.w2_weight.dtype == torch.float8_e4m3fn
            )
            if is_fp8_linear or is_fp8_moe:
                fp8_state.fp8_param_names.add(name)
    return name in fp8_state.fp8_param_names


def is_mxfp8_vllm_ascend(quant_config):
    try:
        from vllm_ascend.quantization.modelslim_config import AscendModelSlimConfig

        if isinstance(quant_config, AscendModelSlimConfig):
            quant_method = quant_config.quant_description.get("quant_method")
            return quant_method in ["ascend"]
        return False
    except ImportError:
        # vllm_ascend not installed, so this can't be an Ascend MXFP8 config
        return False


def restore_mxfp8_weights_for_loading(model):
    for name, module in model.named_modules():
        if (
            hasattr(module, "_mxfp8_transformed")
            and hasattr(module, "quant_method")
            and hasattr(module.quant_method, "quant_method")
            and hasattr(module.quant_method.quant_method, "restore_weights_for_rl_loading")
        ):
            module.quant_method.quant_method.restore_weights_for_rl_loading(module)


def apply_mxfp8_transformation_after_loading(model):
    """Re-apply MXFP8 transformations after weight loading.

    This function iterates through all linear modules in the model and applies
    the MXFP8 transformations (transpose, reshape) that are required for NPU
    inference.

    Must be called AFTER model.load_weights() in RL training loops.
    """
    try:
        from vllm.model_executor.layers.linear import LinearBase
    except ImportError:
        logger.warning("Could not import LinearBase, skipping MXFP8 transformation")
        return

    for name, module in model.named_modules():
        if (isinstance(module, LinearBase) or _is_expert_weight_module(module)) and hasattr(
            module, "_mxfp8_original_shapes"
        ):
            if hasattr(module, "quant_method") and hasattr(module.quant_method, "process_weights_after_loading"):
                logger.debug(f"Applying MXFP8 transformation for module: {name}")
                module.quant_method.process_weights_after_loading(module)


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


def quant_weights(weights, model, quant_config, dtype=torch.bfloat16):
    """Quantize weights to FP8 format using a memory-efficient generator.


    Args:
        weights: Generator or iterable of (name, tensor) pairs
        model: The model to check for FP8 weight names
        quant_config: Quantization configuration with weight_block_size
        dtype: Data type for intermediate computation (default: bfloat16)

    Yields:
        Tuples of (name, tensor) for each weight and its scale
    """

    if is_deepseek_v4_model(model):
        yield from iter_deepseek_v4_weights(weights)
        return

    fp8_state.seen_params.clear()
    fp8_state.fp8_param_names.clear()
    is_mxfp8_npu = is_mxfp8_vllm_ascend(quant_config)
    if is_mxfp8_npu:
        import torch_npu
    for k, v in weights:
        if not is_fp8_weight(k, model):
            yield (k, v)
            continue

        # Cast the weight into fp8 and its scale factor
        if torch.distributed.get_rank() == 0:
            logger.debug(f"Quantizing to FP8 blockwise: {k}")
        if is_mxfp8_npu:
            param_lp, param_scale = torch_npu.npu_dynamic_mx_quant(
                v.to(dtype),
                axis=-1,
                dst_type=torch_npu.float8_e4m3fn,
            )
            param_scale = param_scale.flatten(-2, -1)
        else:
            param_lp, param_scale = scaled_fp8_blockwise(
                v.to(dtype),
                weight_block_size=quant_config.weight_block_size,
            )
        param_scale = param_scale.squeeze(-1)

        # Yield the quantized weight
        yield (k, param_lp)

        if is_mxfp8_npu:
            yield (k + "_scale", param_scale)
        else:
            yield (k + "_scale_inv", param_scale)

        # Explicitly delete original tensor reference to help GC
        del v, param_lp, param_scale


def prepare_quanted_weights_for_loading(model):
    """Restore quantized params to the layout their ``weight_loader`` expects.

    Must run once before the first bucket, and pairs with
    ``process_quanted_weights_after_loading``, which re-applies the inference
    layout once the last bucket has landed. Each backend selects its own work
    by inspecting the model and returns an empty result when it finds nothing,
    so this is safe to call for any quantization scheme. The returned value is
    opaque reload state for the paired call.
    """
    restore_mxfp8_weights_for_loading(model)
    reload_state: dict[str, Any] = {
        "fp8_layers": stage_fp8_params_for_loading(model),
        "mxfp4_moe_modules": stage_mxfp4_moe_params_for_loading(model),
    }
    return reload_state


def process_quanted_weights_after_loading(model, reload_state):
    """Re-apply the inference layout undone by ``prepare_quanted_weights_for_loading``."""
    apply_mxfp8_transformation_after_loading(model)
    reload_state = reload_state or {}
    process_fp8_weights_after_loading(reload_state.get("fp8_layers") or [])
    process_mxfp4_moe_weights_after_loading(reload_state.get("mxfp4_moe_modules") or [])
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
    quant_config = model_runner.vllm_config.quant_config
    vllm_dtype = model_runner.vllm_config.model_config.dtype

    weights = list(weights)
    weights_quantized = quant_weights(weights, model, quant_config, dtype=vllm_dtype)

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
            loaded_params = model.load_weights(weights_quantized)
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
    initial load. The mxfp4 path needs no startup patch -- it derives the
    checkpoint layout from the quant method and patches ``replace_parameter``
    only for the duration of its own post-processing call.
    """
    if fp8_state.vllm_patches:
        logger.debug("vLLM quantization patches already applied")
        return

    for patcher in build_fp8_method_patchers(_get_vllm_version()):
        patcher.start()
        fp8_state.vllm_patches.append(patcher)
