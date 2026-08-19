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

"""Refit support for vLLM's ``Mxfp4MoEMethod`` (DeepSeek V4 routed experts).

Reloading these experts means undoing the kernel's weight conversion: the refit
stream carries checkpoint-layout tensors while the live parameters hold whatever
layout the selected backend rewrote them into. Each expert param is handed a
checkpoint-layout buffer to absorb the reload, then the backend's conversion is
replayed and folded back into the live storage, so the addresses the CUDA graph
captured stay put.

Backends differ only in how much of that buffer has to be a real allocation.
A param whose kernel layout spans the same bytes as the checkpoint one -- the
expert weights under both DeepGEMM and Marlin -- stages as a reinterpret of its
own storage and costs nothing. Only the block scales, which change element
count, need an allocation, and those are a sixteenth of the weights.

``verl/utils/vllm/vllm_quant_utils.py`` is the entry point that drives these.
"""

import logging
from unittest.mock import patch

import torch

logger = logging.getLogger(__name__)

_MXFP4_SF_BLOCK = 32
_MXFP4_LIVE_ATTR = "_verl_mxfp4_live_params"


def is_deepseek_v4_model(model):
    if model is None:
        return False

    for obj in (model, getattr(model, "config", None), getattr(model, "hf_config", None)):
        if obj is not None and getattr(obj, "model_type", None) is not None:
            return obj.model_type == "deepseek_v4"

    text_config = getattr(getattr(model, "config", None), "text_config", None)
    return getattr(text_config, "model_type", None) == "deepseek_v4"


def iter_deepseek_v4_weights(weights):
    """Pass the refit stream through untouched apart from a dtype reinterpret.

    A DSv4 checkpoint already ships quantized experts, so unlike the BF16 path
    there is nothing to quantize here. The expert tensors only need to be seen
    as the raw ``uint8`` byte layout that ``Mxfp4MoEMethod`` allocated.
    """
    for name, weight in weights:
        if ".experts." in name and weight.dtype in (torch.int8, torch.float8_e8m0fnu):
            weight = weight.view(torch.uint8)
        yield name, weight


def _resolve_mxfp4_quant_method(module):
    """Return the ``Mxfp4MoEMethod`` backing ``module``, unwrapping LoRA.

    Under verl merge=False, vLLM's ``FusedMoEWithLoRA`` swaps the layer's
    ``quant_method`` to ``FusedMoEModularMethod`` (a forward-only wrapper that
    stores the original as ``old_quant_method`` and hides the mxfp4 layout attrs
    + repack). Resolve to the original method so staging, layout derivation and
    post-load repack work like the non-LoRA path.
    """
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod

    qm = module.quant_method
    if isinstance(qm, Mxfp4MoEMethod):
        return qm
    inner = getattr(qm, "old_quant_method", None)
    if isinstance(inner, Mxfp4MoEMethod):
        return inner
    return None


def _is_mxfp4_fused_moe_module(module):
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    return isinstance(module, RoutedExperts) and _resolve_mxfp4_quant_method(module) is not None


def _refittable_mxfp4_backends():
    """Backends whose weight conversion can be replayed onto live storage.

    Two properties qualify a backend. Its conversion must be a pure function of
    the checkpoint layout, so replaying it after a reload reproduces the same
    inference layout. And it must publish the result through
    ``replace_parameter``, which is the hook ``_replace_parameter_in_place``
    intercepts to redirect the write into the captured storage.

    Triton is the notable exclusion: it assigns the weights as non-Parameter
    wrappers and parks the scales on the quant method, so the buffers its kernel
    reads are not reachable as parameters at all.
    """
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend

    return (
        Mxfp4MoeBackend.DEEPGEMM_MXFP4,
        Mxfp4MoeBackend.MARLIN,
        Mxfp4MoeBackend.BATCHED_MARLIN,
        Mxfp4MoeBackend.AITER_MXFP4_BF16,
    )


def _mxfp4_checkpoint_layout(module):
    """Layout of the expert params before kernel post-processing rewrites them.

    Mirrors ``Mxfp4MoEMethod.create_weights``. These are not guesses: vLLM
    re-asserts the same shapes at the top of ``_setup_kernel``, so a refit that
    hands back anything else fails loudly there. The third element is the
    ``quant_method`` tag ``RoutedExperts.weight_loader`` dispatches on, which
    ``replace_parameter`` does not carry over and must be reattached.
    """
    quant_method = _resolve_mxfp4_quant_method(module)
    num_experts = quant_method.num_experts
    intermediate = quant_method.intermediate_size
    hidden = quant_method.hidden_size
    return {
        "w13_weight": ((num_experts, 2 * intermediate, hidden // 2), torch.uint8, None),
        "w2_weight": ((num_experts, hidden, intermediate // 2), torch.uint8, None),
        "w13_weight_scale": ((num_experts, 2 * intermediate, hidden // _MXFP4_SF_BLOCK), torch.uint8, "block"),
        "w2_weight_scale": ((num_experts, hidden, intermediate // _MXFP4_SF_BLOCK), torch.uint8, "block"),
        # Listed unconditionally: models without MoE bias never registered these
        # and are skipped when the parameter turns up missing. Leaving them out
        # would instead let Marlin's bias permute fall through to the real
        # ``replace_parameter`` and quietly move the buffer.
        "w13_bias": ((num_experts, 2 * intermediate), torch.bfloat16, None),
        "w2_bias": ((num_experts, hidden), torch.bfloat16, None),
    }


def _mxfp4_staging_data(param, shape, dtype):
    """Checkpoint-layout tensor for ``load_weights`` to write into.

    Reuses the live storage whenever it spans the same number of bytes, which
    covers more than the params a backend left alone. Marlin's repack is a
    permutation of 4-bit values, so its int32 tiles occupy exactly the bytes the
    uint8 checkpoint weight did; reinterpreting them costs nothing and keeps the
    reload landing where the kernel reads. That is what stops a refit from
    needing a second copy of the experts, which for the expert weights is the
    difference between a negligible allocation and doubling MoE memory.

    Replaying the conversion reads this buffer in full and builds its result in a
    fresh tensor before ``_replace_parameter_in_place`` writes back, so aliasing
    the destination is safe.
    """
    data = param.data
    shape = torch.Size(shape)
    if data.dtype == dtype and data.shape == shape:
        return data
    if data.is_contiguous() and data.numel() * data.element_size() == shape.numel() * dtype.itemsize:
        return data.flatten().view(dtype).reshape(shape)
    # Zero rather than empty: a param the refit stream happens to skip then
    # collapses the layer's output instead of feeding the kernel whatever was
    # left in the freed memory.
    return torch.zeros(shape, dtype=dtype, device=param.device)


def _stage_mxfp4_moe_params(module):
    """Expose checkpoint-layout buffers without moving any live storage.

    Each expert param is handed a buffer in the layout its ``weight_loader``
    expects, preferring a reinterpret of the param's own storage so the reload
    lands where the kernel reads. ``_process_mxfp4_moe_params`` then replays the
    backend's conversion and folds the result back into the live storage.
    """
    live = {}
    for name, (shape, dtype, scale_kind) in _mxfp4_checkpoint_layout(module).items():
        param = getattr(module, name, None)
        if not isinstance(param, torch.nn.Parameter):
            continue
        live[name] = param

        staged = torch.nn.Parameter(_mxfp4_staging_data(param, shape, dtype), requires_grad=False)
        staged.weight_loader = module.weight_loader
        if scale_kind is not None:
            staged.quant_method = scale_kind
        setattr(module, name, staged)

    setattr(module, _MXFP4_LIVE_ATTR, live)


def _replace_parameter_in_place(layer, param_name, new_data, prefer_copy=False):
    """Fold post-processing output into the parameter the CUDA graph captured.

    Reinstating the live parameter here rather than after
    ``process_weights_after_loading`` returns is deliberate: the tail of
    ``_setup_kernel`` rebuilds ``moe_quant_config`` and ``moe_kernel`` from
    whatever is on the layer at that moment, and those cache tensor references.
    Swapping back later would leave the kernel pointing at the temporaries.
    """
    from vllm.model_executor.utils import replace_parameter

    live = getattr(layer, _MXFP4_LIVE_ATTR, None) or {}
    param = live.pop(param_name, None)
    if param is None or new_data is None:
        return replace_parameter(layer, param_name, new_data, prefer_copy)

    if isinstance(new_data, torch.nn.Parameter):
        new_data = new_data.data

    if new_data.shape != param.shape or new_data.dtype != param.dtype:
        raise RuntimeError(
            f"mxfp4 refit re-derived {param_name} as {tuple(new_data.shape)}/{new_data.dtype}, "
            f"but the live parameter is {tuple(param.shape)}/{param.dtype}; "
            "its storage cannot be updated in place."
        )
    if new_data.data_ptr() != param.data_ptr():
        param.data.copy_(new_data)
    setattr(layer, param_name, param)


def _process_mxfp4_moe_params(module):
    from vllm.model_executor.layers.quantization import mxfp4 as vllm_mxfp4

    # Use the legacy Mxfp4MoEMethod for the repack: the modular wrapper that
    # LoRA installs (FusedMoEModularMethod) inherits a no-op
    # process_weights_after_loading, so calling it would skip the repack and
    # leave live storage in the checkpoint layout. See _resolve_mxfp4_quant_method.
    quant_method = _resolve_mxfp4_quant_method(module)
    with patch.object(vllm_mxfp4, "replace_parameter", _replace_parameter_in_place):
        quant_method.process_weights_after_loading(module)

    # Every staged param is consumed through the patched replace_parameter, so
    # a leftover means post-processing took a path that assigns weights some
    # other way and the live storage was never refreshed.
    leftover = getattr(module, _MXFP4_LIVE_ATTR, None) or {}
    delattr(module, _MXFP4_LIVE_ATTR)
    if leftover:
        raise RuntimeError(
            f"mxfp4 refit left {sorted(leftover)} un-reinstated; every backend listed in "
            "_refittable_mxfp4_backends is expected to route each expert param through "
            "replace_parameter."
        )


def stage_mxfp4_moe_params_for_loading(model):
    """Hand ``load_weights`` checkpoint-layout expert buffers.

    Returns the staged modules for ``process_mxfp4_moe_weights_after_loading``.
    A model with no mxfp4 experts yields an empty list, which makes this safe
    to call unconditionally.
    """
    supported = _refittable_mxfp4_backends()
    staged_modules = []
    backends = set()
    for module in model.modules():
        if not _is_mxfp4_fused_moe_module(module):
            continue

        backend = getattr(_resolve_mxfp4_quant_method(module), "mxfp4_backend", None)
        if backend not in supported:
            # Refusing beats skipping: an unsupported backend would quietly load
            # checkpoint-layout data into rewritten parameters, and the rollout
            # would drift with nothing pointing at the cause.
            raise NotImplementedError(
                f"mxfp4 MoE refit does not support the {backend} backend selected by "
                f"{type(module).__name__}. Supported backends: "
                f"{', '.join(str(b) for b in supported)}."
            )

        _stage_mxfp4_moe_params(module)
        staged_modules.append(module)
        backends.add(backend)

    logger.info(
        "Staged %d mxfp4 MoE modules for in-place refit (backends: %s)",
        len(staged_modules),
        ", ".join(sorted(str(b) for b in backends)) or "none",
    )
    return staged_modules


def process_mxfp4_moe_weights_after_loading(modules):
    """Repack the loaded experts into the kernel layout inside the live storage."""
    for module in modules:
        _process_mxfp4_moe_params(module)
