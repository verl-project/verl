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

Only the DeepGEMM backend (``--moe-backend deep_gemm``) is refit in place. It
is the one backend whose weight conversion leaves the expert weights in
checkpoint layout, so the live parameters can absorb a reload directly and only
the block scales need a detour through a temporary buffer. The other backends
reshuffle the weights themselves, which would require a full-size staging
allocation per layer, so they are left alone here.

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


def _is_mxfp4_fused_moe_module(module):
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod

    return isinstance(module, RoutedExperts) and isinstance(module.quant_method, Mxfp4MoEMethod)


def _is_deepgemm_mxfp4_moe_module(module):
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend

    return (
        _is_mxfp4_fused_moe_module(module)
        and getattr(module.quant_method, "mxfp4_backend", None) == Mxfp4MoeBackend.DEEPGEMM_MXFP4
    )


def _mxfp4_checkpoint_layout(module):
    """Layout of the expert params before kernel post-processing rewrites them.

    Mirrors ``Mxfp4MoEMethod.create_weights``. These are not guesses: vLLM
    re-asserts the same shapes at the top of ``_setup_kernel``, so a refit that
    hands back anything else fails loudly there. The third element is the
    ``quant_method`` tag ``RoutedExperts.weight_loader`` dispatches on, which
    ``replace_parameter`` does not carry over and must be reattached.
    """
    quant_method = module.quant_method
    num_experts = quant_method.num_experts
    intermediate = quant_method.intermediate_size
    hidden = quant_method.hidden_size
    return {
        "w13_weight": ((num_experts, 2 * intermediate, hidden // 2), torch.uint8, None),
        "w2_weight": ((num_experts, hidden, intermediate // 2), torch.uint8, None),
        "w13_weight_scale": ((num_experts, 2 * intermediate, hidden // _MXFP4_SF_BLOCK), torch.uint8, "block"),
        "w2_weight_scale": ((num_experts, hidden, intermediate // _MXFP4_SF_BLOCK), torch.uint8, "block"),
    }


def _stage_mxfp4_moe_params(module):
    """Expose checkpoint-layout buffers without moving any live storage.

    Under the DeepGEMM backend ``convert_weight_to_mxfp4_moe_kernel_format``
    returns the expert weights untouched, so their live parameters already
    accept checkpoint data and only need their loader attributes back. The
    scales are repacked into DeepGEMM's layout, so those get a temporary buffer
    to land in; ``_process_mxfp4_moe_params`` folds the repacked result back
    into the live storage and drops the buffer.
    """
    live = {}
    for name, (shape, dtype, scale_kind) in _mxfp4_checkpoint_layout(module).items():
        param = getattr(module, name, None)
        if not isinstance(param, torch.nn.Parameter):
            continue
        live[name] = param

        if param.shape == torch.Size(shape) and param.dtype == dtype:
            data = param.data
        else:
            # Zero rather than empty: a scale the refit stream happens to skip
            # then collapses the layer's output instead of reading whatever was
            # in the freed memory.
            data = torch.zeros(shape, dtype=dtype, device=param.device)

        staged = torch.nn.Parameter(data, requires_grad=False)
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

    with patch.object(vllm_mxfp4, "replace_parameter", _replace_parameter_in_place):
        module.quant_method.process_weights_after_loading(module)

    # Every staged param is consumed through the patched replace_parameter, so
    # a leftover means post-processing took a path that assigns weights some
    # other way and the live storage was never refreshed.
    leftover = getattr(module, _MXFP4_LIVE_ATTR, None) or {}
    delattr(module, _MXFP4_LIVE_ATTR)
    if leftover:
        raise RuntimeError(
            f"mxfp4 refit left {sorted(leftover)} un-reinstated; the DeepGEMM path is expected to "
            "route every expert param through replace_parameter."
        )


def stage_mxfp4_moe_params_for_loading(model):
    """Hand ``load_weights`` checkpoint-layout expert buffers.

    Returns the staged modules for ``process_mxfp4_moe_weights_after_loading``.
    A model with no mxfp4 experts yields an empty list, which makes this safe
    to call unconditionally.
    """
    staged_modules = []
    for module in model.modules():
        if not _is_mxfp4_fused_moe_module(module):
            continue
        if not _is_deepgemm_mxfp4_moe_module(module):
            # Refusing beats skipping: the other backends reshuffle the expert
            # weights, so a refit would quietly load checkpoint-layout data
            # into rewritten parameters and the rollout would drift.
            raise NotImplementedError(
                "mxfp4 MoE refit only supports the DeepGEMM backend, but "
                f"{type(module).__name__} selected "
                f"{getattr(module.quant_method, 'mxfp4_backend', None)}. "
                "Launch the rollout engine with --moe-backend deep_gemm."
            )
        _stage_mxfp4_moe_params(module)
        staged_modules.append(module)

    logger.info("Staged %d DeepGEMM mxfp4 MoE modules for in-place refit", len(staged_modules))
    return staged_modules


def process_mxfp4_moe_weights_after_loading(modules):
    """Repack the loaded experts into DeepGEMM layout inside the live storage."""
    for module in modules:
        _process_mxfp4_moe_params(module)
