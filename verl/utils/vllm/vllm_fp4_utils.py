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

The refit stream carries checkpoint-layout tensors; live params hold whatever the
backend rewrote them into. Each expert param is handed a checkpoint-layout buffer to
absorb the reload, then the backend's conversion is replayed and folded back into the
live storage (so the addresses the CUDA graph captured stay put). Only block scales
(change element count) need a real allocation; expert weights stage as a reinterpret of
their own storage. Entry point: ``verl/utils/vllm/vllm_quant_utils.py``.
"""

import logging
from unittest.mock import patch

import torch

from verl.utils.vllm.vllm_fp8_utils import _scale_from_amax, dsv4_fp8_linear_leaf, quantize_dsv4_fp8_linear

logger = logging.getLogger(__name__)

_MXFP4_SF_BLOCK = 32
_MXFP4_LIVE_ATTR = "_verl_mxfp4_live_params"

# MXFP4 quantization — inlined from Megatron-Bridge quantization_utils (no megatron import here).
_FP4_E2M1_MAX = 6.0


def _quantize_mxfp4_e2m1_like_scale(weight, source_scale, *, name="", block_size=_MXFP4_SF_BLOCK):
    """Quantize a 2-D weight to packed MXFP4 E2M1 using source scale geometry."""
    if weight.ndim != 2:
        label = f" for {name!r}" if name else ""
        raise RuntimeError(f"MXFP4 quantized export expects a 2-D weight{label}, got {weight.ndim}D")
    rows, cols = weight.shape
    if cols % 2 != 0 or cols % block_size != 0 or source_scale.shape != (rows, cols // block_size):
        label = f" for {name!r}" if name else ""
        raise RuntimeError(
            f"Unsupported MXFP4 geometry{label}: weight={tuple(weight.shape)} scale={tuple(source_scale.shape)}"
        )
    weight_f32 = weight.to(torch.float32)
    packed = torch.empty((rows, cols // 2), dtype=torch.uint8, device=weight.device)
    scale_f32 = torch.empty(tuple(source_scale.shape), dtype=torch.float32, device=weight.device)
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32, device=weight.device)
    max_chunk_elements = 16_000_000
    rows_per_chunk = max(1, min(rows, max_chunk_elements // max(cols, 1)))
    scale_cols = source_scale.shape[1]
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        chunk = weight_f32[row_start:row_end].reshape(-1, scale_cols, block_size)
        chunk_amax = chunk.abs().amax(dim=-1)
        if source_scale.dtype == torch.uint8:
            unrounded_scale = torch.where(
                chunk_amax > 0,
                chunk_amax / _FP4_E2M1_MAX,
                torch.ones_like(chunk_amax),
            )
            chunk_scale = torch.exp2(torch.ceil(torch.log2(unrounded_scale)).clamp(min=-127, max=127))
        else:
            chunk_scale = _scale_from_amax(chunk_amax, _FP4_E2M1_MAX, source_scale.dtype)
        scale_f32[row_start:row_end] = chunk_scale
        normalized = chunk / chunk_scale[:, :, None]
        codes = torch.bucketize(normalized.abs(), boundaries).to(torch.uint8)
        codes = (codes | ((normalized < 0).to(torch.uint8) * 8)).reshape(row_end - row_start, cols)
        lo = codes[:, 0::2].to(torch.int16)
        hi = codes[:, 1::2].to(torch.int16)
        packed[row_start:row_end] = (lo | (hi << 4)).to(torch.uint8)
    if source_scale.dtype == torch.uint8:
        output_scale = (torch.log2(scale_f32).round() + 127).clamp(min=0, max=254).to(torch.uint8)
    else:
        output_scale = scale_f32.to(dtype=source_scale.dtype)
    return packed.contiguous().view(torch.int8), output_scale


def is_deepseek_v4_model(model):
    if model is None:
        return False

    for obj in (model, getattr(model, "config", None), getattr(model, "hf_config", None)):
        if obj is not None and getattr(obj, "model_type", None) is not None:
            return obj.model_type == "deepseek_v4"

    text_config = getattr(getattr(model, "config", None), "text_config", None)
    return getattr(text_config, "model_type", None) == "deepseek_v4"


def _quantize_expert_to_mxfp4(weight):
    """Quantize a 2-D bf16 expert weight to packed MXFP4 E2M1 + e8m0 scale (geometry [out, in] -> scale [out, in//32])."""
    rows, cols = weight.shape
    scale_geom = torch.empty(rows, cols // _MXFP4_SF_BLOCK, dtype=torch.uint8, device=weight.device)
    packed, scale = _quantize_mxfp4_e2m1_like_scale(weight, scale_geom, name="dsv4_expert")
    return packed, scale


def iter_deepseek_v4_weights(weights):
    """Normalize the refit stream to the packed layout vLLM expects.

    Automodel loads the checkpoint through ``from_hf``, which unpacks FP4
    experts -> bf16 and dequantizes the non-expert FP8 linears -> bf16
    (stripping their ``.scale``). vLLM wants experts packed ``[out, in // 2]``
    int8 + ``[out, in // 32]`` e8m0 scale, and the FP8 linears as fp8_e4m3 +
    block scale — so both must be re-quantized here before ``load_weights``
    (else a 2x shape mismatch, or NaN from the staging-fill scale). The
    paired ``.scale`` is emitted right after each ``.weight``. The
    Megatron-Bridge path already streams quantized tensors and is handled by
    the ``int8`` branch (``uint8`` reinterpret only).
    ``dsv4_fp8_linear_leaf`` (from ``vllm_fp8_utils``) selects the
    checkpoint-FP8 linears; bf16-on-disk layers (compressor, norms, embed,
    gate, head) pass through untouched.
    """
    for name, weight in weights:
        if ".experts." in name and name.endswith(".weight") and weight.dtype == torch.bfloat16:
            packed, scale = _quantize_expert_to_mxfp4(weight)
            yield name, packed
            yield name[: -len(".weight")] + ".scale", scale
            continue
        if ".experts." in name and weight.dtype in (torch.int8, torch.float8_e8m0fnu):
            weight = weight.view(torch.uint8)
            yield name, weight
            continue
        fp8_leaf = dsv4_fp8_linear_leaf(name)
        if fp8_leaf is not None and weight.dtype == torch.bfloat16:
            fp8_weight, scale = quantize_dsv4_fp8_linear(weight, fp8_leaf)
            yield name, fp8_weight
            yield name[: -len(".weight")] + ".scale", scale
            continue
        yield name, weight


def _is_mxfp4_fused_moe_module(module):
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod

    return isinstance(module, RoutedExperts) and isinstance(module.quant_method, Mxfp4MoEMethod)


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
    quant_method = module.quant_method
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

    with patch.object(vllm_mxfp4, "replace_parameter", _replace_parameter_in_place):
        module.quant_method.process_weights_after_loading(module)

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

        backend = getattr(module.quant_method, "mxfp4_backend", None)
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
