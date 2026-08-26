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

"""FP8 support for the RL weight-refit stream.

DSV4 checkpoint re-quantization: ``from_hf`` dequantizes the non-expert FP8
linears (attn wq_a/wkv/wq_b/wo_a/wo_b, indexer.wq_b, shared experts) to bf16
and strips their ``.scale``; ``quantize_dsv4_fp8_linear`` / \
``dsv4_fp8_linear_leaf`` re-pack them to fp8_e4m3 + e8m0 scale before
``load_weights`` (called from ``vllm_fp4_utils.iter_deepseek_v4_weights``).

vLLM-native refit (``Fp8LinearMethod`` / ``Fp8MoEMethod``): keep parameter
subclass metadata across vLLM's post-load rebuild, and stage checkpoint-layout
buffers so kernel post-processing folds back into the storage a CUDA graph
captured. ``vllm_quant_utils.py`` is the entry point.
"""

import inspect
import logging
import math
from unittest.mock import patch

import torch
import torch.nn.functional as F
from packaging import version

logger = logging.getLogger(__name__)


# DSV4 checkpoint-FP8 re-quantization (bf16 -> fp8_e4m3 + e8m0 scale).
# Inlined from Megatron-Bridge quantization_utils (no megatron import here).

_DSV4_FP8_LINEAR_LEAVES = frozenset(
    {
        "wq_a",
        "wkv",
        "wq_b",
        "wo_a",
        "wo_b",
        "w1",
        "w2",
        "w3",
    }
)

# w1/w2/w3 are shared-expert leaves; routed experts use the same names but are
# matched by the ``.experts.`` branch in ``iter_deepseek_v4_weights`` (MXFP4).
_DSV4_FP8_LINEAR_PARENTS = frozenset({"shared_experts", "indexer", "attn"})

_FP8_BLOCK_SIZE = 128
_FP8_E4M3_MAX = 448.0


def _is_float8_e8m0_dtype(dtype: torch.dtype) -> bool:
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    return e8m0 is not None and dtype == e8m0


def _scale_from_amax(amax: torch.Tensor, max_quantized_value: float, scale_dtype: torch.dtype) -> torch.Tensor:
    """Per-block amax -> quantization scale (shared with the MXFP4 path)."""
    scale = torch.where(amax > 0, amax / max_quantized_value, torch.ones_like(amax))
    if _is_float8_e8m0_dtype(scale_dtype):
        scale = scale.clamp(min=2.0**-127, max=2.0**127)
        return torch.exp2(torch.ceil(torch.log2(scale)))
    return scale


def _quantize_fp8_2d_blocks(weight, source_scale, *, name="", block_size=_FP8_BLOCK_SIZE):
    rows, cols = weight.shape
    scale_rows, scale_cols = source_scale.shape
    expected_shape = (
        (rows + block_size - 1) // block_size,
        (cols + block_size - 1) // block_size,
    )
    if (scale_rows, scale_cols) != expected_shape:
        label = f" for {name!r}" if name else ""
        raise RuntimeError(
            f"Unsupported FP8 scale geometry{label}: "
            f"weight={tuple(weight.shape)} scale={tuple(source_scale.shape)} expected={expected_shape}"
        )
    weight_f32 = weight.to(torch.float32)
    pad_rows = scale_rows * block_size - rows
    pad_cols = scale_cols * block_size - cols
    if pad_rows or pad_cols:
        weight_f32 = F.pad(weight_f32, (0, pad_cols, 0, pad_rows))
    blocks = weight_f32.view(scale_rows, block_size, scale_cols, block_size).transpose(1, 2)
    scale_f32 = _scale_from_amax(blocks.abs().amax(dim=(-1, -2)), _FP8_E4M3_MAX, source_scale.dtype)
    q_blocks = (blocks / scale_f32[:, :, None, None]).to(torch.float8_e4m3fn)
    q_weight = q_blocks.transpose(1, 2).reshape(scale_rows * block_size, scale_cols * block_size)[:rows, :cols]
    return q_weight.contiguous(), scale_f32.to(dtype=source_scale.dtype)


def _quantize_fp8_per_row_tiles(weight, source_scale, *, name=""):
    rows, cols = weight.shape
    scale_rows, scale_cols = source_scale.shape
    if scale_rows != rows or scale_cols <= 0 or cols % scale_cols != 0:
        label = f" for {name!r}" if name else ""
        raise RuntimeError(
            f"Unsupported per-row FP8 scale geometry{label}: "
            f"weight={tuple(weight.shape)} scale={tuple(source_scale.shape)}"
        )
    tile_cols = cols // scale_cols
    grouped = weight.to(torch.float32).reshape(rows, scale_cols, tile_cols)
    scale_f32 = _scale_from_amax(grouped.abs().amax(dim=-1), _FP8_E4M3_MAX, source_scale.dtype)
    q_weight = (grouped / scale_f32[:, :, None]).to(torch.float8_e4m3fn).reshape(rows, cols)
    return q_weight.contiguous(), scale_f32.to(dtype=source_scale.dtype)


def _quantize_fp8_1d_scale(weight, source_scale, *, name="", block_size=_FP8_BLOCK_SIZE):
    rows, _ = weight.shape
    scale_len = source_scale.numel()
    weight_f32 = weight.to(torch.float32)
    if scale_len == rows:
        scale_f32 = _scale_from_amax(weight_f32.abs().amax(dim=1), _FP8_E4M3_MAX, source_scale.dtype)
        q_weight = (weight_f32 / scale_f32[:, None]).to(torch.float8_e4m3fn)
        return q_weight.contiguous(), scale_f32.to(dtype=source_scale.dtype)
    expected_len = (rows + block_size - 1) // block_size
    if scale_len != expected_len:
        label = f" for {name!r}" if name else ""
        raise RuntimeError(
            f"Unsupported 1-D FP8 scale geometry{label}: "
            f"weight={tuple(weight.shape)} scale={tuple(source_scale.shape)} expected_len={expected_len}"
        )
    q_weight = torch.empty_like(weight_f32, dtype=torch.float8_e4m3fn)
    scale_f32 = torch.empty(scale_len, dtype=torch.float32, device=weight.device)
    for block_idx in range(scale_len):
        row_start = block_idx * block_size
        row_end = min(row_start + block_size, rows)
        block = weight_f32[row_start:row_end]
        block_scale = _scale_from_amax(block.abs().amax().reshape(()), _FP8_E4M3_MAX, source_scale.dtype)
        q_weight[row_start:row_end] = (block / block_scale).to(torch.float8_e4m3fn)
        scale_f32[block_idx] = block_scale
    return q_weight.contiguous(), scale_f32.to(dtype=source_scale.dtype)


def _quantize_fp8_e4m3fn_like_scale(weight, source_scale, *, name="", block_size=_FP8_BLOCK_SIZE):
    """Quantize a 2-D weight to FP8 E4M3 using ``source_scale`` geometry and dtype."""
    if weight.ndim != 2:
        label = f" for {name!r}" if name else ""
        raise RuntimeError(f"FP8 quantized export expects a 2-D weight{label}, got {weight.ndim}D")
    if source_scale.ndim == 1:
        return _quantize_fp8_1d_scale(weight, source_scale, name=name, block_size=block_size)
    if source_scale.ndim != 2:
        label = f" for {name!r}" if name else ""
        raise RuntimeError(f"Unsupported FP8 scale rank{label}: scale={tuple(source_scale.shape)}")
    if source_scale.shape[0] == weight.shape[0]:
        return _quantize_fp8_per_row_tiles(weight, source_scale, name=name)
    return _quantize_fp8_2d_blocks(weight, source_scale, name=name, block_size=block_size)


def dsv4_fp8_linear_leaf(name: str) -> str | None:
    """Return the FP8 linear leaf of a checkpoint-layout stream key, or None."""
    if not name.endswith(".weight"):
        return None
    base = name[: -len(".weight")]
    if ".experts." in base:  # routed experts are MXFP4, handled separately
        return None
    if base.endswith(".base_layer"):  # strip LoRA wrapper segment
        base = base[: -len(".base_layer")]
    parts = base.split(".")
    if len(parts) < 2:
        return None
    leaf = parts[-1]
    parent = parts[-2]
    # ``indexer.wq_b`` -> parent ``indexer``; ``attn.wq_a`` -> parent ``attn``;
    # ``shared_experts.w1`` -> parent ``shared_experts``.
    if leaf in _DSV4_FP8_LINEAR_LEAVES and parent in _DSV4_FP8_LINEAR_PARENTS:
        return leaf
    return None


def quantize_dsv4_fp8_linear(weight, leaf: str):
    """Re-quantize a bf16 non-expert FP8 linear weight to fp8_e4m3 + e8m0 scale."""
    rows, cols = weight.shape
    # weight_block_size is [128, 128] for every DSV4-Flash FP8 linear.
    src_scale = torch.empty(
        (rows + 127) // 128,
        (cols + 127) // 128,
        dtype=torch.float8_e8m0fnu,
        device=weight.device,
    )
    fp8_weight, scale = _quantize_fp8_e4m3fn_like_scale(weight, src_scale, name=f"dsv4_fp8_{leaf}")
    return fp8_weight, scale


# ---------------------------------------------------------------------------
# vLLM-native FP8 refit (Fp8LinearMethod / Fp8MoEMethod) — original contents.
# ---------------------------------------------------------------------------


def _iter_transferable_param_attrs(src_param):
    """Yield the attributes worth moving onto a rebuilt parameter.

    Methods bound to a tensor are excluded, and that exclusion is the whole
    point of this helper. ``dir()`` reports vLLM's loader entry points
    (``load_merged_column_weight`` and friends) alongside the data attributes,
    and copying one lands a method whose ``self`` is some *other* tensor in the
    destination's instance dict, where it shadows the class. Every later load
    then writes into that stale tensor instead of the parameter it was handed --
    silently, because the shapes still agree.

    Note the test is "bound to a tensor", not "bound to the source": these
    methods are copied forward from rebuild to rebuild, so by the time one does
    damage its ``self`` is usually a temporary from several rebuilds back.
    ``weight_loader`` is bound to the layer, so it survives.
    """
    base_param_attrs = dir(torch.nn.Parameter)
    for attr in dir(src_param):
        if attr in base_param_attrs or attr.startswith("__"):
            continue
        try:
            value = getattr(src_param, attr)
        except AttributeError:
            continue
        if inspect.ismethod(value) and isinstance(value.__self__, torch.Tensor):
            continue
        yield attr, value


def _copy_param_subclass_attrs(dst_param, src_param):
    """Carry vLLM's parameter metadata across a parameter rebuild.

    vLLM stores loader dispatch information (``output_dim``, ``weight_loader``,
    tp state, ...) as plain attributes on ``Parameter`` subclasses. Anything
    that re-wraps a tensor has to bring them along or the next refit will not
    know how to shard the incoming weight.
    """
    if src_param is None:
        return

    for attr, value in _iter_transferable_param_attrs(src_param):
        try:
            setattr(dst_param, attr, value)
        except AttributeError:
            pass

    subclass_type = getattr(src_param, "subclass_type", type(src_param))
    if subclass_type is not torch.nn.Parameter:
        dst_param.subclass_type = subclass_type


_FP8_PRISTINE_ATTR = "_verl_fp8_pristine"
_FP8_LIVE_ATTR = "_verl_fp8_live_params"

# ``Fp8LinearMethod`` owns the first group, ``Fp8MoEMethod`` the second. The MoE
# scale is named ``w13_weight_scale_inv`` under block quant and
# ``w13_weight_scale`` otherwise (``self.weight_scale_name``), so both spellings
# are listed; only the names a layer actually has get recorded.
_FP8_LINEAR_PARAM_NAMES = ("weight", "weight_scale_inv", "weight_scale")
_FP8_MOE_PARAM_NAMES = (
    "w13_weight",
    "w2_weight",
    "w13_weight_scale_inv",
    "w2_weight_scale_inv",
    "w13_weight_scale",
    "w2_weight_scale",
)
_FP8_REFIT_PARAM_NAMES = (*_FP8_LINEAR_PARAM_NAMES, *_FP8_MOE_PARAM_NAMES)


def _fold_into_live_param(layer, param_name, param, new_data):
    """Move post-processing output into the storage the CUDA graph captured."""
    if isinstance(new_data, torch.nn.Parameter):
        new_data = new_data.data

    if new_data.shape != param.shape or new_data.dtype != param.dtype:
        raise RuntimeError(
            f"FP8 refit re-derived {param_name} as {tuple(new_data.shape)}/{new_data.dtype}, "
            f"but the live parameter is {tuple(param.shape)}/{param.dtype}; "
            "its storage cannot be updated in place."
        )
    if new_data.data_ptr() != param.data_ptr():
        param.data.copy_(new_data)
    setattr(layer, param_name, param)


def replace_parameter_preserve_subclass(
    layer: torch.nn.Module, param_name: str, new_data: torch.Tensor | None, prefer_copy: bool = False
):
    """Stand-in for vLLM's ``replace_parameter`` inside the fp8 quant module.

    During a refit of a staged layer this folds the result into the live
    parameter and reinstates it immediately, rather than after
    ``process_weights_after_loading`` returns. That timing is required for MoE:
    the tail of ``Fp8MoEMethod._setup_kernel`` rebuilds ``moe_quant_config`` and
    ``moe_kernel`` from whatever is on the layer at that moment, and those cache
    tensor references, so a later swap would leave the kernel pointing at the
    staging buffers.
    """
    live = getattr(layer, _FP8_LIVE_ATTR, None) or {}
    param = live.get(param_name)
    if param is not None and new_data is not None:
        _fold_into_live_param(layer, param_name, param, new_data)
        del live[param_name]
        return

    if new_data is None:
        setattr(layer, param_name, None)
        return

    if isinstance(new_data, torch.nn.Parameter):
        new_data = new_data.data

    old_param = getattr(layer, param_name, None)
    param = torch.nn.Parameter(new_data, requires_grad=False)
    _copy_param_subclass_attrs(param, old_param)
    setattr(layer, param_name, param)


def _restore_layer_param_subclass_attrs(layer: torch.nn.Module, old_params: dict[str, torch.nn.Parameter]):
    for name, old_param in old_params.items():
        new_param = getattr(layer, name, None)
        if isinstance(new_param, torch.nn.Parameter):
            _copy_param_subclass_attrs(new_param, old_param)


def _record_pristine_fp8_layout(layer, params):
    """Remember the checkpoint layout of a layer's FP8 weights and scales.

    Kernel post-processing rewrites these into inference layouts that the
    checkpoint tensors no longer fit. DeepGEMM, for instance, packs a
    ``(N/128, K/128)`` block scale into an ``(N, K/512)`` int32 UE8M0 tensor,
    so a refit feeding checkpoint-layout scales hits a shape assert. Keep
    enough state to rebuild the original buffers before each reload.

    Only the layout is recorded, never a tensor. This runs during model load,
    inside the memory pool vLLM tags as ``weights``, so any buffer kept here
    would be subject to that pool's discard-and-rezero across a sleep and would
    not survive to the next refit anyway.
    """
    if hasattr(layer, _FP8_PRISTINE_ATTR):
        return

    pristine = {}
    for name in _FP8_REFIT_PARAM_NAMES:
        param = params.get(name)
        if not isinstance(param, torch.nn.Parameter):
            continue
        pristine[name] = (tuple(param.shape), param.dtype)

    if pristine:
        setattr(layer, _FP8_PRISTINE_ATTR, pristine)


def _layer_needs_fp8_staging(layer, pristine) -> bool:
    for name, (shape, dtype) in pristine.items():
        param = getattr(layer, name, None)
        if not isinstance(param, torch.nn.Parameter):
            continue
        if tuple(param.shape) != shape or param.dtype != dtype:
            return True
        # ROCm's AITER MoE backend permutes the expert weights into its MFMA
        # layout while leaving shape and dtype untouched, so the comparison
        # above cannot see it. ``is_shuffled``, which that backend sets on the
        # repacked parameter, is the only signal that the live buffer is no
        # longer in checkpoint layout.
        if getattr(param, "is_shuffled", False):
            return True
    return False


def _fp8_staging_data(param, shape, dtype):
    """Checkpoint-layout tensor that ``load_weights`` can write into.

    Reuses the live storage whenever the kernel transform only reshaped it, so
    those writes land straight in the tensor the CUDA graph points at.
    Otherwise the buffer is allocated fresh for this refit and dropped with it,
    which keeps nothing alive across the engine's sleep.

    An all-ones fill is NaN in every dtype staged here, so a buffer the incoming
    stream fails to write blows up on the first forward instead of being repacked
    into the live weights. Uninitialized memory would carry whatever the caching
    allocator last held, which for a scale is most likely a plausible-looking
    value from a previous refit.
    """
    if param.dtype == dtype and param.numel() == math.prod(shape):
        return param.data.reshape(shape)
    staging = torch.empty(shape, dtype=dtype, device=param.device)
    staging.view(torch.uint8).fill_(0xFF)
    return staging


def stage_fp8_params_for_loading(model):
    """Expose checkpoint-layout buffers so ``load_weights`` can write into them.

    Covers both quant methods: ``Fp8LinearMethod`` layers keyed on ``weight`` /
    ``weight_scale*`` and ``Fp8MoEMethod`` expert modules keyed on ``w13_*`` /
    ``w2_*``. The live parameters are set aside and reinstated by
    ``process_fp8_weights_after_loading``, which puts the re-derived inference
    layout back into their original storage. No storage the CUDA graph captured
    is reallocated -- only tensor contents change -- and no forward pass runs
    between the two calls.
    """
    staged_layers = []
    for layer in model.modules():
        pristine = getattr(layer, _FP8_PRISTINE_ATTR, None)
        # A layer left in checkpoint layout (its kernel did not repack, e.g. a
        # Triton or vLLM-CUTLASS MoE backend) already loads as-is.
        if not pristine or not _layer_needs_fp8_staging(layer, pristine):
            continue

        live = {}
        for name, (shape, dtype) in pristine.items():
            param = getattr(layer, name, None)
            if not isinstance(param, torch.nn.Parameter):
                continue
            live[name] = param
            staged = torch.nn.Parameter(_fp8_staging_data(param, shape, dtype), requires_grad=False)
            _copy_param_subclass_attrs(staged, param)
            setattr(layer, name, staged)

        setattr(layer, _FP8_LIVE_ATTR, live)
        staged_layers.append(layer)

    # A zero here means no layer ever recorded a pristine layout, so the
    # post-processing hook never ran and the refit is unprotected.
    logger.info("Staged %d FP8 layers for refit", len(staged_layers))
    return staged_layers


def process_fp8_weights_after_loading(layers):
    """Re-derive the inference layout in place and reinstate the live params."""
    for layer in layers:
        live = getattr(layer, _FP8_LIVE_ATTR, None) or {}

        quant_method = getattr(layer, "quant_method", None)
        process = getattr(quant_method, "process_weights_after_loading", None)
        if process is not None:
            process(layer)

        # Anything routed through the patched ``replace_parameter`` has already
        # been folded and dropped from ``live``; what remains was rewritten by a
        # kernel that imported ``replace_parameter`` into its own module, out of
        # reach of the patch. The block-scaled linear kernels all do that.
        for name, param in list(live.items()):
            _fold_into_live_param(layer, name, param, getattr(layer, name))

        if hasattr(layer, _FP8_LIVE_ATTR):
            delattr(layer, _FP8_LIVE_ATTR)


def _make_process_weights_after_loading_for_vllm20(original_fn):
    def _patched_process_weights_after_loading(self, layer) -> None:
        old_params = dict(layer.named_parameters(recurse=False))
        _record_pristine_fp8_layout(layer, old_params)
        with patch(
            "vllm.model_executor.layers.quantization.fp8.replace_parameter", replace_parameter_preserve_subclass
        ):
            original_fn(self, layer)
        _restore_layer_param_subclass_attrs(layer, old_params)

    return _patched_process_weights_after_loading


def process_weights_after_loading_for_vllm14(self, layer) -> None:
    """This function is used to process the weights after loading for a Linear layer, it is used for vllm v0.18-v0.19.

    Compared to the original process_weights_after_loading in vllm, we just avoid creation of
    new torch.nn.Parameter objects, because that removes the weight_loader attribute which we need for refit.
    """
    from torch.nn import Parameter
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        maybe_post_process_fp8_weight_block,
        process_fp8_weight_block_strategy,
    )
    from vllm.model_executor.parameter import BlockQuantScaleParameter, ModelWeightParameter

    assert self.block_quant and self.quant_config.is_checkpoint_fp8_serialized
    assert self.quant_config.activation_scheme == "dynamic"

    def _create_param_from_subclass_attributes(custom_param):
        param = Parameter(custom_param.data, requires_grad=False)
        _copy_param_subclass_attrs(param, custom_param)
        return param

    weight, weight_scale_inv = process_fp8_weight_block_strategy(layer.weight, layer.weight_scale_inv)

    layer.weight = _create_param_from_subclass_attributes(
        ModelWeightParameter(
            data=weight.data,
            output_dim=0,
            input_dim=1,
            weight_loader=layer.weight.weight_loader,
        )
    )
    layer.weight_scale_inv = _create_param_from_subclass_attributes(
        BlockQuantScaleParameter(
            data=weight_scale_inv.data,
            output_dim=0,
            input_dim=1,
            weight_loader=layer.weight_scale_inv.weight_loader,
        )
    )

    # vLLM v0.17 removed the `else: register_parameter("input_scale", None)` from
    # create_weights() for dynamic activation, but apply() still accesses layer.input_scale.
    # Since block_quant always uses dynamic activation, ensure the attribute exists.
    if not hasattr(layer, "input_scale"):
        layer.input_scale = None

    maybe_post_process_fp8_weight_block(layer)


def process_weights_after_loading_moe_for_vllm14(self, layer) -> None:
    # removed the reentrancy guard here for refit
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import convert_to_fp8_moe_kernel_format, make_fp8_moe_kernel

    # Allow for accessing weights and scales in standard way.
    w13 = layer.w13_weight
    w2 = layer.w2_weight
    w13_scale = getattr(layer, f"w13_{self.weight_scale_name}")
    w2_scale = getattr(layer, f"w2_{self.weight_scale_name}")
    w13_input_scale = layer.w13_input_scale
    w2_input_scale = layer.w2_input_scale

    # Shuffle weights to runtime format and setup kernel.
    w13, w2, w13_scale, w2_scale = convert_to_fp8_moe_kernel_format(
        fp8_backend=self.fp8_backend,
        layer=layer,
        w13=w13,
        w2=w2,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        w13_input_scale=w13_input_scale,
        w2_input_scale=w2_input_scale,
    )
    from torch.nn import Parameter

    def _create_param_from_subclass_attributes(custom_data, custom_weight):
        param = Parameter(custom_data, requires_grad=False)
        for attr, value in _iter_transferable_param_attrs(custom_weight):
            setattr(param, attr, value)
        return param

    # Replace parameters with updated versions. Note that this helper
    # function ensures the replacement is compatible with RL weight reloads.
    layer.w13_weight = _create_param_from_subclass_attributes(w13, layer.w13_weight)
    layer.w2_weight = _create_param_from_subclass_attributes(w2, layer.w2_weight)
    layer.w13_weight_scale_inv = _create_param_from_subclass_attributes(w13_scale, layer.w13_weight_scale_inv)
    layer.w2_weight_scale_inv = _create_param_from_subclass_attributes(w2_scale, layer.w2_weight_scale_inv)

    self.moe_quant_config = self.get_fused_moe_quant_config(layer)
    if self.moe_quant_config:
        assert self.experts_cls is not None

        # Check for the new API by inspecting the function signature, which is more
        # robust than version string comparison, especially for dev/pre-release versions.
        sig = inspect.signature(make_fp8_moe_kernel)
        if "routing_tables" in sig.parameters:
            # vLLM >= 0.16+: routing_tables/shared_experts added, returns kernel directly
            self.moe_kernel = make_fp8_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                fp8_backend=self.fp8_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._maybe_init_expert_routing_tables(),
                shared_experts=layer.shared_experts,
            )
        else:
            # vLLM 0.14/0.15: routing_tables/shared_experts not supported, returns (kernel, use_inplace)
            self.kernel, self.use_inplace = make_fp8_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                fp8_backend=self.fp8_backend,
                experts_cls=self.experts_cls,
            )


def build_fp8_method_patchers(vllm_version):
    """Patchers that make the FP8 quant methods survive a refit, not yet started.

    The caller owns starting and tracking them so all patch lifetime lives in
    one place.
    """
    linear_path = "vllm.model_executor.layers.quantization.fp8.Fp8LinearMethod.process_weights_after_loading"
    moe_path = "vllm.model_executor.layers.quantization.fp8.Fp8MoEMethod.process_weights_after_loading"

    # vLLM 0.20 refactored FP8 post-load handling and removed
    # maybe_post_process_fp8_weight_block. Keep its native transformation logic,
    # but preserve parameter subclass metadata for RL weight reloads.
    if vllm_version >= version.parse("0.20.0"):
        from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod, Fp8MoEMethod

        wrap = _make_process_weights_after_loading_for_vllm20
        return [
            patch(linear_path, wrap(Fp8LinearMethod.process_weights_after_loading)),
            patch(moe_path, wrap(Fp8MoEMethod.process_weights_after_loading)),
        ]

    return [
        patch(linear_path, process_weights_after_loading_for_vllm14),
        patch(moe_path, process_weights_after_loading_moe_for_vllm14),
    ]
