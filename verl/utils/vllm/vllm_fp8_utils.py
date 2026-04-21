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

import inspect
import logging
from dataclasses import dataclass, field
from unittest.mock import patch

import torch
import vllm
from packaging import version

try:
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.linear import LinearBase
except ImportError as e:
    raise ImportError("FP8 quantization not available") from e

from verl.utils.kernel.fp8_kernel import scaled_fp8_blockwise

logger = logging.getLogger(__name__)

MXFP8_BLOCK_QUANT_KWARGS = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "ascend",
    "weight_block_size": [1, 32],
    # Enable dynamic inference mode, no need to pre-generate quant_model_description.json
    # for each model. Just specify default_quant_type, vllm-ascend will automatically
    # infer quant type based on layer type:
    # - lm_head, embedding, norm layers -> FLOAT
    # - Other Linear, MoE layers -> default_quant_type (W8A8_MXFP8)
    "default_quant_type": "W8A8_MXFP8",
    # group_size for MXFP8 quantization
    "group_size": 32,
}


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
            if isinstance(current_module, FusedMoE):
                return current_module
            elif isinstance(current_module, torch.nn.ModuleList):
                current_module = current_module[int(part)]
            else:
                current_module = getattr(current_module, part)
    except (AttributeError, IndexError, ValueError) as e:
        print(f"Warning: Could not find module for parameter '{name}'. Error: {e}")
    return current_module


def is_fp8_weight(name, model):
    if name not in fp8_state.seen_params:
        fp8_state.seen_params.add(name)
        # Filter out bias params
        if name.endswith("weight"):
            module = get_module_from_param_name(model, name)
            # We currently only quantize linear layers

            if (isinstance(module, LinearBase) and module.weight.dtype == torch.float8_e4m3fn) or (
                isinstance(module, FusedMoE)
                and module.w13_weight.dtype == torch.float8_e4m3fn
                and module.w2_weight.dtype == torch.float8_e4m3fn
            ):
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
        if (isinstance(module, LinearBase) or isinstance(module, FusedMoE)) and hasattr(
            module, "_mxfp8_original_shapes"
        ):
            if hasattr(module, "quant_method") and hasattr(module.quant_method, "process_weights_after_loading"):
                logger.debug(f"Applying MXFP8 transformation for module: {name}")
                module.quant_method.process_weights_after_loading(module)


def npu_scaled_mxfp8_blockwise(
    data_hp,
    weight_block_size,
):
    assert data_hp.dim() == 2, "Only 2D tensors supported (M, N)"

    block_size = weight_block_size[1]

    # Constants for MXFP8 / NPU
    FP32_MIN_NORMAL = torch.finfo(torch.float32).tiny
    MAX_NORM = torch.finfo(torch.float8_e4m3fn).max  # 2 ** 8 * 1.75
    EMAX = 8  # 2 ** (4 - 1)
    SCALE_EMAX = 127  # 2 ** (8 - 1) - 1

    data_hp = data_hp.float()
    original_shape = data_hp.shape
    M, N = original_shape
    assert N % block_size == 0, f"Last dimension {N} must be divisible by block_size {block_size}"

    # Reshape to (M, N // block_size, block_size)
    num_blocks_n = N // block_size
    data_blocked = data_hp.reshape(M, num_blocks_n, block_size)

    # Calculate max absolute value per block
    max_val = torch.amax(torch.abs(data_blocked), dim=-1)

    # Shared exponent calculation
    # Handle zero/tiny values to avoid log2(0) -> -inf
    max_val_safe = torch.where(max_val == 0, FP32_MIN_NORMAL, max_val)
    shared_exp = torch.floor(torch.log2(max_val_safe)) - EMAX

    shared_exp[shared_exp > SCALE_EMAX] = float("NaN")

    shared_exp_expanded = shared_exp.unsqueeze(-1)
    scale_factor = torch.pow(2.0, shared_exp_expanded)
    data_normalized = data_blocked / scale_factor
    abs_norm = torch.abs(data_normalized)
    private_exp = torch.floor(torch.log2(abs_norm + (abs_norm == 0).float()))
    min_exp = -6
    private_exp = private_exp.clamp(min=min_exp)

    mantissa_scale = 8.0  # 2 ** (5 - 2)

    scale_private = torch.pow(2.0, private_exp)
    scaled = data_normalized / scale_private * mantissa_scale

    # Round half away from zero: sign * floor(abs + 0.5)
    data_quant = torch.sign(scaled) * torch.floor(torch.abs(scaled) + 0.5)
    data_quant = data_quant / mantissa_scale * scale_private
    data_quant = torch.clamp(data_quant, min=-MAX_NORM, max=MAX_NORM)

    # Restore Inf/NaN
    data_quant = torch.where(torch.isinf(data_normalized), data_normalized, data_quant)
    data_quant = torch.where(torch.isnan(data_normalized), data_normalized, data_quant)

    fp_data = data_quant.reshape(original_shape).to(torch.float8_e4m3fn)

    # Encode scale/exponent for NPU (uint8)
    shared_exp_fixed = torch.nan_to_num(shared_exp, nan=-127.0)
    descale_fp = torch.clamp(shared_exp_fixed + 127, 0, 255).round().to(torch.uint8)

    return fp_data, descale_fp


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

    is_mxfp8_npu = is_mxfp8_vllm_ascend(quant_config)

    weight_block_size = None
    if is_mxfp8_npu:
        weight_block_size = MXFP8_BLOCK_QUANT_KWARGS["weight_block_size"]
    else:
        if quant_config.weight_block_size is None:
            raise ValueError(
                "Currently only support blockwise quantization, please set weight_block_size in quant_config"
            )
        weight_block_size = quant_config.weight_block_size

    # vLLM v0.11-v0.12 renamed weight_scale_inv → weight_scale in process_weights_after_loading,
    # so load_weights expects "_scale" suffix. v0.14+ keeps weight_scale_inv, so expects "_scale_inv".
    _use_scale_not_scale_inv = version.parse("0.11.0") <= version.parse(vllm.__version__) < version.parse("0.14.0")

    for k, v in weights:
        if not is_fp8_weight(k, model):
            yield (k, v)
            continue

        # Cast the weight into fp8 and its scale factor
        if torch.distributed.get_rank() == 0:
            logger.debug(f"Quantizing to FP8 blockwise: {k}")
        if is_mxfp8_npu:
            param_lp, param_scale = npu_scaled_mxfp8_blockwise(
                v.to(dtype),
                weight_block_size=weight_block_size,
            )
        else:
            param_lp, param_scale = scaled_fp8_blockwise(
                v.to(dtype),
                weight_block_size=quant_config.weight_block_size,
            )
        param_scale = param_scale.squeeze(-1)

        # Yield the quantized weight
        yield (k, param_lp)

        # Yield the scale with appropriate naming based on vLLM version
        if _use_scale_not_scale_inv and "expert" not in k:
            yield (k + "_scale", param_scale)
        elif not is_mxfp8_npu:
            yield (k + "_scale_inv", param_scale)

        # Explicitly delete original tensor reference to help GC
        del v, param_lp, param_scale


def load_quanted_weights(weights, model_runner):
    model = model_runner.model
    quant_config = model_runner.vllm_config.quant_config
    vllm_dtype = model_runner.vllm_config.model_config.dtype

    is_mxfp8_npu = is_mxfp8_vllm_ascend(quant_config)

    if is_mxfp8_npu:
        # For MXFP8 on NPU, we need to restore weights to original shapes
        # before loading, then re-apply transformation after loading.
        # This is because process_weights_after_loading transposes the weights,
        # but the weight_loader expects original shapes.
        restore_mxfp8_weights_for_loading(model)

    weights_quantized = quant_weights(weights, model, quant_config, dtype=vllm_dtype)

    # Monkey patch the param class to their subclass, as certain models
    # will check the param type to call the proper weightloader
    for name, param in model.named_parameters():
        if hasattr(param, "subclass_type"):
            param.orig_type = param.__class__
            param.__class__ = param.subclass_type
    # Finally load the weights into vllm
    loaded_params = model.load_weights(weights_quantized)
    # Undo the type change above to the original type
    for name, param in model.named_parameters():
        if hasattr(param, "subclass_type"):
            param.__class__ = param.orig_type

    if is_mxfp8_npu:
        # Re-apply MXFP8 transformations after weight loading
        apply_mxfp8_transformation_after_loading(model)

    return loaded_params


def process_weights_after_loading_for_vllm10(self, layer) -> None:
    """This function is used to process the weights after loading for a Linear layer, it is used for vllm v0.10

    Compared to the original process_weights_after_loading in vllm, we just avoid creation of
    new torch.nn.Parameter objects, because that removes the weight_loader attribute which we need for refit.
    """
    logger.debug("Applying patch process_weights_after_loading")
    try:
        from vllm.model_executor.parameter import (
            BlockQuantScaleParameter,
            ModelWeightParameter,
        )
    except Exception:
        print("error")
    from torch.nn import Parameter

    def _create_param_from_subclass_attributes(custom_param):
        param = Parameter(custom_param.data, requires_grad=False)
        base_param_dir = dir(torch.nn.Parameter)
        custom_param_dir = dir(custom_param)
        # Find the attributes that are unique to the custom parameter
        custom_attributes = [
            attr for attr in custom_param_dir if attr not in base_param_dir and not attr.startswith("__")
        ]
        # Set the custom attributes into the base parameter object
        for attr in custom_attributes:
            setattr(param, attr, getattr(custom_param, attr))

        param.subclass_type = type(custom_param)
        return param

    assert self.block_quant and self.quant_config.is_checkpoint_fp8_serialized
    assert self.quant_config.activation_scheme == "dynamic"
    weight = layer.weight.data
    weight_scale_inv = layer.weight_scale_inv.data
    weight = self._maybe_pad_weight(weight)

    layer.weight = _create_param_from_subclass_attributes(
        ModelWeightParameter(
            data=weight,
            output_dim=0,
            input_dim=1,
            weight_loader=layer.weight.weight_loader,
        )
    )
    layer.weight_scale_inv = _create_param_from_subclass_attributes(
        BlockQuantScaleParameter(
            data=weight_scale_inv,
            output_dim=0,
            input_dim=1,
            weight_loader=layer.weight_scale_inv.weight_loader,
        )
    )


def process_weights_after_loading_for_vllm11(self, layer) -> None:
    """This function is used to process the weights after loading for a Linear layer, it is used for vllm 0.11

    Compared to the original process_weights_after_loading in vllm, we just avoid creation of
    new torch.nn.Parameter objects, because that removes the weight_loader attribute which we need for refit.
    """
    from torch.nn import Parameter
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        maybe_post_process_fp8_weight_block,
        process_fp8_weight_block_strategy,
    )
    from vllm.model_executor.parameter import (
        BlockQuantScaleParameter,
        ModelWeightParameter,
    )

    assert self.block_quant and self.quant_config.is_checkpoint_fp8_serialized
    assert self.quant_config.activation_scheme == "dynamic"

    def _create_param_from_subclass_attributes(custom_param):
        param = Parameter(custom_param.data, requires_grad=False)
        base_param_dir = dir(torch.nn.Parameter)
        custom_param_dir = dir(custom_param)
        # Find the attributes that are unique to the custom parameter
        custom_attributes = [
            attr for attr in custom_param_dir if attr not in base_param_dir and not attr.startswith("__")
        ]
        # Set the custom attributes into the base parameter object
        for attr in custom_attributes:
            setattr(param, attr, getattr(custom_param, attr))

        param.subclass_type = type(custom_param)
        return param

    weight_scale = layer.weight_scale_inv if hasattr(layer, "weight_scale_inv") else layer.weight_scale
    weight, weight_scale = process_fp8_weight_block_strategy(layer.weight, weight_scale)

    layer.weight = _create_param_from_subclass_attributes(
        ModelWeightParameter(
            data=weight.data,
            output_dim=0,
            input_dim=1,
            weight_loader=layer.weight.weight_loader,
        )
    )
    layer.weight_scale = _create_param_from_subclass_attributes(
        BlockQuantScaleParameter(
            data=weight_scale.data,
            output_dim=0,
            input_dim=1,
            weight_loader=layer.weight_scale_inv.weight_loader,
        )
    )

    del layer.weight_scale_inv

    if version.parse(vllm.__version__) == version.parse("0.11.0"):
        maybe_post_process_fp8_weight_block(layer, self.cutlass_block_fp8_supported)
    else:
        maybe_post_process_fp8_weight_block(layer)


def process_weights_after_loading_for_vllm14(self, layer) -> None:
    """process_weights_after_loading for vLLM >= 0.14.

    Starting from v0.14, vLLM keeps the scale parameter as `weight_scale_inv`
    (instead of renaming it to `weight_scale` like v0.11-v0.12), and `apply()`
    accesses `layer.weight_scale_inv`. We preserve `weight_loader` and
    `subclass_type` attributes so that refit (repeated weight sync) works.
    """
    from torch.nn import Parameter
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        maybe_post_process_fp8_weight_block,
        process_fp8_weight_block_strategy,
    )
    from vllm.model_executor.parameter import (
        BlockQuantScaleParameter,
        ModelWeightParameter,
    )

    assert self.block_quant and self.quant_config.is_checkpoint_fp8_serialized
    assert self.quant_config.activation_scheme == "dynamic"

    def _create_param_from_subclass_attributes(custom_param):
        param = Parameter(custom_param.data, requires_grad=False)
        base_param_dir = dir(torch.nn.Parameter)
        custom_param_dir = dir(custom_param)
        custom_attributes = [
            attr for attr in custom_param_dir if attr not in base_param_dir and not attr.startswith("__")
        ]
        for attr in custom_attributes:
            setattr(param, attr, getattr(custom_param, attr))

        param.subclass_type = type(custom_param)
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


def process_weights_after_loading_moe_for_vllm10(self, layer) -> None:
    """This function is used to process the weights after loading for a FusedMoE layer, it is used for vllm v0.10"""
    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import is_rocm_aiter_moe_enabled
    from vllm.model_executor.layers.quantization.fp8 import _is_col_major, _swap_w13_to_w31
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        get_col_major_tma_aligned_tensor,
        requant_weight_ue8m0_inplace,
    )
    from vllm.utils.deep_gemm import is_blackwell_deep_gemm_used

    self.rocm_aiter_moe_enabled = is_rocm_aiter_moe_enabled()
    assert self.quant_config.activation_scheme == "dynamic"
    if self.flashinfer_moe_enabled:
        w13_weight = _swap_w13_to_w31(layer.w13_weight.data)
        w13_weight_scale_inv = _swap_w13_to_w31(layer.w13_weight_scale_inv.data)
        w2_weight = layer.w2_weight.data
        w2_weight_scale_inv = layer.w2_weight_scale_inv.data
    else:
        w13_weight = layer.w13_weight.data
        w13_weight_scale_inv = layer.w13_weight_scale_inv.data
        w2_weight = layer.w2_weight
        w2_weight_scale_inv = layer.w2_weight_scale_inv

    from torch.nn import Parameter

    def _create_param_from_subclass_attributes(custom_data, custom_weight):
        param = Parameter(custom_data, requires_grad=False)
        base_param_dir = dir(torch.nn.Parameter)
        custom_weight_dir = dir(custom_weight)
        # Find the attributes that are unique to the custom parameter
        custom_attributes = [
            attr for attr in custom_weight_dir if attr not in base_param_dir and not attr.startswith("__")
        ]
        # Set the custom attributes into the base parameter object
        for attr in custom_attributes:
            setattr(param, attr, getattr(custom_weight, attr))

        return param

    layer.w13_weight = _create_param_from_subclass_attributes(w13_weight, layer.w13_weight)
    layer.w13_weight_scale_inv = _create_param_from_subclass_attributes(
        w13_weight_scale_inv, layer.w13_weight_scale_inv
    )
    layer.w2_weight = _create_param_from_subclass_attributes(w2_weight, layer.w2_weight)
    layer.w2_weight_scale_inv = _create_param_from_subclass_attributes(w2_weight_scale_inv, layer.w2_weight_scale_inv)

    # DeepGemm scales need to be transposed and aligned.  We try to do
    # it ahead of time for performance reasons.
    if self.allow_deep_gemm and not is_blackwell_deep_gemm_used():
        # Lazy import to avoid CUDA initialization problems.
        if _is_col_major(layer.w13_weight_scale_inv):
            layer.w13_weight_scale_inv = get_col_major_tma_aligned_tensor(layer.w13_weight_scale_inv).contiguous()
        if _is_col_major(layer.w2_weight_scale_inv):
            layer.w2_weight_scale_inv = get_col_major_tma_aligned_tensor(layer.w2_weight_scale_inv).contiguous()

    if is_blackwell_deep_gemm_used():
        assert layer.weight_block_size is not None
        # Re-quantise the expert weights so their scales are UE8M0.
        block_sz = tuple(layer.weight_block_size)
        requant_weight_ue8m0_inplace(
            layer.w13_weight.data,
            layer.w13_weight_scale_inv.data,
            block_sz,
        )
        requant_weight_ue8m0_inplace(
            layer.w2_weight.data,
            layer.w2_weight_scale_inv.data,
            block_sz,
        )

        if _is_col_major(layer.w13_weight_scale_inv):
            layer.w13_weight_scale_inv = get_col_major_tma_aligned_tensor(layer.w13_weight_scale_inv).contiguous()
        if _is_col_major(layer.w2_weight_scale_inv):
            layer.w2_weight_scale_inv = get_col_major_tma_aligned_tensor(layer.w2_weight_scale_inv).contiguous()


def process_weights_after_loading_moe_for_vllm11(self, layer) -> None:
    """This function is used to process the weights after loading for a FusedMoE layer, it is used for vllm 0.11"""
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        swap_w13_to_w31,
    )
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        expert_weight_is_col_major,
        requant_weight_ue8m0_inplace,
    )
    from vllm.utils.deep_gemm import (
        get_col_major_tma_aligned_tensor,
        is_deep_gemm_e8m0_used,
    )

    try:
        from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import is_rocm_aiter_moe_enabled

        self.rocm_aiter_moe_enabled = is_rocm_aiter_moe_enabled()
    except ImportError:
        from vllm._aiter_ops import rocm_aiter_ops

        self.rocm_aiter_moe_enabled = rocm_aiter_ops.is_fused_moe_enabled()

    assert self.block_quant and self.quant_config.is_checkpoint_fp8_serialized
    assert self.quant_config.activation_scheme == "dynamic"

    if self.flashinfer_moe_backend is not None:
        layer.w13_weight.data = swap_w13_to_w31(layer.w13_weight.data)
        layer.w13_weight_scale_inv.data = swap_w13_to_w31(layer.w13_weight_scale_inv.data)

    if self.allow_deep_gemm and not is_deep_gemm_e8m0_used():
        if expert_weight_is_col_major(layer.w13_weight_scale_inv):
            layer.w13_weight_scale_inv = get_col_major_tma_aligned_tensor(layer.w13_weight_scale_inv)
        if expert_weight_is_col_major(layer.w2_weight_scale_inv):
            layer.w2_weight_scale_inv = get_col_major_tma_aligned_tensor(layer.w2_weight_scale_inv)

    if is_deep_gemm_e8m0_used():
        assert layer.weight_block_size is not None
        # Re-quantise the expert weights so their scales are UE8M0.
        block_sz = tuple(layer.weight_block_size)
        requant_weight_ue8m0_inplace(
            layer.w13_weight.data,
            layer.w13_weight_scale_inv.data,
            block_sz,
        )
        requant_weight_ue8m0_inplace(
            layer.w2_weight.data,
            layer.w2_weight_scale_inv.data,
            block_sz,
        )

        # Ensure column-major TMA alignment expected by DeepGEMM.
        if expert_weight_is_col_major(layer.w13_weight_scale_inv):
            layer.w13_weight_scale_inv = get_col_major_tma_aligned_tensor(layer.w13_weight_scale_inv)
        if expert_weight_is_col_major(layer.w2_weight_scale_inv):
            layer.w2_weight_scale_inv = get_col_major_tma_aligned_tensor(layer.w2_weight_scale_inv)


def process_weights_after_loading_moe_for_vllm14(self, layer) -> None:
    # removed the reentrancy guard here for refit
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        convert_to_fp8_moe_kernel_format,
        make_fp8_moe_kernel,
    )

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
        base_param_dir = dir(torch.nn.Parameter)
        custom_weight_dir = dir(custom_weight)
        # Find the attributes that are unique to the custom parameter
        custom_attributes = [
            attr for attr in custom_weight_dir if attr not in base_param_dir and not attr.startswith("__")
        ]
        # Set the custom attributes into the base parameter object
        for attr in custom_attributes:
            setattr(param, attr, getattr(custom_weight, attr))

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


def apply_vllm_fp8_patches():
    logger.info("Applying vllm fp8 patches for blockwise quantization")
    vllm_ver = version.parse(vllm.__version__)

    # Linear patch: v0.14+ keeps weight_scale_inv, v0.11-v0.12 renames to weight_scale
    func1_path = "vllm.model_executor.layers.quantization.fp8.Fp8LinearMethod.process_weights_after_loading"
    if vllm_ver >= version.parse("0.14.0"):
        linear_patch_fn = process_weights_after_loading_for_vllm14
    elif vllm_ver >= version.parse("0.11.0"):
        linear_patch_fn = process_weights_after_loading_for_vllm11
    else:
        linear_patch_fn = process_weights_after_loading_for_vllm10
    patcher1 = patch(func1_path, linear_patch_fn)
    patcher1.start()

    # MoE patch
    func2_path = "vllm.model_executor.layers.quantization.fp8.Fp8MoEMethod.process_weights_after_loading"
    if vllm_ver >= version.parse("0.14.0"):
        moe_patch_fn = process_weights_after_loading_moe_for_vllm14
    elif vllm_ver >= version.parse("0.11.0"):
        moe_patch_fn = process_weights_after_loading_moe_for_vllm11
    else:
        moe_patch_fn = process_weights_after_loading_moe_for_vllm10
    patcher2 = patch(func2_path, moe_patch_fn)
    patcher2.start()
