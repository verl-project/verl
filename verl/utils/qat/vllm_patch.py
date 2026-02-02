# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""
vLLM NVFP4 Patches for Dynamic Weight Updates.

This module enables dynamic weight reloading for quantized models in vLLM,
supporting QAT (Quantization-Aware Training) weight synchronization.

Key features:
- LazyParamsDict for parameter metadata caching and lazy rebuild
- Tensor Swap for parameters with shape changes (CUDA Graph address stability)
- Support for multi-bucket weight loading

Supported schemes:
- Dense: W4A16-FP4, W4A4-FP4
- MoE: W4A16-FP4-MoE (MARLIN backend), W4A4-FP4-MoE (FlashInfer/CUTLASS backend)
"""

import logging
import os
from collections.abc import Callable
from typing import Optional
from unittest.mock import patch

import torch
from torch.nn import Parameter

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# Parameters deleted after process_weights, need lazy rebuild
DENSE_LAZY_REBUILD_PARAMS = {"weight_packed", "weight_global_scale"}
MOE_LAZY_REBUILD_PARAMS = {"w13_weight_packed", "w2_weight_packed"}

# Parameters with shape changes, need Tensor Swap for address stability
DENSE_TENSOR_SWAP_PARAMS = {"weight_scale"}
MOE_TENSOR_SWAP_PARAMS = {"w13_weight_scale", "w2_weight_scale"}


class LazyParamsDict(dict):
    """
    Dict-like class for parameter management with lazy rebuild and tensor swap.

    Supports:
    - Lazy rebuild of deleted parameters from saved metadata
    - Tensor Swap for parameters with shape changes (address stability for CUDA Graph)
    """

    def __init__(self, model: torch.nn.Module, device: Optional[torch.device] = None):
        """
        Initialize LazyParamsDict from a model.

        Args:
            model: vLLM model (may be wrapped in ModelRunner)
            device: Device for created parameters
        """
        super().__init__()
        self.device = device

        # Get the actual model (handle vLLM's wrapper structure)
        actual_model = model
        if hasattr(model, "model"):
            actual_model = model.model
        self._model = actual_model

        # Build mappings by scanning all modules
        self._layer_meta_cache: dict[str, dict] = {}  # Cache of _hf_param_meta
        self._tensor_swap_layers: dict[str, dict] = {}  # Layers needing tensor swap

        self._build_mappings()

        # Initialize with current parameters
        for name, param in actual_model.named_parameters():
            self[name] = param

    def _build_mappings(self):
        """Build layer metadata cache for lazy rebuild and tensor swap."""
        for layer_name, module in self._model.named_modules():
            # Check for _hf_param_meta which indicates this layer has HF format params
            if hasattr(module, "_hf_param_meta"):
                self._layer_meta_cache[layer_name] = {
                    "module": module,
                    "meta": module._hf_param_meta,
                }

                # Check for tensor swap layers (weight_scale with shape change)
                if "weight_scale" in module._hf_param_meta:
                    marlin_refs = getattr(module, "_marlin_tensor_refs", {})
                    if "weight_scale" in marlin_refs:
                        self._tensor_swap_layers[layer_name] = {
                            "module": module,
                            "marlin_ref": marlin_refs["weight_scale"],
                            "hf_meta": module._hf_param_meta["weight_scale"],
                        }

                # MoE layers (w13_weight_scale, w2_weight_scale)
                if "w13_weight_scale" in module._hf_param_meta:
                    marlin_refs = getattr(module, "_marlin_tensor_refs", {})
                    if "w13_weight_scale" in marlin_refs:
                        self._tensor_swap_layers[f"{layer_name}.w13"] = {
                            "module": module,
                            "param_name": "w13_weight_scale",
                            "marlin_ref": marlin_refs["w13_weight_scale"],
                            "hf_meta": module._hf_param_meta["w13_weight_scale"],
                        }
                    if "w2_weight_scale" in marlin_refs:
                        self._tensor_swap_layers[f"{layer_name}.w2"] = {
                            "module": module,
                            "param_name": "w2_weight_scale",
                            "marlin_ref": marlin_refs["w2_weight_scale"],
                            "hf_meta": module._hf_param_meta["w2_weight_scale"],
                        }

    def _try_lazy_rebuild(self, key: str) -> Optional[Parameter]:
        """
        Try to rebuild a parameter from metadata if it was deleted.

        Args:
            key: Full parameter name

        Returns:
            Rebuilt parameter or None if cannot rebuild
        """
        # Extract layer name and param name
        parts = key.rsplit(".", 1)
        if len(parts) != 2:
            return None

        layer_name, param_name = parts

        # Check if we have metadata for this layer
        if layer_name not in self._layer_meta_cache:
            return None

        cache_entry = self._layer_meta_cache[layer_name]
        module = cache_entry["module"]
        meta = cache_entry["meta"]

        # Check if this param needs rebuild
        if param_name not in meta:
            return None

        # Already exists on module?
        if hasattr(module, param_name):
            param = getattr(module, param_name)
            if param is not None:
                return param

        # Rebuild from metadata
        param_meta = meta[param_name]
        shape = param_meta["shape"]
        dtype = param_meta["dtype"]
        device = self.device or param_meta.get("device", "cuda")
        param_class = param_meta.get("param_class", Parameter)

        # Get saved weight_loader
        weight_loaders = getattr(module, "_weight_loaders", {})
        weight_loader = weight_loaders.get(param_name)

        # Create parameter data tensor
        data = torch.empty(shape, dtype=dtype, device=device)

        # Try to use the original parameter class (vLLM special types)
        try:
            if param_class is not Parameter and weight_loader is not None:
                # vLLM parameter types need specific constructor arguments
                kwargs = {"data": data, "weight_loader": weight_loader}

                # Add input_dim/output_dim if saved
                if "input_dim" in param_meta:
                    kwargs["input_dim"] = param_meta["input_dim"]
                if "output_dim" in param_meta:
                    kwargs["output_dim"] = param_meta["output_dim"]

                new_param = param_class(**kwargs)
            else:
                # Fallback to standard Parameter
                new_param = Parameter(data, requires_grad=False)
                if weight_loader is not None:
                    new_param.weight_loader = weight_loader
        except Exception as e:
            # If reconstruction with original class fails, fallback to Parameter
            logger.warning(f"Failed to rebuild {key} with class {param_class}: {e}, using Parameter")
            new_param = Parameter(data, requires_grad=False)
            if weight_loader is not None:
                new_param.weight_loader = weight_loader

        # Restore MoE-specific attributes (quant_method is required by weight_loader)
        if "quant_method" in param_meta:
            new_param.quant_method = param_meta["quant_method"]

        # Register on module
        module.register_parameter(param_name, new_param)

        return new_param

    def prepare_for_reload(self) -> None:
        """Prepare layers for weight reload by swapping Marlin tensors with HF-shape tensors."""
        for layer_name, swap_info in self._tensor_swap_layers.items():
            module = swap_info["module"]
            hf_meta = swap_info["hf_meta"]
            param_name = swap_info.get("param_name", "weight_scale")

            # Get HF shape from metadata
            hf_shape = hf_meta["shape"]
            hf_dtype = hf_meta["dtype"]
            device = self.device or hf_meta.get("device", "cuda")
            param_class = hf_meta.get("param_class", Parameter)

            # Get saved weight_loader
            weight_loaders = getattr(module, "_weight_loaders", {})
            weight_loader = weight_loaders.get(param_name)

            # Create temporary HF shape tensor
            hf_tensor = torch.empty(hf_shape, dtype=hf_dtype, device=device)

            # Try to use the original parameter class (vLLM special types)
            try:
                if param_class is not Parameter and weight_loader is not None:
                    # vLLM parameter types need specific constructor arguments
                    kwargs = {"data": hf_tensor, "weight_loader": weight_loader}

                    # Add input_dim/output_dim if saved
                    if "input_dim" in hf_meta:
                        kwargs["input_dim"] = hf_meta["input_dim"]
                    if "output_dim" in hf_meta:
                        kwargs["output_dim"] = hf_meta["output_dim"]

                    temp_param = param_class(**kwargs)
                else:
                    temp_param = Parameter(hf_tensor, requires_grad=False)
                    if weight_loader is not None:
                        temp_param.weight_loader = weight_loader
            except Exception as e:
                logger.warning(f"Failed to create temp param with class {param_class}: {e}")
                temp_param = Parameter(hf_tensor, requires_grad=False)
                if weight_loader is not None:
                    temp_param.weight_loader = weight_loader

            if "quant_method" in hf_meta:
                temp_param.quant_method = hf_meta["quant_method"]

            setattr(module, param_name, temp_param)

    def __getitem__(self, key: str) -> Parameter:
        """Get parameter with lazy rebuild support."""
        # Try standard lookup first
        if key in dict.keys(self):
            return super().__getitem__(key)

        # Try lazy rebuild
        param = self._try_lazy_rebuild(key)
        if param is not None:
            self[key] = param
            return param

        raise KeyError(f"Parameter not found: {key}")

    def __contains__(self, key: str) -> bool:
        """Check if parameter exists (with lazy rebuild check)."""
        if super().__contains__(key):
            return True

        # Check if can lazy rebuild
        parts = key.rsplit(".", 1)
        if len(parts) == 2:
            layer_name, param_name = parts
            if layer_name in self._layer_meta_cache:
                meta = self._layer_meta_cache[layer_name]["meta"]
                if param_name in meta:
                    return True

        return False

    def get(self, key: str, default=None):
        """Get parameter with default."""
        try:
            return self[key]
        except KeyError:
            return default


def save_param_meta(layer: torch.nn.Module, param_name: str):
    """Save parameter metadata for lazy rebuild."""
    if not hasattr(layer, "_hf_param_meta"):
        layer._hf_param_meta = {}

    param = getattr(layer, param_name, None)
    if param is None:
        return

    meta = {
        "shape": tuple(param.shape),
        "dtype": param.dtype,
        "device": str(param.device),
        "param_class": type(param),  # Save the actual parameter class
    }

    # Save vLLM-specific attributes needed for reconstruction
    if hasattr(param, "_input_dim"):
        meta["input_dim"] = param._input_dim
    if hasattr(param, "_output_dim"):
        meta["output_dim"] = param._output_dim

    # Save MoE-specific attributes (quant_method is required by weight_loader)
    if hasattr(param, "quant_method"):
        meta["quant_method"] = param.quant_method

    layer._hf_param_meta[param_name] = meta


def get_process_call_count(layer: torch.nn.Module) -> int:
    """Get the number of times process_weights_after_loading has been called."""
    if not hasattr(layer, "_process_weights_call_count"):
        layer._process_weights_call_count = 0
    return layer._process_weights_call_count


def increment_process_call_count(layer: torch.nn.Module):
    """Increment the process_weights_after_loading call count."""
    if not hasattr(layer, "_process_weights_call_count"):
        layer._process_weights_call_count = 0
    layer._process_weights_call_count += 1


# Dense W4A16 Patches
def patched_w4a16_create_weights(
    self,
    layer: torch.nn.Module,
    output_partition_sizes: list[int],
    input_size_per_partition: int,
    params_dtype: torch.dtype,
    weight_loader: Callable,
    **kwargs,
):
    """Patched create_weights for W4A16 Dense layer."""
    from vllm.model_executor.parameter import (
        GroupQuantScaleParameter,
        ModelWeightParameter,
        PerTensorScaleParameter,
    )

    output_size_per_partition = sum(output_partition_sizes)
    layer.logical_widths = output_partition_sizes
    layer.input_size_per_partition = input_size_per_partition
    layer.output_size_per_partition = output_size_per_partition
    layer.params_dtype = params_dtype

    weight = ModelWeightParameter(
        data=torch.empty(
            output_size_per_partition,
            input_size_per_partition // 2,
            dtype=torch.uint8,
        ),
        input_dim=1,
        output_dim=0,
        weight_loader=weight_loader,
    )
    layer.register_parameter("weight_packed", weight)

    weight_global_scale = PerTensorScaleParameter(
        data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
        weight_loader=weight_loader,
    )
    layer.register_parameter("weight_global_scale", weight_global_scale)

    weight_scale = GroupQuantScaleParameter(
        data=torch.empty(
            output_size_per_partition,
            input_size_per_partition // self.group_size,
            dtype=torch.float8_e4m3fn,
        ),
        input_dim=1,
        output_dim=0,
        weight_loader=weight_loader,
    )
    layer.register_parameter("weight_scale", weight_scale)

    if self.has_input_global_scale:
        input_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("input_global_scale", input_global_scale)


def patched_w4a16_process_weights_after_loading(self, layer: torch.nn.Module) -> None:
    """Patched process_weights_after_loading for W4A16 Dense layer."""
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        marlin_make_workspace_new,
        marlin_permute_scales,
        nvfp4_marlin_process_global_scale,
        nvfp4_marlin_process_scales,
    )

    call_count = get_process_call_count(layer)
    is_first_call = call_count == 0
    increment_process_call_count(layer)

    group_size = 16
    part_size_n = layer.output_size_per_partition
    part_size_k = layer.input_size_per_partition
    device = layer.weight_packed.device
    param_dtype = getattr(layer, "params_dtype", torch.float16)

    # Save metadata (first call only)
    if is_first_call:
        save_param_meta(layer, "weight_packed")
        save_param_meta(layer, "weight_global_scale")
        save_param_meta(layer, "weight_scale")
        if not hasattr(layer, "_weight_loaders"):
            layer._weight_loaders = {}
        for pname in ["weight_packed", "weight_global_scale", "weight_scale"]:
            param = getattr(layer, pname, None)
            if param is not None and hasattr(param, "weight_loader"):
                layer._weight_loaders[pname] = param.weight_loader

    # Get HF format data
    weight_packed_hf = layer.weight_packed.data
    weight_global_scale_hf = layer.weight_global_scale.data
    weight_scale_hf = layer.weight_scale.data

    # Create workspace (first call only)
    if is_first_call:
        layer.workspace = marlin_make_workspace_new(device)

    # Convert to Marlin format
    perm = torch.empty(0, dtype=torch.int, device=device)
    qweight = weight_packed_hf.view(torch.int32).T.contiguous()
    marlin_weight = ops.gptq_marlin_repack(
        b_q_weight=qweight,
        perm=perm,
        size_k=part_size_k,
        size_n=part_size_n,
        num_bits=4,
        is_a_8bit=False,
    )

    weight_scale = weight_scale_hf.T.contiguous().to(param_dtype)
    weight_scale_permuted = marlin_permute_scales(
        s=weight_scale,
        size_k=part_size_k,
        size_n=part_size_n,
        group_size=group_size,
        is_a_8bit=False,
    )
    marlin_weight_scale = nvfp4_marlin_process_scales(weight_scale_permuted)

    weight_scale_2_raw = (1.0 / weight_global_scale_hf.max()).to(param_dtype)
    marlin_weight_scale_2 = nvfp4_marlin_process_global_scale(weight_scale_2_raw)

    # Update compute parameters
    if is_first_call:
        layer.weight = Parameter(marlin_weight, requires_grad=False)
        layer.weight_scale = Parameter(marlin_weight_scale, requires_grad=False)
        layer.weight_scale_2 = Parameter(marlin_weight_scale_2, requires_grad=False)
        if not hasattr(layer, "_marlin_tensor_refs"):
            layer._marlin_tensor_refs = {}
        layer._marlin_tensor_refs["weight_scale"] = layer.weight_scale.data
    else:
        layer.weight.data.copy_(marlin_weight)
        layer.weight_scale_2.data.copy_(marlin_weight_scale_2)
        marlin_scale_ref = layer._marlin_tensor_refs.get("weight_scale")
        if marlin_scale_ref is not None:
            marlin_scale_ref.copy_(marlin_weight_scale)
            layer.weight_scale = Parameter(marlin_scale_ref, requires_grad=False)
        else:
            logger.warning("W4A16: _marlin_tensor_refs['weight_scale'] not found")
            layer.weight_scale = Parameter(marlin_weight_scale, requires_grad=False)

    # Delete HF parameters
    if hasattr(layer, "weight_packed"):
        delattr(layer, "weight_packed")
    if hasattr(layer, "weight_global_scale"):
        delattr(layer, "weight_global_scale")


def patched_w4a16_apply_weights(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply weights using compute parameters."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        apply_fp4_marlin_linear,
    )

    return apply_fp4_marlin_linear(
        input=x,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        weight_scale_2=layer.weight_scale_2,
        workspace=layer.workspace,
        size_n=layer.output_size_per_partition,
        size_k=layer.input_size_per_partition,
        bias=bias,
    )


def patched_w4a4_create_weights(
    self,
    layer: torch.nn.Module,
    output_partition_sizes: list[int],
    input_size_per_partition: int,
    params_dtype: torch.dtype,
    weight_loader: Callable,
    **kwargs,
):
    """Patched create_weights for W4A4 Dense layer."""
    from vllm.model_executor.parameter import (
        GroupQuantScaleParameter,
        ModelWeightParameter,
        PerTensorScaleParameter,
    )

    output_size_per_partition = sum(output_partition_sizes)
    layer.logical_widths = output_partition_sizes
    layer.input_size_per_partition = input_size_per_partition
    layer.output_size_per_partition = output_size_per_partition
    layer.params_dtype = params_dtype

    weight = ModelWeightParameter(
        data=torch.empty(
            output_size_per_partition,
            input_size_per_partition // 2,
            dtype=torch.uint8,
        ),
        input_dim=1,
        output_dim=0,
        weight_loader=weight_loader,
    )
    layer.register_parameter("weight_packed_hf", weight)

    weight_global_scale = PerTensorScaleParameter(
        data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
        weight_loader=weight_loader,
    )
    layer.register_parameter("weight_global_scale_hf", weight_global_scale)

    weight_scale = GroupQuantScaleParameter(
        data=torch.empty(
            output_size_per_partition,
            input_size_per_partition // self.group_size,
            dtype=torch.float8_e4m3fn,
        ),
        input_dim=1,
        output_dim=0,
        weight_loader=weight_loader,
    )
    layer.register_parameter("weight_scale_hf", weight_scale)

    input_global_scale = PerTensorScaleParameter(
        data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
        weight_loader=weight_loader,
    )
    layer.register_parameter("input_global_scale_hf", input_global_scale)


def patched_w4a4_process_weights_after_loading(self, layer: torch.nn.Module) -> None:
    """Patched process_weights_after_loading for W4A4 Dense layer."""
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        swizzle_blockscale,
    )

    call_count = get_process_call_count(layer)
    is_first_call = call_count == 0
    increment_process_call_count(layer)

    # Save metadata (first call only)
    if is_first_call:
        save_param_meta(layer, "weight_packed_hf")
        save_param_meta(layer, "weight_scale_hf")
        save_param_meta(layer, "weight_global_scale_hf")
        save_param_meta(layer, "input_global_scale_hf")

    # Get HF format data
    weight_packed_hf = layer.weight_packed_hf.data
    weight_scale_hf = layer.weight_scale_hf.data
    weight_global_scale_hf = layer.weight_global_scale_hf.data
    input_global_scale_hf = layer.input_global_scale_hf.data

    # Compute transformed values
    global_input_scale = input_global_scale_hf.max().to(torch.float32)
    global_weight_scale = weight_global_scale_hf.max().to(torch.float32)

    if self.backend == "flashinfer-trtllm":
        from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

        epilogue_tile_m = 128
        weight_shuffled = shuffle_matrix_a(weight_packed_hf.view(torch.uint8), epilogue_tile_m)
        weight_scale_shuffled = (
            shuffle_matrix_sf_a(weight_scale_hf.view(torch.uint8), epilogue_tile_m)
            .reshape(weight_scale_hf.shape)
            .view(torch.float8_e4m3fn)
        )
        processed_weight = weight_shuffled
        processed_weight_scale = weight_scale_shuffled
    else:
        # cutlass or flashinfer-cutlass
        processed_weight_scale = swizzle_blockscale(weight_scale_hf)
        if self.backend == "fbgemm":
            processed_weight_scale = processed_weight_scale.view(-1).view(torch.uint8)
        processed_weight = weight_packed_hf

    alpha = 1.0 / (global_input_scale * global_weight_scale)

    # Update compute parameters
    if is_first_call:
        layer.weight_packed = Parameter(processed_weight, requires_grad=False)
        layer.weight_scale = Parameter(processed_weight_scale, requires_grad=False)
        layer.input_global_scale = Parameter(global_input_scale, requires_grad=False)
        layer.weight_global_scale = Parameter(global_weight_scale, requires_grad=False)
        layer.alpha = Parameter(alpha, requires_grad=False)
    else:
        layer.weight_packed.data.copy_(processed_weight)
        layer.weight_scale.data.copy_(processed_weight_scale)
        layer.input_global_scale.data.copy_(global_input_scale)
        layer.weight_global_scale.data.copy_(global_weight_scale)
        layer.alpha.data.copy_(alpha)

    # Delete HF parameters
    delattr(layer, "weight_packed_hf")
    delattr(layer, "weight_scale_hf")
    delattr(layer, "weight_global_scale_hf")
    delattr(layer, "input_global_scale_hf")


def patched_w4a4_apply_weights(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply weights using compute parameters."""
    import vllm.envs as envs
    from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
        run_nvfp4_emulations,
    )
    from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm

    if envs.VLLM_USE_NVFP4_CT_EMULATIONS:
        out = run_nvfp4_emulations(
            x=x,
            input_global_scale=layer.input_global_scale,
            weight=layer.weight_packed,
            weight_scale_swizzled=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale,
        )
        if bias is not None:
            out = out + bias
        return out

    output_dtype = x.dtype
    output_shape = [*x.shape[:-1], layer.weight_packed.shape[0]]

    # Quantize input
    x_fp4, x_blockscale = scaled_fp4_quant(
        x,
        layer.input_global_scale,
        is_sf_swizzled_layout=True,
        backend=self.backend,
    )

    # Matrix multiply
    mm_args = (
        x_fp4,
        layer.weight_packed,
        x_blockscale,
        layer.weight_scale,
        layer.alpha,
        output_dtype,
    )

    if self.backend.startswith("flashinfer-"):
        backend_name = self.backend[len("flashinfer-") :]
        out = flashinfer_scaled_fp4_mm(*mm_args, backend=backend_name)
    elif self.backend == "fbgemm":
        out = torch.ops.fbgemm.f4f4bf16(
            x_fp4,
            layer.weight_packed,
            x_blockscale.view(-1).view(torch.uint8),
            layer.weight_scale,
            layer.alpha,
            use_mx=False,
        ).to(output_dtype)
    else:
        out = cutlass_scaled_fp4_mm(*mm_args)

    if bias is not None:
        out = out + bias
    return out.view(*output_shape)


def patched_nvfp4_moe_create_weights(
    self,
    layer: torch.nn.Module,
    num_experts: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
    params_dtype: torch.dtype,
    **extra_weight_attrs,
):
    """Patched create_weights for NVFP4 MoE layer."""
    from vllm.model_executor.layers.fused_moe import FusedMoeWeightScaleSupported
    from vllm.model_executor.utils import set_weight_attrs

    layer.num_experts = num_experts
    layer.params_dtype = params_dtype
    layer.hidden_size = hidden_size
    layer.intermediate_size_per_partition = intermediate_size_per_partition
    w13_num_shards = 2 if self.moe.is_act_and_mul else 1

    w13_weight = torch.nn.Parameter(
        torch.empty(
            num_experts,
            w13_num_shards * intermediate_size_per_partition,
            hidden_size // 2,
            requires_grad=False,
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    layer.register_parameter("w13_weight_packed", w13_weight)
    set_weight_attrs(w13_weight, extra_weight_attrs)

    w2_weight = torch.nn.Parameter(
        torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition // 2,
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    layer.register_parameter("w2_weight_packed", w2_weight)
    set_weight_attrs(w2_weight, extra_weight_attrs)

    w13_weight_scale = torch.nn.Parameter(
        torch.empty(
            num_experts,
            w13_num_shards * intermediate_size_per_partition,
            hidden_size // self.group_size,
            dtype=torch.float8_e4m3fn,
        ),
        requires_grad=False,
    )
    layer.register_parameter("w13_weight_scale", w13_weight_scale)
    extra_weight_attrs_scale = dict(extra_weight_attrs)
    extra_weight_attrs_scale["quant_method"] = FusedMoeWeightScaleSupported.GROUP.value
    set_weight_attrs(w13_weight_scale, extra_weight_attrs_scale)

    w2_weight_scale = torch.nn.Parameter(
        torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition // self.group_size,
            dtype=torch.float8_e4m3fn,
        ),
        requires_grad=False,
    )
    layer.register_parameter("w2_weight_scale", w2_weight_scale)
    set_weight_attrs(w2_weight_scale, extra_weight_attrs_scale)

    w13_weight_scale_2 = torch.nn.Parameter(
        torch.empty(num_experts, w13_num_shards, dtype=torch.float32),
        requires_grad=False,
    )
    layer.register_parameter("w13_weight_global_scale", w13_weight_scale_2)
    extra_weight_attrs_tensor = dict(extra_weight_attrs)
    extra_weight_attrs_tensor["quant_method"] = FusedMoeWeightScaleSupported.TENSOR.value
    set_weight_attrs(w13_weight_scale_2, extra_weight_attrs_tensor)

    w2_weight_scale_2 = torch.nn.Parameter(
        torch.empty(num_experts, dtype=torch.float32),
        requires_grad=False,
    )
    layer.register_parameter("w2_weight_global_scale", w2_weight_scale_2)
    set_weight_attrs(w2_weight_scale_2, extra_weight_attrs_tensor)

    w13_input_scale = torch.nn.Parameter(
        torch.empty(num_experts, w13_num_shards, dtype=torch.float32),
        requires_grad=False,
    )
    layer.register_parameter("w13_input_global_scale", w13_input_scale)
    set_weight_attrs(w13_input_scale, extra_weight_attrs_tensor)

    w2_input_scale = torch.nn.Parameter(
        torch.empty(num_experts, dtype=torch.float32),
        requires_grad=False,
    )
    layer.register_parameter("w2_input_global_scale", w2_input_scale)
    set_weight_attrs(w2_input_scale, extra_weight_attrs_tensor)


def patched_nvfp4_moe_process_weights_after_loading(self, layer: torch.nn.Module) -> None:
    """Patched process_weights_after_loading for NVFP4 MoE layer."""
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend

    call_count = get_process_call_count(layer)
    is_first_call = call_count == 0
    increment_process_call_count(layer)

    # Save metadata (first call only)
    if is_first_call:
        save_param_meta(layer, "w13_weight_packed")
        save_param_meta(layer, "w2_weight_packed")
        save_param_meta(layer, "w13_weight_scale")
        save_param_meta(layer, "w2_weight_scale")
        if not hasattr(layer, "_weight_loaders"):
            layer._weight_loaders = {}
        for pname in ["w13_weight_packed", "w2_weight_packed", "w13_weight_scale", "w2_weight_scale"]:
            param = getattr(layer, pname, None)
            if param is not None and hasattr(param, "weight_loader"):
                layer._weight_loaders[pname] = param.weight_loader

    is_marlin = self.nvfp4_backend == NvFp4MoeBackend.MARLIN
    if is_marlin:
        _process_nvfp4_moe_marlin(self, layer, is_first_call)
    else:
        _process_nvfp4_moe_flashinfer_cutlass(self, layer, is_first_call)

    # Delete HF parameters
    if hasattr(layer, "w13_weight_packed"):
        delattr(layer, "w13_weight_packed")
    if hasattr(layer, "w2_weight_packed"):
        delattr(layer, "w2_weight_packed")


def _process_nvfp4_moe_marlin(self, layer: torch.nn.Module, is_first_call: bool) -> None:
    """Process MoE layer with MARLIN backend (W4A16)."""
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import make_nvfp4_moe_kernel
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        marlin_make_workspace_new,
        marlin_permute_scales,
        nvfp4_marlin_process_global_scale,
        nvfp4_marlin_process_scales,
    )

    group_size = 16
    e = layer.num_experts
    k = layer.hidden_size
    n = layer.intermediate_size_per_partition
    device = layer.w13_weight_packed.device
    param_dtype = layer.params_dtype
    w13_num_shards = 2 if self.moe.is_act_and_mul else 1

    # Create workspace
    if is_first_call:
        layer.workspace = marlin_make_workspace_new(device, 4)

    perm = torch.empty(0, dtype=torch.int, device=device)

    w13_packed = layer.w13_weight_packed.data
    w2_packed = layer.w2_weight_packed.data
    w13_scale_hf = layer.w13_weight_scale.data
    w2_scale_hf = layer.w2_weight_scale.data

    if self.moe.is_act_and_mul and not torch.allclose(
        layer.w13_weight_global_scale[:, 0], layer.w13_weight_global_scale[:, 1]
    ):
        logger.warning("w1_weight_global_scale must match w3_weight_global_scale. Accuracy may be affected.")

    # Process w13_weight
    w13_tensor_list = []
    size_n_w13, size_k_w13 = n * w13_num_shards, k
    for i in range(e):
        qweight = w13_packed[i].view(torch.int32).T.contiguous()
        marlin_qweight = ops.gptq_marlin_repack(
            b_q_weight=qweight,
            perm=perm,
            size_k=size_k_w13,
            size_n=size_n_w13,
            num_bits=4,
            is_a_8bit=False,
        )
        w13_tensor_list.append(marlin_qweight)
    w13_weight_marlin = torch.cat([x.unsqueeze(0) for x in w13_tensor_list], 0)

    # Process w2_weight
    w2_tensor_list = []
    size_n_w2, size_k_w2 = k, n
    for i in range(e):
        qweight = w2_packed[i].view(torch.int32).T.contiguous()
        marlin_qweight = ops.gptq_marlin_repack(
            b_q_weight=qweight,
            perm=perm,
            size_k=size_k_w2,
            size_n=size_n_w2,
            num_bits=4,
            is_a_8bit=False,
        )
        w2_tensor_list.append(marlin_qweight)
    w2_weight_marlin = torch.cat([x.unsqueeze(0) for x in w2_tensor_list], 0)

    # Process w13_weight_scale
    w13_scale_list = []
    scales = w13_scale_hf.to(param_dtype)
    for i in range(e):
        scale = scales[i].T
        marlin_scales = marlin_permute_scales(
            s=scale,
            size_k=size_k_w13,
            size_n=size_n_w13,
            group_size=group_size,
            is_a_8bit=False,
        )
        marlin_scales = nvfp4_marlin_process_scales(marlin_scales)
        w13_scale_list.append(marlin_scales)
    w13_weight_scale_marlin = torch.cat([x.unsqueeze(0) for x in w13_scale_list], 0)

    # Process w2_weight_scale
    w2_scale_list = []
    scales = w2_scale_hf.to(param_dtype)
    for i in range(e):
        scale = scales[i].T
        marlin_scales = marlin_permute_scales(
            s=scale,
            size_k=size_k_w2,
            size_n=size_n_w2,
            group_size=group_size,
            is_a_8bit=False,
        )
        marlin_scales = nvfp4_marlin_process_scales(marlin_scales)
        w2_scale_list.append(marlin_scales)
    w2_weight_scale_marlin = torch.cat([x.unsqueeze(0) for x in w2_scale_list], 0)

    # Process global scales
    w13_scale_2 = 1.0 / layer.w13_weight_global_scale[:, 0]
    w2_scale_2 = 1.0 / layer.w2_weight_global_scale.data
    w13_scale_2_processed = nvfp4_marlin_process_global_scale(w13_scale_2.to(param_dtype))
    w2_scale_2_processed = nvfp4_marlin_process_global_scale(w2_scale_2.to(param_dtype))

    # Update parameters
    if is_first_call:
        layer.w13_weight = Parameter(w13_weight_marlin, requires_grad=False)
        layer.w2_weight = Parameter(w2_weight_marlin, requires_grad=False)
        layer.w13_weight_scale = Parameter(w13_weight_scale_marlin, requires_grad=False)
        layer.w2_weight_scale = Parameter(w2_weight_scale_marlin, requires_grad=False)
        layer.w13_weight_scale_2 = Parameter(w13_scale_2_processed, requires_grad=False)
        layer.w2_weight_scale_2 = Parameter(w2_scale_2_processed, requires_grad=False)
        if not hasattr(layer, "_marlin_tensor_refs"):
            layer._marlin_tensor_refs = {}
        layer._marlin_tensor_refs["w13_weight_scale"] = layer.w13_weight_scale.data
        layer._marlin_tensor_refs["w2_weight_scale"] = layer.w2_weight_scale.data
    else:
        layer.w13_weight.data.copy_(w13_weight_marlin)
        layer.w2_weight.data.copy_(w2_weight_marlin)
        layer.w13_weight_scale_2.data.copy_(w13_scale_2_processed)
        layer.w2_weight_scale_2.data.copy_(w2_scale_2_processed)
        w13_marlin_ref = layer._marlin_tensor_refs.get("w13_weight_scale")
        w2_marlin_ref = layer._marlin_tensor_refs.get("w2_weight_scale")
        if w13_marlin_ref is not None:
            w13_marlin_ref.copy_(w13_weight_scale_marlin)
            layer.w13_weight_scale = Parameter(w13_marlin_ref, requires_grad=False)
        else:
            logger.warning("MoE: _marlin_tensor_refs['w13_weight_scale'] not found")
            layer.w13_weight_scale.data.copy_(w13_weight_scale_marlin)
        if w2_marlin_ref is not None:
            w2_marlin_ref.copy_(w2_weight_scale_marlin)
            layer.w2_weight_scale = Parameter(w2_marlin_ref, requires_grad=False)
        else:
            logger.warning("MoE: _marlin_tensor_refs['w2_weight_scale'] not found")
            layer.w2_weight_scale.data.copy_(w2_weight_scale_marlin)

    layer.w13_input_scale = None
    layer.w2_input_scale = None

    # Initialize kernel
    self.moe_quant_config = self.get_fused_moe_quant_config(layer)
    if self.moe_quant_config is not None and (
        (not self.moe.moe_parallel_config.use_all2all_kernels) or self.moe.moe_parallel_config.use_naive_all2all_kernels
    ):
        self.kernel = make_nvfp4_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
        )


def _process_nvfp4_moe_flashinfer_cutlass(self, layer: torch.nn.Module, is_first_call: bool) -> None:
    """Process MoE layer with FlashInfer/CUTLASS backend (W4A4)."""
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        convert_to_nvfp4_moe_kernel_format,
        make_nvfp4_moe_kernel,
    )
    from vllm.model_executor.utils import replace_parameter

    w13_packed = layer.w13_weight_packed.data
    w2_packed = layer.w2_weight_packed.data
    w13_scale_hf = layer.w13_weight_scale.data
    w2_scale_hf = layer.w2_weight_scale.data

    if self.moe.is_act_and_mul and not torch.allclose(
        layer.w13_weight_global_scale[:, 0], layer.w13_weight_global_scale[:, 1]
    ):
        logger.warning("w1_weight_global_scale must match w3_weight_global_scale. Accuracy may be affected.")
    w13_weight_global_scale = layer.w13_weight_global_scale[:, 0].contiguous()

    w13_temp = Parameter(w13_packed.clone(), requires_grad=False)
    w2_temp = Parameter(w2_packed.clone(), requires_grad=False)

    if is_first_call:
        layer.w13_weight = w13_temp
        layer.w2_weight = w2_temp

    (
        w13,
        w13_scale,
        w13_scale_2,
        a13_scale,
        w2,
        w2_scale,
        w2_scale_2,
        a2_scale,
    ) = convert_to_nvfp4_moe_kernel_format(
        nvfp4_backend=self.nvfp4_backend,
        layer=layer,
        w13=w13_temp,
        w13_scale=w13_scale_hf,
        w13_scale_2=(1.0 / w13_weight_global_scale),
        a13_scale=(1.0 / layer.w13_input_global_scale),
        w2=w2_temp,
        w2_scale=w2_scale_hf,
        w2_scale_2=(1.0 / layer.w2_weight_global_scale),
        a2_scale=(1.0 / layer.w2_input_global_scale),
        is_act_and_mul=self.moe.is_act_and_mul,
    )

    # Update parameters
    if is_first_call:
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)
        layer.w13_weight_scale = Parameter(w13_scale, requires_grad=False)
        layer.w2_weight_scale = Parameter(w2_scale, requires_grad=False)
        if not hasattr(layer, "_marlin_tensor_refs"):
            layer._marlin_tensor_refs = {}
        layer._marlin_tensor_refs["w13_weight_scale"] = layer.w13_weight_scale.data
        layer._marlin_tensor_refs["w2_weight_scale"] = layer.w2_weight_scale.data
    else:
        layer.w13_weight.data.copy_(w13.data)
        layer.w2_weight.data.copy_(w2.data)
        w13_scale_ref = layer._marlin_tensor_refs.get("w13_weight_scale")
        w2_scale_ref = layer._marlin_tensor_refs.get("w2_weight_scale")
        if w13_scale_ref is not None:
            w13_scale_ref.copy_(w13_scale)
            layer.w13_weight_scale = Parameter(w13_scale_ref, requires_grad=False)
        else:
            logger.warning("MoE W4A4: _marlin_tensor_refs['w13_weight_scale'] not found")
            layer.w13_weight_scale.data.copy_(w13_scale)
        if w2_scale_ref is not None:
            w2_scale_ref.copy_(w2_scale)
            layer.w2_weight_scale = Parameter(w2_scale_ref, requires_grad=False)
        else:
            logger.warning("MoE W4A4: _marlin_tensor_refs['w2_weight_scale'] not found")
            layer.w2_weight_scale.data.copy_(w2_scale)

    layer.w13_weight_scale_2 = w13_scale_2
    layer.w2_weight_scale_2 = w2_scale_2
    layer.w13_input_scale = a13_scale
    layer.w2_input_scale = a2_scale

    # Initialize kernel
    self.moe_quant_config = self.get_fused_moe_quant_config(layer)
    if self.moe_quant_config is not None and (
        (not self.moe.moe_parallel_config.use_all2all_kernels) or self.moe.moe_parallel_config.use_naive_all2all_kernels
    ):
        self.kernel = make_nvfp4_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
        )


_applied_patches = []


def apply_qat_patches():
    """Apply NVFP4 patches to support dynamic weight updates. Call before model loading."""
    global _applied_patches

    if _applied_patches:
        logger.warning("QAT patches already applied, skipping")
        return _applied_patches

    logger.info("Applying NVFP4 patches for dynamic weight loading...")

    # Dense W4A16 patches
    patch1 = patch(
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
        "compressed_tensors_w4a16_nvfp4.CompressedTensorsW4A16Fp4.create_weights",
        patched_w4a16_create_weights,
    )
    _applied_patches.append(patch1)
    patch1.start()

    patch2 = patch(
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
        "compressed_tensors_w4a16_nvfp4.CompressedTensorsW4A16Fp4.process_weights_after_loading",
        patched_w4a16_process_weights_after_loading,
    )
    _applied_patches.append(patch2)
    patch2.start()

    patch3 = patch(
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
        "compressed_tensors_w4a16_nvfp4.CompressedTensorsW4A16Fp4.apply_weights",
        patched_w4a16_apply_weights,
    )
    _applied_patches.append(patch3)
    patch3.start()

    # Dense W4A4 patches
    patch4 = patch(
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
        "compressed_tensors_w4a4_nvfp4.CompressedTensorsW4A4Fp4.create_weights",
        patched_w4a4_create_weights,
    )
    _applied_patches.append(patch4)
    patch4.start()

    patch5 = patch(
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
        "compressed_tensors_w4a4_nvfp4.CompressedTensorsW4A4Fp4.process_weights_after_loading",
        patched_w4a4_process_weights_after_loading,
    )
    _applied_patches.append(patch5)
    patch5.start()

    patch6 = patch(
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
        "compressed_tensors_w4a4_nvfp4.CompressedTensorsW4A4Fp4.apply_weights",
        patched_w4a4_apply_weights,
    )
    _applied_patches.append(patch6)
    patch6.start()

    # MoE NVFP4 patches
    patch7 = patch(
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe.CompressedTensorsW4A4Nvfp4MoEMethod.create_weights",
        patched_nvfp4_moe_create_weights,
    )
    _applied_patches.append(patch7)
    patch7.start()

    patch8 = patch(
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe.CompressedTensorsW4A4Nvfp4MoEMethod.process_weights_after_loading",
        patched_nvfp4_moe_process_weights_after_loading,
    )
    _applied_patches.append(patch8)
    patch8.start()

    logger.info(f"Applied {len(_applied_patches)} NVFP4 patches for dynamic weight loading")
    return _applied_patches


def remove_qat_patches(patches=None):
    """Remove applied patches."""
    global _applied_patches

    patches_to_remove = patches or _applied_patches
    for p in patches_to_remove:
        p.stop()

    if patches is None:
        _applied_patches = []

    logger.info("Removed NVFP4 patches")


def prepare_qat_for_load_weights(model, device=None):
    """
    Prepare QAT model for weight loading. Call ONCE before multi-bucket weight loading.

    Args:
        model: vLLM model
        device: Device for created parameters
    """
    inner_model = model
    if hasattr(model, "model"):
        inner_model = model.model

    lazy_params = LazyParamsDict(inner_model, device=device)

    # Tensor swap: replace Marlin-format weight_scale with HF-format tensor
    lazy_params.prepare_for_reload()
    logger.info(f"[prepare_qat] Tensor swap prepared for {len(lazy_params._tensor_swap_layers)} layers")

    # Rebuild deleted parameters
    rebuilt_count = 0
    for layer_name, cache_entry in lazy_params._layer_meta_cache.items():
        meta = cache_entry["meta"]
        module = cache_entry["module"]
        for param_name in meta.keys():
            if hasattr(module, param_name) and getattr(module, param_name) is not None:
                continue
            full_name = f"{layer_name}.{param_name}" if layer_name else param_name
            try:
                lazy_params[full_name]
                rebuilt_count += 1
            except KeyError:
                pass

    logger.info(f"[prepare_qat] Rebuilt {rebuilt_count} parameters")
    inner_model._lazy_params_for_restore = lazy_params
    return lazy_params


def manual_process_weights_after_loading(model):
    """Trigger weight post-processing for all quantized layers after load_weights."""
    dense_count = 0
    moe_count = 0

    actual_model = model
    if hasattr(model, "model"):
        actual_model = model.model

    for module in actual_model.modules():
        if hasattr(module, "scheme"):
            module.scheme.process_weights_after_loading(module)
            dense_count += 1

        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None and not hasattr(module, "scheme"):
            if hasattr(quant_method, "process_weights_after_loading"):
                # Skip KV cache quantization methods
                if "KVCache" in quant_method.__class__.__name__:
                    continue
                quant_method.process_weights_after_loading(module)
                moe_count += 1

    logger.debug(f"Processed {dense_count} dense layers, {moe_count} MoE layers")
    return dense_count + moe_count


__all__ = [
    "LazyParamsDict",
    "apply_qat_patches",
    "remove_qat_patches",
    "prepare_qat_for_load_weights",
    "manual_process_weights_after_loading",
]
