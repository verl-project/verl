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

Enables dynamic weight reloading for NVFP4 quantized models in vLLM.

Supported schemes:
- Dense: W4A16-FP4, W4A4-FP4
- MoE: NVFP4-MoE
"""

import logging
import os
from typing import Optional
from unittest.mock import patch

import torch
from torch.nn import Parameter

from verl.utils.device import get_device_name

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ParamMetaDict(dict):
    """
    Dict-like class for parameter management with metadata-based rebuild and tensor swap.

    Supports:
    - Rebuild of deleted parameters from saved metadata
    - Tensor Swap for parameters with shape changes (address stability for CUDA Graph)
    """

    def __init__(self, model: torch.nn.Module, device: Optional[torch.device] = None):
        """
        Initialize ParamMetaDict from a model.

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
        """Build layer metadata cache for rebuild and tensor swap."""
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

    def _try_rebuild(self, key: str) -> Optional[Parameter]:
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
        new_param = _create_param_from_meta(module, param_name, meta[param_name], self.device)
        module.register_parameter(param_name, new_param)
        return new_param

    def prepare_for_reload(self) -> None:
        """Replace Marlin-format tensors with HF-shape tensors for reload."""
        for layer_name, swap_info in self._tensor_swap_layers.items():
            module = swap_info["module"]
            param_name = swap_info.get("param_name", "weight_scale")
            hf_meta = swap_info["hf_meta"]
            if hasattr(module, param_name):
                new_param = _create_param_from_meta(module, param_name, hf_meta, self.device)
                setattr(module, param_name, new_param)

    def __getitem__(self, key: str) -> Parameter:
        """Get parameter with rebuild support."""
        # Try standard lookup first
        if key in dict.keys(self):
            return super().__getitem__(key)

        # Try rebuild from metadata
        param = self._try_rebuild(key)
        if param is not None:
            self[key] = param
            return param

        raise KeyError(f"Parameter not found: {key}")

    def __contains__(self, key: str) -> bool:
        """Check if parameter exists (with rebuild check)."""
        if super().__contains__(key):
            return True

        # Check if can rebuild from metadata
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


def _create_param_from_meta(
    module: torch.nn.Module,
    param_name: str,
    meta: dict,
    device: Optional[torch.device] = None,
) -> Parameter:
    """Create a Parameter from saved metadata. Used by rebuild and tensor swap."""
    shape = meta["shape"]
    dtype = meta["dtype"]
    dev = device or meta.get("device", get_device_name())
    param_class = meta.get("param_class", Parameter)

    weight_loaders = getattr(module, "_weight_loaders", {})
    weight_loader = weight_loaders.get(param_name)

    data = torch.empty(shape, dtype=dtype, device=dev)

    try:
        if param_class is not Parameter and weight_loader is not None:
            kwargs = {"data": data, "weight_loader": weight_loader}
            if "input_dim" in meta:
                kwargs["input_dim"] = meta["input_dim"]
            if "output_dim" in meta:
                kwargs["output_dim"] = meta["output_dim"]
            new_param = param_class(**kwargs)
        else:
            new_param = Parameter(data, requires_grad=False)
            if weight_loader is not None:
                new_param.weight_loader = weight_loader
    except Exception as e:
        logger.warning(f"Failed to create param {param_name} with class {param_class}: {e}, using Parameter")
        new_param = Parameter(data, requires_grad=False)
        if weight_loader is not None:
            new_param.weight_loader = weight_loader

    if "quant_method" in meta:
        new_param.quant_method = meta["quant_method"]

    return new_param


def save_param_meta(layer: torch.nn.Module, param_name: str):
    """Save parameter metadata for rebuild."""
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


def _check_first_call(layer: torch.nn.Module) -> bool:
    """Check if this is the first process_weights call, and increment counter."""
    count = getattr(layer, "_process_weights_call_count", 0)
    layer._process_weights_call_count = count + 1
    return count == 0


# Dense W4A16 Patches
def patched_w4a16_process_weights_after_loading(self, layer: torch.nn.Module) -> None:
    """Patched process_weights_after_loading for W4A16 Dense layer."""
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        marlin_make_workspace_new,
        marlin_permute_scales,
        nvfp4_marlin_process_global_scale,
        nvfp4_marlin_process_scales,
    )

    is_first_call = _check_first_call(layer)

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


def patched_w4a4_process_weights_after_loading(self, layer: torch.nn.Module) -> None:
    """Patched process_weights_after_loading for W4A4 Dense (all backends)."""
    from vllm.model_executor.layers.quantization.utils.quant_utils import swizzle_blockscale

    is_first_call = _check_first_call(layer)

    _W4A4_HF_PARAMS = ["weight_packed", "weight_scale", "weight_global_scale", "input_global_scale"]

    if is_first_call:
        for pname in _W4A4_HF_PARAMS:
            save_param_meta(layer, pname)
        if not hasattr(layer, "_weight_loaders"):
            layer._weight_loaders = {}
        for pname in _W4A4_HF_PARAMS:
            param = getattr(layer, pname, None)
            if param is not None and hasattr(param, "weight_loader"):
                layer._weight_loaders[pname] = param.weight_loader

    weight_packed_data = layer.weight_packed.data
    weight_scale_data = layer.weight_scale.data
    input_global_scale_data = layer.input_global_scale.data
    weight_global_scale_data = layer.weight_global_scale.data

    global_input_scale = input_global_scale_data.max().to(torch.float32)
    global_weight_scale = weight_global_scale_data.max().to(torch.float32)

    if self.backend == "flashinfer-trtllm":
        from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

        epilogue_tile_m = 128
        processed_weight = shuffle_matrix_a(weight_packed_data.view(torch.uint8), epilogue_tile_m)
        processed_weight_scale = (
            shuffle_matrix_sf_a(weight_scale_data.view(torch.uint8), epilogue_tile_m)
            .reshape(weight_scale_data.shape)
            .view(torch.float8_e4m3fn)
        )
    elif self.backend == "fbgemm":
        processed_weight_scale = swizzle_blockscale(weight_scale_data).view(-1).view(torch.uint8)
        processed_weight = weight_packed_data
    else:
        # cutlass / flashinfer-cutlass
        processed_weight_scale = swizzle_blockscale(weight_scale_data)
        processed_weight = weight_packed_data

    alpha = 1.0 / (global_input_scale * global_weight_scale)

    if is_first_call:
        layer.weight_packed = Parameter(processed_weight, requires_grad=False)
        layer.weight_scale = Parameter(processed_weight_scale, requires_grad=False)
        layer.input_global_scale = Parameter(global_input_scale, requires_grad=False)
        layer.weight_global_scale = Parameter(global_weight_scale, requires_grad=False)
        layer.alpha = Parameter(alpha, requires_grad=False)

        if not hasattr(layer, "_marlin_tensor_refs"):
            layer._marlin_tensor_refs = {}
        layer._marlin_tensor_refs["weight_packed"] = layer.weight_packed.data
        layer._marlin_tensor_refs["weight_scale"] = layer.weight_scale.data
        layer._marlin_tensor_refs["input_global_scale"] = layer.input_global_scale.data
        layer._marlin_tensor_refs["weight_global_scale"] = layer.weight_global_scale.data
        layer._marlin_tensor_refs["alpha"] = layer.alpha.data
    else:
        refs = layer._marlin_tensor_refs
        for ref_name, new_data in [
            ("weight_packed", processed_weight),
            ("weight_scale", processed_weight_scale),
            ("input_global_scale", global_input_scale),
            ("weight_global_scale", global_weight_scale),
            ("alpha", alpha),
        ]:
            ref = refs.get(ref_name)
            if ref is not None:
                ref.copy_(new_data)
                setattr(layer, ref_name, Parameter(ref, requires_grad=False))
            else:
                logger.warning(f"W4A4: _marlin_tensor_refs['{ref_name}'] not found, creating new Parameter")
                setattr(
                    layer,
                    ref_name,
                    Parameter(
                        new_data.clone() if isinstance(new_data, torch.Tensor) else torch.tensor(new_data),
                        requires_grad=False,
                    ),
                )


def _marlin_repack_experts(packed, perm, size_k, size_n, num_experts):
    """Repack weight for each expert into Marlin format and stack."""
    import vllm._custom_ops as ops

    result = []
    for i in range(num_experts):
        qweight = packed[i].view(torch.int32).T.contiguous()
        result.append(
            ops.gptq_marlin_repack(
                b_q_weight=qweight,
                perm=perm,
                size_k=size_k,
                size_n=size_n,
                num_bits=4,
                is_a_8bit=False,
            )
        )
    return torch.stack(result)


def _marlin_process_scales_experts(scale_hf, param_dtype, size_k, size_n, group_size, num_experts):
    """Process scales for each expert into Marlin format and stack."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        marlin_permute_scales,
        nvfp4_marlin_process_scales,
    )

    result = []
    scales = scale_hf.to(param_dtype)
    for i in range(num_experts):
        s = marlin_permute_scales(
            s=scales[i].T,
            size_k=size_k,
            size_n=size_n,
            group_size=group_size,
            is_a_8bit=False,
        )
        result.append(nvfp4_marlin_process_scales(s))
    return torch.stack(result)


def _process_nvfp4_moe_marlin(self, layer: torch.nn.Module, is_first_call: bool) -> None:
    """Process MoE layer with MARLIN backend (W4A16)."""
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import make_nvfp4_moe_kernel
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        marlin_make_workspace_new,
        nvfp4_marlin_process_global_scale,
    )

    group_size = 16
    e = layer.num_experts
    k = layer.hidden_size
    n = layer.intermediate_size_per_partition
    device = layer.w13_weight_packed.device
    param_dtype = layer.params_dtype
    w13_num_shards = 2 if self.moe.is_act_and_mul else 1

    if is_first_call:
        layer.workspace = marlin_make_workspace_new(device, 4)

    perm = torch.empty(0, dtype=torch.int, device=device)

    if self.moe.is_act_and_mul and not torch.allclose(
        layer.w13_weight_global_scale[:, 0], layer.w13_weight_global_scale[:, 1]
    ):
        logger.warning("w1_weight_global_scale must match w3_weight_global_scale. Accuracy may be affected.")

    size_n_w13, size_k_w13 = n * w13_num_shards, k
    size_n_w2, size_k_w2 = k, n

    w13_weight_marlin = _marlin_repack_experts(layer.w13_weight_packed.data, perm, size_k_w13, size_n_w13, e)
    w2_weight_marlin = _marlin_repack_experts(layer.w2_weight_packed.data, perm, size_k_w2, size_n_w2, e)
    w13_weight_scale_marlin = _marlin_process_scales_experts(
        layer.w13_weight_scale.data, param_dtype, size_k_w13, size_n_w13, group_size, e
    )
    w2_weight_scale_marlin = _marlin_process_scales_experts(
        layer.w2_weight_scale.data, param_dtype, size_k_w2, size_n_w2, group_size, e
    )

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


# MoE NVFP4 Patches (entry points)
def patched_nvfp4_moe_process_weights_after_loading(self, layer: torch.nn.Module) -> None:
    """Patched process_weights_after_loading for NVFP4 MoE layer."""
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend

    is_first_call = _check_first_call(layer)

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


_PATCH_TARGETS = [
    # Dense W4A16
    (
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
        "compressed_tensors_w4a16_nvfp4.CompressedTensorsW4A16Fp4.process_weights_after_loading",
        patched_w4a16_process_weights_after_loading,
    ),
    # Dense W4A4
    (
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
        "compressed_tensors_w4a4_nvfp4.CompressedTensorsW4A4Fp4.process_weights_after_loading",
        patched_w4a4_process_weights_after_loading,
    ),
    # MoE NVFP4
    (
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe.CompressedTensorsW4A4Nvfp4MoEMethod.process_weights_after_loading",
        patched_nvfp4_moe_process_weights_after_loading,
    ),
]

_applied_patches = []


def apply_qat_patches():
    """Apply NVFP4 patches to support dynamic weight updates. Call before model loading."""
    global _applied_patches

    if _applied_patches:
        logger.warning("QAT patches already applied, skipping")
        return _applied_patches

    logger.info("Applying NVFP4 patches for dynamic weight loading...")

    for target, replacement in _PATCH_TARGETS:
        p = patch(target, replacement)
        _applied_patches.append(p)
        p.start()

    logger.info(f"Applied {len(_applied_patches)} NVFP4 patches for dynamic weight loading")
    return _applied_patches


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

    param_meta = ParamMetaDict(inner_model, device=device)

    param_meta.prepare_for_reload()
    logger.info(f"[prepare_qat] Tensor swap prepared for {len(param_meta._tensor_swap_layers)} layers")

    # Rebuild deleted (W4A16) or overwritten (W4A4) params back to HF format
    rebuilt_count = 0
    for layer_name, cache_entry in param_meta._layer_meta_cache.items():
        module = cache_entry["module"]
        for param_name, pm in cache_entry["meta"].items():
            existing = getattr(module, param_name, None)
            if existing is not None:
                hf_shape = tuple(pm["shape"])
                hf_dtype = pm["dtype"]
                if (
                    tuple(existing.shape) == hf_shape
                    and existing.dtype == hf_dtype
                    and hasattr(existing, "weight_loader")
                ):
                    continue
            new_param = _create_param_from_meta(module, param_name, pm, device)
            module.register_parameter(param_name, new_param)
            rebuilt_count += 1

    logger.info(f"[prepare_qat] Rebuilt {rebuilt_count} parameters")
    inner_model._param_meta_for_restore = param_meta
    return param_meta


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
    "apply_qat_patches",
    "prepare_qat_for_load_weights",
    "manual_process_weights_after_loading",
]
