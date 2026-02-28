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

"""
vLLM ModelOpt NVFP4 Patches for Dynamic Weight Updates (Marlin Backend).

Enables dynamic weight reloading for NVFP4 quantized models in vLLM
using the ModelOpt quantization path with the Marlin kernel backend.

Saves parameter metadata on first load and deletes HF parameters. Before
reload, HF parameters are rebuilt from metadata, loaded, then re-converted
to Marlin format in-place via copy_ (preserving CUDA Graph tensor addresses).

Supported schemes:
- Dense: ModelOptNvFp4LinearMethod (Marlin backend)
- MoE:  ModelOptNvFp4FusedMoE     (Marlin backend)
- KV:   BaseKVCacheMethod          (preserves scales for reload)
"""

import logging
import os
from typing import Optional

import torch
from torch.nn import Parameter

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def save_param_meta(layer: torch.nn.Module, param_name: str):
    """Save parameter metadata (shape, dtype, param_class, dims) for later rebuild."""
    if not hasattr(layer, "_hf_param_meta"):
        layer._hf_param_meta = {}

    param = getattr(layer, param_name, None)
    if param is None:
        return

    meta = {
        "shape": tuple(param.shape),
        "dtype": param.dtype,
        "device": str(param.device),
        "param_class": type(param),
    }

    if hasattr(param, "_input_dim"):
        meta["input_dim"] = param._input_dim
    if hasattr(param, "_output_dim"):
        meta["output_dim"] = param._output_dim

    layer._hf_param_meta[param_name] = meta


def _create_param_from_meta(
    module: torch.nn.Module,
    param_name: str,
    meta: dict,
    device: Optional[torch.device] = None,
) -> Parameter:
    """Create a Parameter from saved metadata. Used by rebuild and tensor swap."""
    shape = meta["shape"]
    dtype = meta["dtype"]
    dev = device or meta.get("device", "cuda")
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

    return new_param


def _check_first_call(layer: torch.nn.Module) -> bool:
    """Check if this is the first process_weights call, and increment counter."""
    count = getattr(layer, "_process_weights_call_count", 0)
    layer._process_weights_call_count = count + 1
    return count == 0


def _save_weight_loaders(layer: torch.nn.Module, param_names: list[str]):
    """Save weight_loader references from parameters before they are overwritten."""
    if not hasattr(layer, "_weight_loaders"):
        layer._weight_loaders = {}
    for pname in param_names:
        param = getattr(layer, pname, None)
        if param is not None and hasattr(param, "weight_loader"):
            layer._weight_loaders[pname] = param.weight_loader


def _update_ref_or_create(layer, ref_name, new_data):
    """Copy new_data into existing tensor ref (CUDA Graph safe), or create new Parameter."""
    refs = getattr(layer, "_marlin_tensor_refs", {})
    ref = refs.get(ref_name)
    if ref is not None:
        ref.copy_(new_data)
        setattr(layer, ref_name, Parameter(ref, requires_grad=False))
    else:
        logger.warning(f"_marlin_tensor_refs['{ref_name}'] not found, creating new Parameter")
        t = new_data.clone() if isinstance(new_data, torch.Tensor) else torch.tensor(new_data)
        setattr(layer, ref_name, Parameter(t, requires_grad=False))


class ModelOptParamMetaDict(dict):
    """
    Dict-like class for parameter management with metadata-based rebuild
    and tensor swap. Supports:
    - Rebuild of deleted parameters from saved metadata
    - Tensor swap for parameters with shape changes (address stability for CUDA Graph)
    """

    def __init__(self, model: torch.nn.Module, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device

        actual_model = model
        if hasattr(model, "model"):
            actual_model = model.model
        self._model = actual_model

        self._layer_meta_cache: dict[str, dict] = {}
        self._tensor_swap_layers: dict[str, dict] = {}

        self._build_mappings()

        for name, param in actual_model.named_parameters():
            self[name] = param

    def _build_mappings(self):
        """Build layer metadata cache for rebuild and tensor swap."""
        for layer_name, module in self._model.named_modules():
            if not hasattr(module, "_hf_param_meta"):
                continue

            self._layer_meta_cache[layer_name] = {
                "module": module,
                "meta": module._hf_param_meta,
            }

            marlin_refs = getattr(module, "_marlin_tensor_refs", {})
            for param_name, meta in module._hf_param_meta.items():
                if param_name in marlin_refs:
                    key = f"{layer_name}.{param_name}" if layer_name else param_name
                    self._tensor_swap_layers[key] = {
                        "module": module,
                        "param_name": param_name,
                        "marlin_ref": marlin_refs[param_name],
                        "hf_meta": meta,
                    }

    def _try_rebuild(self, key: str) -> Optional[Parameter]:
        parts = key.rsplit(".", 1)
        if len(parts) != 2:
            return None
        layer_name, param_name = parts
        if layer_name not in self._layer_meta_cache:
            return None
        cache_entry = self._layer_meta_cache[layer_name]
        module = cache_entry["module"]
        meta = cache_entry["meta"]
        if param_name not in meta:
            return None
        if hasattr(module, param_name):
            param = getattr(module, param_name)
            if param is not None:
                return param
        new_param = _create_param_from_meta(module, param_name, meta[param_name], self.device)
        module.register_parameter(param_name, new_param)
        return new_param

    def prepare_for_reload(self) -> None:
        """Replace kernel-format tensors with HF-shape tensors for reload."""
        for _key, swap_info in self._tensor_swap_layers.items():
            module = swap_info["module"]
            param_name = swap_info["param_name"]
            hf_meta = swap_info["hf_meta"]
            if hasattr(module, param_name):
                new_param = _create_param_from_meta(module, param_name, hf_meta, self.device)
                setattr(module, param_name, new_param)

    def __getitem__(self, key: str) -> Parameter:
        if key in dict.keys(self):
            return super().__getitem__(key)
        param = self._try_rebuild(key)
        if param is not None:
            self[key] = param
            return param
        raise KeyError(f"Parameter not found: {key}")

    def __contains__(self, key: str) -> bool:
        if super().__contains__(key):
            return True
        parts = key.rsplit(".", 1)
        if len(parts) == 2:
            layer_name, param_name = parts
            if layer_name in self._layer_meta_cache:
                if param_name in self._layer_meta_cache[layer_name]["meta"]:
                    return True
        return False

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default


_DENSE_HF_PARAMS = ["weight", "weight_scale", "input_scale", "weight_scale_2"]


def _modelopt_dense_process_weights(self, layer: torch.nn.Module) -> None:
    """
    Replacement for ModelOptNvFp4LinearMethod.process_weights_after_loading.

    First call:  save metadata + weight_loaders, convert HF→Marlin format,
                 save _marlin_tensor_refs for CUDA Graph stability.
    Subsequent:  read reloaded HF data, convert, copy_ into saved refs.
    """
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
        marlin_permute_scales,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        nvfp4_marlin_process_global_scale,
        nvfp4_marlin_process_scales,
    )

    is_first_call = _check_first_call(layer)

    if is_first_call:
        for pname in _DENSE_HF_PARAMS:
            save_param_meta(layer, pname)
        _save_weight_loaders(layer, _DENSE_HF_PARAMS)

    weight_data = layer.weight.data
    weight_scale_data = layer.weight_scale.data
    weight_scale_2_data = layer.weight_scale_2.data

    assert weight_scale_data.dtype == torch.float8_e4m3fn

    device = weight_data.device
    part_size_n = layer.output_size_per_partition
    part_size_k = layer.input_size_per_partition
    param_dtype = layer.params_dtype
    group_size = 16
    weight_scale_2_max = weight_scale_2_data.max().to(torch.float32)

    if is_first_call:
        layer.workspace = marlin_make_workspace_new(device)

    perm = torch.empty(0, dtype=torch.int, device=device)
    qweight = weight_data.view(torch.int32).T.contiguous()
    marlin_weight = ops.gptq_marlin_repack(
        b_q_weight=qweight,
        perm=perm,
        size_k=part_size_k,
        size_n=part_size_n,
        num_bits=4,
    )

    weight_scale = weight_scale_data.T.contiguous().to(param_dtype)
    weight_scale = marlin_permute_scales(
        s=weight_scale,
        size_k=part_size_k,
        size_n=part_size_n,
        group_size=group_size,
    )
    marlin_weight_scale = nvfp4_marlin_process_scales(weight_scale)
    marlin_weight_scale_2 = nvfp4_marlin_process_global_scale(weight_scale_2_max.to(param_dtype))

    if is_first_call:
        layer.weight = Parameter(marlin_weight, requires_grad=False)
        layer.weight_scale = Parameter(marlin_weight_scale, requires_grad=False)
        layer.weight_scale_2 = Parameter(marlin_weight_scale_2, requires_grad=False)
        layer._marlin_tensor_refs = {
            "weight": layer.weight.data,
            "weight_scale": layer.weight_scale.data,
            "weight_scale_2": layer.weight_scale_2.data,
        }
    else:
        _update_ref_or_create(layer, "weight", marlin_weight)
        _update_ref_or_create(layer, "weight_scale", marlin_weight_scale)
        _update_ref_or_create(layer, "weight_scale_2", marlin_weight_scale_2)

    for attr in ["input_scale", "alpha", "input_scale_inv"]:
        if hasattr(layer, attr):
            delattr(layer, attr)


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
            )
        )
    return torch.stack(result)


def _marlin_process_scales_experts(scale_hf, param_dtype, size_k, size_n, group_size, num_experts):
    """Process scales for each expert into Marlin format and stack."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import nvfp4_marlin_process_scales

    result = []
    scales = scale_hf.to(param_dtype)
    for i in range(num_experts):
        s = marlin_permute_scales(s=scales[i].T, size_k=size_k, size_n=size_n, group_size=group_size)
        result.append(nvfp4_marlin_process_scales(s))
    return torch.stack(result)


_MOE_HF_PARAMS = [
    "w13_weight",
    "w2_weight",
    "w13_weight_scale",
    "w2_weight_scale",
    "w13_weight_scale_2",
    "w2_weight_scale_2",
    "w13_input_scale",
    "w2_input_scale",
]


def _modelopt_moe_marlin_convert(self, layer: torch.nn.Module, is_first_call: bool) -> None:
    """Convert MoE layer weights between HF and Marlin format."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import nvfp4_marlin_process_global_scale

    group_size = 16
    e = layer.num_experts
    k = layer.hidden_size
    n = layer.intermediate_size_per_partition
    device = layer.w13_weight.device
    param_dtype = layer.params_dtype

    if is_first_call:
        layer.workspace = marlin_make_workspace_new(device, 4)

    perm = torch.empty(0, dtype=torch.int, device=device)
    size_n_w13, size_k_w13 = n * 2, k
    size_n_w2, size_k_w2 = k, n

    # Repack weights
    w13_weight_marlin = _marlin_repack_experts(layer.w13_weight.data, perm, size_k_w13, size_n_w13, e)
    w2_weight_marlin = _marlin_repack_experts(layer.w2_weight.data, perm, size_k_w2, size_n_w2, e)

    # Process scales
    w13_weight_scale_marlin = _marlin_process_scales_experts(
        layer.w13_weight_scale.data,
        param_dtype,
        size_k_w13,
        size_n_w13,
        group_size,
        e,
    )
    w2_weight_scale_marlin = _marlin_process_scales_experts(
        layer.w2_weight_scale.data,
        param_dtype,
        size_k_w2,
        size_n_w2,
        group_size,
        e,
    )

    # Process global scales  (w13_weight_scale_2 is already (E,) after common processing)
    w13_scale_2_processed = nvfp4_marlin_process_global_scale(layer.w13_weight_scale_2.data.to(param_dtype))
    w2_scale_2_processed = nvfp4_marlin_process_global_scale(layer.w2_weight_scale_2.data.to(param_dtype))

    if is_first_call:
        layer.w13_weight = Parameter(w13_weight_marlin, requires_grad=False)
        layer.w2_weight = Parameter(w2_weight_marlin, requires_grad=False)
        layer.w13_weight_scale = Parameter(w13_weight_scale_marlin, requires_grad=False)
        layer.w2_weight_scale = Parameter(w2_weight_scale_marlin, requires_grad=False)
        layer.w13_weight_scale_2 = Parameter(w13_scale_2_processed, requires_grad=False)
        layer.w2_weight_scale_2 = Parameter(w2_scale_2_processed, requires_grad=False)
        if not hasattr(layer, "_marlin_tensor_refs"):
            layer._marlin_tensor_refs = {}
        for rn in [
            "w13_weight",
            "w2_weight",
            "w13_weight_scale",
            "w2_weight_scale",
            "w13_weight_scale_2",
            "w2_weight_scale_2",
        ]:
            layer._marlin_tensor_refs[rn] = getattr(layer, rn).data
    else:
        for rn, nd in [
            ("w13_weight", w13_weight_marlin),
            ("w2_weight", w2_weight_marlin),
            ("w13_weight_scale", w13_weight_scale_marlin),
            ("w2_weight_scale", w2_weight_scale_marlin),
            ("w13_weight_scale_2", w13_scale_2_processed),
            ("w2_weight_scale_2", w2_scale_2_processed),
        ]:
            _update_ref_or_create(layer, rn, nd)

    for attr in ["w13_input_scale", "w2_input_scale"]:
        if hasattr(layer, attr):
            delattr(layer, attr)


def _modelopt_moe_process_weights(self, layer: torch.nn.Module) -> None:
    """
    Replacement for ModelOptNvFp4FusedMoE.process_weights_after_loading (Marlin).

    First call:  save metadata + weight_loaders, convert HF→Marlin format,
                 save _marlin_tensor_refs for CUDA Graph stability.
    Subsequent:  read reloaded HF data, convert, copy_ into saved refs.
    """
    is_first_call = _check_first_call(layer)

    if is_first_call:
        for pname in _MOE_HF_PARAMS:
            save_param_meta(layer, pname)
        _save_weight_loaders(layer, _MOE_HF_PARAMS)

    w13_weight_scale_2 = layer.w13_weight_scale_2.data
    if w13_weight_scale_2.dim() == 2:
        w13_weight_scale_2 = w13_weight_scale_2[:, 0]
    layer.w13_weight_scale_2 = Parameter(w13_weight_scale_2, requires_grad=False)

    _modelopt_moe_marlin_convert(self, layer, is_first_call)

    self.moe_quant_config = self.get_fused_moe_quant_config(layer)


# ============================================================================
# KV Cache Patch
# ============================================================================


def _modelopt_kv_process_weights(self, layer) -> None:
    """
    Replacement for BaseKVCacheMethod.process_weights_after_loading.
    Doesn't delete k_scale, v_scale, q_scale, prob_scale to allow
    for dynamic updates during refit.
    """
    from vllm.platforms import current_platform

    if layer.kv_cache_dtype != "auto" and not layer.calculate_kv_scales:
        if layer.k_scale > 0.0 and layer.v_scale > 0.0:
            k_scale = layer.k_scale.to("cpu").tolist()
            v_scale = layer.v_scale.to("cpu").tolist()
            if current_platform.is_fp8_fnuz():
                k_scale *= 2
                v_scale *= 2
        elif layer.k_scale < 0.0 and layer.v_scale < 0.0:
            k_scale = 1.0
            v_scale = 1.0
        else:
            assert layer.k_scale > 0.0
            scale_to_duplicate = max(layer.k_scale, layer.v_scale)
            k_scale = scale_to_duplicate.to("cpu").tolist()
            v_scale = scale_to_duplicate.to("cpu").tolist()
            if current_platform.is_fp8_fnuz():
                k_scale *= 2
                v_scale *= 2

        if not isinstance(k_scale, float) or not isinstance(v_scale, float):
            raise ValueError("Only support per-tensor scaling factor for fp8 KV cache")

        if layer.q_scale < 0.0:
            layer._q_scale.copy_(k_scale)
            layer._q_scale_float = k_scale

        layer._k_scale.copy_(k_scale)
        layer._v_scale.copy_(v_scale)
        layer._k_scale_float = k_scale
        layer._v_scale_float = v_scale

    if layer.q_scale > 0.0:
        q_scale = layer.q_scale
        if current_platform.is_fp8_fnuz():
            q_scale *= 2
        layer.calculate_kv_scales = False
    else:
        q_scale = 1.0
    if layer.prob_scale > 0.0:
        prob_scale = layer.prob_scale
        if current_platform.is_fp8_fnuz():
            prob_scale *= 2
    else:
        prob_scale = 1.0

    is_singleton_float = (
        lambda x: isinstance(x, float) or isinstance(x, torch.Tensor) and x.numel() == 1 and x.is_floating_point()
    )
    if not is_singleton_float(q_scale) or not is_singleton_float(prob_scale):
        raise ValueError("Only support per-tensor scaling factor for fp8-quantized Q/prob")

    layer._q_scale.copy_(q_scale)
    layer._q_scale_float = q_scale.item() if isinstance(q_scale, torch.Tensor) else q_scale
    layer._prob_scale.copy_(prob_scale)


# ============================================================================
# Patch Application & Entry Points
# ============================================================================

_patched = False


def prepare_modelopt_for_weight_reload(model, device=None):
    """
    Prepare ModelOpt model for weight reloading. Call ONCE before each reload cycle.

    1. Builds ModelOptParamMetaDict from saved metadata
    2. Swaps kernel-format tensors back to HF-shape for weight_loader compatibility
    3. Rebuilds any deleted parameters from metadata

    Args:
        model: vLLM model
        device: Device for created parameters
    """
    inner_model = model
    if hasattr(model, "model"):
        inner_model = model.model

    param_meta = ModelOptParamMetaDict(inner_model, device=device)

    param_meta.prepare_for_reload()
    logger.info(f"[prepare_modelopt] Tensor swap prepared for {len(param_meta._tensor_swap_layers)} layers")

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

    logger.info(f"[prepare_modelopt] Rebuilt {rebuilt_count} parameters")
    inner_model._param_meta_for_restore = param_meta
    return param_meta


def modelopt_process_weights_after_loading(model):
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
                if "KVCache" in quant_method.__class__.__name__:
                    continue
                quant_method.process_weights_after_loading(module)
                moe_count += 1

    logger.debug(f"Processed {dense_count} dense layers, {moe_count} MoE layers")
    return dense_count + moe_count


def apply_modelopt_nvfp4_patches():
    """Apply ModelOpt NVFP4 patches to support dynamic weight updates. Call before model loading."""
    global _patched

    if _patched:
        logger.warning("ModelOpt NVFP4 patches already applied, skipping")
        return

    logger.info("Applying ModelOpt NVFP4 patches for dynamic weight loading...")

    from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4FusedMoE,
        ModelOptNvFp4LinearMethod,
    )

    ModelOptNvFp4LinearMethod.process_weights_after_loading = _modelopt_dense_process_weights
    ModelOptNvFp4FusedMoE.process_weights_after_loading = _modelopt_moe_process_weights
    BaseKVCacheMethod.process_weights_after_loading = _modelopt_kv_process_weights

    _patched = True
    logger.info("Applied 3 ModelOpt NVFP4 patches (Dense, MoE, KV)")
