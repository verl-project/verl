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
from types import MethodType
from unittest.mock import patch

import torch
import vllm
from packaging import version

try:
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.linear import LinearBase
except ImportError as e:
    raise ImportError("FP8 quantization not available") from e

from verl.utils.device import get_device_name
from verl.utils.kernel.fp8_kernel import scaled_fp8_blockwise

logger = logging.getLogger(__name__)

_MXFP4_E2M1_MAX = 6.0
_MXFP4_E8M0_BIAS = 127
_MXFP4_E8M0_MIN = 1
_MXFP4_E8M0_MAX = 254
_MXFP4_E2M1_THRESHOLDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


# Ref: https://github.com/NVIDIA-NeMo/RL/commit/bc24887c72a6e1b2699a228bc87c588546dfe6b7
@dataclass()
class FP8State:
    # A cache of fp8 parameter names, we can check this cache to see if a
    # param name corresponds to a fp8 weight
    seen_params: set = field(default_factory=lambda: set())
    fp8_param_names: set = field(default_factory=lambda: set())
    vllm_patches: list = field(default_factory=lambda: [])


fp8_state: FP8State = FP8State()


def _copy_param_attributes(dst_param, src_param):
    base_param_dir = dir(torch.nn.Parameter)
    for attr in dir(src_param):
        if attr not in base_param_dir and not attr.startswith("__"):
            try:
                setattr(dst_param, attr, getattr(src_param, attr))
            except Exception:
                pass


def _create_param_from_subclass_attributes(custom_param, source_param=None):
    param = torch.nn.Parameter(custom_param.data, requires_grad=False)
    if source_param is not None:
        _copy_param_attributes(param, source_param)
    _copy_param_attributes(param, custom_param)

    param.subclass_type = type(custom_param)
    return param


def _create_param_from_data_with_attrs(data, source_param):
    param = torch.nn.Parameter(data, requires_grad=False)
    _copy_param_attributes(param, source_param)
    return param


def _get_param_weight_loader(param):
    weight_loader = getattr(param, "weight_loader", None)
    if weight_loader is None:
        weight_loader = getattr(param, "_weight_loader", None)
    return weight_loader


def _param_parallel_dim(param, public_name, private_name, default):
    if hasattr(param, private_name):
        return getattr(param, private_name)
    if hasattr(param, public_name):
        return getattr(param, public_name)
    return default


def _copy_loaded_weight(param, loaded_weight):
    param.data.copy_(loaded_weight.to(device=param.data.device, dtype=param.data.dtype))


def _copy_weight_attrs(param, weight_loader=None, quant_method=None):
    if weight_loader is not None:
        param.weight_loader = weight_loader
    if quant_method is not None:
        param.quant_method = quant_method
    return param


def _is_rank_zero():
    return (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == 0
    )


def _normalize_dim(dim, ndim):
    if dim is None:
        return 0
    if dim < 0:
        dim += ndim
    return dim


def _try_load_grouped_column_weight(param, loaded_weight):
    tp_size = int(getattr(param, "tp_size", 1))
    tp_rank = int(getattr(param, "tp_rank", 0))
    if tp_size <= 1:
        return False

    data = param.data
    if loaded_weight.ndim == data.ndim + 1 and loaded_weight.shape == torch.Size((tp_size, *data.shape)):
        _copy_loaded_weight(param, loaded_weight.select(0, tp_rank))
        return True

    if data.ndim >= 2 and loaded_weight.ndim == data.ndim - 1:
        expected_flat_shape = (tp_size * data.shape[0] * data.shape[1], *data.shape[2:])
        if loaded_weight.shape == torch.Size(expected_flat_shape):
            global_shape = (tp_size, data.shape[0], data.shape[1], *data.shape[2:])
            _copy_loaded_weight(param, loaded_weight.reshape(global_shape).select(0, tp_rank))
            return True

    if data.ndim == loaded_weight.ndim and data.ndim > 0:
        expected_dim0 = tp_size * data.shape[0]
        if loaded_weight.shape[0] == expected_dim0 and loaded_weight.shape[1:] == data.shape[1:]:
            start = tp_rank * data.shape[0]
            _copy_loaded_weight(param, loaded_weight.narrow(0, start, data.shape[0]))
            return True

    return False


def _merged_column_offsets(param, loaded_weight, shard_offset, shard_size, loaded_shard_id):
    dim = _normalize_dim(getattr(param, "output_dim", getattr(param, "_output_dim", 0)), param.data.ndim)
    loaded_dim = loaded_weight.shape[dim]
    offsets = []
    if isinstance(shard_offset, int):
        offsets.append(shard_offset)
        if isinstance(shard_size, int) and shard_size > 0:
            scaled_offset = shard_offset * loaded_dim // shard_size
            offsets.append(scaled_offset)
    if isinstance(loaded_shard_id, int):
        offsets.append(loaded_shard_id * loaded_dim)

    seen = set()
    for offset in offsets:
        if offset in seen:
            continue
        seen.add(offset)
        if offset < 0 or offset + loaded_dim > param.data.shape[dim]:
            continue
        yield dim, offset


def _try_load_merged_column_weight(param, loaded_weight, shard_offset, shard_size, loaded_shard_id):
    for dim, offset in _merged_column_offsets(param, loaded_weight, shard_offset, shard_size, loaded_shard_id):
        target = param.data.narrow(dim, offset, loaded_weight.shape[dim])
        if target.shape == loaded_weight.shape:
            target.copy_(loaded_weight.to(device=target.device, dtype=target.dtype))
            return True
    return False


def _attach_fp8_reload_fallbacks(param):
    subclass_type = getattr(param, "subclass_type", None)
    if subclass_type is None or getattr(param, "_verl_fp8_reload_fallbacks", False):
        return

    original_column_loader = getattr(subclass_type, "load_column_parallel_weight", None)
    original_merged_loader = getattr(subclass_type, "load_merged_column_weight", None)

    if original_column_loader is not None:

        def load_column_parallel_weight(self, *args, **kwargs):
            loaded_weight = kwargs.get("loaded_weight")
            if loaded_weight is None and args:
                loaded_weight = args[0]
            if loaded_weight is not None:
                if self.data.shape == loaded_weight.shape:
                    _copy_loaded_weight(self, loaded_weight)
                    return
                if _try_load_grouped_column_weight(self, loaded_weight):
                    return
            return original_column_loader(self, *args, **kwargs)

        param.load_column_parallel_weight = MethodType(load_column_parallel_weight, param)

    if original_merged_loader is not None:

        def load_merged_column_weight(self, *args, **kwargs):
            loaded_weight = kwargs.get("loaded_weight")
            if loaded_weight is None and args:
                loaded_weight = args[0]
            loaded_shard_id = kwargs.get("loaded_shard_id", kwargs.get("shard_id"))
            if loaded_shard_id is None and len(args) > 1:
                loaded_shard_id = args[1]
            shard_offset = kwargs.get("shard_offset")
            if shard_offset is None and len(args) > 2:
                shard_offset = args[2]
            shard_size = kwargs.get("shard_size")
            if shard_size is None and len(args) > 3:
                shard_size = args[3]

            if loaded_weight is not None and _try_load_merged_column_weight(
                self,
                loaded_weight,
                shard_offset,
                shard_size,
                loaded_shard_id,
            ):
                return
            return original_merged_loader(self, *args, **kwargs)

        param.load_merged_column_weight = MethodType(load_merged_column_weight, param)

    param._verl_fp8_reload_fallbacks = True


def _ensure_linear_params_reloadable(layer):
    """Restore vLLM parameter metadata after fp8 post-processing replaces it."""
    from vllm.model_executor.parameter import (
        BlockQuantScaleParameter,
        ModelWeightParameter,
    )

    if hasattr(layer, "weight") and not hasattr(layer.weight, "subclass_type"):
        source_weight = layer.weight
        weight_loader = _get_param_weight_loader(layer.weight)
        if weight_loader is not None:
            layer.weight = _create_param_from_subclass_attributes(
                ModelWeightParameter(
                    data=source_weight.data,
                    output_dim=_param_parallel_dim(source_weight, "output_dim", "_output_dim", 0),
                    input_dim=_param_parallel_dim(source_weight, "input_dim", "_input_dim", 1),
                    weight_loader=weight_loader,
                ),
                source_weight,
            )

    for scale_name in ("weight_scale_inv", "weight_scale"):
        if not hasattr(layer, scale_name):
            continue
        scale = getattr(layer, scale_name)
        if hasattr(scale, "subclass_type"):
            continue
        weight_loader = _get_param_weight_loader(scale)
        if weight_loader is not None:
            setattr(
                layer,
                scale_name,
                _create_param_from_subclass_attributes(
                    BlockQuantScaleParameter(
                        data=scale.data,
                        output_dim=_param_parallel_dim(scale, "output_dim", "_output_dim", 0),
                        input_dim=_param_parallel_dim(scale, "input_dim", "_input_dim", 1),
                        weight_loader=weight_loader,
                    ),
                    scale,
                ),
            )

    update_param_tp_status = getattr(layer, "update_param_tp_status", None)
    if callable(update_param_tp_status):
        update_param_tp_status()

    for param_name in ("weight", "weight_scale_inv", "weight_scale"):
        param = getattr(layer, param_name, None)
        if param is not None:
            _attach_fp8_reload_fallbacks(param)


def _ensure_model_params_reloadable(model):
    for module in model.modules():
        if isinstance(module, LinearBase):
            _ensure_linear_params_reloadable(module)


def _is_mxfp4_moe_module(module):
    if not isinstance(module, FusedMoE):
        return False
    quant_method = getattr(module, "quant_method", None)
    quant_method_name = type(quant_method).__name__
    return quant_method_name in ("Mxfp4MoEMethod", "GptOssMxfp4MoEMethod") or getattr(
        quant_method, "weight_dtype", None
    ) in ("mxfp4", "gpt_oss_mxfp4")


def _is_deepseek_v4_mega_moe_module(module):
    return type(module).__name__ == "DeepseekV4MegaMoEExperts"


def _mxfp4_moe_load_shape(module):
    quant_method = getattr(module, "quant_method", None)
    shape = (
        getattr(quant_method, "num_experts", getattr(module, "local_num_experts", None)),
        getattr(quant_method, "intermediate_size", getattr(module, "intermediate_size_per_partition", None)),
        getattr(quant_method, "hidden_size", getattr(module, "hidden_size", None)),
    )
    if any(value is None for value in shape):
        return None
    return tuple(int(value) for value in shape)


def _deepseek_v4_mega_moe_load_shape(module):
    shape = (
        getattr(module, "num_local_experts", None),
        getattr(module, "intermediate_size", None),
        getattr(module, "hidden_size", None),
    )
    if any(value is None for value in shape):
        return None
    return tuple(int(value) for value in shape)


def _module_param_device(module):
    for param_name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
        param = getattr(module, param_name, None)
        if param is not None and hasattr(param, "device"):
            return param.device
    return torch.device(get_device_name())


def _make_mxfp4_moe_param(shape, device, weight_loader, quant_method=None):
    param = torch.nn.Parameter(torch.empty(shape, dtype=torch.uint8, device=device), requires_grad=False)
    return _copy_weight_attrs(param, weight_loader=weight_loader, quant_method=quant_method)


def _restore_mxfp4_moe_module_params(module, load_shape):
    num_experts, intermediate_size, hidden_size = load_shape
    device = _module_param_device(module)
    weight_loader = getattr(module, "weight_loader", None)
    module.w13_weight = _make_mxfp4_moe_param(
        (num_experts, 2 * intermediate_size, hidden_size // 2),
        device,
        weight_loader,
    )
    module.w2_weight = _make_mxfp4_moe_param(
        (num_experts, hidden_size, intermediate_size // 2),
        device,
        weight_loader,
    )
    module.w13_weight_scale = _make_mxfp4_moe_param(
        (num_experts, 2 * intermediate_size, hidden_size // 32),
        device,
        weight_loader,
        quant_method="block",
    )
    module.w2_weight_scale = _make_mxfp4_moe_param(
        (num_experts, hidden_size, intermediate_size // 32),
        device,
        weight_loader,
        quant_method="block",
    )


def _restore_mxfp4_moe_params_for_loading(model):
    restored = False
    for module in model.modules():
        if _is_deepseek_v4_mega_moe_module(module):
            load_shape = _deepseek_v4_mega_moe_load_shape(module)
            if load_shape is None:
                continue

            _restore_mxfp4_moe_module_params(module, load_shape)
            restored = True
            continue

        if not _is_mxfp4_moe_module(module):
            continue
        load_shape = _mxfp4_moe_load_shape(module)
        if load_shape is None:
            continue

        _restore_mxfp4_moe_module_params(module, load_shape)
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None:
            for attr in ("moe_kernel", "moe_quant_config", "w13_precision_config", "w2_precision_config"):
                if hasattr(quant_method, attr):
                    setattr(quant_method, attr, None)
        restored = True
    return restored


def _process_mxfp4_moe_weights_after_loading(model):
    for module in model.modules():
        if not _is_mxfp4_moe_module(module):
            continue
        quant_method = getattr(module, "quant_method", None)
        process_weights = getattr(quant_method, "process_weights_after_loading", None)
        if callable(process_weights):
            process_weights(module)


def is_fp8_model(vllm_config):
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    quant_config = getattr(vllm_config, "quant_config", None)
    return isinstance(quant_config, Fp8Config) or is_mxfp8_vllm_ascend(quant_config)


def get_module_from_param_name(model, name: str):
    if name.startswith("mtp."):
        return None

    def _mapped_name(param_name: str) -> str:
        mapper = getattr(model, "hf_to_vllm_mapper", None)
        map_name = getattr(mapper, "_map_name", None)
        if callable(map_name):
            mapped = map_name(param_name)
            if mapped is not None:
                return mapped
        return param_name

    def _candidate_module_paths(param_name: str) -> list[list[str]]:
        # Split the name into parts (e.g., 'layers', '0', 'self_attn', 'q_proj', 'weight')
        # The module path is all but the last part (the parameter's own name)
        path_parts = _mapped_name(param_name).split(".")
        if not path_parts:
            return []
        module_path = path_parts[:-1]

        # Replace with the fused model name
        packed_module_path = list(module_path)
        if packed_module_path and packed_module_path[-1] in reversed_mapping.keys():
            packed_module_path[-1] = reversed_mapping[packed_module_path[-1]]

        candidates = [packed_module_path]

        # DeepSeek-V4 keeps stacked checkpoint names in its load_weights method
        # rather than in packed_modules_mapping, so mirror those aliases here.
        module_path_str = ".".join(packed_module_path)
        deepseek_v4_aliases = {
            ".ffn.shared_experts.w1": ".ffn.shared_experts.gate_up_proj",
            ".ffn.shared_experts.w3": ".ffn.shared_experts.gate_up_proj",
            ".attn.wq_a": ".attn.fused_wqa_wkv",
            ".attn.wkv": ".attn.fused_wqa_wkv",
            ".compressor.wkv": ".compressor.fused_wkv_wgate",
            ".compressor.wgate": ".compressor.fused_wkv_wgate",
        }
        for old, new in deepseek_v4_aliases.items():
            if old in module_path_str:
                candidates.append(module_path_str.replace(old, new, 1).split("."))

        return candidates

    packed_modules_mapping = getattr(model, "packed_modules_mapping", None) or {}
    reversed_mapping = {
        original_name: fused_name
        for fused_name, original_names_list in packed_modules_mapping.items()
        for original_name in original_names_list
    }

    last_error = None
    for module_path in _candidate_module_paths(name):
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
            return current_module
        except (AttributeError, IndexError, ValueError) as e:
            last_error = e
    logger.debug("Could not find module for parameter %r: %s", name, last_error)
    return None


def is_fp8_weight(name, model):
    if name not in fp8_state.seen_params:
        fp8_state.seen_params.add(name)
        # Filter out bias params
        if name.endswith("weight"):
            module = get_module_from_param_name(model, name)
            # We currently only quantize linear layers

            if module is not None and (
                (isinstance(module, LinearBase) and module.weight.dtype == torch.float8_e4m3fn)
                or (
                    isinstance(module, FusedMoE)
                    and module.w13_weight.dtype == torch.float8_e4m3fn
                    and module.w2_weight.dtype == torch.float8_e4m3fn
                )
            ):
                fp8_state.fp8_param_names.add(name)
    return name in fp8_state.fp8_param_names


def is_mxfp4_moe_weight(name, tensor, model):
    if not name.endswith(".weight") or ".experts." not in name:
        return False
    if _is_prequantized_mxfp4_tensor(tensor):
        return False
    module = get_module_from_param_name(model, name)
    return _is_mxfp4_moe_module(module)


def is_prequantized_mxfp4_moe_weight(name, tensor, model):
    if not name.endswith(".weight") or ".experts." not in name:
        return False
    if not _is_prequantized_mxfp4_tensor(tensor):
        return False
    if _model_type(model) == "deepseek_v4":
        return True
    module = get_module_from_param_name(model, name)
    return _is_mxfp4_moe_module(module)


def is_prequantized_mxfp4_moe_scale(name, tensor, model):
    if not name.endswith(".scale") or ".experts." not in name:
        return False
    if tensor.dtype != getattr(torch, "float8_e8m0fnu", None):
        return False
    if not any(f".w{i}.scale" in name for i in (1, 2, 3)):
        return False
    if _model_type(model) == "deepseek_v4":
        return True
    weight_name = name[: -len(".scale")] + ".weight"
    module = get_module_from_param_name(model, weight_name)
    return _is_mxfp4_moe_module(module)


def _is_prequantized_mxfp4_tensor(tensor):
    float4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    return tensor.dtype in (torch.int8, torch.uint8, float4_dtype)


def _as_mxfp4_storage_tensor(tensor):
    if tensor.dtype in (torch.int8, getattr(torch, "float8_e8m0fnu", None)):
        return tensor.view(torch.uint8)
    return tensor


def _is_prequantized_fp8_tensor(tensor):
    fp8_dtypes = tuple(
        dtype
        for dtype in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None))
        if dtype is not None
    )
    return tensor.dtype in fp8_dtypes


def _model_type(model):
    if model is None:
        return None

    for obj in (model, getattr(model, "config", None), getattr(model, "hf_config", None)):
        if obj is None:
            continue
        model_type = getattr(obj, "model_type", None)
        if model_type is not None:
            return model_type
    config = getattr(model, "config", None)
    if config is not None:
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            return getattr(text_config, "model_type", None)
    return None


def _uses_dot_scale_suffix(model):
    return _model_type(model) == "deepseek_v4"


def _map_weight_name_for_vllm(model, name: str) -> str:
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    map_name = getattr(mapper, "_map_name", None)
    if callable(map_name):
        mapped = map_name(name)
        if mapped is not None:
            return mapped

    mapped = name
    if ".shared_experts.w2" in mapped:
        mapped = mapped.replace(".shared_experts.w2", ".shared_experts.down_proj", 1)
    if mapped.endswith(".scale"):
        mapped = mapped[: -len(".scale")] + ".weight_scale_inv"
    if mapped.startswith("layers."):
        mapped = "model." + mapped
    elif mapped.startswith("embed."):
        mapped = "model." + mapped
    elif mapped == "head.weight":
        mapped = "lm_head.weight"
    return mapped


def _cache_deepseek_v4_dense_fp8_scales(model, weights):
    if _model_type(model) != "deepseek_v4":
        return

    scale_dtype = getattr(torch, "float8_e8m0fnu", None)
    if scale_dtype is None:
        return

    cache = getattr(model, "_verl_dense_fp8_scale_cache", None)
    if cache is None:
        cache = {}
        model._verl_dense_fp8_scale_cache = cache

    for name, tensor in weights:
        if not name.endswith(".scale") or ".experts." in name or tensor.dtype != scale_dtype:
            continue
        mapped_name = _map_weight_name_for_vllm(model, name)
        cache[mapped_name] = tensor.detach().clone()


def _copy_scale_shard(param: torch.nn.Parameter, loaded_scale: torch.Tensor) -> None:
    target = param.data
    loaded = loaded_scale.to(device=target.device, dtype=target.dtype)
    if target.shape == loaded.shape:
        target.copy_(loaded)
        return

    if target.ndim != loaded.ndim:
        return

    tp_rank = int(getattr(param, "tp_rank", 0))
    tp_size = int(getattr(param, "tp_size", 1))
    candidate_dims = []
    for attr in ("input_dim", "_input_dim", "output_dim", "_output_dim"):
        if hasattr(param, attr):
            dim = _normalize_dim(int(getattr(param, attr)), target.ndim)
            if dim not in candidate_dims:
                candidate_dims.append(dim)

    for dim in candidate_dims:
        if loaded.shape[:dim] != target.shape[:dim] or loaded.shape[dim + 1 :] != target.shape[dim + 1 :]:
            continue
        if loaded.shape[dim] != target.shape[dim] * tp_size:
            continue
        start = tp_rank * target.shape[dim]
        target.copy_(loaded.narrow(dim, start, target.shape[dim]))
        return


def _reload_cached_deepseek_v4_dense_fp8_scales(model):
    cache = getattr(model, "_verl_dense_fp8_scale_cache", None)
    if not cache:
        return

    params = dict(model.named_parameters())
    for name, scale in cache.items():
        param = params.get(name)
        if param is None:
            continue
        _copy_scale_shard(param, scale)


def _scale_name_for_weight(name, model, *, is_mxfp8_npu=False, use_scale_not_scale_inv=False, force_scale=False):
    if name.endswith(".weight") and _uses_dot_scale_suffix(model):
        return name[: -len(".weight")] + ".scale"
    if force_scale:
        return name + "_scale"
    if is_mxfp8_npu:
        return name + "_scale"
    if use_scale_not_scale_inv and "expert" not in name:
        return name + "_scale"
    return name + "_scale_inv"


def _mxfp4_scale_to_e8m0(scale):
    scale = scale.to(torch.float32)
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    scale_exp = torch.ceil(torch.log2(safe_scale)).to(torch.int32) + _MXFP4_E8M0_BIAS
    return scale_exp.clamp(_MXFP4_E8M0_MIN, _MXFP4_E8M0_MAX).to(torch.uint8)


def quantize_mxfp4_weight(weight, dtype=torch.bfloat16):
    target_dtype = dtype if dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
    weight = weight.to(target_dtype).contiguous()
    *prefix_shape, hidden_dim = weight.shape
    if hidden_dim % 32 != 0:
        raise ValueError(f"MXFP4 weight hidden dimension must be divisible by 32, got shape={tuple(weight.shape)}")

    block_size = 32
    num_blocks = hidden_dim // block_size
    blocks = weight.reshape(-1, num_blocks, block_size).to(torch.float32)

    amax = blocks.abs().amax(dim=-1)
    quant_scale = _mxfp4_scale_to_e8m0(amax / _MXFP4_E2M1_MAX)
    scale = torch.exp2((quant_scale.to(torch.float32) - _MXFP4_E8M0_BIAS).unsqueeze(-1))

    scaled = (blocks / scale).clamp(-_MXFP4_E2M1_MAX, _MXFP4_E2M1_MAX)
    thresholds = torch.tensor(_MXFP4_E2M1_THRESHOLDS, dtype=torch.float32, device=weight.device)
    magnitude = torch.bucketize(scaled.abs(), thresholds).to(torch.uint8)
    sign = torch.where(scaled < 0, torch.full_like(magnitude, 8), torch.zeros_like(magnitude))
    codes = magnitude | sign

    quant_weight = codes[..., 0::2] | (codes[..., 1::2] * 16)
    quant_weight = quant_weight.view(*prefix_shape, hidden_dim // 2)
    quant_scale = quant_scale.view(*prefix_shape, num_blocks)
    return quant_weight, quant_scale


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
    for name, module in model.named_modules():
        if (isinstance(module, LinearBase) or isinstance(module, FusedMoE)) and hasattr(
            module, "_mxfp8_original_shapes"
        ):
            if hasattr(module, "quant_method") and hasattr(module.quant_method, "process_weights_after_loading"):
                logger.debug(f"Applying MXFP8 transformation for module: {name}")
                module.quant_method.process_weights_after_loading(module)


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

    fp8_state.seen_params.clear()
    fp8_state.fp8_param_names.clear()
    is_mxfp8_npu = is_mxfp8_vllm_ascend(quant_config)
    if is_mxfp8_npu:
        import torch_npu
    # vLLM v0.11-v0.12 renamed weight_scale_inv → weight_scale in process_weights_after_loading,
    # so load_weights expects "_scale" suffix. v0.14+ keeps weight_scale_inv, so expects "_scale_inv".
    _use_scale_not_scale_inv = version.parse("0.11.0") <= version.parse(vllm.__version__) < version.parse("0.14.0")

    for k, v in weights:
        if is_prequantized_mxfp4_moe_weight(k, v, model):
            yield (k, _as_mxfp4_storage_tensor(v))
            continue

        if is_prequantized_mxfp4_moe_scale(k, v, model):
            yield (k, _as_mxfp4_storage_tensor(v))
            continue

        if is_mxfp4_moe_weight(k, v, model):
            if _is_rank_zero():
                logger.debug(f"Quantizing to MXFP4 blockwise: {k}")
            param_lp, param_scale = quantize_mxfp4_weight(v, dtype=dtype)
            yield (k, param_lp)
            yield (_scale_name_for_weight(k, model, force_scale=True), param_scale)
            del v, param_lp, param_scale
            continue

        if not is_fp8_weight(k, model):
            yield (k, v)
            continue

        if _is_prequantized_fp8_tensor(v):
            yield (k, v)
            continue

        # Cast the weight into fp8 and its scale factor
        if _is_rank_zero():
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

        # Yield the scale with appropriate naming based on vLLM version
        yield (
            _scale_name_for_weight(
                k,
                model,
                is_mxfp8_npu=is_mxfp8_npu,
                use_scale_not_scale_inv=_use_scale_not_scale_inv,
            ),
            param_scale,
        )

        # Explicitly delete original tensor reference to help GC
        del v, param_lp, param_scale


def _get_quanted_weight_model(model_runner, is_drafter=False):
    if is_drafter:
        drafter = getattr(model_runner, "drafter", None)
        model = drafter.model if drafter is not None and hasattr(drafter, "model") else None
        assert model is not None, (
            "load_quanted_weights(is_drafter=True) requires model_runner.drafter.model "
            "to be present and non-None for FP8 weight loading."
        )
    else:
        model = model_runner.model
    return model


def prepare_quanted_weights_for_loading(model_runner, is_drafter=False):
    model = _get_quanted_weight_model(model_runner, is_drafter=is_drafter)
    quant_config = model_runner.vllm_config.quant_config

    is_mxfp8_npu = is_mxfp8_vllm_ascend(quant_config)
    is_mxfp4_moe = _restore_mxfp4_moe_params_for_loading(model)

    if is_mxfp8_npu:
        # For MXFP8 on NPU, we need to restore weights to original shapes
        # before loading, then re-apply transformation after loading.
        # This is because process_weights_after_loading transposes the weights,
        # but the weight_loader expects original shapes.
        restore_mxfp8_weights_for_loading(model)

    _ensure_model_params_reloadable(model)
    return {"is_mxfp8_npu": is_mxfp8_npu, "is_mxfp4_moe": is_mxfp4_moe}


def process_quanted_weights_after_loading(model_runner, reload_state=None, is_drafter=False):
    model = _get_quanted_weight_model(model_runner, is_drafter=is_drafter)
    quant_config = model_runner.vllm_config.quant_config
    if reload_state is None:
        reload_state = {
            "is_mxfp8_npu": is_mxfp8_vllm_ascend(quant_config),
            "is_mxfp4_moe": any(_is_mxfp4_moe_module(module) for module in model.modules()),
        }
    if reload_state.get("is_mxfp8_npu"):
        # Re-apply MXFP8 transformations after weight loading.
        apply_mxfp8_transformation_after_loading(model)
    if reload_state.get("is_mxfp4_moe"):
        _process_mxfp4_moe_weights_after_loading(model)
    _reload_cached_deepseek_v4_dense_fp8_scales(model)


def load_quanted_weights(weights, model_runner, is_drafter=False, prepare_model=True, process_model=True):
    model = _get_quanted_weight_model(model_runner, is_drafter=is_drafter)
    quant_config = model_runner.vllm_config.quant_config
    vllm_dtype = model_runner.vllm_config.model_config.dtype

    reload_state = None
    if prepare_model:
        reload_state = prepare_quanted_weights_for_loading(model_runner, is_drafter=is_drafter)

    if isinstance(weights, list):
        weights_for_load = weights
    else:
        weights_for_load = list(weights)
    _cache_deepseek_v4_dense_fp8_scales(model, weights_for_load)

    weights_quantized = quant_weights(weights_for_load, model, quant_config, dtype=vllm_dtype)

    # Monkey patch the param class to their subclass, as certain models
    # will check the param type to call the proper weightloader
    for _, param in model.named_parameters():
        if hasattr(param, "subclass_type"):
            param.orig_type = param.__class__
            param.__class__ = param.subclass_type
    # Finally load the weights into vllm
    try:
        loaded_params = model.load_weights(weights_quantized)
        _reload_cached_deepseek_v4_dense_fp8_scales(model)
    finally:
        # Undo the type change above to the original type
        for _, param in model.named_parameters():
            if hasattr(param, "orig_type"):
                param.__class__ = param.orig_type
                del param.orig_type

    if process_model:
        process_quanted_weights_after_loading(model_runner, reload_state, is_drafter=is_drafter)

    return loaded_params


def _copy_param_subclass_attrs(param, source_param):
    if source_param is None:
        return

    base_param_dir = dir(torch.nn.Parameter)
    source_param_dir = dir(source_param)
    custom_attributes = [attr for attr in source_param_dir if attr not in base_param_dir and not attr.startswith("__")]
    for attr in custom_attributes:
        try:
            setattr(param, attr, getattr(source_param, attr))
        except AttributeError:
            pass

    subclass_type = getattr(source_param, "subclass_type", type(source_param))
    if subclass_type is not torch.nn.Parameter:
        param.subclass_type = subclass_type


def replace_parameter_preserve_subclass(layer: torch.nn.Module, param_name: str, new_data: torch.Tensor | None):
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


def _make_process_weights_after_loading_for_vllm20(original_fn):
    def _patched_process_weights_after_loading(self, layer) -> None:
        old_params = dict(layer.named_parameters(recurse=False))
        with patch(
            "vllm.model_executor.layers.quantization.fp8.replace_parameter", replace_parameter_preserve_subclass
        ):
            original_fn(self, layer)
        _restore_layer_param_subclass_attrs(layer, old_params)

    return _patched_process_weights_after_loading


def process_weights_after_loading_for_vllm10(self, layer) -> None:
    """This function is used to process the weights after loading for a Linear layer, it is used for vllm v0.10

    Compared to the original process_weights_after_loading in vllm, we just avoid creation of
    new torch.nn.Parameter objects, because that removes the weight_loader attribute which we need for refit.
    """
    logger.debug("Applying patch process_weights_after_loading")
    from vllm.model_executor.parameter import (
        BlockQuantScaleParameter,
        ModelWeightParameter,
    )

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
    """This function is used to process the weights after loading for a Linear layer, it is used for vllm v0.14-v0.19.

    Compared to the original process_weights_after_loading in vllm, we just avoid creation of
    new torch.nn.Parameter objects, because that removes the weight_loader attribute which we need for refit.
    """
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        process_fp8_weight_block_strategy,
    )
    from vllm.model_executor.parameter import (
        BlockQuantScaleParameter,
        ModelWeightParameter,
    )

    assert self.block_quant and self.quant_config.is_checkpoint_fp8_serialized
    assert self.quant_config.activation_scheme == "dynamic"

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

    try:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import maybe_post_process_fp8_weight_block
    except ImportError:
        maybe_post_process_fp8_weight_block = None

    if maybe_post_process_fp8_weight_block is not None:
        maybe_post_process_fp8_weight_block(layer)
    elif hasattr(self, "fp8_linear"):
        self.fp8_linear.process_weights_after_loading(layer)
    _ensure_linear_params_reloadable(layer)


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

    layer.w13_weight = _create_param_from_data_with_attrs(w13_weight, layer.w13_weight)
    layer.w13_weight_scale_inv = _create_param_from_data_with_attrs(w13_weight_scale_inv, layer.w13_weight_scale_inv)
    layer.w2_weight = _create_param_from_data_with_attrs(w2_weight, layer.w2_weight)
    layer.w2_weight_scale_inv = _create_param_from_data_with_attrs(w2_weight_scale_inv, layer.w2_weight_scale_inv)

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
    # Replace parameters with updated versions. Note that this helper
    # function ensures the replacement is compatible with RL weight reloads.
    layer.w13_weight = _create_param_from_data_with_attrs(w13, layer.w13_weight)
    layer.w2_weight = _create_param_from_data_with_attrs(w2, layer.w2_weight)
    layer.w13_weight_scale_inv = _create_param_from_data_with_attrs(w13_scale, layer.w13_weight_scale_inv)
    layer.w2_weight_scale_inv = _create_param_from_data_with_attrs(w2_scale, layer.w2_weight_scale_inv)

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
    if fp8_state.vllm_patches:
        logger.debug("vLLM FP8 patches already applied")
        return

    logger.info("Applying vllm fp8 patches for blockwise quantization")
    vllm_ver = version.parse(vllm.__version__)

    func1_path = "vllm.model_executor.layers.quantization.fp8.Fp8LinearMethod.process_weights_after_loading"
    func2_path = "vllm.model_executor.layers.quantization.fp8.Fp8MoEMethod.process_weights_after_loading"

    # vLLM 0.20 refactored FP8 post-load handling and removed
    # maybe_post_process_fp8_weight_block. Keep its native transformation logic,
    # but preserve parameter subclass metadata for RL weight reloads.
    if vllm_ver >= version.parse("0.20.0"):
        from vllm.model_executor.layers.quantization.fp8 import (
            Fp8LinearMethod,
            Fp8MoEMethod,
        )

        patcher1 = patch(
            func1_path,
            _make_process_weights_after_loading_for_vllm20(Fp8LinearMethod.process_weights_after_loading),
        )
        patcher2 = patch(
            func2_path,
            _make_process_weights_after_loading_for_vllm20(Fp8MoEMethod.process_weights_after_loading),
        )
        patcher1.start()
        patcher2.start()
        fp8_state.vllm_patches.extend([patcher1, patcher2])
        return

    # Linear patch: v0.14+ keeps weight_scale_inv, v0.11-v0.12 renames to weight_scale
    if vllm_ver >= version.parse("0.14.0"):
        linear_patch_fn = process_weights_after_loading_for_vllm14
    elif vllm_ver >= version.parse("0.11.0"):
        linear_patch_fn = process_weights_after_loading_for_vllm11
    else:
        linear_patch_fn = process_weights_after_loading_for_vllm10
    patcher1 = patch(func1_path, linear_patch_fn)
    patcher1.start()

    # MoE patch
    if vllm_ver >= version.parse("0.14.0"):
        moe_patch_fn = process_weights_after_loading_moe_for_vllm14
    elif vllm_ver >= version.parse("0.11.0"):
        moe_patch_fn = process_weights_after_loading_moe_for_vllm11
    else:
        moe_patch_fn = process_weights_after_loading_moe_for_vllm10
    patcher2 = patch(func2_path, moe_patch_fn)
    patcher2.start()
    fp8_state.vllm_patches.extend([patcher1, patcher2])
