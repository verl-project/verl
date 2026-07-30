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

from types import MethodType

import torch
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.linear import LinearBase


def is_deepseek_v4_model(model):
    if model is None:
        return False

    for obj in (model, getattr(model, "config", None), getattr(model, "hf_config", None)):
        if obj is not None and getattr(obj, "model_type", None) is not None:
            return obj.model_type == "deepseek_v4"

    text_config = getattr(getattr(model, "config", None), "text_config", None)
    return getattr(text_config, "model_type", None) == "deepseek_v4"


def iter_deepseek_v4_weights(weights):
    for name, weight in weights:
        if ".experts." in name and weight.dtype in (torch.int8, torch.float8_e8m0fnu):
            weight = weight.view(torch.uint8)
        yield name, weight


def _is_mega_moe_module(module):
    from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MegaMoEExperts

    return isinstance(module, DeepseekV4MegaMoEExperts)


def _is_mxfp4_fused_moe_module(module):
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod

    return isinstance(module, FusedMoE) and isinstance(module.quant_method, Mxfp4MoEMethod)


def _make_mxfp4_moe_param(shape, device, weight_loader, quant_method=None):
    param = torch.nn.Parameter(torch.empty(shape, dtype=torch.uint8, device=device), requires_grad=False)
    param.weight_loader = weight_loader
    if quant_method is not None:
        param.quant_method = quant_method
    return param


def _wrap_vllm_param(custom_param, source_param, copy_param_subclass_attrs):
    param = torch.nn.Parameter(custom_param.data, requires_grad=False)
    copy_param_subclass_attrs(param, source_param)
    copy_param_subclass_attrs(param, custom_param)
    return param


def _get_param_weight_loader(param):
    return getattr(param, "weight_loader", None) or getattr(param, "_weight_loader", None)


def _param_parallel_dim(param, public_name, private_name, default):
    if hasattr(param, private_name):
        return getattr(param, private_name)
    return getattr(param, public_name, default)


def _copy_loaded_weight(param, loaded_weight):
    param.data.copy_(loaded_weight.to(device=param.device, dtype=param.dtype))


def _try_load_column_weight(param, loaded_weight):
    tp_size = int(getattr(param, "tp_size", 1))
    tp_rank = int(getattr(param, "tp_rank", 0))
    if tp_size <= 1:
        return False

    data = param.data
    if loaded_weight.ndim == data.ndim + 1 and loaded_weight.shape == torch.Size((tp_size, *data.shape)):
        _copy_loaded_weight(param, loaded_weight.select(0, tp_rank))
        return True

    if data.ndim >= 2 and loaded_weight.ndim == data.ndim - 1:
        expected_shape = (tp_size * data.shape[0] * data.shape[1], *data.shape[2:])
        if loaded_weight.shape == torch.Size(expected_shape):
            global_shape = (tp_size, data.shape[0], data.shape[1], *data.shape[2:])
            _copy_loaded_weight(param, loaded_weight.reshape(global_shape).select(0, tp_rank))
            return True

    if data.ndim == loaded_weight.ndim and data.ndim > 0:
        expected_shape = (tp_size * data.shape[0], *data.shape[1:])
        if loaded_weight.shape == torch.Size(expected_shape):
            _copy_loaded_weight(param, loaded_weight.narrow(0, tp_rank * data.shape[0], data.shape[0]))
            return True

    return False


def _try_load_merged_weight(param, loaded_weight, shard_offset, shard_size, shard_id):
    output_dim = int(getattr(param, "output_dim", getattr(param, "_output_dim", 0)))
    loaded_dim = loaded_weight.shape[output_dim]
    offsets = []
    if isinstance(shard_offset, int):
        offsets.append(shard_offset)
        if isinstance(shard_size, int) and shard_size > 0:
            offsets.append(shard_offset * loaded_dim // shard_size)
    if isinstance(shard_id, int):
        offsets.append(shard_id * loaded_dim)

    for offset in dict.fromkeys(offsets):
        if offset < 0 or offset + loaded_dim > param.shape[output_dim]:
            continue
        target = param.data.narrow(output_dim, offset, loaded_dim)
        if target.shape == loaded_weight.shape:
            target.copy_(loaded_weight.to(device=target.device, dtype=target.dtype))
            return True
    return False


def _attach_weight_loaders(param):
    subclass_type = getattr(param, "subclass_type", None)
    if subclass_type is None or getattr(param, "_verl_deepseek_v4_loaders", False):
        return

    original_column_loader = getattr(subclass_type, "load_column_parallel_weight", None)
    if original_column_loader is not None:

        def load_column_parallel_weight(self, *args, **kwargs):
            loaded_weight = kwargs.get("loaded_weight", args[0] if args else None)
            if loaded_weight is not None:
                if self.shape == loaded_weight.shape:
                    _copy_loaded_weight(self, loaded_weight)
                    return
                if _try_load_column_weight(self, loaded_weight):
                    return
            return original_column_loader(self, *args, **kwargs)

        param.load_column_parallel_weight = MethodType(load_column_parallel_weight, param)

    original_merged_loader = getattr(subclass_type, "load_merged_column_weight", None)
    if original_merged_loader is not None:

        def load_merged_column_weight(self, *args, **kwargs):
            loaded_weight = kwargs.get("loaded_weight", args[0] if args else None)
            shard_id = kwargs.get("loaded_shard_id", kwargs.get("shard_id", args[1] if len(args) > 1 else None))
            shard_offset = kwargs.get("shard_offset", args[2] if len(args) > 2 else None)
            shard_size = kwargs.get("shard_size", args[3] if len(args) > 3 else None)
            if loaded_weight is not None and _try_load_merged_weight(
                self, loaded_weight, shard_offset, shard_size, shard_id
            ):
                return
            return original_merged_loader(self, *args, **kwargs)

        param.load_merged_column_weight = MethodType(load_merged_column_weight, param)

    param._verl_deepseek_v4_loaders = True


def _prepare_linear_params_for_loading(model, copy_param_subclass_attrs):
    from vllm.model_executor.parameter import BlockQuantScaleParameter, ModelWeightParameter

    for layer in model.modules():
        if not isinstance(layer, LinearBase):
            continue

        for name, param_type in (
            ("weight", ModelWeightParameter),
            ("weight_scale_inv", BlockQuantScaleParameter),
            ("weight_scale", BlockQuantScaleParameter),
        ):
            param = getattr(layer, name, None)
            if not isinstance(param, torch.nn.Parameter):
                continue
            if not hasattr(param, "subclass_type"):
                weight_loader = _get_param_weight_loader(param)
                if weight_loader is None:
                    continue
                param = _wrap_vllm_param(
                    param_type(
                        data=param.data,
                        output_dim=_param_parallel_dim(param, "output_dim", "_output_dim", 0),
                        input_dim=_param_parallel_dim(param, "input_dim", "_input_dim", 1),
                        weight_loader=weight_loader,
                    ),
                    param,
                    copy_param_subclass_attrs,
                )
                setattr(layer, name, param)

        update_param_tp_status = getattr(layer, "update_param_tp_status", None)
        if callable(update_param_tp_status):
            update_param_tp_status()

        for name in ("weight", "weight_scale_inv", "weight_scale"):
            param = getattr(layer, name, None)
            if param is not None:
                _attach_weight_loaders(param)


def _restore_moe_params_for_loading(model):
    restored = False
    for module in model.modules():
        if _is_mega_moe_module(module):
            num_experts = module.num_local_experts
            intermediate_size = module.intermediate_size
            hidden_size = module.hidden_size
            device = module._transformed_l1_weights[0].device
        elif _is_mxfp4_fused_moe_module(module):
            quant_method = module.quant_method
            num_experts = quant_method.num_experts
            intermediate_size = quant_method.intermediate_size
            hidden_size = quant_method.hidden_size
            device = module.w13_weight.device
        else:
            continue

        weight_loader = module.weight_loader
        module.w13_weight = _make_mxfp4_moe_param(
            (num_experts, 2 * intermediate_size, hidden_size // 2), device, weight_loader
        )
        module.w2_weight = _make_mxfp4_moe_param(
            (num_experts, hidden_size, intermediate_size // 2), device, weight_loader
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
        restored = True
    return restored


def _process_moe_weights_after_loading(model):
    for module in model.modules():
        if _is_mega_moe_module(module):
            module._transformed_l1_weights = None
            module._transformed_l2_weights = None
            module.finalize_weights()
        elif _is_mxfp4_fused_moe_module(module):
            module.quant_method.process_weights_after_loading(module)


def prepare_deepseek_v4_weights_for_loading(model, copy_param_subclass_attrs):
    _prepare_linear_params_for_loading(model, copy_param_subclass_attrs)
    return _restore_moe_params_for_loading(model)


def process_deepseek_v4_weights_after_loading(model, moe_params_restored):
    if moe_params_restored:
        _process_moe_weights_after_loading(model)
    reload_deepseek_v4_dense_fp8_scales(model)


def _normalize_dim(dim: int, ndim: int) -> int:
    return dim + ndim if dim < 0 else dim


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
    if mapped.startswith(("layers.", "embed.")):
        mapped = "model." + mapped
    elif mapped == "head.weight":
        mapped = "lm_head.weight"
    return mapped


def cache_deepseek_v4_dense_fp8_scales(model, weights):
    if not is_deepseek_v4_model(model):
        return

    scale_dtype = getattr(torch, "float8_e8m0fnu", None)
    if scale_dtype is None:
        return

    cache = getattr(model, "_verl_dense_fp8_scale_cache", None)
    if cache is None:
        cache = {}
        model._verl_dense_fp8_scale_cache = cache

    for name, tensor in weights:
        if name.endswith(".scale") and ".experts." not in name and tensor.dtype == scale_dtype:
            cache[_map_weight_name_for_vllm(model, name)] = tensor.detach().clone()


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


def reload_deepseek_v4_dense_fp8_scales(model):
    cache = getattr(model, "_verl_dense_fp8_scale_cache", None)
    if not cache:
        return

    params = dict(model.named_parameters())
    for name, scale in cache.items():
        param = params.get(name)
        if param is not None:
            _copy_scale_shard(param, scale)
