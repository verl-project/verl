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

import functools
import itertools
import json
import math
import os
from abc import ABC
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from typing import cast

import torch
import torch.distributed as dist
import torch.nn as nn
from packaging import version
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._runtime_utils import _lazy_init
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
from transformers.trainer_pt_utils import get_module_class_from_name

from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.model import check_exclude_modules, check_target_modules

if version.parse(torch.__version__) >= version.parse("2.6"):
    from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard
    from torch.distributed.fsdp._fully_shard._fsdp_init import _get_post_forward_mesh_info
    from torch.distributed.tensor import Shard

    fully_shard_module = torch.distributed.fsdp._fully_shard._fully_shard
elif version.parse(torch.__version__) >= version.parse("2.4"):
    from torch.distributed._composable.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard

    fully_shard_module = torch.distributed._composable.fsdp
else:
    fully_shard, MixedPrecisionPolicy, FSDPModule, CPUOffloadPolicy, fully_shard_module = None, None, None, None, None


def init_fn(x: torch.nn.Module):
    if torch.distributed.get_rank() != 0:
        x = x.to_empty(device=get_device_id(), recurse=False)
        get_torch_device().empty_cache()
    return x


def get_init_weight_context_manager(use_meta_tensor=True, mesh: DeviceMesh = None):
    from accelerate import init_empty_weights

    cpu_init_weights = lambda: torch.device("cpu")
    if use_meta_tensor:
        if mesh is None:
            init_context = init_empty_weights if torch.distributed.get_rank() != 0 else cpu_init_weights
        else:
            init_context = init_empty_weights if mesh.get_coordinate()[-1] != 0 else cpu_init_weights
    else:
        init_context = cpu_init_weights
    return init_context


# Copyright 2020-present the HuggingFace Inc. team.
# Adapted from https://github.com/huggingface/transformers/src/transformers/trainer.py
def get_fsdp_wrap_policy(module, config=None, is_lora=False):
    """Get FSDP wrap policy for the module.

    Args:
        module: The module to get wrap policy for
        config: Configuration for wrap policy
        is_lora: Whether to enable lambda policy for LoRA modules
    """
    if config is None:
        config = {}

    # NOTE: This is a temporary workaround to be compatible with the OmegaConf & dataclass. We will remove this
    # once we have make all config in verl from OmegaConf to data class.
    def _get_attr(attr_name, default_value=None):
        if hasattr(config, "get"):
            return config.get(attr_name, default_value)
        else:
            return config.__getattribute__(attr_name)

    if _get_attr("disable", False):
        return None

    default_transformer_cls_names_to_wrap = getattr(module, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = _get_attr(
        "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
    )
    min_num_params = _get_attr("min_num_params", 0)
    auto_wrap_policy = None

    policies = []

    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy

    # Add lambda policy for LoRA modules if is_lora is True
    if is_lora:

        def lambda_policy_fn(module):
            return bool(
                len(list(module.named_children())) == 0
                and getattr(module, "weight", None) is not None
                and module.weight.requires_grad
            )

        lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
        policies.append(lambda_policy)

    if min_num_params > 0:
        size_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
        policies.append(size_policy)
    elif fsdp_transformer_layer_cls_to_wrap is not None:
        transformer_cls_to_wrap = set()
        for layer_class in fsdp_transformer_layer_cls_to_wrap:
            transformer_cls = get_module_class_from_name(module, layer_class)
            if transformer_cls is None:
                raise Exception("Could not find the transformer layer class to wrap in the model.")
            else:
                transformer_cls_to_wrap.add(transformer_cls)

        transformer_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_cls_to_wrap,
        )
        policies.append(transformer_policy)

    if len(policies) > 0:
        auto_wrap_policy = functools.partial(_or_policy, policies=policies)

    return auto_wrap_policy


@torch.no_grad()
def offload_fsdp_model_to_cpu(model: FSDP, empty_cache: bool = True):
    if fsdp_version(model) == 2:
        offload_fsdp2_model_to_cpu(model, empty_cache)
        return

    assert isinstance(model, FSDP)
    # lazy init FSDP model
    _lazy_init(model, model)
    assert model._is_root, "Only support root model offloading to CPU"
    for handle in model._all_handles:
        if handle._offload_params:
            continue
        flat_param = handle.flat_param
        assert (
            flat_param.data.data_ptr() == flat_param._local_shard.data_ptr()
            and id(flat_param.data) != id(flat_param._local_shard)
            and flat_param.data.size() == flat_param._local_shard.size()
        )
        handle.flat_param_to(torch.device("cpu"), non_blocking=True)
        # the following still keeps id(._local_shard) != id(.data)
        flat_param._local_shard = flat_param.data
        assert id(flat_param._local_shard) != id(flat_param.data)
    if empty_cache:
        get_torch_device().empty_cache()


@torch.no_grad()
def offload_fsdp2_model_to_cpu(model, empty_cache: bool = True):
    model.cpu()
    if empty_cache:
        get_torch_device().empty_cache()


@torch.no_grad()
def load_fsdp_model_to_gpu(model: FSDP):
    if fsdp_version(model) == 2:
        load_fsdp2_model_to_gpu(model)
        return

    assert isinstance(model, FSDP)
    # lazy init FSDP model
    _lazy_init(model, model)
    assert model._is_root, "Only support root model loading to GPU"
    device_id = get_device_id()
    for handle in model._all_handles:
        if handle._offload_params:
            continue
        flat_param = handle.flat_param
        handle.flat_param_to(torch.device(f"{get_device_name()}:{device_id}"), non_blocking=True)
        # the following still keeps id(._local_shard) != id(.data)
        flat_param._local_shard = flat_param.data


@torch.no_grad()
def load_fsdp2_model_to_gpu(model):
    device = get_device_id()
    model.to(device)


@torch.no_grad()
def offload_fsdp_optimizer(optimizer):
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)


@torch.no_grad()
def load_fsdp_optimizer(optimizer, device_id):
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device_id, non_blocking=True)


@contextmanager
def meta_device_init():
    """
    Create model parameters with meta device.

    Note buffers in model will still be initialized in default device (e.g., CPU),
    since the buffers can be non-persistent and filled with expected values that can
    NOT be captured in meta device.
    """
    device = torch.device("meta")
    old_register_parameter = nn.Module.register_parameter
    registered = set()

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        # we will skip register shared parameters as it
        # is already registered previously
        if param is not None and param not in registered:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)
            registered.add(module._parameters[name])

    try:
        nn.Module.register_parameter = register_empty_parameter
        yield
    finally:
        registered.clear()
        nn.Module.register_parameter = old_register_parameter


def parallel_load_safetensors(filepath):
    """
    Parallel load safetensors from huggingface checkpoint

    Huggingface checkpoint contains:

    - config.json: a json file for model configuration
    - model.safetensor.index.json: a json file for safetensors (parameters & buffers) index
    - model-000x-of-ooxx.safetensors: a binary file for safetensors (parameters & buffers) chunks

    Or (when model is small),

    - model.safetensors: a binary file for all parameters and buffers

    Each rank will own a part of model chunks and load them directly into GPU memory.
    """
    from safetensors.torch import load_file

    safetensors2param = {}

    index_file = os.path.join(filepath, "model.safetensors.index.json")
    if os.path.exists(index_file):
        index = json.load(open(index_file, "rb"))
        for param_name, filename in index["weight_map"].items():
            safetensors2param.setdefault(filename, []).append(param_name)
    else:
        # in this case, the model is small and we can load it all at once
        param_file = os.path.join(filepath, "model.safetensors")
        assert os.path.exists(param_file), f"Cannot find {param_file}"
        states = load_file(param_file)
        for param_name in states:
            safetensors2param.setdefault("model.safetensors", []).append(param_name)
        del states

    total_files = len(safetensors2param)
    ckpt_chunks = sorted(safetensors2param.keys())
    world_size = dist.get_world_size()
    size = int(math.ceil(total_files / world_size))
    ckpt_chunks = [ckpt_chunks[rank * size : rank * size + size] for rank in range(world_size)]

    shard_states = {}
    device = get_device_id()
    for rank, files in enumerate(ckpt_chunks):
        if rank == dist.get_rank():
            for file in files:
                file = os.path.join(filepath, file)
                states = load_file(file, device=device)
                # print(f"rank {rank} loading {file}...")
                shard_states.update(states)
        else:
            for file in files:
                for param_name in safetensors2param[file]:
                    shard_states[param_name] = rank
    return shard_states


def parallel_init_module_fn(module: torch.nn.Module, shard_states: dict[str, torch.nn.Parameter]):
    """
    Generate a function to initialize sub-modules in the `module` with `shard_states`
    from huggingface checkpoint.

    Args:
        module (torch.nn.Module): the global module to be initialized
        shard_states (Dict[str, torch.nn.Parameter]): the shard states from huggingface checkpoint

    Returns:
        init_fn (Callable): a function to initialize sub-modules in the `module` with `shard_states`
    """

    state2fqn = {}
    for name, state in itertools.chain(
        module.named_parameters(remove_duplicate=False), module.named_buffers(remove_duplicate=False)
    ):
        state2fqn.setdefault(state, []).append(name)
    # remove standalone parameters and buffers
    shared = {s for s, names in state2fqn.items() if len(names) > 1}
    materialized_states = {}

    @torch.no_grad()
    def create_and_sync_state(param_name, state, is_param):
        assert param_name in shard_states, f"{param_name} not loaded"
        device = get_device_id()
        if is_param:
            param = torch.nn.Parameter(torch.empty_like(state.data, device=device), requires_grad=state.requires_grad)
        else:  # buffer
            param = torch.empty_like(state.data, device=device)
        loaded = shard_states[param_name]
        if isinstance(loaded, torch.nn.Parameter | torch.Tensor):
            # NOTE: loaded.dtype can be different with param.dtype
            param.data.copy_(loaded.data)
            dist.broadcast(param.data, src=dist.get_rank())
        else:
            assert isinstance(loaded, int)  # the rank that holds the state
            dist.broadcast(param.data, src=loaded)
        shard_states.pop(param_name)
        del loaded
        return param

    def init_fn(sub_mod: torch.nn.Module, recurse: bool = True):
        param_and_buffers = tuple(sub_mod.named_parameters(recurse=False)) + tuple(sub_mod.named_buffers(recurse=False))
        # param_and_buffers = sorted(sub_mod.named_parameters(recurse=False), key=lambda x: x[0])
        for name, state in param_and_buffers:
            if not state.is_meta:
                continue
            is_param = name in sub_mod._parameters
            fqn = state2fqn[state].pop(0)
            # non-persistent buffers will not be saved in state dict, we can safely skip it
            if (not is_param) and fqn not in shard_states:
                if state.is_meta:
                    raise RuntimeError(
                        f"find a non-persistent buffer ({fqn}) initiated with device meta. Such buffer is not saved "
                        f"in checkpoint and user should guarantee to init in CPU / GPU device."
                    )
                continue
            # for shared parameter, we get it from the first time it is created
            if state in shared:
                if state not in materialized_states:
                    materialized_states[state] = create_and_sync_state(fqn, state, is_param)
                else:
                    if fqn in shard_states:
                        shard_states.pop(fqn)
                materialize_state = materialized_states[state]
            # for not shared parameter, we create it directly
            else:
                materialize_state = create_and_sync_state(fqn, state, is_param)
            if is_param:
                sub_mod._parameters[name] = materialize_state
            else:
                sub_mod._buffers[name] = materialize_state
        if recurse:
            for module in sub_mod.children():
                init_fn(module, recurse=True)

        # for debug
        # if len(shard_states) == 0: print("clear")
        return sub_mod

    return init_fn


def fsdp_version(model):
    if isinstance(model, FSDP):
        return 1
    elif isinstance(model, FSDPModule):
        return 2
    else:
        return 0


def get_fsdp_state_ctx(model, state_type, state_cfg, optim_cfg):
    if fsdp_version(model) == 1:
        return FSDP.state_dict_type(model, state_type, state_cfg, optim_cfg)
    else:
        return nullcontext()


def get_fsdp_full_state_dict(model: torch.nn.Module, offload_to_cpu: bool = True, rank0_only: bool = True):
    """
    Get the full state dict from an FSDP model.

    Args:
        model (torch.nn.Module): The FSDP model to get state dict from
        offload_to_cpu (bool, optional): Whether to offload the state dict to CPU. Defaults to True.
        rank0_only (bool, optional): Whether to only get state dict on rank 0. Defaults to True.

    Returns:
        dict: The full state dict of the model

    Raises:
        NotImplementedError: If the FSDP version is unknown
    """
    if fsdp_version(model) == 1:
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        state_dict_config = FullStateDictConfig(offload_to_cpu=offload_to_cpu, rank0_only=rank0_only)
        with get_fsdp_state_ctx(
            model, state_type=StateDictType.FULL_STATE_DICT, state_cfg=state_dict_config, optim_cfg=None
        ):
            state_dict = model.state_dict()
        return state_dict
    elif fsdp_version(model) == 2:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        state_dict_config = StateDictOptions(
            full_state_dict=True, cpu_offload=offload_to_cpu, broadcast_from_rank0=not rank0_only
        )
        state_dict = get_model_state_dict(model, options=state_dict_config)
        return state_dict
    else:
        raise NotImplementedError(f"Unknown FSDP version {fsdp_version}")


def fsdp2_load_full_state_dict(model: torch.nn.Module, full_state: dict, device_mesh=None, cpu_offload=None):
    """
    Loads the full state dict (could be only on rank 0) into the sharded model. This is done by broadcasting the
    parameters from rank 0 to all other ranks. This function modifies the model in-place.

    Args:
        model (`torch.nn.Module`): The model to load the state dict into
        full_state (`dict`): The full state dict to load, can only be on rank 0
    """

    if version.parse(torch.__version__) >= version.parse("2.7.0"):
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
    else:
        # official torch 2.6.0 set_model_state_dict API leads to OOM
        # use torch 2.7.0 copy from verl/third_party/torch/distributed/checkpoint
        from verl.third_party.torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    # To broadcast, it needs to be instantiated in the GPU.
    if dist.get_rank() == 0:
        model = model.to(device=get_device_id(), non_blocking=True)
    else:
        model = model.to_empty(device=get_device_id())

    cpu_offload = cpu_offload is not None
    options = StateDictOptions(full_state_dict=True, cpu_offload=cpu_offload, broadcast_from_rank0=True)
    set_model_state_dict(model, full_state, options=options)

    # rotary_emb is not in state_dict, so we need to broadcast it manually
    for name, buf in model.named_buffers():
        dist.broadcast(buf, src=0)

    if cpu_offload:
        model.to("cpu", non_blocking=True)
        for buf in model.buffers():
            buf.data = buf.data.to(get_device_id())


@contextmanager
def maybe_patch_fsdp_module(model):
    if fully_shard_module is None:
        yield
        return

    orig_fsdp_module = fully_shard_module.FSDPModule

    class FSDPModuleABC(ABC, orig_fsdp_module):
        pass

    try:
        if isinstance(model, ABC):
            fully_shard_module.FSDPModule = FSDPModuleABC
        yield
    finally:
        fully_shard_module.FSDPModule = orig_fsdp_module


def apply_fsdp2(model, fsdp_kwargs, config):
    """model: AutoModelForCausalLM"""
    assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"

    default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = config.get("wrap_policy", {}).get(
        "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
    )

    if isinstance(fsdp_transformer_layer_cls_to_wrap, str):
        fsdp_transformer_layer_cls_to_wrap = [fsdp_transformer_layer_cls_to_wrap]

    assert len(fsdp_transformer_layer_cls_to_wrap) > 0 and fsdp_transformer_layer_cls_to_wrap[0] is not None

    modules = []
    for name, module in model.named_modules():
        if module.__class__.__name__ in fsdp_transformer_layer_cls_to_wrap or (
            isinstance(module, nn.Embedding) and not model.config.tie_word_embeddings
        ):
            modules.append(module)

    for idx, module in enumerate(modules):
        # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        #     print(f"wrap module {module.__class__.__name__}")
        with maybe_patch_fsdp_module(module):
            fully_shard(module, **fsdp_kwargs)

    # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
    #     print(f"wrap module {model.__class__.__name__}")
    with maybe_patch_fsdp_module(model):
        fully_shard(model, **fsdp_kwargs)  # fsdp2 will not reshard_after_forward for root module


def get_shard_placement_fn(fsdp_size):
    """Choose the dimension that can divide fsdp_size to avoid padding"""

    def shard_placement_fn(param):
        shape = list(param.shape)
        for i in range(len(shape)):
            if shape[i] % fsdp_size == 0:
                return Shard(i)
        return Shard(0)

    return shard_placement_fn


def fsdp2_clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):
    """torch.nn.utils.clip_grad_norm_ cann't run on cpu parameter DTensor"""
    from torch.nn.utils.clip_grad import _clip_grads_with_norm_, _get_total_norm

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        # prevent generators from being exhausted
        parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = _get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
    total_norm = total_norm.to(get_device_id(), non_blocking=True)
    _clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm


def layered_summon_lora_params(fsdp_module) -> OrderedDict:
    from peft.utils.save_and_load import get_peft_model_state_dict

    def __prefix_submodules(module, prefix):
        for name, submodule in module.named_modules():
            if name.startswith(prefix) and "." not in name[len(prefix) :]:
                yield name, submodule

    lora_params = OrderedDict()
    prefix_list = [
        # fsdp
        "_fsdp_wrapped_module.base_model.model.",
        "_fsdp_wrapped_module.base_model.model.model.",
        "_fsdp_wrapped_module.base_model.model.model.layers.",
        "_fsdp_wrapped_module.base_model.model.model.language_model.layers.",
        # fsdp2
        "base_model.model.",
        "base_model.model.model.",
        "base_model.model.model.layers.",
        "base_model.model.model.language_model.layers.",
    ]
    peft_model = getattr(fsdp_module, "_fsdp_wrapped_module", fsdp_module)
    for prefix in prefix_list:
        for name, submodule in __prefix_submodules(fsdp_module, prefix):
            prefix = name.replace("_fsdp_wrapped_module.base_model.model.", "base_model.model.")
            if name.endswith(".model") or name.endswith(".layers"):
                continue
            if fsdp_version(submodule) > 0:
                with FSDP.summon_full_params(submodule, writeback=False):
                    sub_lora_params = get_peft_model_state_dict(peft_model, state_dict=submodule.state_dict())
                    sub_lora_params = {
                        f"{prefix}.{name}": param.full_tensor().detach().cpu()
                        if hasattr(param, "full_tensor")
                        else param.detach().cpu()
                        for name, param in sub_lora_params.items()
                    }
                    lora_params.update(sub_lora_params)
                    submodule._is_root = False
                get_torch_device().empty_cache()
    return lora_params


def collect_lora_params(module: FSDP, layered_summon: bool, base_sync_done: bool) -> OrderedDict:
    """
    collect lora params or full params if base model is not ready in vllm
    work with if isinstance(self.module._fsdp_wrapped_module, PeftModel)
    """
    from peft.utils.save_and_load import get_peft_model_state_dict

    lora_params = OrderedDict()
    peft_model = getattr(module, "_fsdp_wrapped_module", module)
    if fsdp_version(module) > 0:
        if layered_summon:
            if not base_sync_done:
                raise ValueError(
                    "To use layered_summon, you must make sure base-model is preloaded in vllm, e.g. let "
                    "rollout.load_format=safetensors"
                )
            lora_params = layered_summon_lora_params(module)
        else:
            with FSDP.summon_full_params(module, writeback=False):
                if base_sync_done:
                    lora_params = get_peft_model_state_dict(peft_model)
                    lora_params = {
                        name: param.full_tensor().detach().cpu()
                        if hasattr(param, "full_tensor")
                        else param.detach().cpu()
                        for name, param in lora_params.items()
                    }
                else:
                    model = peft_model.base_model.model
                    orig_dev = "cpu" if "cpu" in str(next(model.parameters()).device) else get_device_name()
                    model = model.to("cpu")
                    for name, param in model.state_dict().items():
                        if any(x in name for x in ["_flat_param", "lora_"]):
                            continue
                        name = name.replace("_fsdp_wrapped_module.", "").replace(".base_layer", "")
                        lora_params[name] = (
                            param.full_tensor().detach().cpu()
                            if hasattr(param, "full_tensor")
                            else param.detach().cpu()
                        )
                    model = model.to(orig_dev)
            get_torch_device().empty_cache()
    else:
        if base_sync_done:
            lora_params = get_peft_model_state_dict(peft_model)
        else:
            model = peft_model.base_model.model
            orig_dev = "cpu" if "cpu" in str(next(model.parameters()).device) else get_device_name()
            model = model.to("cpu")
            for name, param in model.state_dict().items():
                if any(x in name for x in ["_flat_param", "lora_"]):
                    continue
                name = name.replace("_fsdp_wrapped_module.", "").replace(".base_layer", "")
                lora_params[name] = param.detach().cpu()
            model = model.to(orig_dev)
    return lora_params


def replace_lora_wrapper(k, peft_config):
    """Replace LoRA parameter keys with base layer equivalents.

    Transforms LoRA parameter names to their corresponding base layer
    names for proper weight loading in vLLM when base model sync is not done.

    Args:
        k (str): Original parameter key name.

    Returns:
        str: Transformed parameter key for base layer.
    """
    stacked_params = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if k.endswith(".weight"):
        module_k = k[: -len(".weight")]
        if check_exclude_modules(peft_config, module_k):
            return k
        elif any([module_k.endswith(s) for s in stacked_params]) or check_target_modules(peft_config, module_k):
            return f"{module_k}.base_layer.weight"
    if k.endswith(".bias"):
        module_k = k[: -len(".bias")]
        if check_exclude_modules(peft_config, module_k):
            return k
        elif any([module_k.endswith(s) for s in stacked_params]) or check_target_modules(peft_config, module_k):
            return f"{module_k}.base_layer.bias"
    return k


def set_reshard_after_forward(module: FSDPModule, reshard_after_forward: bool, recurse: bool = True) -> None:
    """
    Sets if the module should reshard parameters after forward. This can be
    used to change the ``reshard_after_forward`` FSDP arg at runtime. For
    example, this can be used to set the FSDP root module's value to
    ``True`` (since it is otherwise specially set to ``False``), or it can
    set an FSDP module's value to ``False`` for running evals and set back
    to ``True`` for training.

    Args:
        reshard_after_forward (bool): Whether to reshard parameters after
            forward.
        recurse (bool): Whether to set for all FSDP submodules or just the
            passed-in module.

    ---
    Copied from https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_fully_shard/_fully_shard.py to
    address the absence of the set_reshard_after_forward function in torch versions earlier than 2.8.0.
    """

    if not isinstance(reshard_after_forward, bool):
        raise ValueError(f"reshard_after_forward should be a bool, got {type(reshard_after_forward)}")
    self_module = cast(nn.Module, module)
    modules = list(self_module.modules()) if recurse else [self_module]
    for module in modules:
        if isinstance(module, FSDPModule):
            state = module._get_fsdp_state()
            state._auto_reshard_after_forward = False
            if fsdp_param_group := state._fsdp_param_group:
                fsdp_param_group.post_forward_mesh_info = _get_post_forward_mesh_info(
                    reshard_after_forward, fsdp_param_group.mesh_info
                )


def normalize_peft_param_name(params: dict) -> dict:
    """
    Converts peft model parameter name to base parameter name
    For example,
        base_model.model.model.embed_tokens.weight -> model.embed_tokens.weight
        base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight -> model.layers.0.self_attn.q_proj.weight
    and remove params such as base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight,
    base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight
    """

    def _normalize_peft_name(name: str) -> str:
        return name.replace("base_model.model.", "").replace("base_model.", "").replace(".base_layer", "")

    def _is_lora_key(name: str) -> bool:
        # catch typical PEFT keys
        return ("lora_" in name) or (".adapter_" in name)

    params = [(_normalize_peft_name(k), v) for k, v in params.items()]
    # strip any residual LoRA tensors
    params = {k: v for k, v in params if not _is_lora_key(k)}
    return params


def _merge_or_unmerge_lora_(module, merge: bool):
    """Merge or unmerge LoRA adapters in a module.

    Args:
        module: The module containing LoRA layers
        merge: If True, merge LoRA into base model; if False, unmerge LoRA
    """
    from peft.tuners.lora import LoraLayer

    with torch.no_grad():
        for m in module.modules():
            if isinstance(m, LoraLayer):
                is_merged = getattr(m, "merged", False)
                if merge and not is_merged:
                    m.merge()
                elif (not merge) and is_merged:
                    m.unmerge()


# merged_adapters
def _clean_merged_lora_(module):
    """Cleans the merged lora adapters"""
    from peft.tuners.lora import LoraLayer

    with torch.no_grad():
        for m in module.modules():
            if isinstance(m, LoraLayer):
                merged_adapters = getattr(m, "merged_adapters", False)
                if merged_adapters:
                    m.merged_adapters = []


def fsdp_merge_unmerge(module: nn.Module, do_merge: bool):
    """Merge or unmerge LoRA adapters in FSDP module.

    For FSDP (v1), it gathers all model parameters to each device, which may cause OOM.
    For FSDP2, it gathers model parameters layer-by-layer to reduce memory footprint.

    Args:
        module: The FSDP module to merge/unmerge LoRA adapters
        do_merge: If True, merge LoRA into base model; if False, unmerge LoRA
    """
    version = fsdp_version(module)
    assert version in [1, 2], f"fsdp_merge_unmerge requires FSDP module, got version {version}"

    if version == 1:
        # Unshard → merge → Reshard
        with FSDP.summon_full_params(module, writeback=True, with_grads=False):
            _merge_or_unmerge_lora_(module, merge=do_merge)
    else:
        # FSDP2: Unshard → merge → Reshard layer-by-layer
        for name, submodule in module.named_modules():
            if isinstance(submodule, FSDPModule) and name != "":  # skip root model
                with FSDP.summon_full_params(submodule, writeback=True, with_grads=False):
                    _merge_or_unmerge_lora_(submodule, merge=do_merge)


def backup_base_model_weights(module):
    """Backup base model weights to CPU with LoRA temporarily disabled.

    This function temporarily disables LoRA adapters, backs up the clean base model weights
    to CPU, then re-enables the adapters.

    Args:
        module: The PEFT model with LoRA adapters

    Returns:
        dict: Dictionary mapping parameter name to CPU tensor backup of base model weights
    """
    from peft import PeftModel

    backup = {}
    with torch.no_grad():
        # Check if module is a PEFT model
        if isinstance(module, PeftModel):
            # Temporarily disable adapters to get clean base model weights
            with module.disable_adapter():
                # Backup base model weights (excluding lora parameters)
                for name, param in module.named_parameters():
                    if "lora" not in name.lower():
                        backup[name] = param.data.clone().cpu()
        else:
            # For non-PEFT models, just backup all parameters
            for name, param in module.named_parameters():
                backup[name] = param.data.clone().cpu()
    return backup


def restore_base_model_weights(module, backup):
    """Restore base model weights from CPU backup.

    This function restores the base model weights from the CPU backup, effectively
    undoing any LoRA merge operations.

    Args:
        module: The PEFT model with LoRA adapters
        backup: Dictionary mapping parameter name to CPU tensor backup of base model weights
    """
    with torch.no_grad():
        for name, param in module.named_parameters():
            if name in backup:
                param.data.copy_(backup[name].to(param.device))


@contextmanager
def merged_lora_context(actor, backup_adapters=False):
    """Context manager to temporarily merge LoRA adapters.

    This context manager merges LoRA adapters into the base model weights,
    performs operations (like syncing weights to vLLM), then restores the base model
    weights from backup.

    Args:
        actor: The actor module with LoRA adapters to merge
        backup_adapters: If True, backup base model weights (with LoRA disabled) before
            merging and restore them after. This is more numerically stable than unmerging.

    Yields:
        None
    """
    base_weights_backup = None
    if backup_adapters:
        # Backup base model weights with LoRA temporarily disabled
        base_weights_backup = backup_base_model_weights(actor)

    # Merge LoRA adapters into base model
    fsdp_merge_unmerge(actor, do_merge=True)
    try:
        # Do work while merged (sync_to_vllm / generate / etc.)
        yield
    finally:
        if backup_adapters and base_weights_backup is not None:
            # Restore base model weights from CPU backup (effectively undoing the merge)
            restore_base_model_weights(actor, base_weights_backup)
            _clean_merged_lora_(actor)
        else:
            # Fall back to unmerge if no backup was made
            fsdp_merge_unmerge(actor, do_merge=False)


class PerLayerGPUOptimizerStep:
    """Per-layer GPU optimizer step with async H2D/D2H prefetching.

    Instead of running Adam on CPU for all ~67GB of optimizer states,
    streams 1-2 layers at a time (~1.5GB each) to GPU, achieving ~50-80x speedup.

    Requires:
    - optimizer_offload=True (optimizer states reside on CPU between steps)
    - offload_policy=False (params and grads MUST remain on GPU)

    If CPUOffloadPolicy is used (params/grads offloaded to CPU), raise ValueError.
    """

    def __init__(self, model, optimizer, device_id, prefetch_layers=1):
        if not isinstance(optimizer, torch.optim.AdamW):
            raise TypeError(
                f"PerLayerGPUOptimizerStep only supports AdamW optimizer, "
                f"got {type(optimizer).__name__}."
            )
        self.optimizer = optimizer
        self.device = torch.device(f"cuda:{device_id}") if isinstance(device_id, int) else torch.device(device_id)
        self.prefetch_layers = prefetch_layers
        self._layer_param_groups = self._build_layer_groups(model)
        self._validate_gpu_params()
        self._validate_single_hyperparam_set()
        self._init_states_and_pin()
        # Persistent CUDA streams — reused across step() calls
        self.h2d_stream = torch.cuda.Stream(device=self.device)
        self.d2h_stream = torch.cuda.Stream(device=self.device)
        self.last_step_metrics = {}

    def _build_layer_groups(self, model) -> list:
        """Group params by FSDP2-wrapped sub-modules (excluding root).

        After apply_fsdp2, each transformer layer and embedding becomes an
        FSDPModule. We find all non-root FSDPModule instances and group their
        params. Any remaining params (final norm, lm_head) form a residual group.
        """
        fsdp_children = []
        for name, module in model.named_modules():
            if name and isinstance(module, FSDPModule):
                fsdp_children.append(module)

        assigned = set()
        groups = []
        for module in fsdp_children:
            params = [p for p in module.parameters() if p.requires_grad and id(p) not in assigned]
            if params:
                groups.append(params)
                for p in params:
                    assigned.add(id(p))

        # Residual: final norm, lm_head, etc.
        residual = [p for p in model.parameters() if p.requires_grad and id(p) not in assigned]
        if residual:
            groups.append(residual)
        return groups

    def _get_local_tensor(self, t):
        """Extract regular tensor from DTensor, or return as-is."""
        return t._local_tensor if hasattr(t, "_local_tensor") else t

    def _validate_gpu_params(self):
        """Verify all params are on GPU (not CPU-offloaded via CPUOffloadPolicy)."""
        for group in self._layer_param_groups:
            for param in group:
                local_p = self._get_local_tensor(param.data)
                if local_p.device.type != "cuda":
                    raise ValueError(
                        f"PerLayerGPUOptimizerStep requires params on GPU, "
                        f"but found param on {local_p.device}. "
                        f"Disable offload_policy (CPUOffloadPolicy) when using "
                        f"per_layer_optimizer_step."
                    )

    def _validate_single_hyperparam_set(self):
        """Assert all param_groups share identical hyperparameters.

        PerLayerGPUOptimizerStep processes params by layer (not by group),
        so all groups must have the same hyperparams. Fails loudly if not.
        """
        if len(self.optimizer.param_groups) <= 1:
            return
        ref = self.optimizer.param_groups[0]
        keys = ["lr", "betas", "eps", "weight_decay", "amsgrad", "maximize",
                "foreach", "capturable", "differentiable", "fused",
                "decoupled_weight_decay"]
        for i, group in enumerate(self.optimizer.param_groups[1:], 1):
            for key in keys:
                if group[key] != ref[key]:
                    raise ValueError(
                        f"PerLayerGPUOptimizerStep requires all param_groups to have "
                        f"identical hyperparameters, but group[{i}]['{key}']={group[key]} "
                        f"differs from group[0]['{key}']={ref[key]}."
                    )

    def _init_states_and_pin(self):
        """Pre-initialize optimizer states on CPU and pin them for async H2D/D2H.

        Without pinning, .to(non_blocking=True) on CPU tensors is synchronous,
        killing all pipeline overlap. Pinning enables true async DMA transfers.

        States must be created on CPU explicitly — NOT zeros_like(param) which
        would follow param's device (GPU), causing OOM for large models.
        """
        for group in self._layer_param_groups:
            for param in group:
                if not param.requires_grad:
                    continue
                state = self.optimizer.state[param]
                if len(state) == 0:
                    local_p = self._get_local_tensor(param.data)
                    state["step"] = torch.tensor(0.0, dtype=torch.float32)
                    state["exp_avg"] = torch.zeros(
                        local_p.shape, dtype=local_p.dtype, device="cpu"
                    )
                    state["exp_avg_sq"] = torch.zeros(
                        local_p.shape, dtype=local_p.dtype, device="cpu"
                    )
                else:
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key in state:
                            local = self._get_local_tensor(state[key])
                            if local.device.type != "cpu":
                                state[key] = local.to("cpu")
                # Pin optimizer state tensors for async transfers
                for key in ("exp_avg", "exp_avg_sq", "step"):
                    local = self._get_local_tensor(state[key])
                    if local.device.type == "cpu" and not local.is_pinned():
                        local.data = local.pin_memory()

    def _prefetch_layer(self, layer_idx):
        """H2D: copy layer's optimizer states to GPU.

        Params and grads are already on GPU — only optimizer states
        (exp_avg, exp_avg_sq, step) need H2D transfer from pinned CPU memory.
        """
        result = {}
        for param in self._layer_param_groups[layer_idx]:
            if param.grad is None:
                continue
            state = self.optimizer.state[param]
            local_m = self._get_local_tensor(state["exp_avg"])
            local_v = self._get_local_tensor(state["exp_avg_sq"])

            result[id(param)] = {
                "state": state,
                "gpu_p": self._get_local_tensor(param.data),  # already on GPU
                "gpu_g": self._get_local_tensor(param.grad.data),  # already on GPU
                "gpu_m": local_m.to(self.device, non_blocking=True),
                "gpu_v": local_v.to(self.device, non_blocking=True),
                "gpu_step": state["step"].to(self.device, non_blocking=True),
                "cpu_m": local_m,  # pinned CPU tensors for D2H
                "cpu_v": local_v,
            }
        return result

    def _run_adam_for_layer(self, gpu_states):
        """Call torch.optim.adam.adam() functional API on GPU tensors.

        Mirrors Adam.step() (adam.py:248-270): reads all hyperparams from group dict.
        All param_groups guaranteed identical by _validate_single_hyperparam_set().
        """
        from torch.optim.adam import adam

        group = self.optimizer.param_groups[0]
        beta1, beta2 = group["betas"]
        params, grads, exp_avgs, exp_avg_sqs, steps = [], [], [], [], []
        has_complex = False
        for buf in gpu_states.values():
            has_complex |= torch.is_complex(buf["gpu_p"])
            params.append(buf["gpu_p"])
            grads.append(buf["gpu_g"])
            exp_avgs.append(buf["gpu_m"])
            exp_avg_sqs.append(buf["gpu_v"])
            steps.append(buf["gpu_step"])
        if not params:
            return
        adam(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            [],  # max_exp_avg_sqs (empty for non-amsgrad)
            steps,
            amsgrad=group["amsgrad"],
            has_complex=has_complex,
            beta1=beta1,
            beta2=beta2,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            eps=group["eps"],
            maximize=group["maximize"],
            foreach=group["foreach"],
            capturable=group["capturable"],
            differentiable=group["differentiable"],
            fused=group["fused"],
            grad_scale=getattr(self.optimizer, "grad_scale", None),
            found_inf=getattr(self.optimizer, "found_inf", None),
            decoupled_weight_decay=group["decoupled_weight_decay"],
        )

    def _offload_layer(self, gpu_states):
        """D2H: copy updated optimizer states back to pinned CPU memory.

        Params are already updated in-place on GPU by Adam — no param D2H needed.
        Only optimizer states (exp_avg, exp_avg_sq, step) are copied back.
        """
        for buf in gpu_states.values():
            buf["cpu_m"].copy_(buf["gpu_m"], non_blocking=True)
            buf["cpu_v"].copy_(buf["gpu_v"], non_blocking=True)
            buf["state"]["step"].copy_(buf["gpu_step"], non_blocking=True)

    @torch.no_grad()
    def step(self):
        """Per-layer GPU optimizer step with async prefetch pipeline.

        Uses CUDA events (not wait_stream) for fine-grained synchronization,
        allowing H2D prefetch of layer i+2 to overlap with Adam on layer i
        and D2H offload of layer i-1.

        After completion, populates self.last_step_metrics dict with:
        - step_time_s: wall-clock time of the entire step
        - peak_memory_gb: peak GPU memory during the step
        - num_layer_groups: number of layer groups processed
        - compute_stall_count: layers where compute waited for H2D
        - avg_h2d_ms, avg_compute_ms, avg_d2h_ms: per-phase avg timing
        """
        import time

        t_start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(self.device)

        h2d_stream = self.h2d_stream
        d2h_stream = self.d2h_stream
        compute_stream = torch.cuda.current_stream(self.device)
        num_groups = len(self._layer_param_groups)
        gpu_states = [None] * num_groups
        h2d_events = [None] * num_groups

        # CUDA events for pipeline timing
        pre_wait_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_groups)]
        post_wait_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_groups)]
        compute_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_groups)]
        h2d_start_events = [None] * num_groups
        h2d_end_events = [None] * num_groups
        d2h_start_events = [None] * num_groups
        d2h_end_events = [None] * num_groups

        # Prefetch initial layers with per-layer events
        for i in range(min(self.prefetch_layers + 1, num_groups)):
            with torch.cuda.stream(h2d_stream):
                h2d_start_events[i] = h2d_stream.record_event(torch.cuda.Event(enable_timing=True))
                gpu_states[i] = self._prefetch_layer(i)
                ev = torch.cuda.Event(enable_timing=True)
                h2d_stream.record_event(ev)
                h2d_events[i] = ev
                h2d_end_events[i] = ev

        # Process each layer
        for i in range(num_groups):
            # Record BEFORE wait — measures actual GPU-side stall time
            compute_stream.record_event(pre_wait_events[i])
            compute_stream.wait_event(h2d_events[i])
            compute_stream.record_event(post_wait_events[i])

            self._run_adam_for_layer(gpu_states[i])
            compute_stream.record_event(compute_end_events[i])

            # Record compute completion for D2H dependency
            compute_done = compute_end_events[i]

            # Prefetch next layer (overlaps with D2H below)
            next_idx = i + self.prefetch_layers + 1
            if next_idx < num_groups:
                with torch.cuda.stream(h2d_stream):
                    h2d_start_events[next_idx] = h2d_stream.record_event(
                        torch.cuda.Event(enable_timing=True)
                    )
                    gpu_states[next_idx] = self._prefetch_layer(next_idx)
                    ev = torch.cuda.Event(enable_timing=True)
                    h2d_stream.record_event(ev)
                    h2d_events[next_idx] = ev
                    h2d_end_events[next_idx] = ev

            # Offload current layer (waits only for this layer's compute)
            d2h_stream.wait_event(compute_done)
            with torch.cuda.stream(d2h_stream):
                d2h_start_events[i] = d2h_stream.record_event(torch.cuda.Event(enable_timing=True))
                self._offload_layer(gpu_states[i])
                d2h_end_events[i] = d2h_stream.record_event(torch.cuda.Event(enable_timing=True))
            gpu_states[i] = None  # free GPU memory

        d2h_stream.synchronize()

        # Prevent cross-phase cache pollution: return freed optimizer state
        # blocks to CUDA driver so forward/backward can't repurpose them.
        # Without this, each optimizer step leaks ~1.7 GB of device memory
        # because caching allocator blocks get "stolen" by gradient allocation.
        torch.cuda.empty_cache()

        # Collect metrics (all streams synchronized)
        step_time = time.perf_counter() - t_start
        peak_memory_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)

        # Compute per-phase timings from CUDA events
        h2d_times, compute_times, d2h_times, wait_times = [], [], [], []
        for i in range(num_groups):
            wait_times.append(pre_wait_events[i].elapsed_time(post_wait_events[i]))
            compute_times.append(post_wait_events[i].elapsed_time(compute_end_events[i]))
            if h2d_start_events[i] is not None and h2d_end_events[i] is not None:
                h2d_times.append(h2d_start_events[i].elapsed_time(h2d_end_events[i]))
            if d2h_start_events[i] is not None and d2h_end_events[i] is not None:
                d2h_times.append(d2h_start_events[i].elapsed_time(d2h_end_events[i]))

        # Stall detection: wait_time measures how long compute stream blocked on
        # wait_event. Skip layer 0 — it always waits for the initial prefetch.
        stall_threshold_ms = 0.1
        stall_count = sum(1 for w in wait_times[1:] if w > stall_threshold_ms)

        avg_h2d = sum(h2d_times) / len(h2d_times) if h2d_times else 0.0
        avg_compute = sum(compute_times) / len(compute_times) if compute_times else 0.0
        avg_d2h = sum(d2h_times) / len(d2h_times) if d2h_times else 0.0

        self.last_step_metrics = {
            "step_time_s": step_time,
            "peak_memory_gb": peak_memory_gb,
            "num_layer_groups": num_groups,
            "avg_h2d_ms": avg_h2d,
            "avg_compute_ms": avg_compute,
            "avg_d2h_ms": avg_d2h,
            "compute_stall_count": stall_count,
        }
