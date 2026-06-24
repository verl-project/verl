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
import logging
import torch.distributed as dist
import torch
import torch.nn as nn

from hyper_parallel import fully_shard
from hyper_parallel.core.fully_shard.utils import CPUOffloadPolicy

from verl.utils.device import get_device_id, get_torch_device

logger = logging.getLogger(__name__)


def check_hyperparallel_availability():
    """
    Check if HyperParallel is available.
    """
    try:
        import hyper_parallel as hp
        logger.info(f"HyperParallel version {hp.__version__} found")
        return True
    except ImportError:
        logger.warning("HyperParallel not found")
        return False
    except Exception as e:
        logger.error(f"Error checking HyperParallel availability: {e}")
        return False


def get_hyperparallel_version():
    """
    Get HyperParallel version.
    """
    try:
        import hyper_parallel as hp
        return hp.__version__
    except ImportError:
        return None


def check_npu_compatibility():
    """
    Check if HyperParallel supports NPU.
    """
    try:
        if torch.npu.is_available():
            import hyper_parallel as hp
            # Check if hyperparallel has NPU support
            return hasattr(hp, 'npu') or hasattr(hp.device_mesh, 'init_device_mesh')
        return False
    except Exception as e:
        logger.error(f"Error checking NPU compatibility: {e}")
        return False


def safe_full_tensor(dtensor):
    """
    Safely convert DTensor to full tensor.
    """
    try:
        return dtensor.full_tensor()
    except Exception as e:
        logger.error(f"Error converting DTensor to full tensor: {e}")
        return dtensor

_hp_unshard_patched = False
def _patch_hp_unshard_no_grad():
    """Monkey-patch HyperParallel unshard/shard to run under torch.no_grad().

    HyperParallel's internal all_gather buffers (all_gather_outputs,
    _sharded_param_data) are views whose creation and inplace modification
    (storage.resize_) cross no_grad / grad-mode boundaries between inference
    and training.  PyTorch's view-safety check rejects this mixed usage.

    By forcing the buffer-management operations (unshard/shard) to always
    run in no_grad context, all view creation and storage mutation happen
    under a consistent grad mode.  This does NOT affect gradient flow:
    parameter gradients are handled by the DTensor autograd mechanism during
    the model's forward/backward pass, not by the all_gather operation.

    NOTE: The actual class is TorchHSDPParamV2 (not ShardedParameter).
    """
    global _hp_unshard_patched
    if _hp_unshard_patched:
        return
    _hp_unshard_patched = True

    patched_count = 0

    # --- Patch TorchHSDPParamV2 (non-fused per-parameter path) ---
    try:
        from hyper_parallel.platform.torch.fully_shard.param import TorchHSDPParamV2
    except ImportError:
        logger.warning("[HyperParallel] TorchHSDPParamV2 not found, skipping param patch")
        TorchHSDPParamV2 = None

    if TorchHSDPParamV2 is not None:
        for method_name in ('unshard', 'shard', 'wait_for_unshard',
                            '_get_unsharded_param_data'):
            orig = getattr(TorchHSDPParamV2, method_name, None)
            if orig is None:
                continue

            @functools.wraps(orig)
            def _no_grad_param_wrapper(self_param, *args, _orig=orig, **kwargs):
                with torch.no_grad():
                    return _orig(self_param, *args, **kwargs)

            setattr(TorchHSDPParamV2, method_name, _no_grad_param_wrapper)
            patched_count += 1

    # --- Patch HSDPParamGroup (fused communication path) ---
    try:
        from hyper_parallel.platform.torch.fully_shard.param_group import HSDPParamGroup
    except ImportError:
        logger.warning("[HyperParallel] HSDPParamGroup not found, skipping param_group patch")
        HSDPParamGroup = None

    if HSDPParamGroup is not None:
        for method_name in ('unshard', 'wait_for_unshard'):
            orig = getattr(HSDPParamGroup, method_name, None)
            if orig is None:
                continue

            @functools.wraps(orig)
            def _no_grad_group_wrapper(self_group, *args, _orig=orig, **kwargs):
                with torch.no_grad():
                    return _orig(self_group, *args, **kwargs)

            setattr(HSDPParamGroup, method_name, _no_grad_group_wrapper)
            patched_count += 1

    logger.info(f"[HyperParallel] Applied no_grad monkey-patches ({patched_count} methods)")


def _select_fsdp2_wrap_targets(model, fsdp_transformer_layer_cls_to_wrap):
    """Select modules to wrap individually with fully_shard in FSDP2.

    Matches transformer layers by class name, and embed_tokens/lm_head by name
    (with isinstance fallback). Name-based matching is needed because peft wraps
    embed_tokens in ModulesToSaveWrapper, breaking isinstance(module, nn.Embedding).
    When tie_word_embeddings is True, embed_tokens and lm_head share weights and
    must not be wrapped separately.
    """
    _tie = getattr(model.config, "tie_word_embeddings", False)
    _wrap_by_name = set() if _tie else {"embed_tokens", "lm_head"}

    modules = []
    for name, module in model.named_modules():
        leaf_name = name.rsplit(".", 1)[-1] if "." in name else name
        if (
            module.__class__.__name__ in fsdp_transformer_layer_cls_to_wrap
            or (isinstance(module, nn.Embedding) and not _tie)
            or (leaf_name in _wrap_by_name and hasattr(module, "weight"))
        ):
            modules.append(module)
    return modules


def apply_hp_fsdp(model, fsdp_kwargs, config):
    """model: AutoModelForCausalLM"""
    assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"

    default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = config.get("wrap_policy", {}).get(
        "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
    )

    if isinstance(fsdp_transformer_layer_cls_to_wrap, str):
        fsdp_transformer_layer_cls_to_wrap = [fsdp_transformer_layer_cls_to_wrap]

    if isinstance(fsdp_transformer_layer_cls_to_wrap, set):
        fsdp_transformer_layer_cls_to_wrap = list(fsdp_transformer_layer_cls_to_wrap)
    assert len(fsdp_transformer_layer_cls_to_wrap) > 0 and fsdp_transformer_layer_cls_to_wrap[0] is not None

    modules = _select_fsdp2_wrap_targets(model, fsdp_transformer_layer_cls_to_wrap)

    for idx, module in enumerate(modules):
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f"wrap module {module.__class__.__name__}")
        fully_shard(module, **fsdp_kwargs)

    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        print(f"wrap module {model.__class__.__name__}")
    fully_shard(model, **fsdp_kwargs)  # fsdp2 will not reshard_after_forward for root module
    _patch_hp_unshard_no_grad()


def fsdp2_load_full_state_dict(model: torch.nn.Module, full_state: dict, device_mesh=None, cpu_offload=None):
    """
    Loads the full state dict (could be only on rank 0) into the sharded model. This is done by broadcasting the
    parameters from rank 0 to all other ranks. This function modifies the model in-place.

    Args:
        model (`torch.nn.Module`): The model to load the state dict into
        full_state (`dict`): The full state dict to load, can only be on rank 0
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

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


@torch.no_grad()
def offload_hp_model_to_cpu(model, empty_cache: bool = True):
    # reshard: 释放 all_gather 缓冲区，只保留本地分片
    if hasattr(model, 'reshard'):
        model.reshard()
    model.to("cpu")
    if empty_cache:
        get_torch_device().empty_cache()


@torch.no_grad()
def load_hp_model_to_gpu(model):
    device = get_device_id()
    model.to(device)


@torch.no_grad()
def offload_hp_optimizer(optimizer):
    optimizers = []
    # Check if this is a MultiOptimizer (for ep and non-ep parameters when ep+fsdp2 is enabled)
    if hasattr(optimizer, "_is_multi_optimizer") and optimizer._is_multi_optimizer:
        optimizers.extend(optimizer.optimizers_dict.values())
    else:
        optimizers.append(optimizer)

    for opt in optimizers:
        if not opt.state:
            continue
        for param_group in opt.param_groups:
            for param in param_group["params"]:
                state = opt.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to("cpu", non_blocking=True)


@torch.no_grad()
def load_hp_optimizer(optimizer, device_id):
    optimizers = []
    # Check if this is a MultiOptimizer (for ep and non-ep parameters when ep+fsdp2 is enabled)
    if hasattr(optimizer, "_is_multi_optimizer") and optimizer._is_multi_optimizer:
        optimizers.extend(optimizer.optimizers_dict.values())
    else:
        optimizers.append(optimizer)

    for opt in optimizers:
        if not opt.state:
            continue
        for param_group in opt.param_groups:
            for param in param_group["params"]:
                state = opt.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device_id, non_blocking=True)
