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
Clean DeepSpeed-based Workers for PPO Training (Native DeepSpeed API)

This module provides worker implementations using native DeepSpeed API,
similar to how FSDP workers use native PyTorch FSDP API.
"""

import asyncio
import datetime
import hashlib
import logging
import os
import threading
import time
import warnings
from contextlib import nullcontext
from collections import OrderedDict
from typing import Any, Optional

import psutil
import torch
import torch.distributed
import deepspeed
from deepspeed.runtime.engine import DeepSpeedEngine
from torch.distributed.device_mesh import init_device_mesh
from tensordict import TensorDict
from codetiming import Timer
from omegaconf import DictConfig, OmegaConf, open_dict
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForVision2Seq

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.third_party.vllm import vllm_version
from verl.utils import hf_processor, hf_tokenizer
from verl.workers.deepspeed_parallel import (
    ParallelLayout,
    build_parallel_layout,
    normalize_actor_batches,
    normalize_critic_batches,
)
from verl.utils.checkpoint.deepspeed_checkpoint_manager import DeepSpeedCheckpointManager
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.deepspeed_utils import (
    get_deepspeed_config,
    initialize_deepspeed_engine,
    load_deepspeed_model_to_gpu,
    offload_deepspeed_model_to_cpu,
)
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
    set_expandable_segments,
)
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import import_external_libs
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.model import (
    compute_position_id_with_mask,
    convert_weight_keys,
    get_generation_config,
    load_valuehead_model,
    print_model_size,
    update_model_config,
)
from verl.utils.fsdp_utils import collect_lora_params, replace_lora_wrapper
from verl.utils.profiler.performance import reduce_timing, topk_reduce_ratio_min_max
from verl.utils.ray_utils import get_event_loop
from verl.utils.py_functional import append_to_dict, convert_to_regular_types
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import masked_mean
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.workers.config import DeepSpeedCriticConfig, DeepSpeedEngineConfig, HFModelConfig, RolloutConfig
from verl.workers.actor import DataParallelPPOActor
from verl.workers.critic import DataParallelPPOCritic
from verl.workers.rollout import get_rollout_class
from verl.utils.ulysses import get_ulysses_sequence_parallel_group, set_ulysses_sequence_parallel_group

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def _log_ds_step_metrics(prefix, metrics, keys):
    return


def _log_ds_engine_state(prefix, engine):
    return


device_name = get_device_name()


class DeepSpeedUlyssesShardingManager:
    """Scoped switch for the global Ulysses sequence-parallel process group."""

    def __init__(self, process_group: Optional[torch.distributed.ProcessGroup]):
        self.process_group = process_group
        self._prev_group: Optional[torch.distributed.ProcessGroup] = None

    def __enter__(self):
        if self.process_group is None:
            return self

        self._prev_group = get_ulysses_sequence_parallel_group()
        set_ulysses_sequence_parallel_group(self.process_group)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.process_group is None:
            return

        set_ulysses_sequence_parallel_group(self._prev_group)
        self._prev_group = None

_BACKGROUND_LOOP: asyncio.AbstractEventLoop | None = None
_BACKGROUND_LOOP_THREAD: threading.Thread | None = None
_BACKGROUND_LOOP_LOCK = threading.Lock()


def _get_engine_module(engine: Optional[DeepSpeedEngine]):
    """Safely fetch engine.module without triggering DeepSpeed __getattr__ recursion."""
    if engine is None:
        return None
    try:
        return object.__getattribute__(engine, "module")
    except Exception:
        return None


def _safe_zero_grad(engine: Optional[DeepSpeedEngine]):
    """Call engine.zero_grad with a guard against DeepSpeed module recursion errors."""
    if engine is None:
        return

    # DeepSpeed can recurse in __getattr__ when module is absent; skip in that case.
    if _get_engine_module(engine) is None:
        logger.warning("Skip zero_grad because DeepSpeed engine.module is missing")
        return

    try:
        engine.zero_grad()
    except RecursionError:
        logger.warning("Skip zero_grad to avoid DeepSpeed zero_grad recursion (missing module)")
    except AttributeError:
        logger.warning("Skip zero_grad because DeepSpeed engine has no module")


def _ensure_engine_has_module(engine: Optional[DeepSpeedEngine], module: Optional[torch.nn.Module]):
    """Force-set engine.module when DeepSpeed drops the reference (avoids __getattr__ recursion)."""
    if engine is None or module is None:
        return
    if _get_engine_module(engine) is None:
        try:
            object.__setattr__(engine, "module", module)
        except Exception:
            try:
                engine.module = module  # type: ignore
            except Exception:
                logger.warning("Failed to restore DeepSpeed engine.module reference")


def _maybe_move_module_to_device(module: Optional[torch.nn.Module], device: torch.device | int | str):
    """Move module to device if its first parameter is on a different device."""
    if module is None:
        return
    try:
        params = list(module.parameters())
    except Exception:
        return
    if not params:
        return
    try:
        dev = params[0].device
    except Exception:
        return
    target = torch.device(device)
    if dev != target:
        try:
            module.to(target)
        except Exception:
            logger.warning("Failed to move module to device %s (current %s)", target, dev)


def _debug_first_param_device(tag: str, module: Optional[torch.nn.Module]):
    """Print first parameter device for debugging offload/device mismatches."""
    if os.getenv("VERL_DS_DEVICE_LOG", "0") == "0":
        return
    if module is None:
        return
    try:
        params = list(module.parameters())
    except Exception:
        return
    if not params:
        return
    try:
        dev = params[0].device
    except Exception:
        dev = "unknown"
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return
    logger.info("[DS device] %s first_param_device=%s", tag, dev)


def _weight_sync_digest_entry(name: str, tensor: torch.Tensor) -> tuple[int, int]:
    """Compute an order-insensitive digest entry for one tensor."""
    view = tensor.detach()
    numel = int(view.numel())
    hasher = hashlib.sha256()
    hasher.update(name.encode("utf-8"))
    hasher.update(str(tuple(view.shape)).encode("utf-8"))
    hasher.update(str(view.dtype).encode("utf-8"))
    hasher.update(str(numel).encode("utf-8"))

    if numel > 0:
        flat = view.reshape(-1)
        n_sample = min(8, numel)
        if n_sample == 1:
            sample = flat[:1]
        else:
            idx = torch.arange(n_sample, device=flat.device, dtype=torch.long)
            idx = (idx * (numel - 1)) // (n_sample - 1)
            sample = flat.index_select(0, idx)
        hasher.update(sample.float().cpu().numpy().tobytes())

    return int.from_bytes(hasher.digest()[:16], byteorder="big", signed=False), numel


def _weight_sync_digest(items: Any) -> tuple[str, int, int]:
    """Aggregate digest over (name, tensor) pairs."""
    acc = 0
    n_tensors = 0
    total_numel = 0
    for name, tensor in items:
        digest_part, numel = _weight_sync_digest_entry(name, tensor)
        acc ^= digest_part
        n_tensors += 1
        total_numel += numel
    return f"{acc:032x}", n_tensors, total_numel


def _zero3_consolidated_state_dict_iter(module: torch.nn.Module):
    """Iterative ZeRO-3 consolidation to avoid deep recursion in DeepSpeed helper."""
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    state = OrderedDict() if rank == 0 else None
    shared: dict[Any, str] = {}
    stack: list[tuple[torch.nn.Module, str]] = [(module, "")]

    while stack:
        mod, prefix = stack.pop()
        params = list(mod.parameters(recurse=False))
        buffers = list(mod.named_buffers(recurse=False))
        with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
            if rank == 0 and state is not None:
                for name, param in mod.named_parameters(recurse=False):
                    if param is None:
                        continue
                    key = prefix + name
                    ds_id = getattr(param, "ds_id", None)
                    if ds_id is not None and ds_id in shared:
                        state[key] = state[shared[ds_id]]
                    else:
                        state[key] = param.detach().cpu()
                        if ds_id is not None:
                            shared[ds_id] = key
                for name, buf in buffers:
                    if buf is not None and name not in mod._non_persistent_buffers_set:
                        state[prefix + name] = buf.detach().cpu()

        for name, child in mod.named_children():
            if child is not None:
                stack.append((child, prefix + name + "."))

    return state


def _broadcast_object_list_cpu(obj_list, src: int = 0):
    """Broadcast python objects via a temporary gloo group to avoid NCCL object hangs."""
    if not torch.distributed.is_initialized():
        return obj_list
    if torch.distributed.get_world_size() <= 1:
        return obj_list

    group = None
    try:
        try:
            backend = torch.distributed.get_backend()
        except Exception:
            backend = None
        if backend == "nccl":
            group = torch.distributed.new_group(backend="gloo")
            torch.distributed.broadcast_object_list(obj_list, src=src, group=group)
        else:
            torch.distributed.broadcast_object_list(obj_list, src=src)
    finally:
        if group is not None:
            torch.distributed.destroy_process_group(group)
    return obj_list


def _gather_zero3_state_dict(module: torch.nn.Module, engine: DeepSpeedEngine):
    """Gather a full ZeRO-3 state dict (fp16/bf16) without relying on NCCL allgather in vLLM."""
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    _ensure_engine_has_module(engine, module)

    # First try DeepSpeed's consolidated helper (fast path, rank0 only)
    params: dict[str, torch.Tensor] | None = None
    if hasattr(engine, "_zero3_consolidated_16bit_state_dict"):
        try:
            _debug_print("rollout_mode:zero3_consolidated_state:enter")
            consolidated = engine._zero3_consolidated_16bit_state_dict()
            _debug_print("rollout_mode:zero3_consolidated_state:exit")
            if consolidated is not None:
                params = {k: v.detach().cpu() for k, v in consolidated.items()} if rank == 0 else {}
            elif rank != 0:
                # Rank != 0 returns None by design; fill with placeholder for broadcast
                params = {}
        except RecursionError as e:
            logger.warning("zero3 consolidated state dict recursion, using iterative gather: %s", e)
            consolidated = _zero3_consolidated_state_dict_iter(module)
            if consolidated is not None:
                params = consolidated if rank == 0 else {}
            elif rank != 0:
                params = {}
        except Exception as e:
            logger.warning("zero3 consolidated state dict failed: %s", e)
            params = None

    allow_fallback = os.getenv("VERL_ZERO3_ALLOW_GATHER_FALLBACK", "0") == "1"
    if params is None and not allow_fallback:
        raise RuntimeError(
            "ZeRO-3 consolidated weights unavailable; enable stage3_gather_16bit_weights_on_model_save to sync rollout."
        )

    # Optional fallback: explicitly gather parameters on all ranks (may timeout; disabled by default)
    if params is None and allow_fallback:
        gathered: dict[str, torch.Tensor] | None = None
        try:
            param_list = list(module.parameters())
            buffer_list = list(module.buffers())
            with deepspeed.zero.GatheredParameters(param_list + buffer_list, modifier_rank=None):
                full_state = module.state_dict()
                gathered = {k: v.detach().cpu() for k, v in full_state.items()}
        except Exception as e:
            logger.exception("zero3 GatheredParameters fallback failed: %s", e)
        params = gathered or {}

    if torch.distributed.is_initialized() and world_size > 1:
        obj_list = [params]
        _broadcast_object_list_cpu(obj_list, src=0)
        params = obj_list[0] or {}

    return params

def _debug_print(msg: str):
    return


def _debug_async_state(tag: str, start_time: float | None = None):
    return


def _debug_timing(tag: str, start_time: float | None = None) -> float | None:
    return time.perf_counter() if start_time is None else None


def _normalize_ds_config(cfg_section: DictConfig | dict | Any) -> Any:
    """Ensure deepspeed_config exists, falling back to deepspeed when needed."""
    if cfg_section is None:
        return None

    # Try direct access first
    ds_cfg = getattr(cfg_section, "deepspeed_config", None)
    if isinstance(cfg_section, DictConfig) and ds_cfg is None:
        ds_cfg = cfg_section.get("deepspeed_config", None)

    # Fallback to legacy key
    if ds_cfg is None:
        if hasattr(cfg_section, "deepspeed"):
            ds_cfg = getattr(cfg_section, "deepspeed")
        elif isinstance(cfg_section, DictConfig):
            ds_cfg = cfg_section.get("deepspeed", None)
        elif isinstance(cfg_section, dict):
            ds_cfg = cfg_section.get("deepspeed", None)

        if ds_cfg is not None:
            try:
                if isinstance(cfg_section, DictConfig):
                    with open_dict(cfg_section):
                        cfg_section["deepspeed_config"] = ds_cfg
                else:
                    setattr(cfg_section, "deepspeed_config", ds_cfg)
            except Exception:
                pass

    # Normalize offload flags so they align with the requested zero_stage.
    try:
        if isinstance(ds_cfg, DictConfig):
            zero_stage = ds_cfg.get("zero_stage", 0)
            offload_val = ds_cfg.get("offload", "none")
            param_offload = bool(ds_cfg.get("param_offload", False))
            optimizer_offload = bool(ds_cfg.get("optimizer_offload", False))
        elif isinstance(ds_cfg, dict):
            zero_stage = ds_cfg.get("zero_stage", 0)
            offload_val = ds_cfg.get("offload", "none")
            param_offload = bool(ds_cfg.get("param_offload", False))
            optimizer_offload = bool(ds_cfg.get("optimizer_offload", False))
        else:
            zero_stage = getattr(ds_cfg, "zero_stage", 0)
            offload_val = getattr(ds_cfg, "offload", "none")
            param_offload = bool(getattr(ds_cfg, "param_offload", False))
            optimizer_offload = bool(getattr(ds_cfg, "optimizer_offload", False))

        try:
            zero_stage_int = int(zero_stage or 0)
        except Exception:
            zero_stage_int = 0
        offload_str = offload_val.lower() if isinstance(offload_val, str) else ""
        offload_requested = offload_str in {"cpu", "nvme", "auto"}

        allow_param_offload = os.getenv("VERL_ENABLE_PARAM_OFFLOAD", "0") == "1"
        if offload_requested:
            if zero_stage_int >= 3 and allow_param_offload:
                param_offload = True
            if zero_stage_int >= 2:
                optimizer_offload = True

        if zero_stage_int >= 3 and not allow_param_offload:
            param_offload = False
        if zero_stage_int < 3:
            param_offload = False
        if zero_stage_int < 2:
            optimizer_offload = False

        if isinstance(ds_cfg, DictConfig):
            with open_dict(ds_cfg):
                ds_cfg["param_offload"] = param_offload
                ds_cfg["optimizer_offload"] = optimizer_offload
        elif isinstance(ds_cfg, dict):
            ds_cfg["param_offload"] = param_offload
            ds_cfg["optimizer_offload"] = optimizer_offload
        else:
            try:
                setattr(ds_cfg, "param_offload", param_offload)
                setattr(ds_cfg, "optimizer_offload", optimizer_offload)
            except Exception:
                try:
                    object.__setattr__(ds_cfg, "param_offload", param_offload)
                    object.__setattr__(ds_cfg, "optimizer_offload", optimizer_offload)
                except Exception:
                    pass
    except Exception:
        pass

    return ds_cfg


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, DictConfig):
        return cfg.get(key, default)
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _get_zero_stage(cfg: Any) -> int:
    try:
        return int(_cfg_get(cfg, "zero_stage", 0) or 0)
    except Exception:
        return 0


def _ensure_background_loop() -> asyncio.AbstractEventLoop:
    """Return a running event loop, spinning up a background loop if needed."""
    global _BACKGROUND_LOOP, _BACKGROUND_LOOP_THREAD
    with _BACKGROUND_LOOP_LOCK:
        loop = _BACKGROUND_LOOP
        if loop is not None and not loop.is_closed() and loop.is_running():
            return loop

        loop = get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
        if loop.is_running():
            _BACKGROUND_LOOP = loop
            return loop

        def _runner(ev_loop: asyncio.AbstractEventLoop):
            asyncio.set_event_loop(ev_loop)
            ev_loop.run_forever()

        _BACKGROUND_LOOP = loop
        _BACKGROUND_LOOP_THREAD = threading.Thread(
            target=_runner, args=(loop,), name="verl-ds-async-loop", daemon=True
        )
        _BACKGROUND_LOOP_THREAD.start()

    while not loop.is_running():
        time.sleep(0.01)
    return loop


def run_coro(coro):
    """Run a coroutine with a safe loop policy shared across DS workers."""
    loop = _ensure_background_loop()
    return asyncio.run_coroutine_threadsafe(coro, loop).result()


def _parse_mixed_precision_config(mixed_precision):
    """Parse mixed_precision config to determine fp16/bf16 flags.

    Args:
        mixed_precision: Can be None, str ("fp16"/"bf16"), or dict {"param_dtype": "bf16", ...}

    Returns:
        tuple: (fp16_enabled, bf16_enabled)
    """
    if mixed_precision is None:
        return False, False
    elif isinstance(mixed_precision, str):
        mode = mixed_precision.lower()
        return mode in {"fp16", "float16"}, mode in {"bf16", "bfloat16"}
    elif isinstance(mixed_precision, (dict, DictConfig)):
        param_dtype = str(mixed_precision.get("param_dtype", "fp32")).lower()
        return param_dtype in {"fp16", "float16"}, param_dtype in {"bf16", "bfloat16"}
    else:
        return False, False


def _parse_comm_dtype_from_mixed_precision(mixed_precision) -> str | None:
    """Parse DeepSpeed communication_data_type from mixed_precision.reduce_dtype.

    Returns DeepSpeed accepted dtype names: {"fp16", "bfloat16", "fp32"}.
    """
    reduce_dtype = None
    if isinstance(mixed_precision, (dict, DictConfig)):
        reduce_dtype = mixed_precision.get("reduce_dtype", None)
    elif isinstance(mixed_precision, str):
        # For string shorthand, keep comm dtype defaulting to DeepSpeed behavior.
        reduce_dtype = None

    if reduce_dtype is None:
        return None

    mode = str(reduce_dtype).lower()
    if mode in {"fp16", "float16"}:
        return "fp16"
    if mode in {"bf16", "bfloat16"}:
        return "bfloat16"
    if mode in {"fp32", "float32"}:
        return "fp32"
    return None


def _normalize_zero_opt_overrides(zero_opt_overrides: Any) -> dict[str, Any] | None:
    if zero_opt_overrides is None:
        return None
    if isinstance(zero_opt_overrides, DictConfig):
        return OmegaConf.to_container(zero_opt_overrides, resolve=True)  # type: ignore[return-value]
    if isinstance(zero_opt_overrides, dict):
        return zero_opt_overrides
    return None


class ActorRolloutRefWorker(Worker):
    """
    Clean DeepSpeed-based worker using native DeepSpeed API.

    Similar to FSDP worker structure but uses DeepSpeed for training.
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        Worker.__init__(self)

        self.config = config
        actor_cfg = self.config.get("actor", None) if isinstance(self.config, DictConfig) else getattr(self.config, "actor", None)
        ref_cfg = self.config.get("ref", None) if isinstance(self.config, DictConfig) else getattr(self.config, "ref", None)
        self.actor_ds_config = _normalize_ds_config(actor_cfg)
        self.ref_ds_config = _normalize_ds_config(ref_cfg)
        self.actor_sharding_manager = None
        self.ref_sharding_manager = None
        self.actor_layout: ParallelLayout | None = None
        self.ref_layout: ParallelLayout | None = None

        rollout_cfg = self.config.get("rollout", {}) if isinstance(self.config, DictConfig) else {}
        self._skip_rollout = rollout_cfg.get("skip_rollout", False)
        load_format = rollout_cfg.get("load_format", "")
        self._dummy_rollout = isinstance(load_format, str) and load_format.startswith("dummy")
        try:
            self._debug_weight_sync_remaining = max(0, int(os.getenv("VERL_DEBUG_WEIGHT_SYNC_CHECK_STEPS", "0")))
        except Exception:
            self._debug_weight_sync_remaining = 0

        # Ensure CUDA device is bound to LOCAL_RANK before init_process_group
        try:
            if torch.cuda.is_available():
                local_rank_env = os.environ.get("LOCAL_RANK")
                if local_rank_env is not None:
                    lr = int(local_rank_env)
                    if torch.cuda.current_device() != lr:
                        torch.cuda.set_device(lr)
        except Exception:
            pass

        # Initialize distributed environment
        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
                device_id=get_device_id() if torch.cuda.is_available() else None,
            )

        # Parse role
        self.role = role
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        # Ensure the CUDA device matches LOCAL_RANK to avoid NCCL hangs in new groups
        try:
            if torch.cuda.is_available():
                local_rank_env = os.environ.get("LOCAL_RANK")
                if local_rank_env is not None:
                    lr = int(local_rank_env)
                    if torch.cuda.current_device() != lr:
                        torch.cuda.set_device(lr)
        except Exception:
            pass

        # Setup offload flags
        self._is_offload_param = False
        if self._is_actor:
            ds_cfg = self.actor_ds_config
        elif self._is_ref:
            ds_cfg = self.ref_ds_config
        else:
            ds_cfg = None

        if isinstance(ds_cfg, dict):
            self._is_offload_param = ds_cfg.get("param_offload", False)
        else:
            self._is_offload_param = getattr(ds_cfg, "param_offload", False) if ds_cfg is not None else False

        # Build parallel layouts and normalize configs
        if self._is_actor:
            tp_size = rollout_cfg.get("tensor_model_parallel_size", 1) if isinstance(rollout_cfg, dict) else 1
            self.actor_layout = build_parallel_layout(self.config.actor, tp_size=tp_size)
            normalize_actor_batches(
                self.config.actor, self.config.rollout.n, self.actor_layout.dp_size, sp_size=self.actor_layout.sp_size
            )
            self.actor_ulysses_sequence_parallel_size = self.actor_layout.sp_size
            self.ulysses_sequence_parallel_size = self.actor_layout.sp_size  # backward compat
            self._register_dispatch_collect_info(
                "actor", dp_rank=self.actor_layout.dp_rank, is_collect=self.actor_layout.collect
            )
        else:
            self.actor_ulysses_sequence_parallel_size = 1
            self.ulysses_sequence_parallel_size = 1

        if self._is_ref:
            self.ref_layout = build_parallel_layout(self.config.ref)
            self.ref_ulysses_sequence_parallel_size = self.ref_layout.sp_size
            self._register_dispatch_collect_info(
                "ref", dp_rank=self.ref_layout.dp_rank, is_collect=self.ref_layout.collect
            )
        else:
            self.ref_ulysses_sequence_parallel_size = 1

        self._lora_rank = self.config.model.get("lora_rank", 0)
        self._is_lora = self._lora_rank > 0

    def _build_model_optimizer(
        self,
        model_path: str,
        deepspeed_config: DeepSpeedEngineConfig,
        optim_config: Optional[dict],
        override_model_config: dict,
        use_remove_padding: bool = False,
        use_fused_kernels: bool = False,
        enable_gradient_checkpointing: bool = False,
        trust_remote_code: bool = False,
        use_liger: bool = False,
        role: str = "actor",
        layout: ParallelLayout | None = None,
    ):
        """
        Build model and optimizer using native DeepSpeed API.

        Returns:
            tuple: (deepspeed_engine, model, optimizer, lr_scheduler, model_config)
        """
        # Load model config
        _debug_print(f"{role}::_build_model_optimizer: load config start ({model_path})")
        actor_model_config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, attn_implementation="flash_attention_2"
        )
        _debug_print(f"{role}::_build_model_optimizer: load config done")

        # Override config
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)

        if self.rank == 0:
            logger.debug("Model config after override: %s", actor_model_config)

        # Determine torch dtype
        torch_dtype = deepspeed_config.get("model_dtype", "fp32")
        if torch_dtype == "fp32":
            torch_dtype = torch.float32
        elif torch_dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif torch_dtype == "fp16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        # Create model
        _debug_print(f"{role}::_build_model_optimizer: from_pretrained start dtype={torch_dtype}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            has_remote_code = hasattr(actor_model_config, "auto_map") and any(
                actor_model_config.architectures[0] in val for val in actor_model_config.auto_map.values()
            )
            if has_remote_code:
                auto_class = next(
                    k for k, v in actor_model_config.auto_map.items() if actor_model_config.architectures[0] in v
                )
                match auto_class:
                    case "AutoModelForVision2Seq":
                        actor_module_class = AutoModelForVision2Seq
                    case "AutoModelForCausalLM":
                        actor_module_class = AutoModelForCausalLM
                    case "AutoModelForImageTextToText":
                        actor_module_class = AutoModelForImageTextToText
                    case _:
                        actor_module_class = AutoModel
            else:
                if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                    actor_module_class = AutoModelForVision2Seq
                elif type(actor_model_config) in AutoModelForCausalLM._model_mapping.keys():
                    actor_module_class = AutoModelForCausalLM
                elif type(actor_model_config) in AutoModelForImageTextToText._model_mapping.keys():
                    actor_module_class = AutoModelForImageTextToText
                else:
                    actor_module_class = AutoModel

            actor_module = actor_module_class.from_pretrained(
                pretrained_model_name_or_path=model_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                trust_remote_code=trust_remote_code,
            )
        _debug_print(f"{role}::_build_model_optimizer: from_pretrained done")

        # Apply Liger kernel
        if use_liger:
            from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
            _apply_liger_kernel_to_instance(model=actor_module)

        # Apply monkey patches
        fused_kernel_options = self.config.model.get("fused_kernel_options", None)
        fused_kernels_backend = (
            fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
        )

        # Initialize Ulysses SP group for DeepSpeed-HF if requested
        if layout is not None:
            sp_size = layout.sp_size
        elif role == "actor":
            sp_size = self.actor_ulysses_sequence_parallel_size
            self.ulysses_sequence_parallel_size = sp_size
        elif role == "ref":
            sp_size = self.ref_ulysses_sequence_parallel_size
        else:
            sp_size = int(getattr(deepspeed_config, "ulysses_sequence_parallel_size", 1) or 1)
        sp_group = None
        prev_sp_group = get_ulysses_sequence_parallel_group()
        _debug_print(f"{role}::_build_model_optimizer: monkey_patch start sp={sp_size}")
        if sp_size > 1 and torch.distributed.is_initialized():
            # Use layout to build per-DP SP group to avoid cross-role pollution
            if layout is None:
                world = torch.distributed.get_world_size()
                assert world % sp_size == 0, f"world_size {world} must be divisible by ulysses sp_size {sp_size}"
                rank = torch.distributed.get_rank()
                group_id = rank // sp_size
                ranks = list(range(group_id * sp_size, (group_id + 1) * sp_size))
            else:
                ranks = list(range(layout.dp_rank * layout.sp_size, (layout.dp_rank + 1) * layout.sp_size))
            sp_group = torch.distributed.new_group(ranks=ranks, backend=get_nccl_backend())
            set_ulysses_sequence_parallel_group(sp_group)
            # synchronize all ranks inside the SP group before patching
            torch.distributed.barrier(group=sp_group)

        apply_monkey_patch(
            model=actor_module,
            use_remove_padding=use_remove_padding,
            ulysses_sp_size=sp_size,
            use_fused_kernels=use_fused_kernels,
            fused_kernels_backend=fused_kernels_backend,
        )
        _debug_print(f"{role}::_build_model_optimizer: monkey_patch done")

        actor_module.to(torch_dtype)

        # Gradient checkpointing
        if enable_gradient_checkpointing:
            actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # LoRA
        if self._is_lora:
            print("Applying LoRA to actor module")
            actor_module.enable_input_require_grads()
            lora_config = {
                "task_type": TaskType.CAUSAL_LM,
                "r": self.config.model.lora_rank,
                "lora_alpha": self.config.model.lora_alpha,
                "target_modules": convert_to_regular_types(self.config.model.target_modules),
                "exclude_modules": convert_to_regular_types(self.config.model.exclude_modules),
                "bias": "none",
            }
            actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))
        if sp_group is not None:
            torch.distributed.barrier(group=sp_group)
        set_ulysses_sequence_parallel_group(prev_sp_group)

        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)


        if role == "actor":
            self.actor_sharding_manager = DeepSpeedUlyssesShardingManager(sp_group)
        elif role == "ref":
            self.ref_sharding_manager = DeepSpeedUlyssesShardingManager(sp_group)

        # Initialize DeepSpeed
        if optim_config is not None and role == "actor":
            # Build DeepSpeed config
            # Parse mixed precision config (supports str or dict)
            mixed_precision_cfg = deepspeed_config.get("mixed_precision")
            fp16_enabled, bf16_enabled = _parse_mixed_precision_config(mixed_precision_cfg)
            comm_dtype = _parse_comm_dtype_from_mixed_precision(mixed_precision_cfg)
            zero_opt_overrides = _normalize_zero_opt_overrides(
                _cfg_get(deepspeed_config, "zero_optimization_overrides", None)
            )

            zero_stage = getattr(self.config.actor, "zero_stage", deepspeed_config.get("zero_stage", 2))

            dp_size = layout.dp_size if layout is not None else (
                torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            )
            sp_size = layout.sp_size if layout is not None else 1
            # NOTE:
            # - actor.ppo_mini_batch_size has already been normalized by DP in worker init.
            # - DeepSpeed GAS must match actual micro-batches per local mini-batch.
            #   Do not divide by dp_size again here, otherwise GAS becomes too small.
            per_rank_mini = max(1, self.config.actor.ppo_mini_batch_size // max(1, sp_size))
            micro_bsz = self.config.actor.get("ppo_micro_batch_size_per_gpu", 1) or 1
            if per_rank_mini % micro_bsz != 0:
                logger.warning(
                    "Actor local mini-batch (%s) is not divisible by micro-batch (%s); "
                    "DeepSpeed GAS will floor-divide.",
                    per_rank_mini,
                    micro_bsz,
                )
            ds_grad_accum = max(1, per_rank_mini // micro_bsz)
            # DS global train batch = micro_bsz * GAS * world_size(dp*sp)
            ds_train_batch_size = max(1, micro_bsz * ds_grad_accum * dp_size * sp_size)

            ds_config_kwargs = dict(
                optimizer_type=optim_config.get("optimizer", "AdamW"),
                train_batch_size=ds_train_batch_size,
                train_micro_batch_size_per_gpu=micro_bsz,
                gradient_accumulation_steps=ds_grad_accum,
                zero_stage=zero_stage,
                lr=optim_config.get("lr", 1e-5),
                betas=optim_config.get("betas", [0.9, 0.999]),
                eps=optim_config.get("eps", 1e-8),
                weight_decay=optim_config.get("weight_decay", 0.01),
                fp16_enabled=fp16_enabled,
                bf16_enabled=bf16_enabled,
                cpu_offload=deepspeed_config.get("param_offload", False),
                offload_optimizer=deepspeed_config.get("optimizer_offload", False),
                offload_dir=getattr(deepspeed_config, "offload_dir", None),
                gradient_clipping=self.config.actor.get("grad_clip", None),
                zero_optimization_overrides=zero_opt_overrides,
            )
            if comm_dtype is not None:
                ds_config_kwargs["communication_data_type"] = comm_dtype
            ds_config = get_deepspeed_config(**ds_config_kwargs)

            # Initialize DeepSpeed engine
            ds_engine, optimizer, _, lr_scheduler = initialize_deepspeed_engine(
                model=actor_module,
                config=ds_config,
                model_parameters=actor_module.parameters(),
            )


            return ds_engine, ds_engine.module, optimizer, lr_scheduler, actor_model_config
        else:
            # No optimizer for ref or rollout
            return None, actor_module, None, None, actor_model_config

    def _timed_build_actor(
        self,
        model_path: str,
        deepspeed_config: DeepSpeedEngineConfig,
        optim_config: Optional[dict],
        override_model_config: dict,
        use_remove_padding: bool = False,
        use_fused_kernels: bool = False,
        enable_gradient_checkpointing: bool = False,
        trust_remote_code: bool = False,
        use_liger: bool = False,
        role: str = "actor",
        layout: ParallelLayout | None = None,
    ):
        """Wrapper to time actor/ref build when debug is enabled."""
        t_build = _debug_timing(f"init_model:build_{role}")
        res = self._build_model_optimizer(
            model_path=model_path,
            deepspeed_config=deepspeed_config,
            optim_config=optim_config,
            override_model_config=override_model_config,
            use_remove_padding=use_remove_padding,
            use_fused_kernels=use_fused_kernels,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            trust_remote_code=trust_remote_code,
            use_liger=use_liger,
            role=role,
            layout=layout,
        )
        _debug_timing(f"init_model:build_{role}", t_build)
        return res

    def _build_rollout(self, trust_remote_code=False):
        """Build rollout engine (vLLM/SGLang)."""

        t_rollout = _debug_timing("build_rollout")
        rollout_config = omega_conf_to_dataclass(self.config.rollout, dataclass_type=RolloutConfig)
        model_config = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)

        # Initialize RNG snapshots (needed even for dummy mode)
        device = get_torch_device()
        self.torch_random_states = device.get_rng_state()
        self.gen_random_states = self.torch_random_states.clone()

        # Auto-fix: ensure vLLM TP divides model attention heads
        try:
            from transformers import AutoConfig as HF_AutoConfig
            from verl.utils.fs import copy_to_local

            local_path = copy_to_local(model_config.path, use_shm=model_config.get("use_shm", False))
            hf_cfg = HF_AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)

            def _extract_num_heads(cfg):
                for name in ("num_attention_heads", "num_heads", "n_head"):
                    if hasattr(cfg, name) and isinstance(getattr(cfg, name), (int,)):  # type: ignore
                        return int(getattr(cfg, name))
                # nested text_config (e.g., some multimodal models)
                tc = getattr(cfg, "text_config", None)
                if tc is not None:
                    for name in ("num_attention_heads", "num_heads", "n_head"):
                        if hasattr(tc, name) and isinstance(getattr(tc, name), (int,)):
                            return int(getattr(tc, name))
                return None

            def _extract_num_kv_heads(cfg):
                for name in ("num_key_value_heads", "num_kv_heads"):
                    if hasattr(cfg, name) and isinstance(getattr(cfg, name), (int,)):
                        return int(getattr(cfg, name))
                tc = getattr(cfg, "text_config", None)
                if tc is not None:
                    for name in ("num_key_value_heads", "num_kv_heads"):
                        if hasattr(tc, name) and isinstance(getattr(tc, name), (int,)):
                            return int(getattr(tc, name))
                return None

            num_heads = _extract_num_heads(hf_cfg)
            num_kv_heads = _extract_num_kv_heads(hf_cfg)
            # Use DP world (not counting SP) for rollout TP validation when actor uses Ulysses
            world_size = (
                self.actor_layout.dp_size
                if self.actor_layout is not None
                else (torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1)
            )
            req_tp = int(getattr(rollout_config, "tensor_model_parallel_size", 1) or 1)
            def _divides_all(tp):
                cond1 = (num_heads is None) or (num_heads % tp == 0)
                cond2 = (num_kv_heads is None) or (num_kv_heads % tp == 0)
                return cond1 and cond2

            if (num_heads is not None or num_kv_heads is not None) and (not _divides_all(req_tp) or req_tp > world_size):
                # pick the largest divisor of num_heads that does not exceed world_size
                divisors = [d for d in range(world_size, 0, -1) if _divides_all(d)]
                new_tp = divisors[0] if divisors else 1
                if self.rank == 0:
                    logger.info(
                        "Adjust vLLM TP from %s to %s so that heads and kv_heads are divisible (world_size=%s).",
                        req_tp,
                        new_tp,
                        world_size,
                    )
                rollout_config.tensor_model_parallel_size = new_tp
        except Exception as e:  # best-effort; don't block rollout build
            if self.rank == 0:
                logger.warning("Skip vLLM TP auto-adjust due to: %s", e)

        # Build rollout device mesh (align with FSDP rollout topology)
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        infer_tp = int(getattr(rollout_config, "tensor_model_parallel_size", 1) or 1) * int(
            getattr(rollout_config, "data_parallel_size", 1) or 1
        )
        infer_pp = int(getattr(rollout_config, "pipeline_model_parallel_size", 1) or 1)
        infer_world_size = infer_tp * infer_pp
        assert world_size % max(1, infer_world_size) == 0, (
            f"rollout world_size {world_size} is not divisible by infer_world_size {infer_world_size}"
        )
        dp = world_size // max(1, infer_world_size)
        rollout_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
        )
        self.rollout_device_mesh = rollout_device_mesh
        self.layered_summon = getattr(self.config.rollout, "layered_summon", False)

        # Seed per rollout dp rank to keep generation deterministic across tp/pp
        gen_dp_rank = rollout_device_mesh["dp"].get_local_rank()
        get_torch_device().manual_seed(gen_dp_rank + 1000)
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

        _debug_timing("build_rollout", t_rollout)

        self.rollout = get_rollout_class(rollout_config.name, rollout_config.mode)(
            config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh
        )

        # Register dispatch info so Ray routing knows how to gather rollout outputs
        if rollout_config.name == "hf":
            self._register_dispatch_collect_info("rollout", dp_rank=self.rank, is_collect=True)
        else:
            is_collect = (
                rollout_device_mesh["infer_tp"].get_local_rank() == 0
                and rollout_device_mesh["infer_pp"].get_local_rank() == 0
            )
            self._register_dispatch_collect_info(
                "rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )

        self.base_sync_done: bool = "dummy" not in self.config.rollout.load_format

        # Switch to trainer mode for sync rollout
        if rollout_config.mode == "sync" and self._is_actor:
            run_coro(self.trainer_mode())


    async def rollout_mode(self):
        """Context switch to rollout mode."""
        _debug_async_state("rollout_mode:enter")
        if self._skip_rollout:
            # When rollout is skipped, no weight syncing is required.
            self.base_sync_done = True
            return

        deepspeed_config = getattr(self.config.actor, "deepspeed_config", {}) if self._is_actor else {}
        if self.actor_engine is not None:
            _ensure_engine_has_module(self.actor_engine, self.actor_module)
        engine_module = _get_engine_module(self.actor_engine)
        engine_has_module = engine_module is not None
        zero_stage = (
            getattr(self.config.actor, "zero_stage", deepspeed_config.get("zero_stage", 2)) if self._is_actor else 0
        )

        def _prepare_rollout_payload():
            _debug_print("rollout_mode:prepare_payload:start")
            aggressive_empty_cache(force_sync=True)

            if self._is_offload_param and engine_has_module:
                _debug_print("rollout_mode:prepare_payload:load_to_gpu")
                load_deepspeed_model_to_gpu(self.actor_engine)

            # Get model parameters for rollout - ensure we get full tensors
            peft_config = None
            base_model_params = None
            actor_module = engine_module if engine_has_module else self.actor_module
            peft_model = getattr(actor_module, "_fsdp_wrapped_module", actor_module)
            if hasattr(peft_model, "peft_config"):
                peft_config = peft_model.peft_config.get("default", None)

            # DeepSpeed engine - need to handle ZeRO partitioned parameters
            # For ZeRO-2, weights are not partitioned, only optimizer states
            # For ZeRO-3, weights ARE partitioned and need gathering
            if engine_has_module:
                if zero_stage >= 3:
                    params = _gather_zero3_state_dict(actor_module, self.actor_engine)
                elif hasattr(self.actor_engine, "get_full_state_dict"):
                    _debug_print("rollout_mode:prepare_payload:get_full_state_dict")
                    params = self.actor_engine.get_full_state_dict()
                else:
                    params = self.actor_engine.module.state_dict()
            else:
                params = actor_module.state_dict()

            if peft_config is not None:
                # Align with FSDP: when base_sync_done==False, send base weights; when True, send LoRA only
                if not self.base_sync_done:
                    params = collect_lora_params(
                        module=actor_module, layered_summon=self.layered_summon, base_sync_done=self.base_sync_done
                    )
                    params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
                else:
                    params = collect_lora_params(
                        module=actor_module, layered_summon=self.layered_summon, base_sync_done=self.base_sync_done
                    )

            # Critical: Convert weight keys to match vLLM expectations (like FSDP does)
            params = convert_weight_keys(params, actor_module)

            # sleep_level=2: send base model weights separately
            if peft_config is not None and getattr(self.rollout, "sleep_level", None) == 2:
                base_model_params = collect_lora_params(
                    module=actor_module, layered_summon=self.layered_summon, base_sync_done=False
                )
                base_model_params = {replace_lora_wrapper(k, peft_config): v for k, v in base_model_params.items()}
                base_model_params = convert_weight_keys(base_model_params, actor_module)

            if self._is_offload_param and engine_has_module:
                _debug_print("rollout_mode:prepare_payload:offload_to_cpu")
                offload_deepspeed_model_to_cpu(self.actor_engine)

            device = get_device_id()

            # Use generator like FSDP does - avoid eager list creation
            # This may help with memory management and weight format compatibility
            def _yield_params(tensors):
                for name, param in tensors.items():
                    tensor = param.to(device, non_blocking=True)
                    if not tensor.is_contiguous():
                        tensor = tensor.contiguous()
                    yield name, tensor

            if peft_config is not None and self.base_sync_done:
                per_tensor_param = params.items()
            else:
                per_tensor_param = _yield_params(params)

            per_tensor_base_param = None
            if base_model_params is not None:
                per_tensor_base_param = _yield_params(base_model_params)

            # Ensure all transfers and memory operations are complete
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _debug_print("rollout_mode:prepare_payload:done")
            return params, base_model_params, per_tensor_param, per_tensor_base_param, peft_config

        params, base_model_params, per_tensor_param, per_tensor_base_param, peft_config = await asyncio.to_thread(
            _prepare_rollout_payload
        )
        set_expandable_segments(False)

        debug_main_kwargs: dict[str, Any] = {}
        debug_base_kwargs: dict[str, Any] = {}
        if self._debug_weight_sync_remaining > 0:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
            sync_prefix = f"r{rank}_t{int(time.time() * 1000)}"
            main_digest, main_n_tensors, main_numel = _weight_sync_digest(params.items())
            debug_main_kwargs = {
                "debug_weight_sync_id": f"{sync_prefix}_main",
                "debug_weight_sync_label": "main",
                "debug_weight_expected_digest": main_digest,
                "debug_weight_expected_num_tensors": main_n_tensors,
                "debug_weight_expected_numel": main_numel,
            }
            logger.info(
                "[WEIGHT_SYNC_DEBUG][sender] sync_id=%s label=main digest=%s tensors=%s numel=%s",
                debug_main_kwargs["debug_weight_sync_id"],
                main_digest,
                main_n_tensors,
                main_numel,
            )
            if base_model_params is not None:
                base_digest, base_n_tensors, base_numel = _weight_sync_digest(base_model_params.items())
                debug_base_kwargs = {
                    "debug_weight_sync_id": f"{sync_prefix}_base",
                    "debug_weight_sync_label": "base",
                    "debug_weight_expected_digest": base_digest,
                    "debug_weight_expected_num_tensors": base_n_tensors,
                    "debug_weight_expected_numel": base_numel,
                }
                logger.info(
                    "[WEIGHT_SYNC_DEBUG][sender] sync_id=%s label=base digest=%s tensors=%s numel=%s",
                    debug_base_kwargs["debug_weight_sync_id"],
                    base_digest,
                    base_n_tensors,
                    base_numel,
                )
            self._debug_weight_sync_remaining -= 1

        # Critical fix for DeepSpeed dummy mode compatibility
        # Unlike FSDP, DeepSpeed's weights in dummy mode cause CUDA errors in vLLM
        # Root cause: vLLM's dummy weight initialization is incompatible with DeepSpeed weight format
        # Solution: Skip weight update in dummy mode, let vLLM use its own dummy weights
        if self._dummy_rollout:
            # Dummy mode: Skip weight update entirely
            logger.info("Dummy mode: Skipping weight update, vLLM will use its own dummy-initialized weights")
            del params, base_model_params, per_tensor_param, per_tensor_base_param
            aggressive_empty_cache(force_sync=True)
            # Mark as synced to prevent future update attempts
            self.base_sync_done = True
        else:
            # Normal mode: Update weights as usual (this works for DeepSpeed)
            if self.config.rollout.free_cache_engine:
                t_resume = time.perf_counter()
                _debug_async_state("rollout_mode:resume_weights:start")
                await self.rollout.resume(tags=["weights"])
                _debug_async_state("rollout_mode:resume_weights:done", t_resume)

            t_update = time.perf_counter()
            _debug_async_state("rollout_mode:update_weights:start")
            if per_tensor_base_param is not None:
                await self.rollout.update_weights(per_tensor_base_param, base_sync_done=False, **debug_base_kwargs)
            await self.rollout.update_weights(
                per_tensor_param, peft_config=peft_config, base_sync_done=self.base_sync_done, **debug_main_kwargs
            )
            _debug_async_state("rollout_mode:update_weights:done", t_update)
            del params, base_model_params, per_tensor_param, per_tensor_base_param
            aggressive_empty_cache(force_sync=True)

            if self.config.rollout.free_cache_engine:
                t_kv = time.perf_counter()
                _debug_async_state("rollout_mode:resume_kv_cache:start")
                await self.rollout.resume(tags=["kv_cache"])
                _debug_async_state("rollout_mode:resume_kv_cache:done", t_kv)

            # Set base_sync_done to True after first sync
            if not self.base_sync_done:
                self.base_sync_done = True

        self.torch_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.gen_random_states)

    async def trainer_mode(self):
        """Context switch to trainer mode."""
        _debug_async_state("trainer_mode:enter")
        if self._skip_rollout:
            return

        if self.config.rollout.free_cache_engine:
            await self.rollout.release()

        engine_module = _get_engine_module(self.actor_engine)
        if engine_module is not None:
            engine_module.train()
        else:
            self.actor_module.train()

        aggressive_empty_cache(force_sync=True)
        set_expandable_segments(True)

        # Restore random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize models and engines."""
        import_external_libs(self.config.model.get("external_lib", None))
        _debug_print("init_model:enter")

        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)
        trust_remote_code = self.config.model.get("trust_remote_code", False)

        # Load local model path
        t_cp_actor = _debug_timing("init_model:copy_to_local_actor")
        local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
        _debug_timing("init_model:copy_to_local_actor", t_cp_actor)

        # Initialize tokenizer/processor (before model creation)
        t_tok = _debug_timing("init_model:tokenizer_processor")
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)
        _debug_timing("init_model:tokenizer_processor", t_tok)

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template

        # Load generation config
        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        # Build actor model with DeepSpeed
        if self._is_actor or self._is_rollout:
            if self._is_actor:
                optim_config = self.config.actor.optim
                deepspeed_config = omega_conf_to_dataclass(self.config.actor.deepspeed_config)
            else:
                optim_config = None
                deepspeed_config = DeepSpeedEngineConfig()

            _debug_print("init_model:build_actor_enter")
            (
                self.actor_engine,
                self.actor_module,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._timed_build_actor(
                model_path=local_path,
                deepspeed_config=deepspeed_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=trust_remote_code,
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                layout=self.actor_layout,
            )
            _debug_print("init_model:build_actor_exit")

            # Ensure actor_module is always the underlying model, not the DeepSpeed engine.
            if self.actor_engine is not None:
                engine_module = _get_engine_module(self.actor_engine)
                if engine_module is not None:
                    self.actor_module = engine_module

            if self._is_offload_param and self.actor_engine is not None:
                offload_deepspeed_model_to_cpu(self.actor_engine)

        # Build actor wrapper
        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            self.actor = DeepSpeedPPOActor(config=actor_cfg, actor_module=self.actor_module, engine=self.actor_engine)

        # Build rollout
        if self._is_rollout:
            _debug_print("init_model:build_rollout_enter")
            self._build_rollout(trust_remote_code=trust_remote_code)
            _debug_print("init_model:build_rollout_exit")

        # Build reference policy
        if self._is_ref:
            ref_model_path = self.config.model.path
            ref_model = self.config.ref.get("model", None)
            if ref_model is not None:
                ref_model_path = ref_model.get("path", self.config.model.path)

            if self.rank == 0:
                logger.info("reference model: %s", ref_model_path)

            t_cp_ref = _debug_timing("init_model:copy_to_local_ref")
            local_path = copy_to_local(ref_model_path, use_shm=use_shm)
            _debug_timing("init_model:copy_to_local_ref", t_cp_ref)
            (
                self.ref_engine,
                self.ref_module,
                _,
                _,
                _,
            ) = self._timed_build_actor(
                model_path=local_path,
                deepspeed_config=omega_conf_to_dataclass(self.config.ref.deepspeed_config),
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=trust_remote_code,
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
                layout=self.ref_layout,
            )
            _debug_print("init_model:build_ref_exit")

            if self.ref_engine is not None:
                ref_engine_module = _get_engine_module(self.ref_engine)
                if ref_engine_module is not None:
                    self.ref_module = ref_engine_module

            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module)

        # Create checkpoint manager and flops counter
        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)

            # DeepSpeedCheckpointManager expects worker object (accesses self.engine.engine)
            # Store engine reference for checkpoint manager to access
            self.engine = self.actor_engine
            self.checkpoint_manager = DeepSpeedCheckpointManager(engine=self)

        if not self._is_actor and self._is_rollout:
            # Standalone rollout checkpoint manager (load only)
            # Store engine reference for checkpoint manager to access
            self.engine = self.actor_engine if self.actor_engine is not None else None
            if self.engine is not None:
                self.checkpoint_manager = DeepSpeedCheckpointManager(engine=self)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step: int = 0, max_ckpt_to_keep: int | None = None):
        """Save actor (DeepSpeed) checkpoint using native DeepSpeed format.

        Directory layout mirrors FSDP manager style: <root>/step_<global_step>/
        """
        import torch

        if not self._is_actor or self.actor_engine is None:
            return

        # Expose engine handle for checkpoint manager
        self.engine = self.actor_engine
        _ensure_engine_has_module(self.actor_engine, self.actor_module)

        if self._is_offload_param:
            load_deepspeed_model_to_gpu(self.actor_engine)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if self._is_offload_param:
            offload_deepspeed_model_to_cpu(self.actor_engine)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, step: int | None = None, del_local_after_load: bool = True):
        """Load actor (DeepSpeed) checkpoint saved by `save_checkpoint`."""
        import torch

        if not self._is_actor or self.actor_engine is None:
            return {}

        # Compatibility with trainer resume logic: when caller passes None, treat as no-op.
        # This mirrors FSDP manager behavior and avoids TypeError inside the DS checkpoint manager.
        if local_path is None or (isinstance(local_path, str) and local_path.strip() == ""):
            return {}

        # Expose engine handle for checkpoint manager
        self.engine = self.actor_engine

        if self._is_offload_param:
            load_deepspeed_model_to_gpu(self.actor_engine)

        state = self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=None, del_local_after_load=del_local_after_load
        )

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if self._is_offload_param:
            offload_deepspeed_model_to_cpu(self.actor_engine)

        return state

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_actor(self, data: DataProto):
        """Update actor policy using PPO."""
        assert self._is_actor
        if self._is_offload_param:
            load_deepspeed_model_to_gpu(self.actor_engine)

        data = data.to("cpu")

        manager = self.actor_sharding_manager
        ctx = manager if manager is not None else nullcontext()
        with ctx:
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/actor"] = (
                estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            )
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            if self.actor_lr_scheduler is not None:
                lr = self.actor_lr_scheduler.get_last_lr()[0]
                self.actor_lr_scheduler.step()
            else:
                lr = self.config.actor.optim.lr
            metrics["actor/lr"] = lr

            output = DataProto(meta_info={"metrics": metrics})
        output = output.to("cpu")

        if self._is_offload_param:
            offload_deepspeed_model_to_cpu(self.actor_engine)

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    def generate_sequences(self, prompts: DataProto):
        """Generate sequences using vLLM/SGLang rollout."""
        assert self._is_rollout

        prompts = prompts.to(get_device_id())

        eos_id = (
            self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id
        )
        pad_id = (
            self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id
        )

        if prompts.meta_info is None:
            prompts.meta_info = {}
        prompts.meta_info.setdefault("eos_token_id", eos_id)
        prompts.meta_info.setdefault("pad_token_id", pad_id)

        timing_generate: dict[str, float] = {}

        # Critical fix for dummy mode: Skip actual generation, craft lightweight placeholder outputs
        # This keeps the PPO training stack exercised without invoking vLLM kernels.
        if self._dummy_rollout:
            import time
            # Dummy mode also needs mode switching for RNG consistency
            if self._is_actor:
                run_coro(self.rollout_mode())

            start = time.perf_counter()

            idx = prompts.batch["input_ids"]
            batch_size = idx.size(0)
            device = idx.device

            if "attention_mask" in prompts.batch.keys():
                attention_mask = prompts.batch["attention_mask"]
            else:
                attention_mask = torch.ones_like(idx, dtype=torch.int64, device=device)

            if "position_ids" in prompts.batch.keys():
                position_ids = prompts.batch["position_ids"]
            else:
                seq_len = idx.size(-1)
                base_positions = torch.arange(seq_len, device=device, dtype=torch.int64)
                position_ids = base_positions.unsqueeze(0).expand_as(idx)

            eos_token_meta = prompts.meta_info.get("eos_token_id", eos_id)
            if isinstance(eos_token_meta, (list, tuple)):
                eos_token_value = eos_token_meta[0]
            elif isinstance(eos_token_meta, torch.Tensor):
                eos_token_value = eos_token_meta.view(-1)[0].item()
            elif eos_token_meta is None:
                eos_token_value = pad_id
            else:
                eos_token_value = eos_token_meta

            try:
                eos_token_value = int(eos_token_value)
            except (TypeError, ValueError):
                eos_token_value = int(pad_id)

            response_length = 1
            responses = torch.full(
                (batch_size, response_length), eos_token_value, dtype=idx.dtype, device=device
            )
            seq = torch.cat([idx, responses], dim=-1)

            delta_position_id = torch.arange(
                1, response_length + 1, device=position_ids.device, dtype=position_ids.dtype
            )
            delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
            if position_ids.dim() == 3:  # e.g. mrope (batch, num_heads, seq_len)
                delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(
                    batch_size, position_ids.size(1), -1
                )
            response_position_ids = position_ids[..., -1:] + delta_position_id
            extended_position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

            response_attention_mask = torch.ones(
                (batch_size, response_length), dtype=attention_mask.dtype, device=attention_mask.device
            )
            extended_attention_mask = torch.cat([attention_mask, response_attention_mask], dim=-1)

            batch = TensorDict(
                {
                    "prompts": idx,
                    "responses": responses,
                    "input_ids": seq,
                    "attention_mask": extended_attention_mask,
                    "position_ids": extended_position_ids,
                },
                batch_size=batch_size,
            )

            if hasattr(prompts.batch, "get") and prompts.batch.get("rollout_log_probs") is not None:
                batch["rollout_log_probs"] = prompts.batch["rollout_log_probs"]

            non_tensor_batch = (
                prompts.non_tensor_batch.copy()
                if isinstance(prompts.non_tensor_batch, dict)
                else dict(prompts.non_tensor_batch or {})
            )
            non_tensor_batch.pop("raw_prompt_ids", None)
            non_tensor_batch.pop("multi_modal_data", None)

            timing_generate["generate_sequences"] = time.perf_counter() - start

            timing_generate_topk_ratio, timing_generate_min, timing_generate_max = topk_reduce_ratio_min_max(
                timing_generate["generate_sequences"]
            )
            timing_generate = reduce_timing(timing_generate)
            timing_generate.update(
                {
                    "generation_timing/max": timing_generate_max,
                    "generation_timing/min": timing_generate_min,
                    "generation_timing/topk_ratio": timing_generate_topk_ratio,
                }
            )

            meta_info = dict(prompts.meta_info or {})
            meta_info["timing"] = timing_generate

            output = DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)
            logger.info("Dummy mode: Skipped vLLM generation, returned prompts as placeholder responses")

            # Switch back to trainer mode after dummy generation
            if self._is_actor:
                run_coro(self.trainer_mode())

            return output

        # Normal mode: Use vLLM for actual generation
        if self._is_actor:
            run_coro(self.rollout_mode())

        t_generate = time.perf_counter()
        output = self.rollout.generate_sequences(prompts=prompts)
        timing_generate["generate_sequences"] = time.perf_counter() - t_generate

        if self._is_actor:
            run_coro(self.trainer_mode())

        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = topk_reduce_ratio_min_max(
            timing_generate["generate_sequences"]
        )
        timing_generate = reduce_timing(timing_generate)
        timing_generate.update(
            {
                "generation_timing/max": timing_generate_max,
                "generation_timing/min": timing_generate_min,
                "generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )
        if output.meta_info is None:
            output.meta_info = {}
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")
        get_torch_device().empty_cache()

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_log_prob(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_deepspeed_model_to_gpu(self.actor_engine)
        _ensure_engine_has_module(self.actor_engine, self.actor_module)
        engine_module = _get_engine_module(self.actor_engine)
        if engine_module is not None:
            self.actor_module = engine_module
            _maybe_move_module_to_device(engine_module, get_device_id())

        manager = self.actor_sharding_manager
        ctx = manager if manager is not None else nullcontext()

        with ctx:
            is_lora = data.meta_info.pop("is_lora", False)
            adapter_ctx = self.actor.disable_adapter() if is_lora else nullcontext()

            data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
            data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
            data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
            data.meta_info["temperature"] = self.config.rollout.temperature

            with adapter_ctx:
                logprob_output = self.actor.compute_log_prob(data=data, calculate_entropy=True)

            # dp_actor returns a dict; keep tuple compatibility for other actor implementations.
            if isinstance(logprob_output, dict):
                log_probs = logprob_output.get("log_probs")
                if log_probs is None:
                    log_probs = logprob_output.get("log_prob")

                entropys = logprob_output.get("entropys")
                if entropys is None:
                    entropys = logprob_output.get("entropy")
            else:
                log_probs, entropys = logprob_output

            output = DataProto.from_dict(
                tensors={"old_log_probs": log_probs, "entropys": entropys},
                meta_info={"temperature": self.config.rollout.temperature},
            )

        output = output.to("cpu")

        if self._is_offload_param:
            offload_deepspeed_model_to_cpu(self.actor_engine)

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_ref_log_prob(self, data: DataProto):
        if self._is_lora:
            data.meta_info["is_lora"] = True
            data = self.compute_log_prob(data)
            data = DataProto.from_dict(tensors={"ref_log_prob": data.batch["old_log_probs"]})
            return data

        assert self._is_ref

        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz

        manager = self.ref_sharding_manager
        ctx = manager if manager is not None else nullcontext()
        with ctx:
            data = data.to("cpu")
            log_prob, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
            output = DataProto.from_dict(tensors={"ref_log_prob": log_prob})

        return output.to("cpu")

    def get_zeromq_address(self):
        if hasattr(self, "rollout") and self.rollout is not None:
            return getattr(self.rollout, "get_zeromq_address", lambda: None)()
        return None


class DeepSpeedPPOActor(DataParallelPPOActor):
    """PPO actor that integrates DeepSpeed optimizer/ZeRO workflow."""

    def __init__(self, config, actor_module, engine):
        # If a DeepSpeed engine is mistakenly passed as actor_module, unwrap to the underlying module.
        try:
            from deepspeed import DeepSpeedEngine  # type: ignore
        except Exception:
            DeepSpeedEngine = None  # pragma: no cover
        if DeepSpeedEngine is not None and isinstance(actor_module, DeepSpeedEngine):
            actor_module = _get_engine_module(actor_module) or actor_module.module

        # Backward-compat: DataParallelPPOActor expects several FSDP-specific fields.
        ds_cfg = getattr(config, "deepspeed", {}) or {}
        if not hasattr(config, "ulysses_sequence_parallel_size"):
            sp_size = ds_cfg.get("ulysses_sequence_parallel_size", 1) if isinstance(ds_cfg, dict) else getattr(ds_cfg, "ulysses_sequence_parallel_size", 1)
            setattr(config, "ulysses_sequence_parallel_size", sp_size)
        if not hasattr(config, "entropy_from_logits_with_chunking"):
            setattr(config, "entropy_from_logits_with_chunking", False)
        if not hasattr(config, "fsdp_config"):
            dtype = ds_cfg.get("dtype", "bfloat16") if isinstance(ds_cfg, dict) else getattr(ds_cfg, "dtype", "bfloat16")
            setattr(config, "fsdp_config", {"dtype": dtype})

        super().__init__(config=config, actor_module=actor_module, actor_optimizer=engine.optimizer)
        self.deepspeed_engine = engine
        self._use_manual_backward = bool(int(os.getenv("DS_USE_MANUAL_BACKWARD", "0")))
        self._is_offload_param = False
        self._zero_stage = 0
        try:
            ds_cfg_obj = getattr(config, "deepspeed", None) or getattr(config, "deepspeed_config", None)
            if ds_cfg_obj is not None:
                self._zero_stage = _get_zero_stage(ds_cfg_obj)
                self._is_offload_param = bool(_cfg_get(ds_cfg_obj, "param_offload", False)) and self._zero_stage >= 3
        except Exception:
            self._zero_stage = 0
            self._is_offload_param = False
        self._last_grad_layout: list[tuple[str, int]] = []

    def _get_grad_accum_steps(self) -> int:
        engine_attr = getattr(self.deepspeed_engine, "gradient_accumulation_steps", None)
        steps = engine_attr() if callable(engine_attr) else engine_attr
        if steps is None:
            steps = max(1, self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu)
        return steps

    def _should_use_zero2_inactive_micro_workaround(self, *, loss_mode: str, entropy_coeff: float) -> bool:
        """Gate the ZeRO-2-only workaround for inactive vanilla PPO micros.

        In vanilla PPO without KL/entropy terms, some micro-batches can have no
        active policy tokens after clipping. ZeRO-2 may then diverge from other
        stages if accumulation boundaries are advanced without any gradient
        contribution. This guard keeps the workaround tightly scoped.
        """
        if self._zero_stage != 2:
            return False
        if loss_mode != "vanilla":
            return False
        if entropy_coeff != 0:
            return False
        if bool(self.config.use_kl_loss):
            return False
        # Emergency off-switch for troubleshooting.
        if os.getenv("VERL_DS_ZERO2_SKIP_INACTIVE_MICRO", "1").lower() in {"0", "false", "off", "no"}:
            return False
        return True

    def _vanilla_micro_has_policy_grad(
        self,
        *,
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        rollout_is_weights: torch.Tensor | None,
    ) -> bool:
        clip_ratio = float(self.config.clip_ratio)
        clip_ratio_low = float(self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio)
        clip_ratio_high = float(self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio)
        clip_ratio_c = float(self.config.get("clip_ratio_c", 3.0))
        if clip_ratio_c <= 1.0:
            clip_ratio_c = 3.0

        neg_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
        ratio = torch.exp(neg_approx_kl)
        valid = response_mask > 0

        active_pos = (advantages > 0) & (ratio <= (1.0 + clip_ratio_high))
        active_neg = (advantages < 0) & (ratio >= (1.0 - clip_ratio_low)) & (ratio <= clip_ratio_c)
        active = valid & (active_pos | active_neg)
        if rollout_is_weights is not None:
            active = active & (rollout_is_weights != 0)
        return bool(active.any().item())

    def update_policy(self, data: DataProto):
        self.actor_module.train()

        if self._is_offload_param:
            load_deepspeed_model_to_gpu(self.deepspeed_engine)
        _ensure_engine_has_module(self.deepspeed_engine, self.actor_module)
        engine_module = _get_engine_module(self.deepspeed_engine)
        if engine_module is not None:
            self.actor_module = engine_module
            _maybe_move_module_to_device(engine_module, get_device_id())
        _debug_first_param_device("actor.update_policy.before_forward", self.actor_module)

        # Optional per-step RNG seed for reproducibility
        if "rng_seed" in data.meta_info:
            rng_seed = int(data.meta_info["rng_seed"])
            torch.manual_seed(rng_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(rng_seed)

        temperature = data.meta_info["temperature"]

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        tis_cap = getattr(self.config, "tis_imp_ratio_cap", 0)
        if tis_cap > 0:
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        if tis_cap > 0:
            assert "rollout_log_probs" in data.batch.keys(), (
                "Truncated Importance Sampling (TIS) requires `actor_rollout_ref.rollout.calculate_log_probs=True` "
                "and is not currently supported in Server mode (agent loop)."
            )

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        mini_batches = data.split(self.config.ppo_mini_batch_size)
        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                grad_accum_steps = self._get_grad_accum_steps()
                sp_factor = max(1, self.ulysses_sequence_parallel_size)
                dp_sz = torch.distributed.get_world_size() // sp_factor if torch.distributed.is_initialized() else 1
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                if self._use_manual_backward:
                    self.actor_optimizer.zero_grad()
                else:
                    _safe_zero_grad(self.deepspeed_engine)

                loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                policy_loss_fn = get_policy_loss_fn(loss_mode)
                zero2_step_each_micro = (
                    self._zero_stage == 2
                    and os.getenv("VERL_DS_ZERO2_STEP_EACH_MICRO", "0").lower() in {"1", "true", "on", "yes"}
                )
                use_zero2_inactive_micro_workaround = self._should_use_zero2_inactive_micro_workaround(
                    loss_mode=loss_mode,
                    entropy_coeff=float(self.config.entropy_coeff),
                )
                if zero2_step_each_micro:
                    # Canonical DeepSpeed usage: call step() every micro and let engine decide update boundary.
                    # This path is ZeRO-2 specific and bypasses the deferred inactive-micro workaround.
                    use_zero2_inactive_micro_workaround = False
                deferred_policy_loss: torch.Tensor | None = None
                last_inactive_policy_loss: torch.Tensor | None = None
                skipped_inactive_micros = 0
                for idx, micro_batch in enumerate(micro_batches):
                    micro_batch = micro_batch.to(get_device_id())
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    rollout_log_probs = model_inputs.get("rollout_log_probs")
                    advantages = model_inputs["advantages"]
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = entropy_coeff != 0
                    _ensure_engine_has_module(self.deepspeed_engine, self.actor_module)
                    forward_out = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )
                    if isinstance(forward_out, dict):
                        log_prob = forward_out.get("log_probs", None)
                        entropy = forward_out.get("entropys", None)
                    else:
                        entropy, log_prob = forward_out
                    if log_prob is None:
                        raise RuntimeError("DeepSpeed actor forward did not return log_probs")

                    old_log_prob = log_prob.detach() if on_policy else model_inputs["old_log_probs"]

                    policy_loss_out = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )

                    if (
                        isinstance(policy_loss_out, tuple)
                        and len(policy_loss_out) == 2
                        and isinstance(policy_loss_out[1], dict)
                    ):
                        pg_loss, pg_metrics = policy_loss_out
                    else:
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_out
                        pg_metrics = {
                            "actor/pg_clipfrac": pg_clipfrac,
                            "actor/ppo_kl": ppo_kl,
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower,
                        }

                    # Normalize metric values for logging
                    pg_metrics = {
                        k: (v.detach().item() if torch.is_tensor(v) else float(v))
                        for k, v in pg_metrics.items()
                    }

                    if entropy_coeff != 0 and entropy is None:
                        entropy = torch.zeros_like(log_prob)
                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef

                        micro_batch_metrics = {
                            "actor/kl_loss": kl_loss.detach().item(),
                            "actor/kl_coef": self.config.kl_loss_coef,
                        }
                    else:
                        micro_batch_metrics = {}

                    is_last_micro = idx == len(micro_batches) - 1
                    if use_zero2_inactive_micro_workaround:
                        micro_has_grad = self._vanilla_micro_has_policy_grad(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            rollout_is_weights=rollout_is_weights,
                        )
                        if micro_has_grad:
                            if deferred_policy_loss is None:
                                deferred_policy_loss = policy_loss
                            else:
                                self.deepspeed_engine.set_gradient_accumulation_boundary(False)
                                self.deepspeed_engine.backward(deferred_policy_loss, scale_wrt_gas=True)
                                deferred_policy_loss = policy_loss
                        else:
                            skipped_inactive_micros += 1
                            last_inactive_policy_loss = policy_loss

                        if is_last_micro:
                            final_policy_loss = deferred_policy_loss
                            if final_policy_loss is None:
                                # Keep DeepSpeed accumulation state valid even if all micros are inactive.
                                final_policy_loss = (
                                    last_inactive_policy_loss if last_inactive_policy_loss is not None else policy_loss
                                )
                            self.deepspeed_engine.set_gradient_accumulation_boundary(True)
                            self.deepspeed_engine.backward(final_policy_loss, scale_wrt_gas=True)
                            if zero2_step_each_micro:
                                self.deepspeed_engine.step()
                    else:
                        # Keep parity with FSDP path:
                        # - no extra SP scaling on policy loss
                        # - boundary controls only when step_each_micro is disabled
                        if not zero2_step_each_micro:
                            self.deepspeed_engine.set_gradient_accumulation_boundary(is_last_micro)
                        self.deepspeed_engine.backward(policy_loss, scale_wrt_gas=True)
                        if zero2_step_each_micro:
                            self.deepspeed_engine.step()

                    # Collect metrics (loss will be properly scaled for logging)
                    if self.config.use_dynamic_bsz:
                        metric_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        metric_scale_factor = 1.0 / grad_accum_steps

                    if entropy is not None:
                        try:
                            entropy_val = float(torch.nan_to_num(entropy.float().mean(), nan=0.0).item())
                        except Exception:
                            entropy_val = 0.0
                    else:
                        entropy_val = 0.0

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item() * metric_scale_factor,
                            "actor/entropy": entropy_val,
                            "actor/grad_accum_steps": float(grad_accum_steps),
                            "actor/sp_size": float(sp_factor),
                            "actor/dp_size": float(dp_sz if 'dp_sz' in locals() else 1),
                            "actor/micro_batches_per_step": float(len(micro_batches)),
                            "actor/sp_loss_divisor": 1.0,
                            "actor/skipped_inactive_micro": float(skipped_inactive_micros),
                        }
                    )
                    micro_batch_metrics.update(pg_metrics)
                    append_to_dict(metrics, micro_batch_metrics)

                # Prefer DeepSpeed global grad norm when ZeRO is active; otherwise fall back to local clip.
                ds_zero = self._zero_stage
                if ds_zero and hasattr(self.deepspeed_engine, "get_global_grad_norm"):
                    try:
                        grad_norm_val = float(self.deepspeed_engine.get_global_grad_norm())
                    except Exception:
                        grad_norm_val = 0.0
                else:
                    grad_norm_val = float(
                        torch.nn.utils.clip_grad_norm_(
                            self.actor_module.parameters(),
                            max_norm=self.config.grad_clip,
                            norm_type=2.0,
                        )
                    )
                append_to_dict(metrics, {"actor/grad_norm": grad_norm_val})

                # Step only once after all micro batches unless ZeRO-2 canonical micro-stepping is enabled.
                if not zero2_step_each_micro:
                    self.deepspeed_engine.step()

        if not self._use_manual_backward:
            _safe_zero_grad(self.deepspeed_engine)

        _log_ds_step_metrics(
            " actor",
            metrics,
            [
                "actor/pg_loss",
                "actor/ppo_kl",
                "actor/pg_clipfrac",
                "actor/pg_clipfrac_lower",
                "actor/kl_loss",
                "actor/entropy",
                "actor/grad_norm",
            ],
        )
        _log_ds_engine_state(" actor", self.deepspeed_engine)

        if self._is_offload_param:
            offload_deepspeed_model_to_cpu(self.deepspeed_engine)
        return metrics


class DeepSpeedPPOCritic(DataParallelPPOCritic):
    """PPO critic that delegates backward/step to a DeepSpeed engine."""

    def __init__(self, config, critic_module, engine):
        try:
            from deepspeed import DeepSpeedEngine  # type: ignore
        except Exception:
            DeepSpeedEngine = None  # pragma: no cover
        if DeepSpeedEngine is not None and isinstance(critic_module, DeepSpeedEngine):
            critic_module = _get_engine_module(critic_module) or critic_module.module

        super().__init__(config=config, critic_module=critic_module, critic_optimizer=engine.optimizer)
        self.deepspeed_engine = engine
        self._use_manual_backward = bool(int(os.getenv("DS_USE_MANUAL_BACKWARD", "0")))
        self._is_offload_param = False
        self._zero_stage = 0
        try:
            ds_cfg_obj = getattr(config, "deepspeed", None) or getattr(config, "deepspeed_config", None)
            if ds_cfg_obj is not None:
                self._zero_stage = _get_zero_stage(ds_cfg_obj)
                self._is_offload_param = bool(_cfg_get(ds_cfg_obj, "param_offload", False)) and self._zero_stage >= 3
        except Exception:
            self._zero_stage = 0
            self._is_offload_param = False

    def _get_grad_accum_steps(self) -> int:
        engine_attr = getattr(self.deepspeed_engine, "gradient_accumulation_steps", None)
        steps = engine_attr() if callable(engine_attr) else engine_attr
        if steps is None:
            steps = max(1, self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu)
        return steps

    def update_critic(self, data: DataProto):
        self.critic_module.train()

        _ensure_engine_has_module(self.deepspeed_engine, self.critic_module)
        if self._is_offload_param:
            load_deepspeed_model_to_gpu(self.deepspeed_engine)
        engine_module = _get_engine_module(self.deepspeed_engine)
        if engine_module is not None:
            self.critic_module = engine_module
            _maybe_move_module_to_device(engine_module, get_device_id())
        _debug_first_param_device("critic.update_critic.before_forward", self.critic_module)

        if 'rng_seed' in data.meta_info:
            rng_seed = data.meta_info['rng_seed']
            torch.manual_seed(rng_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(rng_seed)

        metrics = {}

        select_keys = [
            "input_ids",
            "responses",
            "response_mask",
            "attention_mask",
            "position_ids",
            "values",
            "returns",
        ]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                grad_accum_steps = self._get_grad_accum_steps()
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                sp_factor = max(1, self.ulysses_sequence_parallel_size)

                if self._use_manual_backward:
                    self.critic_optimizer.zero_grad()
                else:
                    _safe_zero_grad(self.deepspeed_engine)

                zero2_step_each_micro = (
                    self._zero_stage == 2
                    and os.getenv("VERL_DS_ZERO2_STEP_EACH_MICRO", "0").lower() in {"1", "true", "on", "yes"}
                )
                for idx, micro_batch in enumerate(micro_batches):
                    micro_batch = micro_batch.to(get_device_id())
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    values = model_inputs["values"]
                    returns = model_inputs["returns"]

                    _ensure_engine_has_module(self.deepspeed_engine, self.critic_module)
                    vpreds = self._forward_micro_batch(model_inputs)

                    # ----- SP-aware loss aggregation -----
                    vpredclipped = verl_F.clip_by_value(
                        vpreds, values - self.config.cliprange_value, values + self.config.cliprange_value
                    )
                    vf_losses1 = (vpreds - returns) ** 2
                    vf_losses2 = (vpredclipped - returns) ** 2
                    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)

                    loss_agg_mode = getattr(self.config, "loss_agg_mode", "token-mean")
                    token_sum = torch.nan_to_num(response_mask.sum(), nan=0.0)
                    loss_sum = torch.nan_to_num((clipped_vf_losses * response_mask).sum(), nan=0.0)
                    clip_used = torch.gt(vf_losses2, vf_losses1).float()
                    clip_used_sum = torch.nan_to_num((clip_used * response_mask).sum(), nan=0.0)
                    vpred_sum = torch.nan_to_num((vpreds * response_mask).sum(), nan=0.0)
                    seq_token_sum = torch.nan_to_num(response_mask.sum(dim=-1), nan=0.0)
                    seq_mask = (seq_token_sum > 0).float()
                    seq_count = torch.nan_to_num(seq_mask.sum(), nan=0.0)
                    seq_loss_token_sum = torch.nan_to_num((clipped_vf_losses * response_mask).sum(dim=-1), nan=0.0)
                    seq_loss_seqmean_sum = torch.nan_to_num((seq_loss_token_sum * seq_mask).sum(), nan=0.0)
                    seq_loss_tokmean = torch.nan_to_num(seq_loss_token_sum / (seq_token_sum + 1e-8), nan=0.0)
                    seq_loss_tokmean_sum = torch.nan_to_num((seq_loss_tokmean * seq_mask).sum(), nan=0.0)
                    seq_loss_total = torch.nan_to_num(seq_loss_token_sum.sum(), nan=0.0)

                    sp_group = get_ulysses_sequence_parallel_group()
                    if sp_group is not None and sp_factor > 1:
                        for t in (
                            token_sum,
                            loss_sum,
                            clip_used_sum,
                            vpred_sum,
                            seq_count,
                            seq_loss_seqmean_sum,
                            seq_loss_tokmean_sum,
                            seq_loss_total,
                        ):
                            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM, group=sp_group)

                    global_token_sum = token_sum.clamp_min(1.0)
                    dp_size = max(
                        1,
                        (torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1) // sp_factor,
                    )
                    if loss_agg_mode == "token-mean":
                        # ``scale_wrt_gas=True`` handles GA normalization in DeepSpeed.
                        # We still multiply by DP size so token-mean matches global
                        # token semantics used by the FSDP reference path.
                        vf_loss_dp_scale = float(dp_size)
                        vf_loss = 0.5 * (loss_sum / global_token_sum) * vf_loss_dp_scale
                    elif loss_agg_mode == "seq-mean-token-sum":
                        vf_loss_dp_scale = float(dp_size)
                        global_seq_count = seq_count.clamp_min(1.0)
                        vf_loss = 0.5 * (seq_loss_seqmean_sum / global_seq_count) * vf_loss_dp_scale
                    elif loss_agg_mode == "seq-mean-token-mean":
                        vf_loss_dp_scale = float(dp_size)
                        global_seq_count = seq_count.clamp_min(1.0)
                        vf_loss = 0.5 * (seq_loss_tokmean_sum / global_seq_count) * vf_loss_dp_scale
                    elif loss_agg_mode == "seq-mean-token-sum-norm":
                        loss_scale_factor = getattr(self.config, "loss_scale_factor", None)
                        if loss_scale_factor is None:
                            loss_scale_factor = int(response_mask.shape[-1])
                        loss_scale_factor = max(1.0, float(loss_scale_factor))
                        vf_loss_dp_scale = 1.0
                        vf_loss = 0.5 * (seq_loss_total / loss_scale_factor)
                    else:
                        raise ValueError(f"Unsupported critic loss_agg_mode: {loss_agg_mode}")
                    vf_clipfrac = (clip_used_sum / global_token_sum)
                    vpred_mean_global = (vpred_sum / global_token_sum).detach().item()
                    vpred_std = torch.nan_to_num(vpreds.float().std(), nan=0.0).item()
                    returns_std = torch.nan_to_num(returns.float().std(), nan=0.0).item()
                    per_token_mse = loss_sum / global_token_sum
                    per_token_rmse = torch.sqrt(torch.nan_to_num(per_token_mse, nan=0.0)).item()

                    is_last_micro = idx == len(micro_batches) - 1
                    if not zero2_step_each_micro:
                        self.deepspeed_engine.set_gradient_accumulation_boundary(is_last_micro)
                    self.deepspeed_engine.backward(vf_loss, scale_wrt_gas=True)
                    if zero2_step_each_micro:
                        self.deepspeed_engine.step()

                    # Collect metrics (loss will be properly scaled for logging)
                    if self.config.use_dynamic_bsz:
                        metric_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        metric_scale_factor = 1.0 / grad_accum_steps

                    micro_batch_metrics = {
                        "critic/vf_loss": vf_loss.detach().item() * metric_scale_factor,
                        "critic/vf_clipfrac": vf_clipfrac.detach().item(),
                        "critic/vpred_mean": vpred_mean_global,
                        "critic/vpred_std": vpred_std,
                        "critic/returns_std": returns_std,
                        "critic/num_tokens": global_token_sum.detach().item(),
                        "critic/rmse_per_token": per_token_rmse,
                        "critic/grad_accum_steps": grad_accum_steps,
                        "critic/sp_size": sp_factor,
                        "critic/dp_size": max(
                            1,
                            (torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1)
                            // sp_factor,
                        ),
                        "critic/micro_batches_per_step": len(micro_batches),
                        "critic/sp_loss_divisor": 1.0,
                        "critic/vf_loss_dp_scale": vf_loss_dp_scale,
                    }
                    append_to_dict(metrics, micro_batch_metrics)

                # Prefer DeepSpeed global grad norm when ZeRO is active; otherwise fall back to local clip.
                ds_zero = self._zero_stage
                if ds_zero and hasattr(self.deepspeed_engine, "get_global_grad_norm"):
                    try:
                        grad_norm_val = float(self.deepspeed_engine.get_global_grad_norm())
                    except Exception:
                        grad_norm_val = 0.0
                else:
                    grad_norm_val = float(
                        torch.nn.utils.clip_grad_norm_(
                            self.critic_module.parameters(),
                            max_norm=self.config.grad_clip,
                            norm_type=2.0,
                        )
                    )
                append_to_dict(metrics, {"critic/grad_norm": grad_norm_val})

                # Step only once after all micro batches unless ZeRO-2 canonical micro-stepping is enabled.
                if not zero2_step_each_micro:
                    self.deepspeed_engine.step()

        if not self._use_manual_backward:
            _safe_zero_grad(self.deepspeed_engine)

        _log_ds_step_metrics(
            " critic",
            metrics,
            [
                "critic/vf_loss",
                "critic/vf_clipfrac",
                "critic/vpred_mean",
                "critic/vpred_std",
                "critic/returns_std",
                "critic/grad_norm",
                "critic/rmse_per_token",
                "critic/num_tokens",
            ],
        )
        _log_ds_engine_state(" critic", self.deepspeed_engine)

        if self._is_offload_param:
            offload_deepspeed_model_to_cpu(self.deepspeed_engine)
        return metrics



class CriticWorker(Worker):
    """Clean DeepSpeed-based Critic Worker."""

    def __init__(self, config: DictConfig | DeepSpeedCriticConfig, **kwargs):
        Worker.__init__(self)

        if isinstance(config, DictConfig):
            critic_config = omega_conf_to_dataclass(config, dataclass_type=DeepSpeedCriticConfig)
        else:
            critic_config = config

        self.config: DeepSpeedCriticConfig = critic_config
        _normalize_ds_config(self.config)
        self.critic_sharding_manager = None
        self.layout: ParallelLayout | None = None

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
                device_id=get_device_id() if torch.cuda.is_available() else None,
            )

        # Build layout & register dispatch on DP dimension
        self.layout = build_parallel_layout(self.config)
        self._register_dispatch_collect_info("critic", dp_rank=self.layout.dp_rank, is_collect=self.layout.collect)

        self._is_offload_param = self.config.deepspeed_config.get("param_offload", False)
        # Ulysses SP for critic dynamic batching
        self.ulysses_sequence_parallel_size = self.layout.sp_size
        self._lora_rank = getattr(self.config.model, "lora_rank", 0)
        self._is_lora = self._lora_rank > 0

        normalize_critic_batches(self.config, self.layout.dp_size, sp_size=self.layout.sp_size)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize critic model with native DeepSpeed."""

        import_external_libs(self.config.model.get("external_lib", None))

        trust_remote_code = getattr(self.config.model, "trust_remote_code", False)
        use_shm = getattr(self.config.model, "use_shm", False)
        local_path = copy_to_local(self.config.model.path, use_shm=use_shm)

        tokenizer_path = getattr(self.config.model, "tokenizer_path", None) or self.config.model.path
        tokenizer_local_path = copy_to_local(tokenizer_path, use_shm=use_shm)
        self.tokenizer = hf_tokenizer(tokenizer_local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(tokenizer_local_path, trust_remote_code=trust_remote_code)

        custom_template = getattr(self.config.model, "custom_chat_template", None)
        if custom_template is not None:
            if self.processor is not None:
                self.processor.chat_template = custom_template
            elif self.tokenizer is not None:
                self.tokenizer.chat_template = custom_template

        override_config = getattr(self.config.model, "override_config", {}) or {}
        if isinstance(override_config, DictConfig):
            override_config = OmegaConf.to_container(override_config, resolve=True)
        override_config_kwargs = {}
        if self.tokenizer is not None:
            override_config_kwargs.update(
                {
                    "bos_token_id": self.tokenizer.bos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
            )
        override_config_kwargs.update(override_config)

        attn_impl = override_config_kwargs.get("attn_implementation", "flash_attention_2")
        critic_model_config = AutoConfig.from_pretrained(
            local_path,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_impl,
        )
        update_model_config(critic_model_config, override_config_kwargs=override_config_kwargs)
        critic_model_config.num_labels = 1
        critic_model_config.classifier_dropout = 0.0
        critic_model_config.hidden_dropout = 0.0
        critic_model_config.summary_dropout_prob = 0.0

        torch_dtype = PrecisionType.to_dtype(getattr(self.config.deepspeed_config, "model_dtype", "fp32"))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            critic_module = load_valuehead_model(
                local_path=local_path,
                torch_dtype=torch_dtype,
                model_config=critic_model_config,
                trust_remote_code=trust_remote_code,
            )

        use_remove_padding = getattr(self.config.model, "use_remove_padding", False)

        # Initialize Ulysses SP group for Critic if requested
        sp_size = int(getattr(self, "ulysses_sequence_parallel_size", 1))
        sp_group = None
        prev_sp_group = get_ulysses_sequence_parallel_group()
        if sp_size > 1 and torch.distributed.is_initialized():
            if self.layout is not None:
                ranks = list(range(self.layout.dp_rank * sp_size, (self.layout.dp_rank + 1) * sp_size))
            else:
                world = torch.distributed.get_world_size()
                assert world % sp_size == 0, f"world_size {world} must be divisible by ulysses sp_size {sp_size}"
                rank = torch.distributed.get_rank()
                group_id = rank // sp_size
                ranks = list(range(group_id * sp_size, (group_id + 1) * sp_size))
            sp_group = torch.distributed.new_group(ranks=ranks, backend=get_nccl_backend())
            set_ulysses_sequence_parallel_group(sp_group)
            torch.distributed.barrier(group=sp_group)

        apply_monkey_patch(
            model=critic_module,
            use_remove_padding=use_remove_padding,
            ulysses_sp_size=sp_size,
        )

        if sp_group is not None:
            torch.distributed.barrier(group=sp_group)
        set_ulysses_sequence_parallel_group(prev_sp_group)

        critic_module.to(torch_dtype)

        if getattr(self.config.model, "enable_gradient_checkpointing", False):
            critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if self._is_lora:
            print("Applying LoRA to critic module")
            critic_module.enable_input_require_grads()
            lora_config = {
                "task_type": TaskType.CAUSAL_LM,
                "r": self._lora_rank,
                "lora_alpha": getattr(self.config.model, "lora_alpha", 16),
                "target_modules": convert_to_regular_types(getattr(self.config.model, "target_modules", None)),
                "bias": "none",
            }
            exclude_modules = getattr(self.config.model, "exclude_modules", None)
            if exclude_modules is not None:
                lora_config["exclude_modules"] = convert_to_regular_types(exclude_modules)
            critic_module = get_peft_model(critic_module, LoraConfig(**lora_config))

        if self.rank == 0:
            print_model_size(critic_module)

        self.critic_model_config = critic_model_config
        self.critic_sharding_manager = DeepSpeedUlyssesShardingManager(sp_group)

        # Initialize DeepSpeed
        # Parse mixed precision config (supports str or dict)
        mixed_precision_cfg = self.config.deepspeed_config.get("mixed_precision")
        fp16_enabled, bf16_enabled = _parse_mixed_precision_config(mixed_precision_cfg)
        comm_dtype = _parse_comm_dtype_from_mixed_precision(mixed_precision_cfg)
        zero_opt_overrides = _normalize_zero_opt_overrides(
            _cfg_get(self.config.deepspeed_config, "zero_optimization_overrides", None)
        )

        zero_stage = getattr(self.config, "zero_stage", self.config.deepspeed_config.get("zero_stage", 2))

        dp_size = self.layout.dp_size if self.layout is not None else (
            torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        )
        sp_size = self.layout.sp_size if self.layout is not None else 1
        # NOTE:
        # - critic.ppo_mini_batch_size has already been normalized by DP in worker init.
        # - GAS must align with local micro-batch count for each mini-batch update.
        per_rank_mini = max(1, self.config.ppo_mini_batch_size // max(1, sp_size))
        micro_bsz = self.config.get("ppo_micro_batch_size_per_gpu", 1) or 1
        if per_rank_mini % micro_bsz != 0:
            logger.warning(
                "Critic local mini-batch (%s) is not divisible by micro-batch (%s); "
                "DeepSpeed GAS will floor-divide.",
                per_rank_mini,
                micro_bsz,
            )
        ds_grad_accum = max(1, per_rank_mini // micro_bsz)
        ds_train_batch_size = max(1, micro_bsz * ds_grad_accum * dp_size * sp_size)

        ds_config_kwargs = dict(
            optimizer_type=self.config.optim.get("optimizer", "AdamW"),
            train_batch_size=ds_train_batch_size,
            train_micro_batch_size_per_gpu=micro_bsz,
            gradient_accumulation_steps=ds_grad_accum,
            zero_stage=zero_stage,
            lr=self.config.optim.lr,
            betas=self.config.optim.get("betas", [0.9, 0.999]),
            eps=self.config.optim.get("eps", 1e-8),
            weight_decay=self.config.optim.get("weight_decay", 0.01),
            fp16_enabled=fp16_enabled,
            bf16_enabled=bf16_enabled,
            cpu_offload=self.config.deepspeed_config.get("param_offload", False),
            offload_optimizer=self.config.deepspeed_config.get("optimizer_offload", False),
            offload_dir=getattr(self.config.deepspeed_config, "offload_dir", None),
            gradient_clipping=self.config.get("grad_clip", None),
            zero_optimization_overrides=zero_opt_overrides,
        )
        if comm_dtype is not None:
            ds_config_kwargs["communication_data_type"] = comm_dtype
        ds_config = get_deepspeed_config(**ds_config_kwargs)
        self.critic_engine, optimizer, _, lr_scheduler = initialize_deepspeed_engine(
            model=critic_module,
            config=ds_config,
            model_parameters=critic_module.parameters(),
        )


        self.critic_module = self.critic_engine.module
        self.critic_optimizer = optimizer
        self.critic_lr_scheduler = lr_scheduler

        self.critic = DeepSpeedPPOCritic(
            config=self.config, critic_module=self.critic_module, engine=self.critic_engine
        )

        # Setup checkpoint manager for critic (mirror actor behavior)
        # Expose engine handle for DeepSpeedCheckpointManager via self.engine
        self.engine = self.critic_engine
        try:
            self.checkpoint_manager = DeepSpeedCheckpointManager(engine=self)
        except Exception:
            # Defer creation if environment lacks DS helpers; save/load will fallback to engine API
            self.checkpoint_manager = DeepSpeedCheckpointManager(engine=self)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    def compute_values(self, data: DataProto):
        """Run critic forward pass to compute value estimates."""
        if self._is_offload_param:
            load_deepspeed_model_to_gpu(self.critic_engine)

        _ensure_engine_has_module(self.critic_engine, self.critic_module)
        engine_module = _get_engine_module(self.critic_engine)
        if engine_module is not None:
            self.critic_module = engine_module
            _maybe_move_module_to_device(engine_module, get_device_id())
        _debug_first_param_device("critic.compute_values.before_forward", self.critic_module)

        micro_batch_size = getattr(self.config, "forward_micro_batch_size_per_gpu", None)
        if micro_batch_size is None:
            micro_batch_size = getattr(self.config, "ppo_micro_batch_size_per_gpu", 1)

        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = getattr(
            self.config, "forward_max_token_len_per_gpu", self.config.ppo_max_token_len_per_gpu
        )
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz

        data = data.to("cpu")  # tensors move to device inside critic.compute_values

        manager = self.critic_sharding_manager
        ctx = manager if manager is not None else nullcontext()
        with ctx:
            values = self.critic.compute_values(data=data)
        output = DataProto.from_dict(tensors={"values": values}).to("cpu")

        if self._is_offload_param:
            offload_deepspeed_model_to_cpu(self.critic_engine)

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    def update_critic(self, data: DataProto):
        """Update critic value function."""
        if self._is_offload_param:
            load_deepspeed_model_to_gpu(self.critic_engine)

        data = data.to("cpu")

        manager = self.critic_sharding_manager
        ctx = manager if manager is not None else nullcontext()
        with ctx:
            with Timer(name="update_critic", logger=None):
                metrics = self.critic.update_critic(data=data)

        if self.critic_lr_scheduler is not None:
            lr = self.critic_lr_scheduler.get_last_lr()[0]
            self.critic_lr_scheduler.step()
        else:
            lr = self.config.optim.lr
        metrics["critic/lr"] = lr

        output = DataProto(meta_info={"metrics": metrics})
        output = output.to("cpu")

        if self._is_offload_param:
            offload_deepspeed_model_to_cpu(self.critic_engine)

        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step: int = 0, max_ckpt_to_keep: int | None = None):
        """Save critic (DeepSpeed) checkpoint using native DeepSpeed format."""
        import torch

        # Ensure engine handle is set for checkpoint manager
        self.engine = getattr(self, "critic_engine", None)
        if self.engine is None:
            return
        _ensure_engine_has_module(self.critic_engine, self.critic_module)

        if self._is_offload_param:
            load_deepspeed_model_to_gpu(self.critic_engine)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        if torch.distributed.is_initialized():
            try:
                torch.distributed.barrier()
            except Exception:
                pass

        if self._is_offload_param:
            offload_deepspeed_model_to_cpu(self.critic_engine)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, step: int | None = None, del_local_after_load: bool = True):
        """Load critic (DeepSpeed) checkpoint saved by `save_checkpoint`."""
        import torch

        # Treat None or empty path as no-op for compatibility
        if local_path is None or (isinstance(local_path, str) and local_path.strip() == ""):
            return {}

        # Ensure engine handle is set for checkpoint manager
        self.engine = getattr(self, "critic_engine", None)
        if self.engine is None:
            return {}
        _ensure_engine_has_module(self.critic_engine, self.critic_module)

        if self._is_offload_param:
            load_deepspeed_model_to_gpu(self.critic_engine)

        state = self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=None, del_local_after_load=del_local_after_load
        )

        if torch.distributed.is_initialized():
            try:
                torch.distributed.barrier()
            except Exception:
                pass

        if self._is_offload_param:
            offload_deepspeed_model_to_cpu(self.critic_engine)

        return state


class RolloutWorker(Worker):
    """Standalone vLLM/SGLang Rollout Worker."""

    def __init__(self, config: DictConfig, **kwargs):
        Worker.__init__(self)
        self.config = config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize rollout engine only."""
        rollout_config = omega_conf_to_dataclass(self.config.rollout, dataclass_type=RolloutConfig)
        model_config = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)

        self.rollout = get_rollout_class(rollout_config.name, rollout_config.mode)(
            config=rollout_config, model_config=model_config, device_mesh=None
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    def generate_sequences(self, prompts: DataProto):
        """Generate sequences."""
        prompts = prompts.to(get_device_id())
        output = self.rollout.generate_sequences(prompts=prompts)
        return output


# Async variants
class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    """Async variant of ActorRolloutRefWorker."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self):
        await self.rollout_mode()
        return True

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def wake_up(self):
        await self.rollout_mode()
        return True

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def sleep(self):
        await self.trainer_mode()
        return True

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def get_zeromq_address(self):
        if hasattr(self, "rollout") and self.rollout is not None:
            getter = getattr(self.rollout, "get_zeromq_address", None)
            if callable(getter):
                return getter()
        return None


class RewardModelWorker(Worker):
    """
    DeepSpeed-based Reward Model Worker.

    Implements reward model inference using DeepSpeed ZeRO for memory efficiency.
    Supports AutoModelForTokenClassification models.
    """

    def __init__(self, config):
        Worker.__init__(self)

        self.config = config
        _normalize_ds_config(self.config)
        self.layout: ParallelLayout | None = None
        self.reward_sharding_manager: DeepSpeedUlyssesShardingManager | None = None
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
                device_id=get_device_id() if torch.cuda.is_available() else None,
            )

        # Build layout (supports DP; optional SP via ulysses_sequence_parallel_size)
        self.layout = build_parallel_layout(self.config)
        self.ulysses_sequence_parallel_size = self.layout.sp_size

        # Create training dispatch
        self._register_dispatch_collect_info("reward", dp_rank=self.layout.dp_rank, is_collect=self.layout.collect)

        self.use_remove_padding = self.config.model.get("use_remove_padding", False)

        # Normalize config
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= self.layout.dp_size
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

        self._is_offload_param = self.config.deepspeed_config.get("param_offload", False)

    def _build_model(self, config):
        """Build reward model with DeepSpeed."""
        from transformers import AutoConfig, AutoModelForTokenClassification

        use_shm = config.model.get("use_shm", False)
        local_path = copy_to_local(config.model.path, use_shm=use_shm)

        # Handle chat template switching if needed
        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_to_local(config.model.input_tokenizer, use_shm=use_shm)
            self.input_tokenizer = hf_tokenizer(
                input_tokenizer_local_path, trust_remote_code=config.model.get("trust_remote_code", False)
            )
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get("trust_remote_code", False))

        trust_remote_code = config.model.get("trust_remote_code", False)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        model_config.num_labels = 1

        # Determine torch dtype
        torch_dtype = self.config.deepspeed_config.get("model_dtype", "fp32")
        if torch_dtype == "fp32":
            torch_dtype = torch.float32
        elif torch_dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif torch_dtype == "fp16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.bfloat16  # default to bf16

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_config.classifier_dropout = 0.0
            reward_module = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=local_path,
                config=model_config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            # Initialize Ulysses SP group if requested
            sp_size = self.ulysses_sequence_parallel_size
            sp_group = None
            prev_sp_group = get_ulysses_sequence_parallel_group()
            if sp_size > 1 and torch.distributed.is_initialized():
                ranks = list(range(self.layout.dp_rank * sp_size, (self.layout.dp_rank + 1) * sp_size))
                sp_group = torch.distributed.new_group(ranks=ranks, backend=get_nccl_backend())
                set_ulysses_sequence_parallel_group(sp_group)
                torch.distributed.barrier(group=sp_group)

            apply_monkey_patch(
                model=reward_module,
                use_remove_padding=config.model.get("use_remove_padding", False),
                ulysses_sp_size=sp_size,
            )

            reward_module.to(torch_dtype)

        if sp_group is not None:
            torch.distributed.barrier(group=sp_group)
        set_ulysses_sequence_parallel_group(prev_sp_group)
        self.reward_sharding_manager = DeepSpeedUlyssesShardingManager(sp_group)

        # Initialize DeepSpeed for inference (no optimizer)
        # Parse mixed precision config
        fp16_enabled, bf16_enabled = _parse_mixed_precision_config(
            self.config.deepspeed_config.get("mixed_precision")
        )

        zero_stage = getattr(self.config, "zero_stage", self.config.deepspeed_config.get("zero_stage", 2))

        ds_config = get_deepspeed_config(
            optimizer_type="AdamW",  # Not used but required by config generator
            train_batch_size=1,
            train_micro_batch_size_per_gpu=1,
            gradient_accumulation_steps=1,
            zero_stage=zero_stage,
            lr=1e-5,  # Not used but required
            fp16_enabled=fp16_enabled,
            bf16_enabled=bf16_enabled,
            cpu_offload=self.config.deepspeed_config.get("param_offload", False),
            offload_optimizer=False,  # No optimizer for inference
            offload_dir=getattr(self.config.deepspeed_config, "offload_dir", None),
            disable_scheduler=True,  # No scheduler for inference
        )

        # Remove optimizer from config since this is inference only
        if "optimizer" in ds_config:
            del ds_config["optimizer"]

        # Initialize DeepSpeed engine without optimizer
        ds_engine, _, _, _ = initialize_deepspeed_engine(
            model=reward_module,
            config=ds_config,
            model_parameters=None,  # No parameters needed for inference
        )

        return ds_engine

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize reward model."""
        import_external_libs(self.config.model.get("external_lib", None))
        self.reward_engine = self._build_model(config=self.config)
        self.reward_module = self.reward_engine.module

    def _forward_micro_batch(self, micro_batch):
        """Forward pass for a single micro batch."""
        from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input

        with torch.no_grad(), torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)
            sp_size = self.ulysses_sequence_parallel_size

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                pad_size = 0
                if sp_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=sp_size
                    )

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.reward_module(
                    input_ids=input_ids_rmpad, attention_mask=None, position_ids=position_ids_rmpad, use_cache=False
                )
                reward_rmpad = output.logits
                reward_rmpad = reward_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if sp_size > 1:
                    reward_rmpad = gather_outputs_and_unpad(
                        reward_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )

                # pad it back
                rm_score = pad_input(reward_rmpad, indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)
            else:
                output = self.reward_module(
                    input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
                )
                rm_score = output.logits  # (batch_size, seq_len, 1)
                rm_score = rm_score.squeeze(-1)

            # extract the result of the last valid token
            eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
            rm_score = rm_score[torch.arange(batch_size), eos_mask_idx]
            return rm_score

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        """Expand sentence-level scores to token-level."""
        batch_size = data.batch.batch_size[0]
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        response_length = data.batch["responses"].shape[-1]
        if position_ids.dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
            position_ids = position_ids[:, 0, :]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    def _switch_chat_template(self, data: DataProto):
        """Switch chat template if input_tokenizer is different from reward model tokenizer."""
        import numpy as np

        src_max_length = data.batch["attention_mask"].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            if not isinstance(data.non_tensor_batch["raw_prompt"][i], list | np.ndarray):
                raise TypeError(
                    f"raw_prompt must be a list or numpy array, got {type(data.non_tensor_batch['raw_prompt'][i])}"
                )

            # extract raw prompt
            chat: list = list(data.non_tensor_batch["raw_prompt"][i])

            # extract response
            response_ids = data.batch["responses"][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            response = response.replace(src_tokenizer.eos_token, "")

            chat.append({"role": "assistant", "content": response})

            prompt_with_chat_template = target_tokenizer.apply_chat_template(
                chat, add_generation_prompt=False, tokenize=False
            )
            if self.rank == 0 and i == 0:
                logger.debug("Switch template. chat: %s", prompt_with_chat_template)

            prompt_ids = target_tokenizer.encode(prompt_with_chat_template, add_special_tokens=False)

            # pad or truncate
            if len(prompt_ids) < src_max_length:
                prompt_ids = prompt_ids + [target_tokenizer.pad_token_id] * (src_max_length - len(prompt_ids))
                attn_mask = [1] * len(prompt_ids) + [0] * (src_max_length - len(prompt_ids))
            else:
                prompt_ids = prompt_ids[:src_max_length]
                attn_mask = [1] * src_max_length

            rm_input_ids.append(prompt_ids)
            rm_attention_mask.append(attn_mask)

        # convert to tensors
        rm_input_ids = torch.tensor(rm_input_ids, dtype=torch.long, device=data.batch["input_ids"].device)
        rm_attention_mask = torch.tensor(
            rm_attention_mask, dtype=torch.long, device=data.batch["attention_mask"].device
        )

        # update data
        data.batch["input_ids"] = rm_input_ids
        data.batch["attention_mask"] = rm_attention_mask
        # recompute position_ids
        data.batch["position_ids"] = compute_position_id_with_mask(rm_attention_mask)

        return data

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward"))
    def compute_rm_score(self, data: DataProto):
        """Compute reward model scores."""
        if self._is_offload_param:
            load_deepspeed_model_to_gpu(self.reward_engine)

        data = data.to("cpu")

        # Switch chat template if needed
        if self._do_switch_chat_template:
            data = self._switch_chat_template(data)

        # Move data to device
        data = data.to(get_device_id())

        # Compute scores for each micro batch
        micro_batch_size = self.config.get("micro_batch_size_per_gpu", 1)
        batch_size = data.batch.batch_size[0]
        num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size

        all_scores = []
        manager = self.reward_sharding_manager if self.reward_sharding_manager is not None else nullcontext()
        with manager:
            for i in range(num_micro_batches):
                start_idx = i * micro_batch_size
                end_idx = min((i + 1) * micro_batch_size, batch_size)

                micro_batch = {
                    "input_ids": data.batch["input_ids"][start_idx:end_idx],
                    "attention_mask": data.batch["attention_mask"][start_idx:end_idx],
                    "position_ids": data.batch["position_ids"][start_idx:end_idx],
                }

                scores = self._forward_micro_batch(micro_batch)
                all_scores.append(scores)

        # Concatenate all scores
        rm_scores = torch.cat(all_scores, dim=0)

        # Expand to token level if needed
        token_level_scores = self._expand_to_token_level(data, rm_scores)

        output = DataProto.from_dict(tensors={"token_level_scores": token_level_scores})

        if self._is_offload_param:
            offload_deepspeed_model_to_cpu(self.reward_engine)

        return output
