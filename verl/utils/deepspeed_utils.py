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
Lightweight DeepSpeed helpers used by the DeepSpeed engine integration.

Only the pieces needed for ZeRO-1/2/3, optional CPU/NVMe offload, and
activation checkpointing are implemented here.
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from typing import Any, Optional, Sequence

import torch

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
    from deepspeed.runtime.zero import GatheredParameters
    try:
        # Newer DeepSpeed versions keep the ZeRO-1/2 optimizer here.
        from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
    except Exception:  # pragma: no cover - compatibility with older DeepSpeed
        from deepspeed.runtime.zero.stage2 import DeepSpeedZeroOptimizer

    DEEPSPEED_AVAILABLE = True
except Exception:  # pragma: no cover - DeepSpeed not installed
    deepspeed = None
    DeepSpeedEngine = Any  # type: ignore
    DeepSpeedZeroOptimizer = Any  # type: ignore
    GatheredParameters = nullcontext  # type: ignore
    DEEPSPEED_AVAILABLE = False


def _maybe_patch_zero2_grad_accum_dtype() -> None:
    """Patch ZeRO-2 grad accumulation to honor ``gradient_accumulation_dtype``.

    Why this exists:
    - ZeRO-2 may accumulate partitioned grads in the model dtype (bf16/fp16),
      which can diverge from ZeRO-1/3 when accumulation precision differs.
    - We upcast accumulation tensors to the requested accumulation dtype and keep
      micro-batch accumulation state intact across non-boundary micros.
    """
    if not DEEPSPEED_AVAILABLE:
        return
    # Emergency off-switch.
    if os.getenv("VERL_DS_ZERO2_FP32_ACCUM_PATCH", "1").lower() in {"0", "false", "off", "no"}:
        return
    try:
        import torch
        from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer as Zero12Opt
    except Exception:
        return
    if getattr(Zero12Opt, "_verl_zero2_grad_accum_patched", False):
        return

    original_get_all_grad_tensors = Zero12Opt.get_all_grad_tensors
    original_ipg_epilogue = Zero12Opt.independent_gradient_partition_epilogue
    _accum_marker = object()

    def _patched_get_all_grad_tensors(self, tensor_list, dtype):  # type: ignore[override]
        out = original_get_all_grad_tensors(self, tensor_list, dtype)
        # ZeRO-2 (partition_gradients=True) can otherwise accumulate in model dtype
        # (bf16/fp16). Upcast to gradient_accumulation_dtype to reduce drift.
        if not getattr(self, "partition_gradients", False):
            return out
        casted = []
        for grad in out:
            if grad is None:
                casted.append(grad)
            elif grad.dtype != dtype:
                casted.append(grad.to(dtype))
            else:
                casted.append(grad)
        return casted

    def _patched_independent_gradient_partition_epilogue(self):  # type: ignore[override]
        # ZeRO-2 should accumulate per-micro partition grads in all_grad_tensors.
        # Upstream condition keys off averaged_gradients, which stays None until
        # boundary and can overwrite previously accumulated micro grads.
        try:
            if not getattr(self, "cpu_offload", False) and getattr(self, "partition_gradients", False):
                averaged = getattr(self, "averaged_gradients", None)
                all_grad = getattr(self, "all_grad_tensors", None)
                if isinstance(averaged, dict) and isinstance(all_grad, dict):
                    for i, _ in enumerate(getattr(self, "bit16_groups", [])):
                        if averaged.get(i, None) is None and all_grad.get(i, None) is not None:
                            averaged[i] = _accum_marker
        except Exception:
            # Fall back to upstream behavior if internals differ.
            pass
        return original_ipg_epilogue(self)

    Zero12Opt.get_all_grad_tensors = _patched_get_all_grad_tensors  # type: ignore[assignment]
    Zero12Opt.independent_gradient_partition_epilogue = _patched_independent_gradient_partition_epilogue  # type: ignore[assignment]
    Zero12Opt._verl_zero2_grad_accum_patched = True  # type: ignore[attr-defined]
    logger.info("Applied VERL ZeRO-2 grad accumulation dtype patch")


_maybe_patch_zero2_grad_accum_dtype()


def _ensure_deepspeed():
    global DEEPSPEED_AVAILABLE, deepspeed, DeepSpeedEngine, GatheredParameters
    if DEEPSPEED_AVAILABLE:
        return
    try:
        import deepspeed as _ds
        from deepspeed import DeepSpeedEngine as _DSE
        from deepspeed.runtime.zero import GatheredParameters as _GatheredParameters
    except Exception as exc:  # pragma: no cover - runtime import guard
        raise ImportError("DeepSpeed is not available. Please install deepspeed to use the DeepSpeed engine.") from exc
    deepspeed = _ds
    DeepSpeedEngine = _DSE
    GatheredParameters = _GatheredParameters
    DEEPSPEED_AVAILABLE = True
    _maybe_patch_zero2_grad_accum_dtype()


def get_deepspeed_config(
    *,
    optimizer_type: str = "AdamW",
    train_batch_size: Optional[int] = None,
    train_micro_batch_size_per_gpu: int = 1,
    gradient_accumulation_steps: int = 1,
    zero_stage: int = 2,
    lr: float = 1e-5,
    betas: Optional[Sequence[float]] = None,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    fp16_enabled: bool = False,
    bf16_enabled: bool = True,
    offload_param_device: Optional[str] = None,
    offload_optimizer_device: Optional[str] = None,
    offload_param_nvme_path: Optional[str] = None,
    offload_optimizer_nvme_path: Optional[str] = None,
    gradient_clipping: Optional[float] = None,
    zero_optimization_overrides: Optional[dict[str, Any]] = None,
    disable_scheduler: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a DeepSpeed configuration dictionary."""

    # Backward-compatible shorthands used by worker configs
    cpu_offload = kwargs.pop("cpu_offload", False)
    offload_optimizer_flag = kwargs.pop("offload_optimizer", False)
    offload_dir = kwargs.pop("offload_dir", None)

    if cpu_offload and offload_param_device is None:
        offload_param_device = "cpu"
        if offload_param_nvme_path is None and offload_dir is not None:
            offload_param_nvme_path = offload_dir
    if offload_optimizer_flag and offload_optimizer_device is None:
        offload_optimizer_device = "cpu"
        if offload_optimizer_nvme_path is None and offload_dir is not None:
            offload_optimizer_nvme_path = offload_dir

    if betas is None:
        betas = (0.9, 0.999)

    config: dict[str, Any] = {
        "train_micro_batch_size_per_gpu": train_micro_batch_size_per_gpu,
        "gradient_accumulation_steps": max(1, gradient_accumulation_steps),
        "optimizer": {
            "type": optimizer_type,
            "params": {
                "lr": lr,
                "betas": list(betas),
                "eps": eps,
                "weight_decay": weight_decay,
            },
        },
        "steps_per_print": 1,
    }

    if train_batch_size is not None:
        config["train_batch_size"] = train_batch_size

    if bf16_enabled:
        config["bf16"] = {"enabled": True}
    elif fp16_enabled:
        config["fp16"] = {"enabled": True}

    if gradient_clipping is not None:
        config["gradient_clipping"] = gradient_clipping

    if zero_stage > 0:
        zero_opt = {
            "stage": zero_stage,
            # Keep communication behavior explicit. We intentionally avoid extra
            # overlap knobs here to reduce stage1/2 comparison noise.
            "overlap_comm": False,
            "contiguous_gradients": True,
            "reduce_scatter": True,
        }

        if zero_stage >= 2 and offload_optimizer_device is not None:
            zero_opt["offload_optimizer"] = {"device": offload_optimizer_device}
            if offload_optimizer_device == "cpu":
                zero_opt["offload_optimizer"]["pin_memory"] = True
            if offload_optimizer_nvme_path is not None:
                zero_opt["offload_optimizer"]["nvme_path"] = offload_optimizer_nvme_path

        if zero_stage >= 3 and offload_param_device is not None:
            zero_opt["offload_param"] = {"device": offload_param_device}
            if offload_param_device == "cpu":
                zero_opt["offload_param"]["pin_memory"] = True
            if offload_param_nvme_path is not None:
                zero_opt["offload_param"]["nvme_path"] = offload_param_nvme_path
            # a conservative bucket size to keep memory use predictable
            zero_opt["stage3_prefetch_bucket_size"] = 2e7

        if zero_stage >= 3:
            # Enable CPU gathering so _zero3_consolidated_16bit_state_dict works
            zero_opt.setdefault("gather_16bit_weights_on_model_save", True)

        if zero_optimization_overrides:
            zero_opt.update(zero_optimization_overrides)

        config["zero_optimization"] = zero_opt

    if disable_scheduler:
        config.pop("scheduler", None)

    config.update(kwargs)
    return config


def initialize_deepspeed_engine(model: torch.nn.Module, config: dict[str, Any], *, model_parameters=None, **kwargs):
    """Initialize a DeepSpeed engine."""
    _ensure_deepspeed()
    init_kwargs = {"model": model, "config": config, "model_parameters": model_parameters or model.parameters()}
    init_kwargs.update(kwargs)
    return deepspeed.initialize(**init_kwargs)


def get_zero_stage(engine: DeepSpeedEngine) -> int:
    if hasattr(engine, "zero_optimization_stage"):
        try:
            return int(engine.zero_optimization_stage())
        except Exception:
            pass
    try:
        zero_cfg = getattr(engine, "_config", {}).get("zero_optimization", {})
        return int(zero_cfg.get("stage", 0))
    except Exception:
        return 0


def is_zero3_engine(engine: DeepSpeedEngine) -> bool:
    try:
        return get_zero_stage(engine) >= 3
    except Exception:
        return False


def _engine_has_param_offload(engine: DeepSpeedEngine) -> bool:
    """Best-effort check for ZeRO-3 param offload in the engine config."""
    try:
        zero_cfg = getattr(engine, "_config", {}).get("zero_optimization", {})
        offload_cfg = zero_cfg.get("offload_param", None)
        return bool(offload_cfg)
    except Exception:
        return False


def deepspeed_gather_params(engine: DeepSpeedEngine, gather_16bit: bool = True):
    """Context manager that gathers params when running ZeRO-3."""
    if is_zero3_engine(engine) and gather_16bit and GatheredParameters is not None:
        return GatheredParameters(engine.module.parameters(), modifier_rank=None)
    return nullcontext()


def load_deepspeed_model_to_gpu(engine: DeepSpeedEngine):
    """Best-effort helper to move the wrapped module to GPU when offloaded."""
    try:
        module = engine.module
    except Exception:
        return
    try:
        if next(module.parameters()).device.type == "cpu":
            engine.module = module.to(torch.device("cuda"))
    except StopIteration:
        return


def offload_deepspeed_model_to_cpu(engine: DeepSpeedEngine):
    """Best-effort helper to move the wrapped module to CPU."""
    try:
        module = engine.module
    except Exception:
        return
    try:
        if next(module.parameters()).device.type != "cpu":
            engine.module = module.to(torch.device("cpu"))
    except StopIteration:
        return


def get_global_grad_norm(engine: DeepSpeedEngine) -> Optional[float]:
    try:
        norm = engine.get_global_grad_norm()
        return norm if norm is None else float(norm)
    except Exception:
        return None


def save_deepspeed_checkpoint(engine: DeepSpeedEngine, save_dir: str, tag: Optional[str] = None, **kwargs):
    """Thin wrapper over engine.save_checkpoint with a consistent signature."""
    if hasattr(engine, "save_checkpoint"):
        client_state = kwargs.get("client_state")
        if client_state is None:
            return engine.save_checkpoint(save_dir, tag=tag)
        return engine.save_checkpoint(save_dir, client_state=client_state, tag=tag)
    raise RuntimeError("DeepSpeed engine does not expose save_checkpoint")


def load_deepspeed_checkpoint(
    engine: DeepSpeedEngine,
    load_dir: str,
    tag: Optional[str] = None,
    load_module_strict: bool = True,
    load_optimizer_states: bool = True,
    load_lr_scheduler_states: bool = True,
):
    """Thin wrapper over engine.load_checkpoint."""
    if hasattr(engine, "load_checkpoint"):
        _, client_state = engine.load_checkpoint(
            load_dir,
            tag=tag,
            load_module_strict=load_module_strict,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states,
        )
        return client_state
    raise RuntimeError("DeepSpeed engine does not expose load_checkpoint")
