# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
import os
from dataclasses import asdict
from typing import Optional

from verl.utils.import_utils import is_msprobe_available
from verl.utils.profiler.config import PrecisionDebuggerToolConfig

logger = logging.getLogger(__name__)

_STAGE_TO_ROLE = {
    "actor_update": "actor",
    "actor_compute_log_prob": "actor",
    "ref_compute_log_prob": "ref",
    "compute_values": "critic",
    "critic_update": "critic",
    "compute_rm_score": "reward_model",
}

_MODEL_ATTRS_BY_ROLE = {
    "actor": (
        "actor.engine.module",
        "actor.actor_module",
        "actor.actor_module_fsdp",
        "actor_module_fsdp",
        "actor_module",
    ),
    "ref": (
        "ref.engine.module",
        "ref.actor_module",
        "ref_policy.actor_module",
        "ref_module_fsdp",
        "ref_module",
        "ref_policy.ref_module",
    ),
    "critic": (
        "critic.engine.module",
        "critic.critic_module",
        "critic_module_fsdp",
        "critic_module",
    ),
    "reward_model": (
        "reward_model.engine.module",
        "reward_model_module_fsdp",
        "reward_model_module",
        "rm.reward_model_module",
    ),
}

_SKIP_STAGES = {"rollout_generate"}


class PrecisionDebuggerProfiler:
    """Minimal msprobe PrecisionDebugger integration."""

    def __init__(self, precision_cfg, rank: Optional[int] = None):
        self.rank = rank
        self.precision_cfg = self._normalize_config(precision_cfg)
        self._enabled = bool(self.precision_cfg.enable)
        self._available = is_msprobe_available()
        self._debugger = None
        self._stages = self._normalize_stages(self.precision_cfg.stages)
        self._current_global_step = None

    @staticmethod
    def _normalize_config(precision_cfg) -> PrecisionDebuggerToolConfig:
        if precision_cfg is None:
            return PrecisionDebuggerToolConfig()
        if isinstance(precision_cfg, PrecisionDebuggerToolConfig):
            return precision_cfg
        if hasattr(precision_cfg, "to_container"):
            precision_cfg = precision_cfg.to_container(resolve=True)
        if isinstance(precision_cfg, dict):
            return PrecisionDebuggerToolConfig(**precision_cfg)
        return PrecisionDebuggerToolConfig(**asdict(precision_cfg))

    @staticmethod
    def _normalize_stage(stage: Optional[str]) -> Optional[str]:
        return stage

    def _normalize_stages(self, stages: Optional[list[str]]) -> Optional[set[str]]:
        if stages is None:
            return None
        normalized = {self._normalize_stage(stage) for stage in stages}
        if _SKIP_STAGES & normalized:
            logger.warning("Ignoring precision_debugger stages: %s", sorted(_SKIP_STAGES & normalized))
        normalized = normalized - _SKIP_STAGES
        unknown = normalized - set(_STAGE_TO_ROLE.keys())
        if unknown:
            msg = f"Unknown precision_debugger stages: {sorted(unknown)}"
            if self.precision_cfg.strict:
                raise ValueError(msg)
            logger.warning(msg)
        return normalized & set(_STAGE_TO_ROLE.keys())

    @staticmethod
    def _resolve_attr(obj, attr_path: str):
        current = obj
        for part in attr_path.split("."):
            current = getattr(current, part, None)
            if current is None:
                return None
        return current

    @staticmethod
    def _is_valid_model(model) -> bool:
        return model is not None and callable(getattr(model, "forward", None))

    def _get_candidate_attrs(self, stage: str) -> tuple[str, ...]:
        role = _STAGE_TO_ROLE.get(stage)
        if role is None:
            return ()
        return _MODEL_ATTRS_BY_ROLE.get(role, ())

    def _resolve_model(self, self_instance, stage: str):
        for attr in self._get_candidate_attrs(stage):
            value = self._resolve_attr(self_instance, attr)
            if self._is_valid_model(value):
                return value
        fallback = getattr(self_instance, "module", None)
        return fallback if self._is_valid_model(fallback) else None

    def _resolve_global_step(self, self_instance, args, kwargs):
        for val in list(args) + list(kwargs.values()):
            if hasattr(val, "meta_info"):
                meta = val.meta_info
                if isinstance(meta, dict) and "global_steps" in meta:
                    return meta.get("global_steps")
            if isinstance(val, dict) and "global_steps" in val:
                return val.get("global_steps")
        for attr in ("global_step", "_global_step"):
            if hasattr(self_instance, attr):
                return getattr(self_instance, attr)
        return self._current_global_step

    def _should_collect(self, stage: str, global_step: Optional[int]) -> bool:
        if not self._enabled:
            return False
        if stage in _SKIP_STAGES:
            return False
        if stage not in _STAGE_TO_ROLE:
            msg = f"Unknown precision_debugger stage: {stage}"
            if self.precision_cfg.strict:
                raise ValueError(msg)
            logger.warning(msg)
            return False
        if self._stages is not None and stage not in self._stages:
            return False
        if self.precision_cfg.steps is not None and global_step is not None:
            if int(global_step) not in set(self.precision_cfg.steps):
                return False
        return True

    def start(self, stage: Optional[str] = None, global_step: Optional[int] = None, model=None, **kwargs) -> bool:
        profile_step = kwargs.get("global_step", kwargs.get("profile_step"))
        if profile_step is not None:
            self._current_global_step = profile_step
        stage = self._normalize_stage(stage)
        if stage is None:
            return False
        if global_step is None:
            global_step = self._current_global_step
        if not self._should_collect(stage, global_step):
            return False
        if not self._available:
            if self.precision_cfg.strict:
                raise ImportError("msprobe is not available but precision_debugger.strict is True")
            return False
        if not self.precision_cfg.config_path or not self.precision_cfg.data_dir:
            return False
        if not self._is_valid_model(model):
            msg = f"PrecisionDebugger model not resolved for stage '{stage}'"
            if self.precision_cfg.strict:
                raise ValueError(msg)
            logger.warning(msg)
            return False

        try:
            from msprobe.pytorch import PrecisionDebugger

            step_tag = f"step_{global_step}" if global_step is not None else "step_unknown"
            dump_path = os.path.join(self.precision_cfg.data_dir, step_tag, stage)
            os.makedirs(dump_path, exist_ok=True)

            if self._debugger is None:
                self._debugger = PrecisionDebugger(config_path=self.precision_cfg.config_path, dump_path=dump_path)
                if self._debugger is None:
                    if self.precision_cfg.strict:
                        raise RuntimeError("Failed to create PrecisionDebugger instance")
                    return False
            if hasattr(self._debugger, "service") and hasattr(self._debugger.service, "config"):
                self._debugger.service.config.dump_path = dump_path
            self._debugger.start(model)
            return True
        except Exception:
            if self.precision_cfg.strict:
                raise
            return False

    def stop(self, started: bool = False) -> None:
        if not started:
            return
        if not self._available:
            return
        if self._debugger is None:
            return
        self._debugger.stop()
        self._reset_debugger_status()

    def annotate(
        self,
        message: Optional[str] = None,
        color: Optional[str] = None,
        domain: Optional[str] = None,
        category: Optional[str] = None,
        **kwargs_outer,
    ):
        _ = (message, color, domain, category)
        stage = self._normalize_stage(kwargs_outer.get("role"))
        if stage is None:
            return lambda func: func

        def decorator(func):
            @functools.wraps(func)
            def wrapper(self_instance, *args, **kwargs_inner):
                global_step = self._resolve_global_step(self_instance, args, kwargs_inner)
                model = self._resolve_model(self_instance, stage)
                started = self.start(stage=stage, global_step=global_step, model=model)
                try:
                    return func(self_instance, *args, **kwargs_inner)
                finally:
                    self.stop(started=started)

            return wrapper

        return decorator

    def _reset_debugger_status(self) -> None:
        service = getattr(self._debugger, "service", None)
        if service is None:
            return

        reset_status = getattr(service, "reset_status", None)
        if callable(reset_status):
            reset_status()
            return

        reset_status = getattr(service, "_reset_status", None)
        if callable(reset_status):
            reset_status()
