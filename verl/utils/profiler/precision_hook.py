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

from typing import Optional

from .precision_debugger_profile import PrecisionDebuggerProfiler


def _resolve_attr(obj, attr):
    if not isinstance(attr, str):
        return None
    if "." in attr:
        current = obj
        for part in attr.split("."):
            current = getattr(current, part, None)
            if current is None:
                return None
        return current
    return getattr(obj, attr, None)


def _get_model(self_instance, precision_model_attr):
    if precision_model_attr is None:
        return None
    if isinstance(precision_model_attr, (list, tuple)):
        for attr in precision_model_attr:
            val = _resolve_attr(self_instance, attr)
            if val is not None:
                return val
        return None
    return _resolve_attr(self_instance, precision_model_attr)


def _get_global_step(self_instance, args, kwargs, precision_global_step_attr: Optional[str]):
    for val in list(args) + list(kwargs.values()):
        if hasattr(val, "meta_info"):
            meta = getattr(val, "meta_info")
            if isinstance(meta, dict) and "global_steps" in meta:
                return meta.get("global_steps")
        if isinstance(val, dict) and "global_steps" in val:
            return val.get("global_steps")
    if precision_global_step_attr and hasattr(self_instance, precision_global_step_attr):
        return getattr(self_instance, precision_global_step_attr)
    if hasattr(self_instance, "precision_global_step"):
        return getattr(self_instance, "precision_global_step")
    return None


def build_precision_impl(self_instance, precision_stage: Optional[str]):
    precision_cfg = getattr(self_instance, "precision_debugger_cfg", None)
    if not precision_cfg or not precision_stage:
        return None
    rank = getattr(getattr(self_instance, "profiler", None), "rank", None)
    return PrecisionDebuggerProfiler(precision_cfg, rank=rank)


def start_precision(
    precision_impl: Optional[PrecisionDebuggerProfiler],
    self_instance,
    args,
    kwargs,
    precision_stage: Optional[str],
    precision_model_attr,
    precision_global_step_attr: Optional[str],
) -> bool:
    if precision_impl is None:
        return False
    global_step = _get_global_step(self_instance, args, kwargs, precision_global_step_attr)
    model = _get_model(self_instance, precision_model_attr)
    return precision_impl.start(stage=precision_stage, global_step=global_step, model=model)


def stop_precision(
    precision_impl: Optional[PrecisionDebuggerProfiler],
    started: bool,
    precision_step: bool,
) -> None:
    if precision_impl is None:
        return
    precision_impl.stop(started=started, step=precision_step)


class PrecisionDebuggerLogger:
    """Decorator to run PrecisionDebugger around a method call.

    Example:
        >>> @PrecisionDebuggerLogger(stage="train", model_attr="actor_module")
        >>> def update_policy(self, batch): ...
    """

    def __init__(
        self,
        stage: str,
        model_attr: Optional[object] = None,
        global_step_attr: Optional[str] = None,
        step: bool = False,
    ):
        self.stage = stage
        self.model_attr = model_attr
        self.global_step_attr = global_step_attr
        self.step = step

    def __call__(self, decorated_function: callable):
        def wrapper(self_instance, *args, **kwargs):
            precision_impl = build_precision_impl(self_instance, self.stage)
            started = start_precision(
                precision_impl,
                self_instance,
                args,
                kwargs,
                self.stage,
                self.model_attr,
                self.global_step_attr,
            )
            try:
                return decorated_function(self_instance, *args, **kwargs)
            finally:
                stop_precision(precision_impl, started, self.step)

        return wrapper
