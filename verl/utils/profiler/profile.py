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
import inspect
from typing import Callable, Optional

from ..memory_utils import MemorySnapshotSampler, enable_memory_visualize
from .config import ProfilerConfig, TorchMemoryToolConfig


def mark_start_range(
    message: Optional[str] = None,
    color: Optional[str] = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
) -> None:
    """Start a profiling range marker (no-op implementation).

    Args:
        message (Optional[str]): Message to associate with the range marker.
        color (Optional[str]): Color for the marker visualization.
        domain (Optional[str]): Domain for the marker.
        category (Optional[str]): Category for the marker.
    """
    pass


def mark_end_range(range_id: str) -> None:
    """End a profiling range marker (no-op implementation).

    Args:
        range_id (str): Identifier of the range to end.
    """
    pass


def mark_annotate(
    message: Optional[str] = None,
    color: Optional[str] = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
) -> Callable:
    """Decorator to annotate a function with profiling markers (no-op implementation).

    Args:
        message (Optional[str]): Message to associate with the annotation.
        color (Optional[str]): Color for the marker visualization.
        domain (Optional[str]): Domain for the marker.
        category (Optional[str]): Category for the marker.

    Returns:
        Callable: Decorator function that returns the original function unchanged.
    """

    def decorator(func):
        return func

    return decorator


class DistProfiler:
    """A dispatcher that delegates to specific profilers based on config.tool.

    Supported tools:
    - nsys: NsightSystemsProfiler
    - npu: NPUProfiler (Ascend)
    - torch: PyTorch torch.profiler wrapper
    - torch_memory: Torch CUDA memory snapshot dump
    - precision_debugger: msprobe precision debugger
    """

    def __init__(
        self, rank: int, config: Optional[ProfilerConfig] = None, tool_config: Optional[object] = None, **kwargs
    ):
        # Default config
        if not config:
            config = ProfilerConfig(ranks=[], enable=False, tool_config=None)

        if tool_config is None:
            tool_config = config.tool_config

        self.config = config
        self.tool_config = tool_config

        self._impl = None
        self._tool = getattr(config, "tool", None)
        self._enable = config.enable
        self._this_step = False
        self.rank = rank

        # Normalize rank selection
        self._this_rank = False
        if config.all_ranks:
            self._this_rank = True
        elif config.ranks:
            self._this_rank = rank in config.ranks
        else:
            # default rank 0 if enabled but ranks unspecified
            self._this_rank = (rank == 0) if self._enable else False

        # TorchMemoryProfiler currently do not support discrete mode.
        self._discrete = getattr(tool_config, "discrete", False) if tool_config else False

        # Lazy import to avoid circular deps
        if self._tool == "nsys":
            from .nvtx_profile import NsightSystemsProfiler as _Nsight

            self._impl = _Nsight(rank=rank, config=config, tool_config=tool_config, **kwargs)
        elif self._tool == "npu":
            from .mstx_profile import NPUProfiler as _Npu

            self._impl = _Npu(rank=rank, config=config, tool_config=tool_config, **kwargs)
        elif self._tool == "torch":
            from .torch_profile import Profiler as _Torch

            self._impl = _Torch(rank=rank, config=config, tool_config=tool_config)
        elif self._tool == "torch_memory":
            self._impl = TorchMemoryProfiler(rank=rank, config=config, tool_config=tool_config)
        elif self._tool == "precision_debugger":
            from .precision_debugger_profile import PrecisionDebuggerProfiler as _Precision

            self._impl = _Precision(precision_cfg=tool_config, rank=rank)
        else:
            # Fallback to a no-op impl
            self._impl = _NoOpProfiler()

    def check_enable(self):
        return self._enable

    def check_this_rank(self):
        return self._this_rank

    def check_this_step(self):
        return self._this_step

    def is_discrete_mode(self):
        return self._discrete

    def start(self, **kwargs):
        if self.check_enable() and self.check_this_rank():
            self._this_step = True
            return getattr(self._impl, "start", lambda **_: None)(**kwargs)

    def stop(self):
        if self.check_enable() and self.check_this_rank():
            self._this_step = False
            return getattr(self._impl, "stop", lambda: None)()


    @classmethod
    def annotate(
        cls,
        message: Optional[str] = None,
        color: Optional[str] = None,
        domain: Optional[str] = None,
        category: Optional[str] = None,
        precision_stage: Optional[str] = None,
        precision_model_attr: Optional[object] = None,
        precision_global_step_attr: Optional[str] = None,
        precision_step: bool = False,
        **kwargs_outer,
    ) -> Callable:
        def _get_model(self_instance):
            if precision_model_attr is None:
                return None
            if isinstance(precision_model_attr, (list, tuple)):
                for attr in precision_model_attr:
                    val = _resolve_attr(self_instance, attr)
                    if val is not None:
                        return val
                return None
            return _resolve_attr(self_instance, precision_model_attr)

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

        def _get_global_step(self_instance, args, kwargs):
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

        def _build_precision_impl(self_instance):
            precision_cfg = getattr(self_instance, "precision_debugger_cfg", None)
            if not precision_cfg or not precision_stage:
                return None
            from .precision_debugger_profile import PrecisionDebuggerProfiler

            rank = getattr(getattr(self_instance, "profiler", None), "rank", None)
            return PrecisionDebuggerProfiler(precision_cfg, rank=rank)

        def _precision_start(precision_impl, self_instance, args, kwargs_inner):
            if precision_impl is None:
                return False
            global_step = _get_global_step(self_instance, args, kwargs_inner)
            model = _get_model(self_instance)
            return precision_impl.start(stage=precision_stage, global_step=global_step, model=model)

        def _decorate_with_profiler(impl, func_inner):
            if hasattr(impl, "annotate"):
                return impl.annotate(message=message, color=color, domain=domain, category=category, **kwargs_outer)(
                    func_inner
                )
            return func_inner

        def _should_profile(self_instance) -> bool:
            profiler = getattr(self_instance, "profiler", None)
            return (
                profiler
                and profiler.check_enable()
                and profiler.check_this_step()
                and profiler.check_this_rank()
            )

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(self_instance, *args, **kwargs_inner):
                precision_impl = _build_precision_impl(self_instance)
                precision_started = _precision_start(precision_impl, self_instance, args, kwargs_inner)
                try:
                    if _should_profile(self_instance):
                        impl = self_instance.profiler._impl
                        wrapped = _decorate_with_profiler(impl, func)
                        return await wrapped(self_instance, *args, **kwargs_inner)
                    return await func(self_instance, *args, **kwargs_inner)
                finally:
                    if precision_impl is not None and precision_stage:
                        precision_impl.stop(started=precision_started, step=precision_step)

            return async_wrapper

        def decorator(func_inner):
            @functools.wraps(func_inner)
            def wrapper(self_instance, *args, **kwargs_inner):
                precision_impl = _build_precision_impl(self_instance)
                precision_started = _precision_start(precision_impl, self_instance, args, kwargs_inner)
                try:
                    if _should_profile(self_instance):
                        impl = self_instance.profiler._impl
                        wrapped = _decorate_with_profiler(impl, func_inner)
                        return wrapped(self_instance, *args, **kwargs_inner)
                    return func_inner(self_instance, *args, **kwargs_inner)
                finally:
                    if precision_impl is not None and precision_stage:
                        precision_impl.stop(started=precision_started, step=precision_step)

            return wrapper

        return decorator


class _NoOpProfiler:
    def start(self, **kwargs):
        return

    def stop(self):
        return




class TorchMemoryProfiler:
    """Profiler that dumps CUDA memory snapshots at step boundaries.

    Behavior:
    - On first construction (per process), enable memory history recording if CUDA is available
    - On start(step=X), remember sub_dir for this step
    - On stop(), dump a memory snapshot into config.save_path under the remembered sub_dir
    """

    _memory_history_enabled: bool = False

    def __init__(
        self, rank: int, config: Optional[ProfilerConfig], tool_config: Optional[TorchMemoryToolConfig] = None
    ):
        # Always respond to explicit start/stop calls for torch_memory tool,
        # regardless of per-role enable flag, to align with global step control.
        self.enable = True
        if not config:
            config = ProfilerConfig(ranks=[])
        self.config = config
        self.rank = rank
        self.this_step = False
        self.sub_dir = None
        self.sampler = MemorySnapshotSampler()

        # Get parameters from tool_config, with fallback to defaults
        if tool_config:
            trace_alloc_max_entries = tool_config.trace_alloc_max_entries
            stack_depth = tool_config.stack_depth
        else:
            trace_alloc_max_entries = 100_000
            stack_depth = 32

        # Best-effort enable memory history once
        if not TorchMemoryProfiler._memory_history_enabled:
            try:
                enable_memory_visualize(trace_alloc_max_entries=trace_alloc_max_entries, stack_depth=stack_depth)
            except Exception:
                # silently ignore if not supported
                pass
            TorchMemoryProfiler._memory_history_enabled = True

    def start(self, **kwargs):
        if not self.enable:
            return
        if not self._should_profile_this_rank():
            return
        profile_step = kwargs.get("profile_step", None)
        # Keep ranks aligned under same folder name
        self.sub_dir = f"step{profile_step}" if profile_step is not None else None
        self.this_step = True

    def stop(self):
        if not self.enable or not self.this_step:
            return
        self.this_step = False
        if not self._should_profile_this_rank():
            return
        out_dir = self.config.save_path or "outputs/profile"
        tag = "torch_memory"
        # Dump snapshot; all ranks write into same sub_dir
        try:
            self.sampler.dump_memory_snapshot(out_dir=out_dir, tag=tag, sub_dir=self.sub_dir)
        except Exception:
            pass

    def _should_profile_this_rank(self) -> bool:
        if self.config.all_ranks:
            return True
        if self.config.ranks:
            return self.rank in self.config.ranks
        # default rank 0
        return self.rank == 0


class DistProfilerExtension:
    """An extension class for DistProfiler that provides distributed profiling capabilities.

    It is intended for workers in verl that single controller invokes.

    This class wraps a DistProfiler instance and provides methods to start/stop profiling
    that can be dispatched across multiple ranks in a distributed training environment.

    Args:
        profiler (DistProfiler): The base distributed profiler instance to extend
    """

    def __init__(self, profiler: DistProfiler):
        self.profiler = profiler

    from verl.single_controller.base.decorator import Dispatch, register

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def start_profile(self, **kwargs) -> None:
        """Start profiling for the current rank in the current training step."""
        self.profiler.start(**kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def stop_profile(self) -> None:
        """Stop profiling for the current rank in the current training step."""
        self.profiler.stop()
