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
from typing import Callable, Optional

from ..tracking import RLInsightLogger
from .config import ProfilerConfig


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
        self,
        rank: int,
        config: Optional[ProfilerConfig] = None,
        tool_config: Optional[object] = None,
        save_file_prefix: Optional[str] = None,
        **kwargs,
    ):
        # Default config
        if config is None:
            config = ProfilerConfig(ranks=[], enable=False, tool_config=None)

        if tool_config is None:
            tool_config = config.tool_config

        self.rank = rank
        self.config = config
        self.tool_config = tool_config
        # Optional label (typically the worker role, e.g. "actor"/"critic"/"ref") embedded
        # in per-process trace filenames so results from different roles are distinguishable.
        self.save_file_prefix = save_file_prefix

        self._impl = None
        self._tool = getattr(config, "tool", None)
        self._enable = config.enable
        self._this_step = False

        # Normalize rank selection
        self._this_rank = False
        if config.all_ranks:
            self._this_rank = True
        elif config.ranks:
            self._this_rank = rank in config.ranks
        else:
            # default rank 0 if enabled but ranks unspecified
            self._this_rank = (rank == 0) if self._enable else False

        # precision_debugger delegates rank filtering to msprobe config.json.
        # Keep verl-side rank gate open when profiler is enabled.
        if self._tool == "precision_debugger" and self._enable:
            self._this_rank = True

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

            self._impl = _Torch(rank=rank, config=config, tool_config=tool_config, save_file_prefix=save_file_prefix)
        elif self._tool == "torch_memory":
            from .torch_memory_profile import TorchMemoryProfiler

            self._impl = TorchMemoryProfiler(rank=rank, config=config, tool_config=tool_config)
        elif self._tool == "precision_debugger":
            from .precision_debugger_profile import PrecisionDebuggerProfiler as _Precision

            self._impl = _Precision(precision_cfg=tool_config, rank=rank, save_path=config.save_path)
        else:
            # Fallback to a no-op impl
            self._impl = _NoOpProfiler()

    def check_enable(self):
        """Return whether profiling is enabled by configuration."""
        return self._enable

    def check_this_rank(self):
        """Return whether current rank should perform profiling."""
        return self._this_rank

    def check_this_step(self):
        """Return whether current global step is marked for profiling."""
        return self._this_step

    def is_discrete_mode(self):
        """Return whether profiler backend runs in discrete mode."""
        return self._discrete

    def start(self, **kwargs):
        """Profiler switch for the Ray main flow; sets `this_step=True`.

        Args:
            **kwargs: Runtime arguments forwarded to backend `start`.
        """
        if self.check_enable() and self.check_this_rank():
            self._this_step = True
            return getattr(self._impl, "start", lambda **_: None)(**kwargs)

    def stop(self):
        """Profiler switch for the Ray main flow; sets `this_step=False`."""
        if self.check_enable() and self.check_this_rank():
            self._this_step = False
            return getattr(self._impl, "stop", lambda: None)()

    def step(self):
        """Advance the profiler schedule by one step, intended to be called per mini-batch.

        Delegates to the backend `step` when the tool supports scheduling (currently the
        torch profiler with a configured `wait/warmup/active/repeat` schedule); for all
        other backends this is a no-op.

        Gated on enable/rank only (not `this_step`): the training loop may run inside a
        nested worker whose profiler was never explicitly started, while the underlying
        torch profiler is process-global. The backend keeps `step` safe (no-op) whenever
        no profiler is actively running.
        """
        if self.check_enable() and self.check_this_rank():
            return getattr(self._impl, "step", lambda: None)()

    @classmethod
    def annotate(
        cls,
        message: Optional[str] = None,
        color: Optional[str] = None,
        domain: Optional[str] = None,
        category: Optional[str] = None,
        **kwargs_outer,
    ) -> Callable:
        """Decorate instance methods with backend profiler annotations.

        The wrapped function is executed directly if profiling is disabled,
        not selected for current rank/step, or backend annotate fails.
        """

        def decorator(func):
            @functools.wraps(func)
            def wrapper(self_instance, *args, **kwargs_inner):
                profiler = getattr(self_instance, "profiler", None)
                if profiler is None:
                    return func(self_instance, *args, **kwargs_inner)

                with RLInsightLogger.trace_state(
                    kwargs_outer.get("role", func.__qualname__), state_lane_id=f"rank_{profiler.rank}"
                ):
                    if not profiler.check_enable() or not profiler.check_this_step() or not profiler.check_this_rank():
                        return func(self_instance, *args, **kwargs_inner)

                    impl = profiler._impl
                    if hasattr(impl, "annotate"):
                        try:
                            actual_decorator = impl.annotate(
                                message=message, color=color, domain=domain, category=category, **kwargs_outer
                            )
                            wrapped = actual_decorator(func)
                        except Exception:
                            # Only fall back when *setting up* backend profiling fails.
                            # Never guard the call to func itself here: doing so would
                            # swallow real stage errors and re-run func (executing the
                            # stage twice with duplicated side effects).
                            wrapped = func
                        return wrapped(self_instance, *args, **kwargs_inner)
                    return func(self_instance, *args, **kwargs_inner)

            return wrapper

        return decorator


class _NoOpProfiler:
    def start(self, **kwargs):
        return

    def stop(self):
        return

    def step(self):
        return


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

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def step_profile(self) -> None:
        """Advance the profiler schedule by one step (typically once per mini-batch)."""
        self.profiler.step()
