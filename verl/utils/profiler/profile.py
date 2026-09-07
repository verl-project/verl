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

import copy
import dataclasses
import functools
import logging
import os
import subprocess
from typing import Callable, Optional

from ..tracking import RLInsightLogger
from .config import ProfilerConfig

logger = logging.getLogger(__name__)


def _hook_print(msg: str) -> None:
    """Report finish-hook activity on stdout.

    Worker processes do not necessarily inherit the ``verl`` logger configuration, so hook
    progress is printed rather than logged to guarantee it reaches the Ray worker logs.
    """
    print(f"[Profiler][finish_hook] {msg}", flush=True)


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

        # Finish-hook rank selection (independent of the profiled ranks). The command runs on
        # `finish_hook_all_ranks`/`finish_hook_ranks`, defaulting to the profiled ranks when unset.
        self._finish_hook_cmd = getattr(config, "finish_hook_cmd", None)
        self._relocate_results = getattr(config, "relocate_results", False)
        if getattr(config, "finish_hook_all_ranks", False):
            self._finish_hook_this_rank = True
        elif getattr(config, "finish_hook_ranks", None):
            self._finish_hook_this_rank = rank in config.finish_hook_ranks
        else:
            self._finish_hook_this_rank = self._this_rank

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

    def stop(self, run_command: bool = True):
        """Profiler switch for the Ray main flow; sets `this_step=False`.

        Stops the backend profiler and relocates its artifacts into ``save_path`` on every profiled
        step (so traces accumulate there). The user's finish command, however, runs only when
        ``run_command`` is True -- the trainer sets that on the *last* profiled step -- so the command
        (typically a one-shot upload of the whole ``save_path``) fires once at the end instead of once
        per step. The hook is invoked outside the ``check_this_rank`` gate so the command may target
        ranks that were not themselves profiled.
        """
        result = None
        if self.check_enable() and self.check_this_rank():
            self._this_step = False
            result = getattr(self._impl, "stop", lambda: None)()
        if self.check_enable():
            self._run_finish_hook(run_command=run_command)
        return result

    def run_finish_hook(self, run_command: bool = True) -> None:
        """Relocate backend artifacts and, when ``run_command`` is True, run the user's command.

        Kept for external drivers that stop their own backend and then want verl to relocate
        artifacts and optionally upload. Inference servers relocate their own traces into
        ``save_path`` and rely on the training worker's end-of-run upload, so they no longer
        call this.
        """
        if self.check_enable():
            self._run_finish_hook(run_command=run_command)

    def _run_finish_hook(self, run_command: bool = True) -> None:
        """Relocate backend artifacts every step; run the user's finish command once, at the end.

        Relocation (currently nsys) runs on every ``stop()`` on the ranks that profiled, so a step's
        artifacts land in ``save_path`` as they are produced. The user command runs only when
        ``run_command`` is True -- the trainer passes that on the last profiled step -- so a command
        that uploads the whole ``save_path`` sends each trace exactly once, at the end, instead of
        re-uploading the accumulating directory every step. Both are best-effort: failures are
        reported and never interrupt training.
        """
        save_path = getattr(self.config, "save_path", None)

        # (1) Relocate backend artifacts (currently nsys) out of framework-fixed dirs into save_path,
        # every step, so they accumulate there for the end-of-run upload.
        if self._relocate_results and self.check_this_rank():
            relocate = getattr(self._impl, "relocate_results", None)
            if callable(relocate):
                try:
                    relocate(save_path, rank=self.rank, save_file_prefix=self.save_file_prefix)
                except Exception as e:  # never let post-processing crash the training loop
                    _hook_print(f"rank {self.rank}: relocating results failed: {e}")

        # (2) Run the user's command, but only on the final stop and only on the selected ranks.
        if not run_command:
            return
        _hook_print(
            f"rank {self.rank}: profiling finished; cmd={self._finish_hook_cmd!r}, "
            f"run_on_this_rank={self._finish_hook_this_rank}, save_path={save_path!r}"
        )
        if not self._finish_hook_cmd:
            return
        if not self._finish_hook_this_rank:
            _hook_print(f"rank {self.rank}: command skipped, rank not selected by finish_hook_ranks/all_ranks")
            return
        self._run_finish_command(self._finish_hook_cmd, save_path)

    def _run_finish_command(self, cmd: str, save_path: Optional[str]) -> None:
        """Execute the user finish command in a shell once profiling is done.

        The command is meant to ship the accumulated traces off the node, e.g.
        ``mlx asset upload "$VERL_PROFILE_SAVE_PATH"``. Because it runs a single time (after the last
        profiled step) rather than once per step, uploading the whole ``save_path`` sends each trace
        exactly once. The run context is exported as ``VERL_PROFILE_SAVE_PATH``, ``VERL_PROFILE_TOOL``,
        ``VERL_PROFILE_RANK``, ``VERL_PROFILE_PID``, ``VERL_PROFILE_ROLE`` and (nsys only)
        ``VERL_PROFILE_RAY_NSIGHT_DIR``. Merged stdout/stderr is streamed so long uploads are visible.
        """
        env = os.environ.copy()
        env["VERL_PROFILE_SAVE_PATH"] = str(save_path) if save_path else ""
        env["VERL_PROFILE_TOOL"] = str(self._tool) if self._tool else ""
        env["VERL_PROFILE_RANK"] = str(self.rank)
        env["VERL_PROFILE_PID"] = str(os.getpid())
        if self.save_file_prefix:
            env["VERL_PROFILE_ROLE"] = str(self.save_file_prefix)
        if self._tool == "nsys":
            from .nvtx_profile import RAY_NSIGHT_LOG_DIR

            env["VERL_PROFILE_RAY_NSIGHT_DIR"] = RAY_NSIGHT_LOG_DIR

        _hook_print(f"rank {self.rank}: running command: {cmd}")
        _hook_print(f"rank {self.rank}: VERL_PROFILE_SAVE_PATH={env['VERL_PROFILE_SAVE_PATH']}")
        # Route through the shell so $VERL_PROFILE_* and shell syntax in `cmd` work.
        try:
            proc = subprocess.Popen(
                ["/bin/sh", "-c", cmd],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as e:  # command failed to launch; do not crash training
            _hook_print(f"rank {self.rank}: command failed to launch: {e}")
            logger.warning("profiler finish hook: command failed to launch on rank %s: %s", self.rank, e)
            return
        with proc:
            for line in proc.stdout:
                _hook_print(f"rank {self.rank}: {line.rstrip()}")
        returncode = proc.returncode
        _hook_print(f"rank {self.rank}: command exited with {returncode}")
        if returncode != 0:
            logger.warning("profiler finish hook: command exited with %s on rank %s: %s", returncode, self.rank, cmd)

    def step(self):
        """Mark the end of one training step in the trace, i.e. one whole RL cycle.

        Delegates to backends that label step boundaries (currently the torch profiler, which
        writes torch's ``ProfilerStep#<n>``); for all others this is a no-op. It never changes
        what gets collected -- ``global_profiler.steps`` decides that -- and the mini-batches a
        step is made of are annotated inside the step instead of advancing it.

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


def build_rollout_dist_profiler(
    replica_rank: int,
    replica_world_size: int,
    config: Optional[ProfilerConfig] = None,
    tool_config: Optional[object] = None,
) -> "DistProfiler":
    """Build a :class:`DistProfiler` for a rollout replica, treating ``config.ranks`` as global GPU ranks.

    Training workers key their ``DistProfiler`` on the worker's own global GPU rank, so ``ranks``
    lists global ranks there. A rollout replica instead drives a single inference engine that spans
    ``replica_world_size`` GPUs (``tensor_model_parallel_size * data_parallel_size *
    pipeline_model_parallel_size``); replica ``r`` owns the global ranks
    ``[r * replica_world_size, (r + 1) * replica_world_size)``.

    To keep ``ranks`` meaning the same thing for rollout as for the training roles, every rank in
    ``config.ranks`` is mapped to the replica that owns it (``rank // replica_world_size``) and the
    replica is profiled when it owns at least one requested rank. For example with
    ``replica_world_size == 8`` (e.g. ``tp=8``), ``ranks=[0, 8]`` profiles replica 0 (owns global
    rank 0) and replica 1 (owns global rank 8) instead of replica indices 0 and 8. ``all_ranks`` and
    the empty-``ranks`` default (profile the replica owning global rank 0, i.e. replica 0) are
    preserved.
    """
    if config is not None and not getattr(config, "all_ranks", False):
        ranks = list(getattr(config, "ranks", None) or [])
        if ranks and replica_world_size and replica_world_size > 0:
            replica_ranks = sorted({int(rank) // replica_world_size for rank in ranks})
            if dataclasses.is_dataclass(config) and not isinstance(config, type):
                config = dataclasses.replace(config, ranks=replica_ranks)
            else:
                # OmegaConf DictConfig or other mutable mapping-style config.
                config = copy.deepcopy(config)
                config.ranks = replica_ranks
    return DistProfiler(rank=replica_rank, config=config, tool_config=tool_config)


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
    def stop_profile(self, run_command: bool = True) -> None:
        """Stop profiling for the current rank in the current training step.

        ``run_command`` is True only on the last profiled step, so the finish command (e.g. the
        trace upload) fires once at the end rather than once per step. Backend stop and artifact
        relocation happen every step regardless.
        """
        self.profiler.stop(run_command=run_command)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def step_profile(self) -> None:
        """Mark the end of one training step in the trace."""
        self.profiler.step()
