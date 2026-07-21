# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import os
import re
from datetime import datetime, timezone
from typing import Callable, Optional

import torch

from .config import ProfilerConfig, TorchProfilerToolConfig
from .profile import DistProfiler


def get_dist_topology() -> dict:
    """Best-effort snapshot of the current process's distributed topology.

    Used to make per-process profiler trace files self-describing. The returned dict
    may contain ``rank``/``world_size`` (from ``torch.distributed``) and the
    ``tp``/``pp``/``dp``/``cp`` parallel ranks (from Megatron's ``parallel_state`` when
    initialized). Every lookup is guarded, so this never raises and simply omits the
    pieces that are unavailable (e.g. plain FSDP data parallelism only exposes rank).
    """
    info: dict = {}
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            info["rank"] = dist.get_rank()
            info["world_size"] = dist.get_world_size()
    except Exception:
        pass

    try:
        from megatron.core import parallel_state as mpu

        if mpu.model_parallel_is_initialized():
            info["tp"] = mpu.get_tensor_model_parallel_rank()
            info["pp"] = mpu.get_pipeline_model_parallel_rank()
            info["dp"] = mpu.get_data_parallel_rank()
            try:
                info["cp"] = mpu.get_context_parallel_rank()
            except Exception:
                pass
    except Exception:
        pass

    return info


def _sanitize_name_part(text: str) -> str:
    """Make an arbitrary label safe to embed in a filename."""
    return re.sub(r"[^0-9A-Za-z.=+-]+", "-", str(text)).strip("-")


def build_trace_basename(
    rank: int,
    role: Optional[str] = None,
    save_file_prefix: Optional[str] = None,
    topology: Optional[dict] = None,
) -> str:
    """Build a descriptive, per-process trace filename stem.

    Encodes -- when available -- the worker role (``save_file_prefix``, e.g. ``actor``),
    the profiling scope role (``role``, e.g. ``e2e``), the global rank and world size,
    and the tensor/pipeline/data/context parallel ranks, followed by pid and a
    timestamp so that files written by different processes never collide.
    """
    topology = get_dist_topology() if topology is None else topology
    current_time = datetime.now(tz=timezone.utc).astimezone()
    timestamp = current_time.strftime("%Y%m%d%H%M%S%f")[:-3]
    pid = os.getpid()

    parts: list[str] = []
    if save_file_prefix:
        parts.append(_sanitize_name_part(save_file_prefix))
    if role:
        parts.append(_sanitize_name_part(role))

    global_rank = topology.get("rank", rank)
    world_size = topology.get("world_size")
    rank_part = f"rank{global_rank}"
    if world_size:
        rank_part += f"-of-{world_size}"
    parts.append(rank_part)

    parallel_part = "-".join(f"{dim}{topology[dim]}" for dim in ("tp", "pp", "dp", "cp") if dim in topology)
    if parallel_part:
        parts.append(parallel_part)

    parts.append(f"pid{pid}")
    parts.append(timestamp)
    return "_".join(parts)


def get_torch_profiler(
    contents: list[str],
    save_path: str,
    role: Optional[str] = None,
    save_file_prefix: Optional[str] = None,
    rank: int = 0,
    schedule: Optional[dict] = None,
):
    """Build a ``torch.profiler.profile`` instance.

    Args:
        contents: Selects the other ``torch.profiler.profile`` arguments -- ``cpu``/``cuda``
            map to ``activities``, ``shapes`` to ``record_shapes``, ``memory`` to
            ``profile_memory`` and ``stack`` to ``with_stack``.
        save_path: Directory (optionally suffixed by ``role``) to write chrome traces to.
        role: Optional sub-directory / logical scope name, also embedded in the filename.
        save_file_prefix: Optional filename prefix, typically the worker role (``actor``/
            ``critic``/``ref``) so per-process traces are distinguishable.
        rank: Global rank, embedded in the trace filename (a fallback when
            ``torch.distributed`` is not initialized).
        schedule: Optional kwargs for ``torch.profiler.schedule``
            (``wait``/``warmup``/``active``/``repeat``/``skip_first``). When provided, the
            caller must drive ``prof.step()`` once per step to advance the schedule.
    """
    save_dir = os.path.join(save_path, role) if role else save_path

    os.makedirs(save_dir, exist_ok=True)

    base_file_name = build_trace_basename(rank=rank, role=role, save_file_prefix=save_file_prefix)

    # A scheduled profiler can fire on_trace_ready multiple times (one per active
    # cycle), so keep an invocation counter to avoid overwriting earlier cycles.
    handler_state = {"count": 0}

    def _trace_handler(prof):
        idx = handler_state["count"]
        handler_state["count"] += 1
        suffix = "" if idx == 0 else f"_cycle{idx}"
        out_path = os.path.join(save_dir, f"{base_file_name}{suffix}.json.gz")
        print(f"[Profiler] Saving trace to {out_path}")
        prof.export_chrome_trace(out_path)

    contents = set(contents) if contents else set()
    activities = []
    if not contents or "cpu" in contents:
        activities.append(torch.profiler.ProfilerActivity.CPU)
    if not contents or "cuda" in contents:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    profile_kwargs = dict(
        activities=activities,
        with_stack="stack" in contents,
        record_shapes="shapes" in contents,
        profile_memory="memory" in contents,
        on_trace_ready=_trace_handler,
    )

    # torch.profiler.schedule drives the wait/warmup/active/repeat state machine
    # via prof.step(); without it the profiler collects continuously.
    if schedule:
        profile_kwargs["schedule"] = torch.profiler.schedule(**schedule)

    return torch.profiler.profile(**profile_kwargs)


class Profiler(DistProfiler):
    """A PyTorch profiler wrapper class for collecting performance metrics.

    This profiler provides a convenient interface for profiling PyTorch operations,
    with support for:

    - CPU and CUDA activity profiling
    - Configurable profiling schedule (wait/warmup/active steps)
    - Multi-rank profiling support
    - Chrome trace export

    Args:
        config: Configuration object containing profiling parameters
    """

    _define_count = 0
    # Process-global handle to the currently running torch profiler. torch.profiler is
    # process-wide, so a step() issued by one Profiler instance (e.g. the inner actor
    # TrainingWorker running the mini-batch loop) must advance the profiler that another
    # instance started (e.g. the outer ActorRolloutRefWorker).
    _active_prof = None

    def __init__(
        self,
        rank,
        config: ProfilerConfig,
        tool_config: Optional[TorchProfilerToolConfig] = None,
        save_file_prefix=None,
    ):
        # note : if we do not set use_profile, it will be set as None, so that all function will be skip
        config = config or ProfilerConfig(ranks=[], enable=False)
        self.save_file_prefix = save_file_prefix

        if not tool_config:
            assert not config.enable, "tool_config must be provided when profiler is enabled"

        self.prof = None
        self.rank = rank
        self.config = config
        self.tool_config = tool_config
        self.contents = self.tool_config.contents
        self.save_path = self.config.save_path
        # Align with other profilers: read discrete mode, default to False for torch profiler
        self.discrete = getattr(self.tool_config, "discrete", False)
        # Resolved torch.profiler.schedule kwargs for the active run (None => continuous).
        self._schedule_kwargs = None

    def check(self):
        return self.prof is not None

    def _resolve_schedule_kwargs(self) -> Optional[dict]:
        """Build torch.profiler.schedule kwargs from tool_config, or None to disable."""
        sched = getattr(self.tool_config, "schedule", None) if self.tool_config else None
        if sched is None:
            return None
        active = int(getattr(sched, "active", 0) or 0)
        if active <= 0:
            return None
        return {
            "skip_first": int(getattr(sched, "skip_first", 0) or 0),
            "wait": int(getattr(sched, "wait", 0) or 0),
            "warmup": int(getattr(sched, "warmup", 0) or 0),
            "active": active,
            "repeat": int(getattr(sched, "repeat", 0) or 0),
        }

    def start(self, **kwargs):
        role = kwargs.get("role", None)
        if not self.discrete and Profiler._define_count == 0:
            self._schedule_kwargs = self._resolve_schedule_kwargs()
            self.prof = get_torch_profiler(
                contents=self.contents,
                save_path=self.save_path,
                role=role,
                save_file_prefix=self.save_file_prefix,
                rank=self.rank,
                schedule=self._schedule_kwargs,
            )
            print(f"[Profiler] started for rank {self.rank}")
            self.prof.start()
            Profiler._active_prof = self.prof
            Profiler._define_count += 1

    def step(self):
        """Advance the process-global active profiler by one step (per mini-batch).

        No-op when no torch profiler is currently running.
        """
        if Profiler._active_prof is not None:
            Profiler._active_prof.step()

    def stop(self):
        if not self.discrete and Profiler._define_count == 1:
            # Continuous mode emits a trailing step to flush the final window; when a
            # schedule is configured, stepping is driven per mini-batch instead.
            if not self._schedule_kwargs:
                self.step()
            print(f"[Profiler] stopped for rank {self.rank}")
            self.prof.stop()
            Profiler._active_prof = None
            self._schedule_kwargs = None
            Profiler._define_count -= 1

    def annotate(self, message: Optional[str] = None, role: Optional[str] = None, **kwargs_outer) -> Callable:
        """Decorate a Worker member function to profile the current rank in the current training step.

        Requires the target function to be a member function of a Worker,
        which has a member field `profiler` with Profiler type.

        Args:
            message (str, optional):
                The message to be displayed in the profiler. Defaults to None.
            role (str, optional):
                The role of the current data collection. Defaults to None.
        """

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs_inner):
                profile_name = message or func.__name__

                if not self.discrete:
                    # In continuous mode, we just record function, profiler started globally
                    with torch.profiler.record_function(profile_name):
                        return func(*args, **kwargs_inner)

                # In discrete mode, we start/stop profiler around the function.
                # torch.profiler is process-global, so wrap the call in try/finally:
                # if func raises, we must still stop the profiler. Otherwise it leaks
                # and the next stage's prof.start() fails with "Profiler is already
                # enabled on this thread", plus the process aborts at teardown.
                prof = get_torch_profiler(
                    contents=self.contents,
                    save_path=self.save_path,
                    role=role,
                    save_file_prefix=self.save_file_prefix,
                    rank=self.rank,
                )
                prof.start()
                try:
                    with torch.profiler.record_function(profile_name):
                        return func(*args, **kwargs_inner)
                finally:
                    prof.stop()

            return wrapper

        return decorator
