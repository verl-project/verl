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

import json
import os
import re
import shutil
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig


@dataclass
class NsightToolConfig(BaseConfig):
    """Nsight tool config."""

    "True for each task has its own database, False for all tasks in one training step share one database."
    discrete: bool = False
    name: str = "nsight"

    def __post_init__(self) -> None:
        pass


@dataclass
class TorchProfilerScheduleConfig(BaseConfig):
    """Schedule for ``torch.profiler.schedule``, applied at the granularity of update
    mini-batches (``profiler.step()`` is advanced once per mini-batch of the actor/critic
    update loop, never on the whole-batch stages like the log-prob forwards or rollout).

    The profiler cycles through ``skip_first`` -> (``wait`` -> ``warmup`` -> ``active``) x
    ``repeat``. Scheduling is only enabled when ``active > 0``; otherwise the update loop is
    recorded in full.

    How it interacts with ``discrete`` matters:

    * ``discrete: True`` -- the update stage is profiled in its own trace, so the full schedule
      applies to its mini-batches: ``skip_first``/``wait``/``warmup`` drop the leading ones and
      only the ``active`` window is kept. The other stages (log-prob, rollout, ref) each get
      their own full trace, untouched by the schedule.
    * ``discrete: False`` -- everything the worker runs in the step shares one continuous trace.
      The schedule still selects which update mini-batches are kept: ``skip_first``/``wait``/
      ``warmup`` skip ahead to a later mini-batch and ``active`` sets how many are recorded. Because
      torch only persists a window that ends in ``RECORD_AND_SAVE``, skipping ahead also drops the
      log-prob/rollout stages that run before the active window. Leave ``skip_first``/``wait``/
      ``warmup`` at 0 (the default) to keep every earlier stage in full plus the first ``active``
      mini-batches. ``repeat`` defaults to a single window per step.

    Either way the schedule never writes torch's ``ProfilerStep#<n>`` rows: verl advances the
    profiler once per mini-batch (not per RL step), so those rows would tag mini-batch boundaries
    and only add noise. The sub-sampling still happens; the boundary labels are just turned off.
    """

    # Number of mini-batches to skip at the very beginning (not counted in the cycle).
    skip_first: int = 0
    # Number of mini-batches to idle (no collection) at the start of each cycle.
    wait: int = 0
    # Number of mini-batches to warm up (tracing on, data discarded) each cycle.
    warmup: int = 0
    # Number of mini-batches to actively record each cycle. <= 0 disables scheduling (record all).
    active: int = 0
    # Number of cycles to repeat. 0 means repeat until profiling stops (continuous mode caps this
    # at a single window per step unless set explicitly).
    repeat: int = 0
    name: str = "torch_schedule"

    def __post_init__(self) -> None:
        """config validation logics go here"""
        for field_name in ("skip_first", "wait", "warmup", "active", "repeat"):
            value = getattr(self, field_name)
            assert isinstance(value, int), f"{field_name} must be int, got {type(value)}"
            assert value >= 0, f"{field_name} must be >= 0, got {value}"

    @property
    def enabled(self) -> bool:
        """Scheduling only takes effect when at least one active mini-batch is requested."""
        return self.active > 0


@dataclass
class TorchProfilerToolConfig(BaseConfig):
    """Torch profiler tool config.

    By default a profiled step is collected whole: ``global_profiler.steps`` picks which RL
    steps to collect, and everything the worker runs in such a step lands in one trace, with
    each stage and each update mini-batch annotated inside it.

    ``schedule`` optionally sub-samples the update loop's mini-batches, so a step with many
    identical mini-batches does not bloat the trace. Only the update loop is sub-sampled
    (``profiler.step()`` is advanced there only); whether the earlier log-prob/rollout stages
    survive depends on ``discrete`` and on how far the schedule skips ahead
    (see :class:`TorchProfilerScheduleConfig`).
    """

    # options: cuda, cpu, memory, shapes, stack. Empty means collect everything.
    # CPU activity is collected either way (the per-stage record_function markers are CPU-side
    # events), so listing "cpu" here is redundant; the other options are honored as written.
    contents: list[str] = field(default_factory=list)
    discrete: bool = False
    # Start collecting profiler data from this response-token index.
    # None means collect from the beginning.
    profile_token_start: Optional[int] = None
    # Stop collecting profiler data at this response-token index (exclusive).
    # None means collect until the end.
    profile_token_end: Optional[int] = None
    # Optional torch.profiler.schedule over update mini-batches (see the class docstring above).
    # When active > 0, DistProfiler.step() advances it once per mini-batch.
    schedule: Optional[TorchProfilerScheduleConfig] = None
    name: str = "torch"

    def __post_init__(self) -> None:
        """config validation logics go here"""
        __support_contents = ["cuda", "cpu", "memory", "shapes", "stack"]
        for content in self.contents:
            assert content in __support_contents, (
                f"Profiler contents only supports {__support_contents}, but gets {content}"
            )
        assert isinstance(self.contents, list), f"Profiler contents must be of type list, got {type(self.contents)}"
        start = self.profile_token_start
        stop = self.profile_token_end
        for name, value in (("profile_token_start", start), ("profile_token_end", stop)):
            if value is not None:
                assert isinstance(value, int), f"{name} must be int or None, got {type(value)}"
                assert value >= 0, f"{name} must be >= 0, got {value}"
        if start is not None and stop is not None:
            assert stop > start, f"profile_token_end must be > profile_token_start, got start={start}, stop={stop}"


@dataclass
class TorchMemoryToolConfig(BaseConfig):
    """Torch memory profiler tool config.

    Args:
        trace_alloc_max_entries (int): Maximum number of memory allocation entries to track.
        stack_depth (int): Stack trace depth for memory allocations.
    """

    trace_alloc_max_entries: int = 100_000
    stack_depth: int = 32
    name: str = "torch_memory"

    def __post_init__(self) -> None:
        """config validation logics go here"""
        assert isinstance(self.trace_alloc_max_entries, int), (
            f"trace_alloc_max_entries must be int, got {type(self.trace_alloc_max_entries)}"
        )
        assert isinstance(self.stack_depth, int), f"stack_depth must be int, got {type(self.stack_depth)}"
        assert self.trace_alloc_max_entries > 0, (
            f"trace_alloc_max_entries must be positive, got {self.trace_alloc_max_entries}"
        )
        assert self.stack_depth > 0, f"stack_depth must be positive, got {self.stack_depth}"


@dataclass
class PrecisionDebuggerToolConfig(BaseConfig):
    """Precision debugger tool config (msprobe)."""

    name: str = "precision_debugger"
    config_path: Optional[str] = None
    # Deprecated: precision_debugger no longer maintains an independent step filter.
    # Collection window is controlled by global_profiler.steps.
    steps: Optional[list[int]] = None
    # Supported stages:
    # actor_update, actor_compute_log_prob, ref_compute_log_prob,
    # compute_values, critic_update, compute_rm_score
    stages: Optional[list[str]] = None
    strict: bool = False

    def __post_init__(self) -> None:
        if self.config_path is not None:
            assert isinstance(self.config_path, str), f"config_path must be str, got {type(self.config_path)}"
        if self.steps is not None:
            assert isinstance(self.steps, list), f"steps must be list[int], got {type(self.steps)}"
        if self.stages is not None:
            assert isinstance(self.stages, list), f"stages must be list[str], got {type(self.stages)}"
        assert isinstance(self.strict, bool), f"strict must be bool, got {type(self.strict)}"


@dataclass
class NPUToolConfig(NsightToolConfig):
    """NPU profiler too; config."""

    # options: npu, cpu, memory, shapes, module, stack
    # CPU activity is collected either way (the per-stage mstx markers are CPU-side events),
    # so listing "cpu" here is redundant; the other options are honored as written.
    contents: list[str] = field(default_factory=list)

    # Collection level, optional values: level_none, level0, level1, level2.
    level: str = "level0"

    # Whether to automatically parse the data.
    analysis: bool = False
    # Start collecting profiler data from this response-token index.
    # None means collect from the beginning.
    profile_token_start: Optional[int] = None
    # Stop collecting profiler data at this response-token index (exclusive).
    # None means collect until the end.
    profile_token_end: Optional[int] = None

    name: str = "npu"

    def __post_init__(self) -> None:
        """config validation logics go here"""
        assert isinstance(self.contents, list), f"Profiler contents must be of type list, got {type(self.contents)}"
        assert isinstance(self.level, str), f"Profiler level must be of type str, got {type(self.level)}"
        assert isinstance(self.analysis, bool), f"Profiler analysis must be of type bool, got {type(self.analysis)}"
        for content in self.contents:
            assert content in ["npu", "cpu", "memory", "shapes", "module", "stack"], (
                f"Profiler contents only supports npu, cpu, memory, shapes, module, stack, but gets {content}"
            )
        assert self.level in ["level_none", "level0", "level1", "level2"], (
            f"Profiler level only supports level0, 1, 2, and level_none, but gets {self.level}"
        )
        start = self.profile_token_start
        stop = self.profile_token_end
        for name, value in (("profile_token_start", start), ("profile_token_end", stop)):
            if value is not None:
                assert isinstance(value, int), f"{name} must be int or None, got {type(value)}"
                assert value >= 0, f"{name} must be >= 0, got {value}"
        if start is not None and stop is not None:
            assert stop > start, f"profile_token_end must be > profile_token_start, got start={start}, stop={stop}"


@dataclass
class ProfilerConfig(BaseConfig):
    """Worker profiler config.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        discrete (bool): True for each task has its own database, False for all tasks in one training step
          share one database.
        all_ranks (bool): Whether to profile all ranks.
        ranks (list[int]): The ranks that will be profiled. Defaults to [].
        global_tool_config (Any): Global tool configuration for all profiling tools.
        relocate_results (bool): When profiling finishes, move backend artifacts that a framework writes
          outside the flat layout into ``save_path``: ``nsys`` reports, which Ray hardcodes under
          ``/tmp/ray/session_latest/logs/nsight`` (the directory is not configurable, see
          https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html), and rollout engine
          traces, which land in a per-replica sub-directory (see :func:`relocate_rollout_traces`). Runs on
          the profiled ranks.
        finish_hook_cmd (Optional[str]): Shell command executed on the selected ranks **once**, after the last
          profiled step (not once per step). Backend ``stop()`` and ``relocate_results`` still run every profiled
          step, so traces accumulate in ``save_path``; this command then runs a single time to ship them off the
          node, e.g. ``mlx asset upload "$VERL_PROFILE_SAVE_PATH"``. Because it runs once rather than per step,
          uploading the whole ``save_path`` sends each trace exactly once (no re-uploading of the accumulating
          directory). The run context is exported as ``VERL_PROFILE_SAVE_PATH`` (the configured ``save_path``),
          ``VERL_PROFILE_TOOL``, ``VERL_PROFILE_RANK``, ``VERL_PROFILE_PID``, ``VERL_PROFILE_ROLE`` (if known) and
          ``VERL_PROFILE_RAY_NSIGHT_DIR`` (nsys only). ``save_path`` is usually node-local, so the command runs on
          every selected rank/node; pick ``finish_hook_ranks`` so one rank per node uploads that node's directory.
        finish_hook_all_ranks (bool): Run ``finish_hook_cmd`` on every rank.
        finish_hook_ranks (list[int]): Ranks that run ``finish_hook_cmd``. Ignored when ``finish_hook_all_ranks``
          is True. When both are unset the hook falls back to the profiled ranks (``all_ranks``/``ranks``).
    """

    tool: Optional[str] = MISSING
    enable: bool = False
    all_ranks: bool = False
    ranks: list[int] = field(default_factory=list)
    save_path: Optional[str] = MISSING
    tool_config: Any = MISSING  # Just a placeholder, will use configs above directly
    global_tool_config: Optional[Any] = None  # Global tool configuration for all profiling tools
    # --- Finish hook (runs when profiling stops) ---
    relocate_results: bool = False
    finish_hook_cmd: Optional[str] = None
    finish_hook_all_ranks: bool = False
    finish_hook_ranks: list[int] = field(default_factory=list)

    def union(self, other: "ProfilerConfig") -> "ProfilerConfig":
        assert self.tool == other.tool, f"Cannot union ProfilerConfig with different tools: {self.tool} vs {other.tool}"
        return ProfilerConfig(
            tool=self.tool,
            enable=self.enable or other.enable,
            all_ranks=self.all_ranks or other.all_ranks,
            ranks=list(set(self.ranks or []) | set(other.ranks or [])),
            save_path=self.save_path,
            tool_config=self.tool_config,
            global_tool_config=self.global_tool_config or other.global_tool_config,
            relocate_results=self.relocate_results or other.relocate_results,
            finish_hook_cmd=self.finish_hook_cmd or other.finish_hook_cmd,
            finish_hook_all_ranks=self.finish_hook_all_ranks or other.finish_hook_all_ranks,
            finish_hook_ranks=list(set(self.finish_hook_ranks or []) | set(other.finish_hook_ranks or [])),
        )

    def intersect(self, other: "ProfilerConfig") -> "ProfilerConfig":
        assert self.tool == other.tool, (
            f"Cannot intersect ProfilerConfig with different tools: {self.tool} vs {other.tool}"
        )
        return ProfilerConfig(
            tool=self.tool,
            enable=self.enable and other.enable,
            all_ranks=self.all_ranks and other.all_ranks,
            ranks=list(set(self.ranks or []) & set(other.ranks or [])),
            save_path=self.save_path,
            tool_config=self.tool_config,
            global_tool_config=self.global_tool_config if self.global_tool_config else other.global_tool_config,
            relocate_results=self.relocate_results and other.relocate_results,
            finish_hook_cmd=self.finish_hook_cmd if self.finish_hook_cmd else other.finish_hook_cmd,
            finish_hook_all_ranks=self.finish_hook_all_ranks and other.finish_hook_all_ranks,
            finish_hook_ranks=list(set(self.finish_hook_ranks or []) & set(other.finish_hook_ranks or [])),
        )

    def __post_init__(self) -> None:
        """config validation logics go here"""
        assert isinstance(self.ranks, set | list | tuple), (
            f"Profiler ranks must be of type list, got {type(self.ranks)}"
        )
        assert isinstance(self.finish_hook_ranks, set | list | tuple), (
            f"Profiler finish_hook_ranks must be of type list, got {type(self.finish_hook_ranks)}"
        )
        assert self.finish_hook_cmd is None or isinstance(self.finish_hook_cmd, str), (
            f"finish_hook_cmd must be str or None, got {type(self.finish_hook_cmd)}"
        )


def rollout_trace_dir(profiler_config: ProfilerConfig, rank: int) -> str:
    """Return the directory an inference engine writes the traces of ``rank`` to.

    Engine-side profiling is per replica, so each replica gets its own sub-directory of
    ``save_path`` instead of writing into the flat layout the training workers use.
    """
    return os.path.join(profiler_config.save_path, f"agent_loop_rollout_replica_{rank}")


def rollout_profiler_global_ranks(profiler_config: Optional["ProfilerConfig"]) -> Optional[set[int]]:
    """The exact global GPU ranks a rollout replica should surface traces for, or ``None`` to keep
    every trace.

    ``ranks`` in a rollout profiler config are global GPU ranks (as in the training roles). Because a
    ``tp>1`` engine cannot profile a subset of its tensor-parallel group, it traces every GPU in the
    replica; this returns the ranks the user actually asked for so :func:`relocate_rollout_traces`
    can surface only those (via its ``keep_global_ranks`` argument). ``None`` means "keep all" and is
    returned when profiling is off, ``all_ranks`` is set, or ``ranks`` is empty (the default).
    """
    if profiler_config is None or getattr(profiler_config, "all_ranks", False):
        return None
    ranks = {int(r) for r in (getattr(profiler_config, "ranks", None) or [])}
    return ranks or None


# The engines name one trace file per GPU and encode that GPU's tensor-parallel rank in the name:
# vLLM uses ``...-rank-<tp>...`` (or ``...rank<tp>...``) and SGLang uses ``...-TP-<tp>...``.
_ROLLOUT_TRACE_TP_RANK_RES = (
    re.compile(r"(?:^|[-_.])TP-(\d+)", re.IGNORECASE),
    re.compile(r"(?:^|[-_.])rank[-_]?(\d+)", re.IGNORECASE),
)
# SGLang only appends these when the corresponding parallel size is > 1; their presence means the
# GPU's linear index within the replica is not just the tp rank, and we do not guess the layout.
_ROLLOUT_TRACE_MULTI_DIM_RE = re.compile(r"(?:^|[-_.])(?:DP|PP|EP)-\d+", re.IGNORECASE)


def rollout_trace_local_rank(name: str, world_size: int) -> Optional[int]:
    """The GPU's linear rank within its replica (``0..world_size-1``), read from an engine trace
    file name, or ``None`` when it cannot be determined unambiguously.

    A rollout replica spans ``world_size`` GPUs and its engine writes one trace per GPU, naming it by
    tensor-parallel rank (vLLM ``...-rank-<tp>...``, SGLang ``...-TP-<tp>...``). For a tp-only replica
    that tp rank is exactly the GPU's offset within the replica, so ``global_rank = base + this``.
    When the name carries other parallel dimensions (SGLang ``-DP-``/``-PP-``/``-EP-``) the linear
    offset is layout-dependent, so ``None`` is returned rather than risk a wrong mapping; engines that
    name traces by host/pid (older vLLM) likewise yield ``None``. Callers treat ``None`` as "keep".
    """
    if world_size <= 1:
        return None
    if _ROLLOUT_TRACE_MULTI_DIM_RE.search(name):
        return None
    for pattern in _ROLLOUT_TRACE_TP_RANK_RES:
        match = pattern.search(name)
        if match:
            local_rank = int(match.group(1))
            return local_rank if 0 <= local_rank < world_size else None
    return None


def relocate_rollout_traces(
    profiler_config: ProfilerConfig,
    rank: int,
    world_size: int = 1,
    keep_global_ranks: Optional[Iterable[int]] = None,
) -> list[str]:
    """Move the engine traces of replica ``rank`` up into ``save_path``. No-op unless
    ``relocate_results`` is set.

    The engines only take a directory, so replicas write into sub-directories of ``save_path``
    (see :func:`rollout_trace_dir`) and post-processing that does not walk sub-directories never
    sees them. Relocating puts a step's rollout traces in the same directory as its training
    traces, which is the one directory the run configured. The replica moves into the file name,
    since it is no longer in the path.

    ``world_size`` is the number of GPUs the replica spans (``tp * dp * pp``). A replica drives one
    engine, which writes one trace per GPU and encodes that GPU's tensor-parallel rank in the file
    name (vLLM ``...-rank-<tp>...``, SGLang ``...-TP-<tp>...``; see :func:`rollout_trace_local_rank`).
    When ``world_size > 1`` and that tp rank can be read, the file's absolute global GPU rank
    (``rank * world_size + tp_rank``) is added to the relocated name so the flattened files line up
    with the global ``ranks`` the user configured -- e.g. with ``tp=2`` replica 4's tp rank 0 becomes
    ``rollout-replica4-globalrank8_...``. When the tp rank is not encoded in the name (or
    ``world_size <= 1``, where the replica index is already the global rank) only the replica index
    is used.

    ``keep_global_ranks`` is the exact set of global GPU ranks the user asked to profile. A ``tp>1``
    engine has to profile its whole replica (there is no way to profile a subset of a tensor-parallel
    group), so it writes a trace for every GPU it spans; passing ``keep_global_ranks`` relocates only
    the traces for those GPUs and leaves the unrequested tp-mates in the sub-directory. So with
    ``tp=2`` and ``ranks=[0, 8]`` (``keep_global_ranks={0, 8}``) exactly global GPUs 0 and 8 land in
    ``save_path``, not their tp-mates 1 and 9. Pass ``None`` (the default, and what ``all_ranks`` or an
    empty ``ranks`` should use) to keep every trace. A file whose global rank cannot be read is always
    kept, so filtering never silently drops a requested rank.

    Anything an engine flushes after this stays in the sub-directory, so late writes are kept
    rather than lost. Returns the new paths; files that cannot be moved are reported and skipped.
    """
    if not getattr(profiler_config, "relocate_results", False):
        return []

    src_dir = rollout_trace_dir(profiler_config, rank)
    if not os.path.isdir(src_dir):
        return []

    keep = {int(r) for r in keep_global_ranks} if keep_global_ranks is not None else None
    base_global_rank = rank * world_size if world_size and world_size > 0 else rank

    relocated = []
    for name in sorted(os.listdir(src_dir)):
        prefix = f"rollout-replica{rank}_"
        local_rank = rollout_trace_local_rank(name, world_size) if world_size else None
        global_rank = base_global_rank + local_rank if local_rank is not None else None
        if global_rank is not None:
            prefix = f"rollout-replica{rank}-globalrank{global_rank}_"
        # Keep only the GPUs the user asked for; a tp>1 engine also traced its other ranks, but those
        # are left in the sub-directory rather than surfaced. Unknown ranks are kept, never dropped.
        if keep is not None and global_rank is not None and global_rank not in keep:
            continue
        dst = os.path.join(profiler_config.save_path, prefix + name)
        try:
            shutil.move(os.path.join(src_dir, name), dst)
        except Exception as e:
            print(f"[Profiler] rank {rank}: relocating rollout trace {name} failed: {e}", flush=True)
            continue
        relocated.append(dst)
    return relocated


def build_vllm_profiler_args(
    profiler_config: ProfilerConfig, tool_config: BaseConfig, rank: int, legacy_env: bool = True
) -> dict:
    """
    Build arguments and environment variables for vLLM profiler.

    Acts as an adapter to bridge verl's unified profiler config and vLLM's specific requirements.
    It sets environment variables for compatibility and constructs arguments for vLLM >= 0.13.0.

    Args:
        profiler_config (ProfilerConfig): The unified profiler configuration.
        tool_config (BaseConfig): The tool configuration.
        rank (int): The rank of the replica.
        legacy_env (bool): Whether to export the ``VLLM_TORCH_PROFILER_*`` environment variables,
            which is how vLLM < 0.13.0 is configured. Newer versions read the returned
            ``profiler_config`` argument instead and reject these names as unknown, so callers
            running vLLM >= 0.13.0 should pass False to avoid noisy startup warnings.

    Returns:
        dict: A dictionary of arguments to be passed to vLLM's start_profile method.
    """
    if not profiler_config or not tool_config or not hasattr(tool_config, "contents"):
        return {}

    contents = tool_config.contents
    with_stack = True if "stack" in contents or "module" in contents else False
    record_shapes = True if "shapes" in contents else False
    with_memory = True if "memory" in contents else False
    save_path = rollout_trace_dir(profiler_config, rank)

    # vLLM < 0.13.0 supports controlling profiler via environment variables
    if legacy_env:
        os.environ["VLLM_TORCH_PROFILER_DIR"] = save_path
        os.environ["VLLM_TORCH_PROFILER_WITH_STACK"] = "1" if with_stack else "0"
        os.environ["VLLM_TORCH_PROFILER_RECORD_SHAPES"] = "1" if record_shapes else "0"
        os.environ["VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY"] = "1" if with_memory else "0"

    # vLLM >= 0.13.0 supports controlling profiler via arguments.
    # While it maintains backward compatibility with environment variables,
    # we provide arguments explicitly to align with the new API style.
    profile_token_start = getattr(tool_config, "profile_token_start", None)
    profile_token_end = getattr(tool_config, "profile_token_end", None)

    # vLLM uses 0 to indicate immediate start / no upper bound.
    delay_iterations = profile_token_start if profile_token_start is not None else 0
    max_iterations = (profile_token_end - delay_iterations) if profile_token_end is not None else 0

    return {
        "profiler_config": json.dumps(
            {
                "profiler": "torch",
                "torch_profiler_dir": save_path,
                "torch_profiler_with_memory": with_memory,
                "torch_profiler_with_stack": with_stack,
                "torch_profiler_record_shapes": record_shapes,
                "delay_iterations": delay_iterations,
                "max_iterations": max_iterations,
                "ignore_frontend": "true",
            }
        )
    }


def build_sglang_profiler_args(profiler_config: ProfilerConfig, tool_config: BaseConfig, rank: int) -> dict:
    """
    Build arguments for SGLang profiler.

    Args:
        profiler_config (ProfilerConfig): The unified profiler configuration.
        tool_config (BaseConfig): The tool configuration.
        rank (int): The rank of the replica.

    Returns:
        dict: A dictionary of arguments suitable for starting the SGLang profiler.
    """
    if not profiler_config or not tool_config or not hasattr(tool_config, "contents"):
        return {}

    contents = tool_config.contents
    if "memory" in contents:
        warnings.warn("SGLang profiler does not support memory profiling. Ignoring memory content.", stacklevel=2)

    profile_token_start = getattr(tool_config, "profile_token_start", None)
    profile_token_end = getattr(tool_config, "profile_token_end", None)
    # SGLang API uses Optional[int], keep None for "not set" and map 0 to "no upper bound".
    start_step = profile_token_start
    num_steps = (
        profile_token_end - (profile_token_start if profile_token_start is not None else 0)
        if profile_token_end is not None
        else None
    )

    return {
        "output_dir": rollout_trace_dir(profiler_config, rank),
        "with_stack": "stack" in contents or "module" in contents,
        "record_shapes": "shapes" in contents,
        "start_step": start_step,
        "num_steps": num_steps,
    }
