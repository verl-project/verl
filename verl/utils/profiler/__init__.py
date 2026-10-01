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

from verl.plugin.platform import get_platform

from ..device import is_npu_available
from ..import_utils import is_nvtx_available
from .config import (
    build_sglang_profiler_args,
    build_vllm_profiler_args,
    relocate_rollout_traces,
    rollout_profiler_global_ranks,
    rollout_trace_dir,
    rollout_trace_local_rank,
)
from .performance import GPUMemoryLogger, log_gpu_memory_usage, simple_timer
from .profile import DistProfiler, DistProfilerExtension, ProfilerConfig, build_rollout_dist_profiler

_mark_start_range, _mark_end_range, _mark_annotate, _marked_timer = None, None, None, None


def _resolve_markers() -> None:
    """Select marker implementations by availability, but keep DistProfiler as our dispatcher.

    nvtx (package) and npu (built-in device) are checked first for backward compatibility;
    any other platform (built-in or plugin-supplied) can opt in via profiler_markers().

    Resolved lazily on first use rather than at import time: get_platform() runs hardware
    auto-detection (smi probes) and caches the result for the process, and this module is
    imported from far too many places to pay that cost -- and lock in the platform choice --
    just from being imported.
    """
    global _mark_start_range, _mark_end_range, _mark_annotate, _marked_timer

    if is_nvtx_available():
        from .nvtx_profile import mark_annotate, mark_end_range, mark_start_range, marked_timer
    elif is_npu_available:
        from .mstx_profile import mark_annotate, mark_end_range, mark_start_range, marked_timer
    elif (platform_markers := get_platform().profiler_markers()) is not None:
        mark_start_range, mark_end_range, mark_annotate, marked_timer = platform_markers
    else:
        from .performance import marked_timer
        from .profile import mark_annotate, mark_end_range, mark_start_range

    _mark_start_range, _mark_end_range, _mark_annotate, _marked_timer = (
        mark_start_range,
        mark_end_range,
        mark_annotate,
        marked_timer,
    )


def mark_start_range(*args, **kwargs):
    """Start a profiling range using the resolved platform marker implementation."""
    if _mark_start_range is None:
        _resolve_markers()
    return _mark_start_range(*args, **kwargs)


def mark_end_range(*args, **kwargs):
    """End a profiling range using the resolved platform marker implementation."""
    if _mark_end_range is None:
        _resolve_markers()
    return _mark_end_range(*args, **kwargs)


def mark_annotate(*args, **kwargs):
    """Annotate a function with a profiling range using the resolved platform marker implementation."""
    if _mark_annotate is None:
        _resolve_markers()
    return _mark_annotate(*args, **kwargs)


def marked_timer(*args, **kwargs):
    """Time a code block and mark it as a profiling range using the resolved platform marker implementation."""
    if _marked_timer is None:
        _resolve_markers()
    return _marked_timer(*args, **kwargs)


__all__ = [
    "GPUMemoryLogger",
    "log_gpu_memory_usage",
    "mark_start_range",
    "mark_end_range",
    "mark_annotate",
    "DistProfiler",
    "DistProfilerExtension",
    "ProfilerConfig",
    "build_rollout_dist_profiler",
    "simple_timer",
    "marked_timer",
    "build_vllm_profiler_args",
    "build_sglang_profiler_args",
    "rollout_trace_dir",
    "relocate_rollout_traces",
    "rollout_profiler_global_ranks",
    "rollout_trace_local_rank",
]
