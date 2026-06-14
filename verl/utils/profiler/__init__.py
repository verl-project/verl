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

from ..device import is_npu_available
from ..import_utils import is_nvtx_available
from .config import build_sglang_profiler_args, build_vllm_profiler_args
from .performance import GPUMemoryLogger, log_gpu_memory_usage, simple_timer
from .profile import DistProfiler, DistProfilerExtension, ProfilerConfig

# Select marker implementations by availability, but keep DistProfiler as our dispatcher
if is_nvtx_available():
    from .nvtx_profile import mark_annotate, mark_end_range, mark_start_range, marked_timer
elif is_npu_available:
    from .mstx_profile import mark_annotate, mark_end_range, mark_start_range, marked_timer
else:
    from .performance import marked_timer
    from .profile import mark_annotate, mark_end_range, mark_start_range

__all__ = [
    "GPUMemoryLogger",
    "log_gpu_memory_usage",
    "mark_start_range",
    "mark_end_range",
    "mark_annotate",
    "DistProfiler",
    "DistProfilerExtension",
    "ProfilerConfig",
    "simple_timer",
    "marked_timer",
    "build_vllm_profiler_args",
    "build_sglang_profiler_args",
]
