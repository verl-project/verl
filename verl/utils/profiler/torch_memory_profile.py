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

from typing import Optional

from .config import ProfilerConfig, TorchMemoryToolConfig
from ..memory_utils import MemorySnapshotSampler, enable_memory_visualize

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