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

import logging
import traceback
from datetime import datetime
from typing import Optional

import torch

from ..device import get_device_name, get_torch_device
from ..memory_utils import MemorySnapshotSampler, clear_memory_history, enable_memory_visualize, get_memory_info
from .config import ProfilerConfig, TorchMemoryToolConfig

logger = logging.getLogger(__name__)


class TorchMemoryProfiler:
    """Profiler that dumps CUDA/NPU memory snapshots at step boundaries.

    Behavior:
    - On first construction (per process), enable memory history recording if an accelerator is available
    - Automatically register an OOM snapshot callback on selected ranks when supported
    - On start(step=X), begin or extend a configured memory-history window
    - On stop(), dump a memory snapshot once the window reaches its configured number of steps
    """

    _memory_history_enabled: bool = False
    _oom_observer_attached: bool = False

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
        self._device_name = get_device_name()
        self.this_step = False
        self._window_start_step = None
        self._window_end_step = None
        self._steps_in_window = 0
        self.sampler = MemorySnapshotSampler()

        # Get parameters from tool_config, with fallback to defaults
        if tool_config:
            self.trace_alloc_max_entries = tool_config.trace_alloc_max_entries
            self.stack_depth = tool_config.stack_depth
            self.memory_snapshot_num_steps = tool_config.memory_snapshot_num_steps
        else:
            self.trace_alloc_max_entries = 100_000
            self.stack_depth = 32
            self.memory_snapshot_num_steps = 1

        # Best-effort enable memory history once
        if not TorchMemoryProfiler._memory_history_enabled:
            try:
                enable_memory_visualize(
                    trace_alloc_max_entries=self.trace_alloc_max_entries, stack_depth=self.stack_depth
                )
            except Exception:
                # silently ignore if not supported
                pass
            TorchMemoryProfiler._memory_history_enabled = True

        if self._should_profile_this_rank():
            self._attach_oom_observer()

    def _attach_oom_observer(self) -> None:
        """Register one process-local callback that writes a snapshot at allocator OOM time."""
        if TorchMemoryProfiler._oom_observer_attached:
            return

        if self._device_name not in ("cuda", "npu"):
            logger.warning("[torch_memory] automatic OOM snapshots are only available on CUDA/NPU devices")
            return

        try:
            if not get_torch_device().is_available():
                logger.warning("[torch_memory] automatic OOM snapshots require an available accelerator")
                return

            if self._device_name == "npu":
                import torch_npu

                backend = torch_npu._C
            else:
                backend = torch._C
            attach_observer = getattr(backend, f"_{self._device_name}_attach_out_of_memory_observer", None)
            if attach_observer is None:
                logger.warning("[torch_memory] this build does not support %s OOM observers", self._device_name.upper())
                return

            attach_observer(self._on_out_of_memory)
            TorchMemoryProfiler._oom_observer_attached = True
            logger.info("[torch_memory] %s OOM snapshot observer attached", self._device_name.upper())
        except Exception as exc:
            logger.warning(
                "[torch_memory] failed to attach %s OOM snapshot observer: %s", self._device_name.upper(), exc
            )

    def _on_out_of_memory(self, device: int, alloc: int, device_total: int, device_free: int) -> None:
        """Best-effort snapshot callback invoked by the CUDA/NPU allocator."""
        out_dir = self.config.save_path or "outputs/profile"
        sub_dir = f"oom_{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
        # NPU reports the configured process memory limit, or device total if no limit is set.
        total_label = "total_or_limit" if self._device_name == "npu" else "total"
        logger.error(
            "[torch_memory] %s OOM on device %s: requested=%s %s=%s free=%s; dumping allocator snapshot",
            self._device_name.upper(),
            device,
            alloc,
            total_label,
            device_total,
            device_free,
        )
        logger.error("[torch_memory] Python stack at OOM:\n%s", "".join(traceback.format_stack()))
        try:
            logger.error("[torch_memory] allocator memory at OOM: %s", get_memory_info())
        except Exception as exc:
            logger.warning(f"[torch_memory] failed to collect allocator memory at OOM: {exc}")
        try:
            # Do not synchronize here: an OOM may have left the device stream in an error state.
            self.sampler.dump_memory_snapshot(
                out_dir=out_dir, tag="torch_memory_oom", sub_dir=sub_dir, synchronize=False
            )
        except Exception as exc:
            logger.warning("[torch_memory] failed to dump %s OOM snapshot: %s", self._device_name.upper(), exc)

    def start(self, **kwargs):
        if not self.enable:
            return
        if not self._should_profile_this_rank():
            return
        profile_step = kwargs.get("profile_step", kwargs.get("global_step"))
        if self._steps_in_window == 0:
            self._window_start_step = profile_step
        self._window_end_step = profile_step
        self.this_step = True

    def stop(self):
        if not self.enable or not self.this_step:
            return
        self.this_step = False
        if not self._should_profile_this_rank():
            return
        self._steps_in_window += 1
        if self._steps_in_window < self.memory_snapshot_num_steps:
            return

        out_dir = self.config.save_path or "outputs/profile"
        tag = "torch_memory"
        # Dump snapshot; all ranks write into the same window directory.
        try:
            self.sampler.dump_memory_snapshot(out_dir=out_dir, tag=tag, sub_dir=self._window_sub_dir())
        except Exception:
            pass
        # Clear memory history
        if TorchMemoryProfiler._memory_history_enabled:
            clear_memory_history(trace_alloc_max_entries=self.trace_alloc_max_entries, stack_depth=self.stack_depth)
        self._steps_in_window = 0
        self._window_start_step = None
        self._window_end_step = None

    def _window_sub_dir(self) -> str | None:
        if self._window_start_step is None:
            return None
        if self._window_start_step == self._window_end_step:
            return f"step{self._window_start_step}"
        return f"steps{self._window_start_step}-{self._window_end_step}"

    def _should_profile_this_rank(self) -> bool:
        if self.config.all_ranks:
            return True
        if self.config.ranks:
            return self.rank in self.config.ranks
        # default rank 0
        return self.rank == 0
