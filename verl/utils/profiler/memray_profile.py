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
import os
from pathlib import Path
from typing import Optional

from .config import MemrayToolConfig, ProfilerConfig

logger = logging.getLogger(__name__)


class MemrayProfiler:
    """Record native process allocations across configured profiling windows."""

    def __init__(self, rank: int, config: Optional[ProfilerConfig], tool_config: Optional[MemrayToolConfig] = None):
        self.enable = True
        if not config:
            config = ProfilerConfig(ranks=[])
        self.config = config
        self.rank = rank
        self.memory_snapshot_num_steps = tool_config.memory_snapshot_num_steps if tool_config else 1
        self.this_step = False
        self._window_start_step = None
        self._window_end_step = None
        self._steps_in_window = 0
        self._tracker = None
        self._output_path: Path | None = None
        self._disabled = False

    def start(self, **kwargs) -> None:
        if not self.enable or not self._should_profile_this_rank():
            return
        profile_step = kwargs.get("profile_step", kwargs.get("global_step"))
        if self._steps_in_window == 0:
            self._window_start_step = profile_step
        self._window_end_step = profile_step
        self._start_tracker()
        self.this_step = True

    def stop(self) -> None:
        if not self.enable or not self.this_step:
            return
        self.this_step = False
        if not self._should_profile_this_rank():
            return
        self._steps_in_window += 1
        if self._steps_in_window < self.memory_snapshot_num_steps:
            return

        self._stop_tracker()
        self._steps_in_window = 0
        self._window_start_step = None
        self._window_end_step = None

    def _start_tracker(self) -> None:
        if self._tracker is not None or self._disabled:
            return

        out_dir = Path(self.config.save_path or "outputs/profile")
        # The end step is not known until stop(), so record to a staging directory
        # and move the complete trace into the final window directory afterwards.
        staging_dir = out_dir / ".memray"
        self._output_path = staging_dir / f"memray_rank{self.rank}_pid{os.getpid()}.bin"
        try:
            import memray
        except ImportError:
            logger.warning("[memray] profiler requested but memray is not installed; install verl[memray]")
            self._disabled = True
            return

        try:
            staging_dir.mkdir(parents=True, exist_ok=True)
            self._tracker = memray.Tracker(file_name=str(self._output_path), native_traces=True)
            self._tracker.__enter__()
            logger.info("[memray] recording started: %s", self._output_path)
        except Exception as exc:
            logger.warning("[memray] failed to start recorder: %s", exc)
            self._tracker = None
            self._disabled = True

    def _stop_tracker(self) -> None:
        if self._tracker is None:
            return

        try:
            self._tracker.__exit__(None, None, None)
            assert self._output_path is not None
            final_dir = Path(self.config.save_path or "outputs/profile") / (self._window_sub_dir() or "memray")
            final_dir.mkdir(parents=True, exist_ok=True)
            final_path = final_dir / self._output_path.name
            self._output_path.replace(final_path)
            logger.info("[memray] trace saved: %s", final_path)
        except Exception as exc:
            logger.warning("[memray] failed to stop recorder: %s", exc)
        finally:
            self._tracker = None
            self._output_path = None

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
        return self.rank == 0
