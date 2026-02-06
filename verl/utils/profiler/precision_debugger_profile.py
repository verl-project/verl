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

import os
import threading
from dataclasses import asdict
from typing import Optional

from verl.utils.import_utils import is_msprobe_available
from verl.utils.profiler.config import PrecisionDebuggerToolConfig


_GLOBAL_LOCK = threading.Lock()


class PrecisionDebuggerProfiler:
    """Precision debugger wrapper for msprobe.

    This class implements a minimal start/stop contract and is intentionally
    not a DistProfiler subclass to keep the dependency one-way.
    """

    def __init__(self, precision_cfg, rank: Optional[int] = None):
        self.rank = rank
        self.precision_cfg = self._normalize_config(precision_cfg)
        self._active_lock: Optional[threading.Lock] = None
        self._enabled = self._is_enabled(self.precision_cfg)
        self._available = is_msprobe_available()
        self._debugger = None

    @staticmethod
    def _normalize_config(precision_cfg) -> PrecisionDebuggerToolConfig:
        if precision_cfg is None:
            return PrecisionDebuggerToolConfig()
        if isinstance(precision_cfg, PrecisionDebuggerToolConfig):
            return precision_cfg
        if hasattr(precision_cfg, "to_container"):
            precision_cfg = precision_cfg.to_container(resolve=True)
        if isinstance(precision_cfg, dict):
            return PrecisionDebuggerToolConfig(**precision_cfg)
        return PrecisionDebuggerToolConfig(**asdict(precision_cfg))

    @staticmethod
    def _is_enabled(precision_cfg: PrecisionDebuggerToolConfig) -> bool:
        return bool(precision_cfg.enable)

    def _should_collect(self, stage: str, global_step: Optional[int]) -> bool:
        if not self._enabled:
            return False
        if self.precision_cfg.stages is not None and stage not in set(self.precision_cfg.stages):
            return False
        if self.precision_cfg.steps is not None and global_step is not None:
            if int(global_step) not in set(self.precision_cfg.steps):
                return False
        return True

    def _get_lock(self) -> threading.Lock:
        return _GLOBAL_LOCK

    def start(self, stage: str, global_step: Optional[int] = None, model=None) -> bool:
        if not self._should_collect(stage=stage, global_step=global_step):
            return False
        if not self._available:
            if self.precision_cfg.strict:
                raise ImportError("msprobe is not available but precision_debugger.strict is True")
            return False

        config_path = self.precision_cfg.config_path
        data_dir = self.precision_cfg.data_dir
        if not config_path or not data_dir:
            return False

        step_tag = f"step_{global_step}" if global_step is not None else "step_unknown"
        rank_tag = f"rank_{self.rank}" if self.rank is not None else "rank_unknown"
        dump_path = os.path.join(data_dir, step_tag, stage, rank_tag)
        os.makedirs(dump_path, exist_ok=True)

        lock = self._get_lock()
        lock.acquire()
        self._active_lock = lock
        try:
            from msprobe.pytorch import PrecisionDebugger

            debugger = None
            if self._debugger is None:
                debugger = PrecisionDebugger(config_path=config_path, dump_path=dump_path)
                if debugger is None:
                    if self.precision_cfg.strict:
                        raise RuntimeError("Failed to create PrecisionDebugger instance")
                    return False
                self._debugger = debugger
            else:
                debugger = self._debugger
            if hasattr(debugger, "service") and hasattr(debugger.service, "config"):
                debugger.service.config.dump_path = dump_path
            debugger.start(model)
            return True
        except Exception:
            self._release_lock()
            if self.precision_cfg.strict:
                raise
            return False

    def stop(self, started: bool = False, step: bool = False) -> None:
        if not started:
            self._release_lock()
            return
        if not self._available:
            self._release_lock()
            return
        try:
            debugger = self._debugger
            if debugger is None:
                return
            debugger.stop()
            if step:
                if hasattr(debugger, "step"):
                    debugger.step()
        finally:
            self._release_lock()

    def _release_lock(self) -> None:
        lock = self._active_lock
        self._active_lock = None
        if lock is not None:
            lock.release()
