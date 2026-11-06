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
"""Fire-and-forget batched trajectory pushes from agent-loop workers to the service.

``enqueue`` must never block or fail the rollout path: pushes are buffered,
flushed by size or age, sent as un-awaited Ray calls, and dropped (counted)
on overflow.
"""

from __future__ import annotations

import asyncio
import logging
import os

from verl.experimental.das.contracts import TrajectoryPush
from verl.workers.config.das import DASConfig

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DASPushClient:
    """Buffers ``TrajectoryPush`` records and ships them to ``SuffixDataService.add_trajectories``."""

    def __init__(self, service_handle, worker_id: str, config: DASConfig | None = None) -> None:
        self._service = service_handle
        self._worker_id = worker_id
        self._cfg = config or DASConfig()
        self._buffer: list[TrajectoryPush] = []
        self._flush_task: asyncio.Task | None = None
        self.pushed = 0
        self.dropped = 0
        self.flush_errors = 0

    def enqueue(self, push: TrajectoryPush) -> None:
        if len(self._buffer) >= self._cfg.push_queue_capacity:
            self.dropped += 1
            return
        self._buffer.append(push)
        if len(self._buffer) >= self._cfg.push_batch_size:
            self.flush()
        else:
            self._ensure_flusher()

    def flush(self) -> None:
        if not self._buffer:
            return
        batch, self._buffer = self._buffer, []
        try:
            # Un-awaited on purpose: the ObjectRef is discarded.
            self._service.add_trajectories.remote(self._worker_id, batch)
            self.pushed += len(batch)
        except Exception as e:
            self.flush_errors += 1
            if self.flush_errors == 1 or self.flush_errors % 100 == 0:
                logger.warning("DAS push flush failed (%d so far): %s", self.flush_errors, e)

    def _ensure_flusher(self) -> None:
        if self._flush_task is not None and not self._flush_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.flush()
            return
        self._flush_task = loop.create_task(self._delayed_flush())

    async def _delayed_flush(self) -> None:
        await asyncio.sleep(self._cfg.push_flush_interval_s)
        self.flush()
