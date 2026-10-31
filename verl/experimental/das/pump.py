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
"""Per-engine delta pump: SuffixDataService -> every TP-rank worker of one engine.

Runs on the ``vLLMHttpServer`` actor's event loop (node rank 0 of a replica).
It polls the store, prefetches servable delta batches into a bounded buffer,
and applies them in order with ``collective_rpc("das_apply_deltas")`` so all
TP ranks stay byte-identical. The store withholds the in-progress training
iteration, so mid-step polls return nothing and applies land once per step,
right after the driver advances the iteration (the new rollout's prefill
window). Only the store actor is touched between boundaries; the engine is
never called while there is nothing to apply.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import socket
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from verl.experimental.das.contracts import deserialize_delta_batch
from verl.workers.config.das import DASConfig

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Beyond this, leave data at the store until the next poll rather than growing
# the server process' memory.
DEFAULT_BUFFER_CAP_BYTES = 256 * 1024 * 1024
_SERVICE_RETRY_S = 5.0


class DeltaPump:
    """Polls one store and feeds one engine. See the module docstring.

    Args:
        service_name: Ray actor name of the ``SuffixDataService`` (job-scoped).
        replica_id: identifier the store uses to track this engine's version.
        collective_rpc: ``async (method, args) -> Any`` reaching every TP rank
            (``vLLMHttpServer.collective_rpc`` bound method).
        poll_interval_s: store poll cadence.
        get_actor: ``name -> handle`` lookup, defaults to ``ray.get_actor``;
            injectable for tests.
    """

    def __init__(
        self,
        service_name: str,
        replica_id: str,
        collective_rpc: Callable[..., Awaitable[Any]],
        poll_interval_s: float = 2.0,
        buffer_cap_bytes: int = DEFAULT_BUFFER_CAP_BYTES,
        get_actor: Callable[[str], Any] | None = None,
    ) -> None:
        self.service_name = service_name
        self.replica_id = replica_id
        self.poll_interval_s = poll_interval_s
        self.buffer_cap_bytes = buffer_cap_bytes
        self._collective_rpc = collective_rpc
        self._get_actor = get_actor
        self._service = None
        self._buffer: list[tuple[int, bytes]] = []  # (to_version, payload), in order
        self._buffer_bytes = 0
        self._apply_lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self.version = -1
        self.applied_batches = 0
        self.consecutive_errors = 0

    # ------------------------------------------------------------ lifecycle

    def start(self) -> asyncio.Task:
        """Schedule the poll loop on the running event loop. Idempotent."""
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._run(), name="verl-das-delta-pump")
            logger.info(
                "DAS: delta pump started for replica %s (poll %.1fs, store %r)",
                self.replica_id,
                self.poll_interval_s,
                self.service_name,
            )
        return self._task

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()

    # ------------------------------------------------------------- one poll

    async def run_once(self) -> int:
        """Prefetch everything servable and apply it in order; returns batches applied."""
        if not self._ensure_service():
            return 0
        await self._prefetch_available()
        return await self._apply_buffered()

    def _ensure_service(self) -> bool:
        if self._service is not None:
            return True
        try:
            if self._get_actor is not None:
                self._service = self._get_actor(self.service_name)
            else:
                import ray

                self._service = ray.get_actor(self.service_name)
        except Exception:
            # Store not up yet: engines launch before the trainer creates it.
            return False
        return True

    async def _prefetch_available(self) -> int:
        """Pull delta payloads from the store into the local buffer (store only, never the engine)."""
        fetched = 0
        while self._buffer_bytes < self.buffer_cap_bytes:
            next_version = self._buffer[-1][0] if self._buffer else self.version
            payload = self._service.get_deltas.remote(self.replica_id, next_version)
            if inspect.isawaitable(payload):
                payload = await payload
            if payload is None:
                return fetched
            batch = deserialize_delta_batch(payload)
            self._buffer.append((batch.to_version, payload))
            self._buffer_bytes += len(payload)
            fetched += 1
        return fetched

    async def _apply_buffered(self) -> int:
        applied = 0
        async with self._apply_lock:
            while self._buffer:
                to_version, payload = self._buffer[0]
                result = self._collective_rpc("das_apply_deltas", args=(payload,))
                if inspect.isawaitable(result):
                    await result
                self._buffer.pop(0)
                self._buffer_bytes -= len(payload)
                self.version = to_version
                applied += 1
                self.applied_batches += 1
        if applied:
            logger.info("DAS: applied %d delta batches, engine at store version %d", applied, self.version)
        return applied

    async def _run(self) -> None:
        while True:
            try:
                if not self._ensure_service():
                    await asyncio.sleep(_SERVICE_RETRY_S)
                    continue
                await self.run_once()
                self.consecutive_errors = 0
                await asyncio.sleep(self.poll_interval_s)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.consecutive_errors += 1
                if self.consecutive_errors == 1 or self.consecutive_errors % 30 == 0:
                    logger.warning("DAS: delta pump error (%d consecutive): %s", self.consecutive_errors, e)
                self._service = None  # re-resolve the actor on the next attempt
                await asyncio.sleep(min(30.0, self.poll_interval_s * (2 ** min(self.consecutive_errors, 4))))


def start_delta_pump(server) -> DeltaPump | None:
    """Create and start the pump for a ``vLLMHttpServer`` whose ``config.das.enable`` is set."""
    cfg: DASConfig = server.config.das
    if not cfg.enable:
        return None
    pump = DeltaPump(
        service_name=cfg.service_name,
        replica_id=f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}",
        collective_rpc=server.collective_rpc,
        poll_interval_s=cfg.poll_interval_s,
    )
    pump.start()
    return pump
