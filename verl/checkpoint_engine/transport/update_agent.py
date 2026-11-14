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
"""Rollout-side agent that decides on its own when a replica loads a newer weight version."""

import asyncio
import logging
import os
import time
from typing import Any

from .registry import RayWeightRegistry, WeightRegistry

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class WeightUpdateAgent:
    """Keeps one vLLM replica on a recent published weight version, without the trainer.

    Each update has two phases, and only the second pauses generation:

    1. Prefetch: once the newest READY version is at least `min_version_lag` ahead of the loaded
       one, every checkpoint engine worker of the replica pulls it into its local mirror while
       the replica keeps generating.
    2. Apply: abort in-flight requests (clients resubmit them with their partial output, which
       re-prefills under the new weights), release the KV cache, load the mirror into the
       engine, then restore the KV cache and resume. This is the pause/resume sequence that
       CheckpointEngineManager.update_weights runs today, triggered by the replica instead.

    Run it as a Ray actor next to the replica: `ray.remote(WeightUpdateAgent).remote(...)`.

    Args:
        servers: Handles of the replica's vLLMHttpServer actors; servers[0] is node rank 0.
        workers: Handles of the replica's CheckpointEngineWorker actors (backend transport_pull).
        registry_name: Ray actor name of the weight registry.
        min_version_lag: Apply once the newest version is this many versions ahead.
        poll_interval: Seconds between registry polls.
        registry: Registry override, for tests.
    """

    def __init__(
        self,
        servers: list,
        workers: list,
        registry_name: str = "verl_weight_registry",
        min_version_lag: int = 1,
        poll_interval: float = 0.5,
        registry: WeightRegistry | None = None,
    ):
        assert min_version_lag >= 1
        self.servers = servers
        self.workers = workers
        self.registry = registry if registry is not None else RayWeightRegistry(name=registry_name)
        self.min_version_lag = min_version_lag
        self.poll_interval = poll_interval

        self.version = -1
        # Highest version a caller of wait_for_version needs; loaded even below min_version_lag.
        self._demanded = -1
        self._version_changed = asyncio.Condition()
        self._stopped = False
        self._error: BaseException | None = None
        self.metrics: dict[str, float] = {}

    async def run(self):
        """Poll and update until `stop`. Re-raises the error that stopped it, if any."""
        try:
            while not self._stopped:
                manifest = await self.registry.latest_ready()
                if manifest is not None and self._should_update(manifest.version):
                    await self._update(manifest.version)
                else:
                    await asyncio.sleep(self.poll_interval)
        except BaseException as e:
            self._error = e
            logger.exception("Weight update agent stopped")
            async with self._version_changed:
                self._version_changed.notify_all()
            raise

    def _should_update(self, newest: int) -> bool:
        # A replica that has not loaded any published version yet takes the first one: the trainer waits
        # for it before generation starts. A caller waiting for a version also gets it regardless of lag.
        if self.version < 0 or self.version < self._demanded <= newest:
            return True
        return newest - self.version >= self.min_version_lag

    async def _update(self, version: int):
        start = time.perf_counter()
        prefetched = await asyncio.gather(*[w.prefetch_weights.remote(version) for w in self.workers])
        if not all(prefetched):
            # Retired or torn on some worker: a newer version is already out, take that one instead.
            return
        prefetch_done = time.perf_counter()

        # From here on generation is paused. A failure leaves the replica paused rather than
        # serving a mix of old and new weights.
        await asyncio.gather(*[s.abort_all_requests.remote(reset_prefix_cache=True) for s in self.servers])
        await self.servers[0].wait_for_requests_to_drain.remote()
        await asyncio.gather(*[s.release_kv_cache.remote() for s in self.servers])
        await asyncio.gather(*[w.update_weights.remote(global_steps=version) for w in self.workers])
        await asyncio.gather(*[s.resume_kv_cache.remote() for s in self.servers])
        await self.servers[0].resume_engine_generation.remote()
        await asyncio.gather(*[s.open_submission_gate.remote() for s in self.servers])
        done = time.perf_counter()

        self.metrics = {
            "weight_update/version": version,
            "weight_update/version_lag": version - self.version,
            "weight_update/prefetch_seconds": prefetch_done - start,
            "weight_update/pause_seconds": done - prefetch_done,
        }
        logger.info(
            f"Replica updated to version {version}: prefetch {prefetch_done - start:.2f}s, "
            f"paused {done - prefetch_done:.2f}s"
        )
        async with self._version_changed:
            self.version = version
            self._version_changed.notify_all()

    async def wait_for_version(self, version: int, timeout: float | None = None) -> int:
        """Block until the replica has loaded `version` or newer; returns the loaded version.

        The agent loads `version` once it is published even if it is fewer than `min_version_lag` versions
        ahead of the loaded one.
        """
        self._demanded = max(self._demanded, version)

        async def wait():
            async with self._version_changed:
                await self._version_changed.wait_for(lambda: self.version >= version or self._error is not None)
            if self._error is not None:
                raise RuntimeError("Weight update agent failed") from self._error
            return self.version

        return await asyncio.wait_for(wait(), timeout)

    def get_state(self) -> dict[str, Any]:
        return {"version": self.version, "failed": self._error is not None, **self.metrics}

    def stop(self):
        self._stopped = True
