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

import asyncio

import pytest

from verl.checkpoint_engine.transport import InMemoryWeightRegistry, WeightManifest
from verl.checkpoint_engine.transport.update_agent import WeightUpdateAgent


class _Method:
    def __init__(self, handle, name):
        self.handle, self.name = handle, name

    async def remote(self, *args, **kwargs):
        self.handle.calls.append((self.handle.label, self.name, kwargs.get("global_steps", args[0] if args else None)))
        return self.handle.results.get(self.name, True)


class _Handle:
    """Stands in for a Ray actor handle and records calls in a shared log."""

    def __init__(self, label, calls, results=None):
        self.label, self.calls, self.results = label, calls, results or {}

    def __getattr__(self, name):
        return _Method(self, name)


def _manifest(version):
    return WeightManifest(version=version, endpoint=b"", total_bytes=0, tensors=[], buckets=[])


def _agent(registry, calls, worker_results=None, **kwargs):
    servers = [_Handle("server0", calls), _Handle("server1", calls)]
    workers = [_Handle("worker0", calls, worker_results), _Handle("worker1", calls, worker_results)]
    return WeightUpdateAgent(servers, workers, registry=registry, poll_interval=0.01, **kwargs)


async def _run_until(agent, version):
    task = asyncio.create_task(agent.run())
    try:
        return await agent.wait_for_version(version, timeout=5)
    finally:
        agent.stop()
        await task


def test_prefetch_happens_before_pause_and_apply_in_order():
    async def main():
        registry, calls = InMemoryWeightRegistry(), []
        agent = _agent(registry, calls)
        await registry.publish(_manifest(3))
        assert await _run_until(agent, 3) == 3

        names = [name for _, name, _ in calls]
        # Prefetch runs while generation continues; everything after abort is the pause.
        assert names == [
            "prefetch_weights",
            "prefetch_weights",
            "abort_all_requests",
            "abort_all_requests",
            "wait_for_requests_to_drain",
            "release_kv_cache",
            "release_kv_cache",
            "update_weights",
            "update_weights",
            "resume_kv_cache",
            "resume_kv_cache",
            "resume_engine_generation",
            "open_submission_gate",
            "open_submission_gate",
        ]
        assert {v for _, name, v in calls if name in ("prefetch_weights", "update_weights")} == {3}
        assert [
            label for label, name, _ in calls if name in ("wait_for_requests_to_drain", "resume_engine_generation")
        ] == [
            "server0",
            "server0",
        ]
        state = agent.get_state()
        assert state["version"] == 3 and not state["failed"] and state["weight_update/version_lag"] == 4

    asyncio.run(main())


def test_min_version_lag_skips_small_steps():
    async def main():
        registry, calls = InMemoryWeightRegistry(), []
        agent = _agent(registry, calls, min_version_lag=2)
        agent.version = 1
        await registry.publish(_manifest(2))
        task = asyncio.create_task(agent.run())
        await asyncio.sleep(0.1)
        assert agent.version == 1 and not calls, "lag 1 is below min_version_lag"

        await registry.publish(_manifest(3))
        assert await agent.wait_for_version(3, timeout=5) == 3
        agent.stop()
        await task

    asyncio.run(main())


def test_failed_prefetch_does_not_pause_generation():
    async def main():
        registry, calls = InMemoryWeightRegistry(), []
        agent = _agent(registry, calls, worker_results={"prefetch_weights": False})
        await registry.publish(_manifest(1))
        task = asyncio.create_task(agent.run())
        await asyncio.sleep(0.1)
        agent.stop()
        await task
        assert agent.version == -1
        assert calls and {name for _, name, _ in calls} == {"prefetch_weights"}

    asyncio.run(main())


def test_wait_for_version_surfaces_agent_failure():
    async def main():
        registry, calls = InMemoryWeightRegistry(), []
        agent = _agent(registry, calls)

        async def broken_update(version):
            raise RuntimeError("engine died")

        agent._update = broken_update
        await registry.publish(_manifest(1))
        task = asyncio.create_task(agent.run())
        with pytest.raises(RuntimeError, match="agent failed"):
            await agent.wait_for_version(1, timeout=5)
        with pytest.raises(RuntimeError, match="engine died"):
            await task
        assert agent.get_state()["failed"]

    asyncio.run(main())
