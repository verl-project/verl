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
import concurrent.futures

import pytest
import ray
import torch

from tests.checkpoint_engine.d2h_checkpoint_engine import TestD2HCheckpointEngine
from verl.checkpoint_engine import CheckpointEngineManager
from verl.checkpoint_engine import base as checkpoint_engine_base


class _FakeTicket:
    def __init__(self):
        self.committed_waited = False

    async def wait_published(self):
        return [{"d2h_publish_seconds": 1.0}]

    async def wait_committed(self):
        self.committed_waited = True
        return {"rollout_commit_seconds": 2.0}


class _StrongSyncBackend:
    support_publish_weights = False


@pytest.mark.asyncio
async def test_manager_returns_after_d2h_publish_without_waiting_for_rollout():
    manager = object.__new__(CheckpointEngineManager)
    manager.backend = "test_d2h"
    manager.backend_cls = TestD2HCheckpointEngine
    manager._update_ticket = None
    ticket = _FakeTicket()

    async def submit_update(global_steps):
        assert global_steps == 3
        return ticket

    manager.submit_update = submit_update

    metrics = await manager.update_weights(global_steps=3)

    assert metrics == {"d2h_publish_seconds": 1.0}
    assert not ticket.committed_waited


@pytest.mark.asyncio
async def test_manager_waits_for_strong_sync_commit():
    manager = object.__new__(CheckpointEngineManager)
    manager.backend = "nccl"
    manager.backend_cls = _StrongSyncBackend
    manager._update_ticket = None
    ticket = _FakeTicket()

    async def submit_update(global_steps):
        assert global_steps == 4
        return ticket

    manager.submit_update = submit_update

    metrics = await manager.update_weights(global_steps=4)

    assert metrics == {"rollout_commit_seconds": 2.0}
    assert ticket.committed_waited


@pytest.mark.asyncio
async def test_strong_sync_retry_after_failed_commit():
    """A failed coupled update raises to its caller once; a retry of the same step must not raise it
    again while waiting for the previous update."""

    class FailedTicket(_FakeTicket):
        async def wait_committed(self):
            raise RuntimeError("commit failed")

    manager = object.__new__(CheckpointEngineManager)
    manager.backend = "nccl"
    manager.backend_cls = _StrongSyncBackend
    manager._update_ticket = None
    tickets = [FailedTicket(), _FakeTicket()]

    async def submit_update(global_steps):
        return tickets.pop(0)

    manager.submit_update = submit_update

    with pytest.raises(RuntimeError, match="commit failed"):
        await manager.update_weights(global_steps=5)
    assert await manager.update_weights(global_steps=5) == {"rollout_commit_seconds": 2.0}


@pytest.mark.asyncio
async def test_strong_sync_quiesces_rollout_before_build_and_publish(monkeypatch):
    calls = []

    class FakeRayWorkerGroup:
        def __init__(self, **kwargs):
            calls.append("create_rollout_group")

        def fetch_rollout_layout(self):
            calls.append("fetch_rollout_layout")

    class FakeActorWorkerGroup:
        world_size = 1

        def update_weights(self, global_steps, mode):
            calls.append("publish")
            return ["published"]

    manager = object.__new__(CheckpointEngineManager)
    manager.backend = "nccl"
    manager.backend_cls = _StrongSyncBackend
    manager.actor_wg = FakeActorWorkerGroup()
    manager.replicas = [type("Replica", (), {"workers": ["worker"]})()]
    manager._background_loop = object()

    async def abort_replicas():
        calls.append("abort")

    async def release_kv_cache_replicas():
        calls.append("release_kv_cache")

    def build_process_group(rollout):
        calls.append("build_process_group")

    def run_coroutine_threadsafe(coro, loop):
        calls.append("schedule_commit")
        coro.close()
        return concurrent.futures.Future()

    manager.abort_replicas = abort_replicas
    manager.release_kv_cache_replicas = release_kv_cache_replicas
    manager.build_process_group = build_process_group
    monkeypatch.setattr(checkpoint_engine_base, "RayWorkerGroup", FakeRayWorkerGroup)
    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", run_coroutine_threadsafe)

    ticket = await manager.submit_update(global_steps=4)

    assert ticket.trainer_published == ["published"]
    assert calls == [
        "create_rollout_group",
        "abort",
        "release_kv_cache",
        "build_process_group",
        "fetch_rollout_layout",
        "publish",
        "schedule_commit",
    ]


@pytest.mark.asyncio
async def test_d2h_snapshot_is_published_before_rollout_consumes_it():
    ray.init(num_cpus=2)
    try:
        sender = TestD2HCheckpointEngine(bucket_size=1, is_master=True)
        receiver = TestD2HCheckpointEngine(bucket_size=1)
        metadata = [sender.prepare(), receiver.prepare()]
        actor_kwargs, rollout_kwargs = TestD2HCheckpointEngine.build_topology(1, 1, metadata)
        sender.init_process_group(**{name: values[0] for name, values in actor_kwargs.items()})
        receiver.init_process_group(**{name: values[0] for name, values in rollout_kwargs.items()})

        async def receive():
            return {name: weight async for name, weight in receiver.receive_weights(global_steps=1)}

        receive_task = asyncio.create_task(receive())
        await asyncio.sleep(0)
        assert not receive_task.done()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        source = torch.arange(8, dtype=torch.float32, device=device)
        expected = source.cpu().clone()
        await sender.send_weights(iter([("weight", source)]), global_steps=1)

        # Publishing owns an immutable CPU snapshot; later trainer mutations
        # must not alter what rollout reads.
        source.zero_()
        received = await receive_task
        assert received["weight"].device.type == "cpu"
        torch.testing.assert_close(received["weight"], expected)
    finally:
        ray.shutdown()
