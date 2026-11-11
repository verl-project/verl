# Copyright 2026 Individual Contributor
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
from types import SimpleNamespace

import pytest

from verl.checkpoint_engine import base
from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineManager, CheckpointEngineRegistry


class Replica:
    def __init__(self, server, size=1):
        self.server = server
        self.size = size
        self.events = []

    async def abort_all_requests(self):
        self.events.append("pause")

    async def release_kv_cache(self):
        self.events.append("release")

    async def resume_kv_cache(self):
        self.events.append("restore")

    async def resume_generation(self):
        self.events.append("resume")


class Transport(CheckpointEngine):
    direct_rollout = True

    @staticmethod
    def get_rollout_endpoints(replicas):
        if not replicas:
            raise ValueError("no receiver")
        endpoints = []
        offset = 1
        for replica in replicas:
            endpoints.append((replica.server, offset))
            offset += replica.size
        return offset - 1, endpoints

    @classmethod
    def build_topology(cls, actor_wg_world_size, rollout_world_size, metadata):
        assert metadata == [("host", 1234)] + [None] * (actor_wg_world_size - 1)
        address, port = metadata[0]
        return {
            "master_address": [address] * actor_wg_world_size,
            "master_port": [port] * actor_wg_world_size,
            "world_size": [1 + rollout_world_size] * actor_wg_world_size,
        }, {}

    @staticmethod
    async def finish_rollout_update(replicas, step):
        for replica in replicas:
            replica.events.append(("cache_and_step", step))


class Trainers:
    world_size = 4

    def __init__(self):
        self.generations = []
        self.updates = []
        self.gate = concurrent.futures.Future()
        self.gate.set_result(None)
        self.fail = False

    def execute_checkpoint_engine(self, method, **kwargs):
        assert len(method) == self.world_size
        if method[0] == "prepare":
            return [asyncio.sleep(0, result=addr) for addr in [("host", 1234), None, None, None]]
        assert method == ["init_process_group"] * self.world_size
        assert all(len(v) == self.world_size for v in kwargs.values())
        settings = [{k: vals[i] for k, vals in kwargs.items()} for i in range(self.world_size)]
        assert all(s == settings[0] for s in settings)
        assert settings[0]["master_address"] == "host"
        assert settings[0]["master_port"] == 1234
        self.generations.append((settings[0]["generation"], settings[0]["world_size"], settings[0]["endpoints"]))
        return [asyncio.sleep(0) for _ in method]

    def update_weights(self, global_steps, mode):
        self.updates.append(global_steps)

        async def send(rank):
            await asyncio.wrap_future(self.gate)
            if self.fail and rank == 0:
                raise RuntimeError("P3 receive failed")
            return {"installed_step": global_steps} if rank == 0 else {}

        return [asyncio.create_task(send(rank)) for rank in range(self.world_size)]


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setitem(CheckpointEngineRegistry._registry, "direct_test", Transport)
    replicas = [Replica("A", size=2), Replica("B")]
    config = SimpleNamespace(backend="direct_test", custom_backend_module=None)
    manager = CheckpointEngineManager(config, Trainers(), replicas)
    yield manager
    manager.close()


@pytest.mark.asyncio
async def test_direct_updates_use_model_workers_and_reuse_topology(manager):
    for step in (1, 2):
        assert await manager.update_weights(step) == {"installed_step": step}
    assert manager.actor_wg.updates == [1, 2]
    assert manager.actor_wg.generations == [(1, 4, [("A", 1), ("B", 3)])]
    first, second = manager.replicas
    expected = ["pause", "release", ("cache_and_step", 1), "restore", "resume"]
    assert first.events[:5] == second.events[:5] == expected

    manager.remove_replicas([first])
    await manager.update_weights(3)
    assert manager.actor_wg.generations[-1] == (2, 2, [("B", 1)])
    assert ("cache_and_step", 3) not in first.events
    assert second.events[-3:] == [("cache_and_step", 3), "restore", "resume"]


def test_synchronous_trainer_waits_until_direct_commit(manager):
    assert manager.update_weights(1) == {"installed_step": 1}
    manager.wait_for_weight_update()
    assert all(replica.events[-1] == "resume" for replica in manager.replicas)


@pytest.mark.asyncio
async def test_direct_ticket_serializes_updates_and_survives_cancelled_waiter(manager):
    manager.actor_wg.gate = concurrent.futures.Future()
    first = await manager.submit_update(1)
    assert len(first.trainer_published) == manager.actor_wg.world_size
    waiter = asyncio.create_task(first.wait_committed())
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    second_update = asyncio.create_task(manager.update_weights(2))
    await asyncio.sleep(0)
    assert manager.actor_wg.updates == [1]
    manager.actor_wg.gate.set_result(None)
    assert await first.wait_committed() == {"installed_step": 1}
    assert await second_update == {"installed_step": 2}


@pytest.mark.asyncio
async def test_failed_direct_install_leaves_replicas_paused(manager):
    manager.actor_wg.fail = True
    with pytest.raises(RuntimeError, match="P3 receive failed"):
        await manager.update_weights(1)
    assert all(replica.events == ["pause", "release"] for replica in manager.replicas)
    with pytest.raises(RuntimeError, match="P3 receive failed"):
        await manager.wait_for_weight_update()


@pytest.mark.asyncio
async def test_invalid_membership_is_rejected_before_quiescing_or_submitting(manager):
    manager.remove_replicas(list(manager.replicas))
    with pytest.raises(ValueError, match="no receiver"):
        await manager.update_weights(1)
    assert manager.actor_wg.generations == []
    assert manager.actor_wg.updates == []


@pytest.mark.asyncio
async def test_unreachable_trainer_preflight_can_be_retried_before_quiescing(manager, monkeypatch):
    def fail_prepare(method, **kwargs):
        raise RuntimeError("unreachable trainer")

    with monkeypatch.context() as preflight:
        preflight.setattr(manager.actor_wg, "execute_checkpoint_engine", fail_prepare)
        with pytest.raises(RuntimeError, match="unreachable trainer"):
            await manager.update_weights(1)
    assert all(not replica.events for replica in manager.replicas)
    assert await manager.update_weights(1) == {"installed_step": 1}


@pytest.mark.asyncio
async def test_interrupted_ray_dispatch_preserves_failed_ticket(manager, monkeypatch):
    calls = []

    def fail_dispatch(global_steps, mode):
        calls.append(global_steps)
        raise RuntimeError("failed dispatch to trainer 2")

    monkeypatch.setattr(manager.actor_wg, "update_weights", fail_dispatch)
    for step in (1, 2):
        with pytest.raises(RuntimeError, match="failed dispatch"):
            await manager.update_weights(step)
    assert calls == [1]
    assert all(replica.events == ["pause", "release"] for replica in manager.replicas)


@pytest.mark.asyncio
async def test_rollout_worker_still_receives_for_relayed_transports(monkeypatch):
    class Relay:
        def __init__(self, bucket_size):
            assert bucket_size == 1 << 20

        def receive_weights(self, global_steps):
            return iter([global_steps])

    class Adapter:
        async def update_weights(self, weights, global_steps, wire_format):
            assert list(weights) == [global_steps]
            assert wire_format == "named_tensors"
            applied.append(global_steps)

    applied = []
    monkeypatch.setattr(base.Worker, "__init__", lambda self: None)
    monkeypatch.setattr(base, "initialize_global_process_group_ray", lambda **kwargs: None)
    monkeypatch.setitem(CheckpointEngineRegistry._registry, "relay_test", Relay)
    config = SimpleNamespace(
        backend="relay_test", update_weights_bucket_megabytes=1, engine_kwargs={}, custom_backend_module=None
    )
    rollout = SimpleNamespace(name="vllm", checkpoint_engine=config)
    worker = base.CheckpointEngineWorker(rollout, None, server_adapter=Adapter())
    await worker.update_weights(13)
    assert applied == [13]

    config.backend = "vllm_native_hccl"
    placement_worker = base.CheckpointEngineWorker(rollout, None, server_adapter=Adapter())
    assert placement_worker.checkpoint_engine is None
    with pytest.raises(RuntimeError, match="directly"):
        await placement_worker.update_weights(14)
    assert applied == [13]
