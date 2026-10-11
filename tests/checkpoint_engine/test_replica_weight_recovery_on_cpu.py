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
from threading import Lock
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch

pytest.importorskip("vllm")

from verl.checkpoint_engine import base as checkpoint_base
from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter


def _make_manager(monkeypatch):
    events = []
    live_weights = torch.tensor([1.0, 2.0])
    wire = SimpleNamespace(weights=None, version=None)
    new_handle = SimpleNamespace(
        check_health=SimpleNamespace(remote=AsyncMock(side_effect=lambda: events.append("health"))),
        snapshot=SimpleNamespace(remote=AsyncMock(side_effect=RuntimeError("metrics logging is disabled"))),
    )
    receiver = checkpoint_base.CheckpointEngineWorker.__new__(checkpoint_base.CheckpointEngineWorker)
    receiver.server_adapter = ServerAdapter.__new__(ServerAdapter)
    receiver.server_adapter.server_handle = object()
    target = SimpleNamespace(
        rollout_mode=RolloutMode.STANDALONE,
        nnodes=1,
        config=SimpleNamespace(name="vllm", disaggregation=SimpleNamespace(enabled=False)),
        model_config=SimpleNamespace(lora={}, lora_rank=0),
        workers=[receiver],
        resource_pool=object(),
        server_handle=receiver.server_adapter.server_handle,
        abort_all_requests=AsyncMock(),
        release_kv_cache=AsyncMock(),
        resume_kv_cache=AsyncMock(),
        resume_generation=AsyncMock(side_effect=lambda: events.append("resume")),
    )

    async def restart(timeout):
        events.append("restart")
        target.server_handle = new_handle

    target.restart = AsyncMock(side_effect=restart)
    healthy = SimpleNamespace(
        workers=[object()],
        restart=AsyncMock(),
        abort_all_requests=AsyncMock(),
        release_kv_cache=AsyncMock(),
        resume_kv_cache=AsyncMock(),
        resume_generation=AsyncMock(),
    )

    def bind(handle):
        checkpoint_base.CheckpointEngineWorker.bind_server_handle.__wrapped__(receiver, handle)
        events.append("bind")
        return []

    def send(*, global_steps, mode):
        wire.weights, wire.version = [("w", live_weights.clone())], global_steps
        events.append("send")
        return [{}]

    def load(*, global_steps):
        assert wire.version == global_steps
        assert receiver.server_adapter.server_handle is new_handle
        receiver.server_adapter.loaded_weights = wire.weights
        events.append("load")
        return [None, None]

    target_group = SimpleNamespace(bind_server_handle=Mock(side_effect=bind))
    all_group = SimpleNamespace(
        world_size=2,
        update_weights=Mock(side_effect=load),
        execute_checkpoint_engine=Mock(return_value=[]),
    )

    def worker_group(*, worker_handles, ray_cls_with_init):
        if worker_handles is target.workers:
            return target_group
        assert worker_handles == target.workers + healthy.workers
        return all_group

    manager = checkpoint_base.CheckpointEngineManager.__new__(checkpoint_base.CheckpointEngineManager)
    manager.backend = "nccl"
    manager.backend_cls = SimpleNamespace(
        wire_format="named_tensors",
        build_topology=Mock(return_value=({"rank": [0]}, {"rank": [1, 2]})),
    )
    manager.actor_wg = SimpleNamespace(
        world_size=1,
        update_weights=Mock(side_effect=send),
        execute_checkpoint_engine=Mock(return_value=[]),
    )
    manager.replicas = [target, healthy]
    manager._weight_sync_lock = Lock()
    manager._pending_restarts = []
    monkeypatch.setattr(checkpoint_base, "RayWorkerGroup", worker_group)
    monkeypatch.setattr(checkpoint_base.ray, "get", lambda values: values)
    return manager, target, healthy, receiver.server_adapter, live_weights, events


def test_restart_rebinds_target_then_normal_sync_loads_current_trainer_weights(monkeypatch):
    manager, target, healthy, adapter, live, events = _make_manager(monkeypatch)
    workers, pool = target.workers, target.resource_pool
    asyncio.run(checkpoint_base.CheckpointEngineManager.restart_replica.__wrapped__(manager, target))
    manager.actor_wg.update_weights.assert_not_called()
    target.abort_all_requests.assert_awaited_once_with(reject_request=True)
    target.resume_generation.assert_not_awaited()
    assert adapter.server_handle is target.server_handle

    live.fill_(17)
    asyncio.run(checkpoint_base.CheckpointEngineManager.update_weights.__wrapped__(manager, global_steps=17))
    assert events == ["restart", "bind", "send", "load", "health", "resume"]
    torch.testing.assert_close(adapter.loaded_weights[0][1], torch.tensor([17.0, 17.0]))
    manager.actor_wg.update_weights.assert_called_once_with(global_steps=17, mode="nccl")
    assert manager._pending_restarts == []
    target.server_handle.snapshot.remote.assert_not_awaited()
    assert target.workers is workers and target.resource_pool is pool
    healthy.restart.assert_not_awaited()


def test_unsupported_restart_has_no_effect_and_failed_health_keeps_target_pending(monkeypatch):
    manager, target, _healthy, _adapter, _live, events = _make_manager(monkeypatch)
    target.rollout_mode = RolloutMode.COLOCATED
    with pytest.raises(NotImplementedError):
        asyncio.run(checkpoint_base.CheckpointEngineManager.restart_replica.__wrapped__(manager, target))
    target.restart.assert_not_awaited()
    assert events == [] and manager._pending_restarts == []

    target.rollout_mode = RolloutMode.STANDALONE
    asyncio.run(checkpoint_base.CheckpointEngineManager.restart_replica.__wrapped__(manager, target))
    failure = RuntimeError("replacement engine is unhealthy")
    target.server_handle.check_health.remote.side_effect = failure
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(checkpoint_base.CheckpointEngineManager.update_weights.__wrapped__(manager, global_steps=17))
    assert caught.value is failure
    target.resume_generation.assert_not_awaited()
    assert manager._pending_restarts == [target]
    assert not manager._weight_sync_lock.locked()
