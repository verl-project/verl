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
from types import SimpleNamespace

import pytest

from verl.checkpoint_engine.base import CheckpointEngineWorker
from verl.workers import engine_workers
from verl.workers.engine_workers import ActorRolloutRefWorker


class _FakeTrainerEngine:
    def __init__(self, events=None):
        self.weights = [("w", object())]
        self.events = events

    def get_per_tensor_param(self):
        return iter(self.weights), None

    def finalize_weight_sync(self):
        if self.events is not None:
            self.events.append("finalize_weight_sync")


class _FakeCheckpointEngine:
    def __init__(self, events=None):
        self.sent_global_steps = None
        self.received_global_steps = None
        self.sent_weights = None
        self.events = events

    async def send_weights(self, weights, global_steps=None):
        self.sent_global_steps = global_steps
        self.sent_weights = list(weights)
        if self.events is not None:
            self.events.append("send_weights_complete")

    def receive_weights(self, global_steps=None):
        self.received_global_steps = global_steps

        async def _weights():
            yield "w", object()

        return _weights()


class _FailingCheckpointEngine(_FakeCheckpointEngine):
    async def send_weights(self, weights, global_steps=None):
        list(weights)
        self.events.append("send_weights_failed")
        raise RuntimeError("weight transfer failed")


class _FakeServerAdapter:
    def __init__(self):
        self.global_steps = None
        self.weights = None

    async def update_weights(self, weights, global_steps=None, **kwargs):
        self.global_steps = global_steps
        self.weights = [item async for item in weights]


class _ConfigNode(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def test_actor_worker_passes_global_steps_to_checkpoint_engine_send():
    events = []
    checkpoint_engine = _FakeCheckpointEngine(events)
    worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
    worker.config = SimpleNamespace(
        rollout=SimpleNamespace(
            checkpoint_engine=SimpleNamespace(backend="modelexpress"),
        ),
    )
    worker.actor = SimpleNamespace(engine=_FakeTrainerEngine(events))
    worker.checkpoint_engine = checkpoint_engine

    asyncio.run(
        ActorRolloutRefWorker.update_weights.__wrapped__(
            worker,
            global_steps=17,
            mode="auto",
        )
    )

    assert checkpoint_engine.sent_global_steps == 17
    assert checkpoint_engine.sent_weights == worker.actor.engine.weights
    assert events == ["send_weights_complete", "finalize_weight_sync"]


def test_actor_worker_finalizes_weight_sync_when_send_fails():
    events = []
    worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
    worker.config = SimpleNamespace(
        rollout=SimpleNamespace(
            checkpoint_engine=SimpleNamespace(backend="modelexpress"),
        ),
    )
    worker.actor = SimpleNamespace(engine=_FakeTrainerEngine(events))
    worker.checkpoint_engine = _FailingCheckpointEngine(events)

    with pytest.raises(RuntimeError, match="weight transfer failed"):
        asyncio.run(ActorRolloutRefWorker.update_weights.__wrapped__(worker, mode="auto"))

    assert events == ["send_weights_failed", "finalize_weight_sync"]


def test_naive_lora_sync_offloads_after_base_and_adapter_consumers(monkeypatch):
    events = []
    peft_config = object()

    class _NaiveTrainerEngine:
        is_param_offload_enabled = True

        def get_per_tensor_param(self, *, layered_summon, base_sync_done):
            label = "adapter" if base_sync_done else "base"
            events.append(f"create_{label}")

            def _weights():
                events.append(f"consume_{label}")
                yield label, object()

            return _weights(), peft_config

        def to(self, device, *, model, optimizer, grad):
            assert (device, model, optimizer, grad) == ("cpu", True, False, False)
            events.append("offload")

    class _NaiveRollout:
        async def update_weights(self, weights, *, peft_config, base_sync_done, global_steps):
            assert "offload" not in events
            list(weights)
            label = "adapter" if base_sync_done else "base"
            events.append(f"finish_{label}")
            assert "offload" not in events

    device = SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None)
    monkeypatch.setattr(engine_workers, "set_expandable_segments", lambda _enabled: None)
    monkeypatch.setattr(engine_workers, "get_torch_device", lambda: device)
    monkeypatch.setattr(engine_workers, "log_gpu_memory_usage", lambda *_args, **_kwargs: None)

    worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
    worker.config = _ConfigNode(
        rollout=_ConfigNode(
            checkpoint_engine=_ConfigNode(backend="naive"),
            free_cache_engine=False,
            name="sglang",
        )
    )
    worker.actor = SimpleNamespace(engine=_NaiveTrainerEngine())
    worker.rollout = _NaiveRollout()
    worker.layered_summon = False
    worker.peft_merge = False
    worker.base_sync_done = False

    asyncio.run(ActorRolloutRefWorker.update_weights.__wrapped__(worker, mode="auto"))

    assert events == [
        "create_adapter",
        "create_base",
        "consume_base",
        "finish_base",
        "consume_adapter",
        "finish_adapter",
        "offload",
    ]
    assert worker.base_sync_done is True


def test_checkpoint_worker_passes_global_steps_to_receive_and_rollout_update():
    checkpoint_engine = _FakeCheckpointEngine()
    server_adapter = _FakeServerAdapter()
    worker = CheckpointEngineWorker.__new__(CheckpointEngineWorker)
    worker.checkpoint_engine = checkpoint_engine
    worker.server_adapter = server_adapter

    asyncio.run(
        CheckpointEngineWorker.update_weights.__wrapped__(
            worker,
            global_steps=17,
        )
    )

    assert checkpoint_engine.received_global_steps == 17
    assert server_adapter.global_steps == 17
    assert [name for name, _tensor in server_adapter.weights] == ["w"]
