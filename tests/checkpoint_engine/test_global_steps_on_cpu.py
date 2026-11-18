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

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry, CheckpointEngineWorker
from verl.workers.engine_workers import ActorRolloutRefWorker


class _FakeTrainerEngine:
    def __init__(self, events=None):
        self.weights = [("w", object())]
        self.events = events
        self.is_param_offload_enabled = True

    def get_per_tensor_param(self):
        return iter(self.weights), None

    def to(self, device, *, model, optimizer, grad):
        assert (device, model, optimizer, grad) == ("cpu", True, False, False)
        if self.events is not None:
            self.events.append("offload")


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


class _FakeServerAdapter:
    def __init__(self):
        self.global_steps = None
        self.weights = None

    async def update_weights(self, weights, global_steps=None, **kwargs):
        self.global_steps = global_steps
        self.weights = [item async for item in weights]


def _actor_worker(backend, checkpoint_engine, events):
    worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
    worker.config = SimpleNamespace(
        rollout=SimpleNamespace(
            checkpoint_engine=SimpleNamespace(backend=backend),
        ),
    )
    worker.actor = SimpleNamespace(engine=_FakeTrainerEngine(events))
    worker.checkpoint_engine = checkpoint_engine
    return worker


def test_actor_worker_passes_global_steps_to_checkpoint_engine_send():
    events = []
    checkpoint_engine = _FakeCheckpointEngine(events)
    worker = _actor_worker("modelexpress", checkpoint_engine, events)

    asyncio.run(
        ActorRolloutRefWorker.update_weights.__wrapped__(
            worker,
            global_steps=17,
            mode="auto",
        )
    )

    assert checkpoint_engine.sent_global_steps == 17
    assert checkpoint_engine.sent_weights == worker.actor.engine.weights
    assert events == ["send_weights_complete", "offload"]


@pytest.mark.parametrize("backend", ["vllm_native_hccl", "model_aware_relay", "delta_sharded"])
def test_actor_worker_passes_model_engine_to_backends_that_request_it(monkeypatch, backend):
    if backend == "vllm_native_hccl":
        checkpoint_engine = CheckpointEngineRegistry.new(backend, bucket_size=128, is_master=True)
        assert isinstance(checkpoint_engine, CheckpointEngine)
    else:
        # Legacy delta plugins predate the flag; other relays can opt in to the same input.
        checkpoint_engine = _FakeCheckpointEngine()
        if backend == "model_aware_relay":
            checkpoint_engine.requires_model_engine = True
    events = []
    worker = _actor_worker(backend, checkpoint_engine, events)

    async def send(model_engine, global_steps):
        assert model_engine is worker.actor.engine
        events.append(("send", global_steps))

    monkeypatch.setattr(checkpoint_engine, "send_weights", send)
    asyncio.run(ActorRolloutRefWorker.update_weights.__wrapped__(worker, global_steps=17, mode="auto"))
    assert events == [("send", 17), "offload"]


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
