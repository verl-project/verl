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

from verl.checkpoint_engine.base import CheckpointEngineWorker
from verl.workers.engine_workers import ActorRolloutRefWorker


class _FakeTrainerEngine:
    def __init__(self):
        self.weights = [("w", object())]

    def get_per_tensor_param(self):
        return iter(self.weights), None


class _FakeCheckpointEngine:
    def __init__(self):
        self.sent_global_steps = None
        self.received_global_steps = None
        self.sent_weights = None

    async def send_weights(self, weights, global_steps=None):
        self.sent_global_steps = global_steps
        self.sent_weights = list(weights)

    def receive_weights(self, global_steps=None):
        self.received_global_steps = global_steps

        async def _weights():
            yield "w", object()

        return _weights()


class _FakeServerAdapter:
    def __init__(self):
        self.global_steps = None
        self.weights = None

    async def update_weights(self, weights, global_steps=None):
        self.global_steps = global_steps
        self.weights = [item async for item in weights]


def test_actor_worker_passes_global_steps_to_checkpoint_engine_send():
    checkpoint_engine = _FakeCheckpointEngine()
    worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
    worker.config = SimpleNamespace(
        rollout=SimpleNamespace(
            checkpoint_engine=SimpleNamespace(backend="modelexpress"),
        ),
    )
    worker.actor = SimpleNamespace(engine=_FakeTrainerEngine())
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
