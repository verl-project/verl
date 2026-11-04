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
    assert events == ["send_weights_complete", "offload"]


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


class _EngineConsumingCheckpointEngine(_FakeCheckpointEngine):
    consumes_training_engine = True

    async def send_weights(self, weights, global_steps=None):
        self.sent_global_steps = global_steps
        self.sent_weights = weights
        if self.events is not None:
            self.events.append("send_weights_complete")


def test_actor_worker_hands_its_training_engine_to_an_engine_consuming_checkpoint_engine():
    events = []
    checkpoint_engine = _EngineConsumingCheckpointEngine(events)
    worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
    worker.config = SimpleNamespace(rollout=SimpleNamespace(checkpoint_engine=SimpleNamespace(backend="sharded")))
    worker.actor = SimpleNamespace(engine=_FakeTrainerEngine(events))
    worker.checkpoint_engine = checkpoint_engine

    asyncio.run(ActorRolloutRefWorker.update_weights.__wrapped__(worker, global_steps=5, mode="auto"))

    assert checkpoint_engine.sent_weights is worker.actor.engine  # not a generator of full tensors
    assert checkpoint_engine.sent_global_steps == 5
    assert events == ["send_weights_complete", "offload"]


@pytest.mark.parametrize("failure", ["error", "cancel"])
def test_checkpoint_worker_closes_the_receive_stream_when_the_adapter_fails(failure):
    """An engine learns at once that its consumer gave up, even before the first tensor, instead of when the stream is
    garbage-collected (the traceback keeps it alive), so it can tell its peers right away."""
    closed = []

    class _Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def aclose(self):
            closed.append(True)

    class _FailingAdapter:
        async def update_weights(self, weights, global_steps=None, **kwargs):
            if failure == "error":
                raise RuntimeError("adapter failed")
            await asyncio.sleep(30)

    worker = CheckpointEngineWorker.__new__(CheckpointEngineWorker)
    worker.checkpoint_engine = SimpleNamespace(receive_weights=lambda global_steps=None: _Stream())
    worker.server_adapter = _FailingAdapter()

    async def run():
        task = asyncio.ensure_future(CheckpointEngineWorker.update_weights.__wrapped__(worker, global_steps=3))
        if failure == "cancel":
            await asyncio.sleep(0.01)
            task.cancel()
        await task

    with pytest.raises(RuntimeError if failure == "error" else asyncio.CancelledError):
        asyncio.run(run())
    assert closed == [True]
