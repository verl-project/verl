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

import asyncio
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from verl.workers.engine_workers import ActorRolloutRefWorker


class _DummyEngine:
    def __init__(self, *, is_param_offload_enabled: bool):
        self.is_param_offload_enabled = is_param_offload_enabled
        self.get_per_tensor_param_calls = []
        self.to_calls = []

    def get_per_tensor_param(self, **kwargs):
        self.get_per_tensor_param_calls.append(kwargs)

        def _weights():
            yield ("model.embed_tokens.weight", torch.tensor([1.0]))

        return _weights(), None

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        self.to_calls.append((device, model, optimizer, grad))


class _DummyRollout:
    def __init__(self):
        self.sleep_level = 0
        self.update_calls = []
        self.resume_calls = []

    async def resume(self, tags):
        self.resume_calls.append(tags)

    async def update_weights(self, weights, **kwargs):
        self.update_calls.append({"weights": list(weights), **kwargs})


def _build_worker(*, is_param_offload_enabled: bool):
    worker = object.__new__(ActorRolloutRefWorker)
    worker.config = OmegaConf.create(
        {
            "rollout": {
                "checkpoint_engine": {"backend": "naive"},
                "free_cache_engine": False,
            }
        }
    )
    worker.actor = SimpleNamespace(engine=_DummyEngine(is_param_offload_enabled=is_param_offload_enabled))
    worker.rollout = _DummyRollout()
    worker.base_sync_done = True
    worker.layered_summon = False
    worker.peft_merge = False
    return worker


def test_update_weights_does_not_offload_actor_when_param_offload_disabled(monkeypatch):
    monkeypatch.setattr("verl.workers.engine_workers.set_expandable_segments", lambda *_: None)
    monkeypatch.setattr("verl.workers.engine_workers.log_gpu_memory_usage", lambda *args, **kwargs: None)
    monkeypatch.setattr("verl.workers.engine_workers.aggressive_empty_cache", lambda *args, **kwargs: None)

    worker = _build_worker(is_param_offload_enabled=False)

    asyncio.run(ActorRolloutRefWorker.update_weights(worker))

    assert worker.actor.engine.to_calls == []
    assert worker.actor.engine.get_per_tensor_param_calls == [{"layered_summon": False, "base_sync_done": True}]
    assert len(worker.rollout.update_calls) == 1
    assert worker.rollout.update_calls[0]["base_sync_done"] is True


def test_update_weights_offloads_actor_when_param_offload_enabled(monkeypatch):
    monkeypatch.setattr("verl.workers.engine_workers.set_expandable_segments", lambda *_: None)
    monkeypatch.setattr("verl.workers.engine_workers.log_gpu_memory_usage", lambda *args, **kwargs: None)
    monkeypatch.setattr("verl.workers.engine_workers.aggressive_empty_cache", lambda *args, **kwargs: None)

    worker = _build_worker(is_param_offload_enabled=True)

    asyncio.run(ActorRolloutRefWorker.update_weights(worker))

    assert worker.actor.engine.to_calls == [("cpu", True, False, False)]
