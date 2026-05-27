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
import pytest

from verl.checkpoint_engine import base as checkpoint_base
from verl.checkpoint_engine.base import CheckpointEngineManager


class _FakeReplica:
    def __init__(self, events: list[str]):
        self.events = events
        self.workers = ["worker"]

    async def abort_all_requests(self):
        self.events.append("abort")

    async def release_kv_cache(self):
        self.events.append("release_kv_cache")

    async def resume_kv_cache(self):
        self.events.append("resume_kv_cache")

    async def resume_generation(self):
        self.events.append("resume_generation")


class _FakeWorkerGroup:
    def __init__(
        self,
        events: list[str],
        *,
        name: str,
        fail_update: bool = False,
        fail_finalize: bool = False,
    ):
        self.events = events
        self.name = name
        self.world_size = 1
        self.workers = ["worker"]
        self.fail_update = fail_update
        self.fail_finalize = fail_finalize

    def update_weights(self, **kwargs):
        self.events.append(f"{self.name}.update_weights")
        if self.fail_update:
            return [RuntimeError(f"{self.name} update failed")]
        return [f"{self.name}.update_weights.done"]

    def execute_checkpoint_engine(self, methods):
        assert methods == ["finalize"]
        self.events.append(f"{self.name}.finalize")
        if self.fail_finalize:
            return [RuntimeError(f"{self.name} finalize failed")]
        return [f"{self.name}.finalize.done"]


def _manager(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    *,
    fail_build: bool = False,
    fail_update: bool = False,
    fail_finalize: bool = False,
) -> CheckpointEngineManager:
    trainer = _FakeWorkerGroup(
        events,
        name="trainer",
        fail_update=fail_update,
        fail_finalize=fail_finalize,
    )
    rollout = _FakeWorkerGroup(events, name="rollout")

    def fake_ray_get(refs):
        for ref in refs:
            if isinstance(ref, Exception):
                raise ref
        return refs

    def fake_ray_worker_group(*, worker_handles, ray_cls_with_init):
        assert worker_handles == ["worker"]
        return rollout

    monkeypatch.setattr(checkpoint_base.ray, "get", fake_ray_get)
    monkeypatch.setattr(checkpoint_base, "RayWorkerGroup", fake_ray_worker_group)

    manager = CheckpointEngineManager.__new__(CheckpointEngineManager)
    manager.backend = "nccl"
    manager.trainer = trainer
    manager.replicas = [_FakeReplica(events)]

    def build_process_group(rollout_group):
        assert rollout_group is rollout
        events.append("build_process_group")
        if fail_build:
            raise RuntimeError("build process group failed")

    manager.build_process_group = build_process_group
    return manager


@pytest.mark.asyncio
async def test_update_weights_restores_rollout_state_when_process_group_build_fails(monkeypatch):
    events = []
    manager = _manager(monkeypatch, events, fail_build=True)

    with pytest.raises(RuntimeError, match="build process group failed"):
        await manager.update_weights(global_steps=7)

    assert events == [
        "abort",
        "release_kv_cache",
        "build_process_group",
        "trainer.finalize",
        "rollout.finalize",
        "resume_kv_cache",
        "resume_generation",
    ]


@pytest.mark.asyncio
async def test_update_weights_restores_rollout_state_when_transfer_fails(monkeypatch):
    events = []
    manager = _manager(monkeypatch, events, fail_update=True)

    with pytest.raises(RuntimeError, match="trainer update failed"):
        await manager.update_weights(global_steps=7)

    assert events == [
        "abort",
        "release_kv_cache",
        "build_process_group",
        "trainer.update_weights",
        "rollout.update_weights",
        "trainer.finalize",
        "rollout.finalize",
        "resume_kv_cache",
        "resume_generation",
    ]


@pytest.mark.asyncio
async def test_update_weights_raises_cleanup_error_after_success(monkeypatch):
    events = []
    manager = _manager(monkeypatch, events, fail_finalize=True)

    with pytest.raises(RuntimeError, match="trainer finalize failed"):
        await manager.update_weights(global_steps=7)

    assert events == [
        "abort",
        "release_kv_cache",
        "build_process_group",
        "trainer.update_weights",
        "rollout.update_weights",
        "trainer.finalize",
        "rollout.finalize",
        "resume_kv_cache",
        "resume_generation",
    ]


@pytest.mark.asyncio
async def test_update_weights_preserves_primary_error_when_cleanup_fails(monkeypatch):
    events = []
    manager = _manager(monkeypatch, events, fail_update=True, fail_finalize=True)

    with pytest.raises(RuntimeError, match="trainer update failed"):
        await manager.update_weights(global_steps=7)

    assert "trainer.finalize" in events
    assert "resume_kv_cache" in events
    assert "resume_generation" in events
