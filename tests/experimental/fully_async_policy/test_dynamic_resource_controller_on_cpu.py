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

"""CPU-only lifecycle contract tests for ``DynamicResourceController``."""

import asyncio

import verl.experimental.fully_async_policy.dynamic_schedule.dynamic_resource_controller as controller_module
from verl.experimental.fully_async_policy.dynamic_schedule import DynamicResourceController


class _RemoteMethod:
    def __init__(self, function):
        self.function = function

    def remote(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class _RecordingRollouter:
    def __init__(self, events: list[str], replicas: dict[str, object]):
        self.events = events
        self.replicas = replicas
        self.get_all_hybrid_replicas = _RemoteMethod(lambda: self.replicas)
        self.add_replicas = _RemoteMethod(self._add_replicas)
        self.remove_replicas = _RemoteMethod(self._remove_replicas)

    async def _add_replicas(self, resource_ids):
        self.events.append(f"add:{','.join(resource_ids)}")

    async def _remove_replicas(self, resource_ids):
        self.events.append(f"remove:{','.join(resource_ids)}")


class _RecordingCheckpointManager:
    def __init__(self, events: list[str], replicas: list[object] | None = None):
        self.events = events
        self.replicas = replicas or []

    async def abort_replicas(self):
        self.events.append("abort")

    async def update_weights(self, global_steps: int):
        self.events.append(f"update:{global_steps}")

    async def resume_generation_replicas(self):
        self.events.append("resume")

    async def sleep_replicas(self):
        self.events.append("sleep")


def _controller(monkeypatch, *, replicas: dict[str, object] | None = None, registered: bool = True):
    events: list[str] = []
    rollouter = _RecordingRollouter(events, replicas or {})
    checkpoint_manager = _RecordingCheckpointManager(events, replicas=[object()] if registered else [])
    monkeypatch.setattr(controller_module.ray, "get", lambda value: value)
    controller = DynamicResourceController(
        rollouter=rollouter,
        hybrid_checkpoint_manager=checkpoint_manager,
        num_standalone_replicas=1,
        num_hybrid_replicas=len(rollouter.replicas),
    )
    return controller, events


def test_sync_hybrid_weights_skips_without_registered_replicas(monkeypatch):
    controller, events = _controller(monkeypatch, registered=False)

    asyncio.run(controller.sync_hybrid_weights(global_steps=3))

    assert events == []


def test_sync_hybrid_weights_guards_update_with_abort_and_resume(monkeypatch):
    controller, events = _controller(monkeypatch)

    asyncio.run(controller.sync_hybrid_weights(global_steps=3))

    assert events == ["abort", "update:3", "resume"]


def test_activation_registers_before_resuming_and_updates_state(monkeypatch):
    controller, events = _controller(monkeypatch, replicas={"hybrid-0": object(), "hybrid-1": object()})

    asyncio.run(controller.activate_hybrid_replicas(global_steps=4))

    assert events == ["add:hybrid-0,hybrid-1", "resume"]
    assert controller.is_hybrid_active
    assert controller.activate_count == 1


def test_activation_skips_when_rollouter_has_no_hybrid_replicas(monkeypatch):
    controller, events = _controller(monkeypatch)

    asyncio.run(controller.activate_hybrid_replicas(global_steps=4))

    assert events == []
    assert not controller.is_hybrid_active
    assert controller.activate_count == 0


def test_deactivation_removes_before_abort_and_sleep(monkeypatch):
    controller, events = _controller(monkeypatch, replicas={"hybrid-0": object()})
    controller._hybrid_active = True

    asyncio.run(controller.deactivate_hybrid_replicas(global_steps=5))

    assert events == ["remove:hybrid-0", "abort", "sleep"]
    assert not controller.is_hybrid_active
    assert controller.deactivate_count == 1


def test_deactivation_clears_state_when_replicas_disappear(monkeypatch):
    controller, events = _controller(monkeypatch)
    controller._hybrid_active = True

    asyncio.run(controller.deactivate_hybrid_replicas(global_steps=5))

    assert events == []
    assert not controller.is_hybrid_active
    assert controller.deactivate_count == 0
