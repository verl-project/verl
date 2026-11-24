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
"""CPU tests for the fail-closed preparation bound of the fault-tolerant
weight sync (CheckpointEngineManager).

A sync worker whose collective is blocked on a dead peer never returns the
ray tasks the quiesce / group-build steps submit. The preparation bound
converts that silent hang into WeightSyncWedgedError; the retry loop fails
closed on it (no retry) while ordinary failures still retry and a fast
preparation is never fenced."""

import asyncio
import time
from types import SimpleNamespace

import pytest

import verl.checkpoint_engine.base as ckpt_base
from verl.checkpoint_engine.base import CheckpointEngineManager, WeightSyncWedgedError
from verl.workers.config import CheckpointEngineConfig


class _FakeBackend:
    support_publish_weights = False

    @staticmethod
    def build_topology(actor_world_size, rollout_world_size, metadata):
        return {}, {}


def _make_actor_worker_group():
    return SimpleNamespace(
        world_size=1,
        execute_checkpoint_engine=lambda *args, **kwargs: [],
        update_weights=lambda **kwargs: [],
    )


async def _async_noop():
    return None


@pytest.fixture()
def make_manager(monkeypatch):
    monkeypatch.setattr(ckpt_base, "_SYNC_PREPARE_TIMEOUT_S", 0.05)
    managers = []

    def _make(*, fault_tolerant=True):
        backend_cls = _FakeBackend
        monkeypatch.setattr(ckpt_base.CheckpointEngineRegistry, "get", lambda backend: backend_cls)
        manager = CheckpointEngineManager(
            config=CheckpointEngineConfig(), actor_wg=_make_actor_worker_group(), replicas=[]
        )
        if fault_tolerant:
            manager.enable_fault_tolerance(pre_sync_hook=None, sync_succeeded_callback=None)
        managers.append(manager)
        return manager

    yield _make

    for manager in managers:
        manager.close()


def test_wedged_preparation_fails_closed(make_manager, monkeypatch):
    manager = make_manager()

    async def wedged_quiesce():
        await asyncio.sleep(5)

    async def wedged_build(_rollout):
        raise AssertionError("the group build must never run when the quiesce is wedged")

    monkeypatch.setattr(manager, "_quiesce_replicas", wedged_quiesce)
    monkeypatch.setattr(manager, "build_process_group", wedged_build)
    with pytest.raises(WeightSyncWedgedError, match="replica quiesce"):
        asyncio.run(manager._prepare_sync_bounded(rollout=SimpleNamespace()))

    monkeypatch.setattr(manager, "_quiesce_replicas", _async_noop)
    monkeypatch.setattr(manager, "build_process_group", lambda rollout: time.sleep(0.5))
    with pytest.raises(WeightSyncWedgedError, match="group build"):
        asyncio.run(manager._prepare_sync_bounded(rollout=SimpleNamespace()))


def test_prepare_within_the_bound_passes(make_manager, monkeypatch):
    manager = make_manager()
    rollout = SimpleNamespace()
    built = []
    monkeypatch.setattr(manager, "_quiesce_replicas", _async_noop)
    monkeypatch.setattr(manager, "build_process_group", built.append)

    asyncio.run(manager._prepare_sync_bounded(rollout=rollout))

    assert built == [rollout]


def test_wedged_error_is_not_retried(make_manager, monkeypatch):
    manager = make_manager()
    manager.replicas = [object()]
    attempts = []

    async def wedged_pipeline(_global_steps=None):
        attempts.append(1)
        raise WeightSyncWedgedError("weight-sync group build stalled for over 120s: test")

    async def unexpected_detection():
        raise AssertionError("a wedged worker must fail closed without a retry")

    monkeypatch.setattr(manager, "_update_weights_pipeline", wedged_pipeline)
    monkeypatch.setattr(manager, "_wait_for_detection", unexpected_detection)

    with pytest.raises(WeightSyncWedgedError, match="group build"):
        asyncio.run(manager._update_weights_with_retry(global_steps=None))

    assert attempts == [1]


def test_ordinary_failure_still_retries(make_manager, monkeypatch):
    manager = make_manager()
    manager.replicas = [object()]
    attempts = []
    detections = []

    async def flaky_pipeline(_global_steps=None):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("boom")
        return {"ok": True}

    async def fast_detection():
        detections.append(1)

    monkeypatch.setattr(manager, "_update_weights_pipeline", flaky_pipeline)
    monkeypatch.setattr(manager, "_wait_for_detection", fast_detection)

    result = asyncio.run(manager._update_weights_with_retry(global_steps=None))

    assert result == {"ok": True}
    assert len(attempts) == 2
    assert detections == [1]
