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

from __future__ import annotations

import asyncio
import gc
import os
import threading

import pytest

from verl.single_controller.base.worker import Worker
from verl.single_controller.ray.actor import _runtime_bound_actor_class

_EVENTS: list[tuple[str, str, int]] = []


def test_detached_workers_survive_group_gc(ray_only_runtime):
    import ray

    from tests.runtime.workers import EchoWorker
    from verl.runtime import ClassWithInitArgs
    from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup

    pool = ray_only_runtime.create_resource_pool(nnodes=1, processes_per_node=1, device_type="cpu")
    actor = RayClassWithInitArgs.from_class_init(ClassWithInitArgs(EchoWorker))
    group = RayWorkerGroup(pool, actor, detached=True)
    names = group.worker_names
    try:
        assert group.ping(1) == [1]
        del group
        gc.collect()
        borrowed = RayWorkerGroup.from_detached(worker_names=names, ray_cls_with_init=actor)
        assert borrowed.ping(2) == [2]
    finally:
        for name in names:
            ray.kill(ray.get_actor(name), no_restart=True)


class _SyncOwner(Worker):
    def __init__(self, *, fail_close: bool = False) -> None:
        _EVENTS.append(("sync", "construct", threading.get_ident()))
        super().__init__()
        self.owner_thread = threading.get_ident()
        self.fail_close = fail_close

    def _setup_visible_devices(self) -> None:
        _EVENTS.append(("sync", "setup", threading.get_ident()))

    def call(self) -> int:
        _EVENTS.append(("sync", "call", threading.get_ident()))
        return threading.get_ident()

    def close(self) -> None:
        _EVENTS.append(("sync", "close", threading.get_ident()))
        if self.fail_close:
            raise RuntimeError("sync-close")


class _AsyncOwner(Worker):
    def __init__(self, *, fail_close: bool = False) -> None:
        _EVENTS.append(("async", "construct", threading.get_ident()))
        super().__init__()
        self.owner_thread = threading.get_ident()
        self.fail_close = fail_close

    def _setup_visible_devices(self) -> None:
        _EVENTS.append(("async", "setup", threading.get_ident()))

    def sync_call(self) -> int:
        _EVENTS.append(("async", "sync-call", threading.get_ident()))
        return threading.get_ident()

    async def async_call(self) -> int:
        _EVENTS.append(("async", "async-call", threading.get_ident()))
        return threading.get_ident()

    def close(self) -> None:
        _EVENTS.append(("async", "close", threading.get_ident()))
        if self.fail_close:
            raise RuntimeError("async-close")


class _LiveGPUOwner(Worker):
    def __init__(self) -> None:
        self.constructor_local_rank = os.environ.get("LOCAL_RANK")
        super().__init__()

    def observed_constructor_local_rank(self) -> str | None:
        return self.constructor_local_rank


def _worker_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in Worker._spmd_environment(
        world_size=1,
        rank=0,
        local_world_size=1,
        local_rank=0,
        master_addr="127.0.0.1",
        master_port="29500",
    ).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("REDIS_STORE_SERVER_HOST", "")


def test_ordinary_workers_keep_construction_setup_calls_and_close_on_one_owner(monkeypatch):
    _worker_environment(monkeypatch)
    _EVENTS.clear()
    monkeypatch.setattr(
        "verl.single_controller.ray.actor._assign_ray_visible_devices",
        lambda: _EVENTS.append(("ray", "assign", threading.get_ident())),
    )
    actor_thread = threading.get_ident()

    sync_actor = _runtime_bound_actor_class(_SyncOwner)()
    async_actor = _runtime_bound_actor_class(_AsyncOwner)()
    sync_owner = asyncio.run(sync_actor.call())
    async_sync_owner = asyncio.run(async_actor.sync_call())
    async_async_owner = asyncio.run(async_actor.async_call())

    assert sync_owner != actor_thread
    assert async_sync_owner == async_async_owner == actor_thread
    for role, owner in (("sync", sync_owner), ("async", actor_thread)):
        owner_events = [(event, thread) for event_role, event, thread in _EVENTS if event_role == role]
        assert owner_events[:2] == [("construct", owner), ("setup", owner)]
        assign_index = next(index for index, event in enumerate(_EVENTS) if event == ("ray", "assign", owner))
        construct_index = next(index for index, event in enumerate(_EVENTS) if event == (role, "construct", owner))
        assert assign_index < construct_index

    asyncio.run(sync_actor._shutdown())
    asyncio.run(async_actor._shutdown())
    assert ("sync", "close", sync_owner) in _EVENTS
    assert ("async", "close", actor_thread) in _EVENTS


@pytest.mark.skipif(
    os.getenv("VERL_RUN_RAY_GPU_TESTS") != "1",
    reason="set VERL_RUN_RAY_GPU_TESTS=1 on a GPU host",
)
def test_live_gpu_assignment_precedes_user_constructor():
    import ray
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    from verl.runtime import Runtime

    runtime = Runtime.from_config(
        {
            "backend": "ray",
            "env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"},
            "ray": {"ray_init": {"num_cpus": 16, "num_gpus": 1}},
        }
    )
    try:
        pool = runtime.create_resource_pool(nnodes=1, processes_per_node=1, device_type="gpu")
        worker = runtime.create_worker_group(_LiveGPUOwner, on=pool)
        assert worker.observed_constructor_local_rank() == "0"
    finally:
        runtime.close()
        if ray.is_initialized():
            ray.shutdown()
