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
"""CPU tests for the unified replica restart (verl/workers/rollout/replica.py).

A minimal RolloutReplica subclass with real Ray actors for the server and
checkpoint-engine worker roles, and a stub RayWorkerGroup for the respawn
step. The five-segment teardown runs for real against Ray's actor lifecycle;
only the "engine" behind each actor is fake. The weight_source passing case
records what launch_servers received instead of asserting on a real engine.
"""

import asyncio
from types import SimpleNamespace

import psutil
import pytest
import ray

import verl.workers.rollout.replica as replica_module
from verl.workers.rollout.fault_tolerance import ReplicaRestartError, WeightSourceOverride
from verl.workers.rollout.replica import RolloutMode, RolloutReplica


def _fake_rollout_config(tp=1):
    return SimpleNamespace(
        tensor_model_parallel_size=tp,
        data_parallel_size=1,
        pipeline_model_parallel_size=1,
        prefix_cache_namespace=None,
    )


@ray.remote
class _FakeServerActor:
    async def shutdown(self):
        proc = psutil.Process()
        return proc.pid, proc.create_time()

    async def check_health(self):
        return None


@ray.remote
class _FakeWorkerActor:
    pass


class _RespawningWorkerGroup:
    """Stands in for RayWorkerGroup: hands out the next batch of fake checkpoint-engine workers."""

    spawn_calls: list[dict] = []

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        type(self).spawn_calls.append(dict(kwargs))
        self.workers = [_FakeWorkerActor.remote(), _FakeWorkerActor.remote()]


class _FakeReplica(RolloutReplica):
    """Minimal concrete replica: records the relaunch instead of starting engines."""

    def __init__(self, **kwargs):
        super().__init__(config=_fake_rollout_config(), model_config=SimpleNamespace(), **kwargs)
        self.relaunch_calls: list[dict] = []

    async def launch_servers(self, weight_source=None):
        self.relaunch_calls.append({"weight_source": weight_source})
        self.servers = [_FakeServerActor.remote()]
        self._server_handle = self.servers[0]
        self._server_address = f"fake-{self.replica_rank}"


@pytest.fixture()
def replica():
    ray.init(ignore_reinit_error=True)
    instance = _FakeReplica(replica_rank=0, gpus_per_node=2)
    instance.rollout_mode = RolloutMode.STANDALONE
    instance.resource_pool = object()
    instance.workers = [_FakeWorkerActor.remote() for _ in range(2)]
    instance.servers = [_FakeServerActor.remote()]
    return instance


def test_restart_replaces_workers_and_servers(replica, monkeypatch):
    monkeypatch.setattr(replica_module, "RayWorkerGroup", _RespawningWorkerGroup)
    old_workers = list(replica.workers)
    old_server = replica.servers[0]

    asyncio.run(replica.restart())

    assert replica.relaunch_calls == [{"weight_source": None}]
    assert replica.workers != old_workers
    assert replica.servers != [old_server]
    assert replica._restart_in_progress is False
    assert _RespawningWorkerGroup.spawn_calls[-1]["name_prefix"] == "rollout_standalone_0"


def test_restart_passes_weight_source_through_the_second_hop(replica, monkeypatch):
    """Checkpoint cold start: the weight source rides restart -> launch_servers;
    a None source keeps the plain launch call."""
    monkeypatch.setattr(replica_module, "RayWorkerGroup", _RespawningWorkerGroup)
    override = WeightSourceOverride(model_path="/ckpt/global_step_8/actor/huggingface", global_steps=8)

    asyncio.run(replica.restart(weight_source=override))

    assert replica.relaunch_calls == [{"weight_source": override}]


def test_restart_rejects_non_standalone_mode(replica):
    replica.rollout_mode = RolloutMode.HYBRID
    with pytest.raises(NotImplementedError):
        asyncio.run(replica.restart())
    assert replica._restart_in_progress is False


def test_restart_rejects_missing_resource_pool(replica):
    replica.resource_pool = None
    with pytest.raises(ReplicaRestartError):
        asyncio.run(replica.restart())


def test_restart_reentry_guard(replica, monkeypatch):
    monkeypatch.setattr(replica_module, "RayWorkerGroup", _RespawningWorkerGroup)
    replica._restart_in_progress = True
    with pytest.raises(ReplicaRestartError, match="already in progress"):
        asyncio.run(replica.restart())


def test_restart_failure_resets_server_state(replica, monkeypatch):
    monkeypatch.setattr(replica_module, "RayWorkerGroup", _RespawningWorkerGroup)

    async def broken_launch():
        raise RuntimeError("engine failed to start")

    replica.launch_servers = broken_launch
    with pytest.raises(RuntimeError, match="engine failed to start"):
        asyncio.run(replica.restart())
    assert replica.servers == []
    assert replica._server_handle is None
    assert replica._server_address is None
    assert replica._restart_in_progress is False


def test_restart_witness_refuses_when_server_process_survives(replica, monkeypatch):
    """Fail-closed: the shutdown() identity never dies -> no relaunch."""

    @ray.remote
    class _IdentityActor:
        def identity(self):
            proc = psutil.Process()
            return proc.pid, proc.create_time()

    @ray.remote
    class _UndyingServerActor:
        def __init__(self, foreign_identity):
            # Reports a foreign, healthy process (a live sibling actor): the
            # witness probe must observe it alive and refuse to relaunch.
            self._foreign_identity = foreign_identity

        async def shutdown(self):
            return self._foreign_identity

    buddy = _IdentityActor.remote()
    foreign = ray.get(buddy.identity.remote())
    replica.servers = [_UndyingServerActor.remote(foreign)]

    async def must_not_relaunch():
        raise AssertionError("relaunch must not happen after a failed witness")

    replica.launch_servers = must_not_relaunch
    with pytest.raises(ReplicaRestartError, match="witness probe"):
        asyncio.run(replica.restart())


def test_check_health_without_servers_raises(replica):
    replica.servers = []
    with pytest.raises(ReplicaRestartError, match="no live server actor"):
        asyncio.run(replica.check_health())


def test_check_health_probes_head_server(replica):
    asyncio.run(replica.check_health())
