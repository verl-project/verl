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
"""CPU smoke tests for the actorized LLMServerManager (zero-CPU Ray actor).

The factories create the manager over RPC; every accessor is exercised as an
actor RPC from both async and sync call sites (the two driver-side patterns
used by the trainers). The manager subclass and its stub replica class are
function-local so cloudpickle ships them by value — a pytest module is not
importable from Ray worker processes, while real subclasses (fully_async)
live in the verl package and stay importable there.
"""

import asyncio
from types import SimpleNamespace

import ray

from verl.workers.rollout.llm_server import LLMServerClient, LLMServerManager


def _make_manager_cls() -> type[LLMServerManager]:
    """Build a manager subclass with a stub replica class (engine imports stay out)."""

    class _StubReplica:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _ManagerForTests(LLMServerManager):
        rollout_replica_class = _StubReplica  # pre-set: skip engine-class resolution

    return _ManagerForTests


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(
                name="vllm",
                nnodes=1,
                disaggregation=SimpleNamespace(enabled=False),
            ),
            model=SimpleNamespace(),
        )
    )


def test_create_empty_actor_rpc_surface():
    """Async driver pattern: factories return an actor handle; accessors are
    RPCs returning values (addresses, registry, client). The manager is also
    the load-balancer host: the embedded balancer's surface is the manager's
    own RPC surface."""
    ray.init(ignore_reinit_error=True)
    manager_cls = _make_manager_cls()

    async def scenario():
        handle = await manager_cls.create_empty(config=_config())
        assert await handle.get_addresses.remote() == []
        assert await handle.get_replicas.remote() == []
        assert await handle.get_server_registry.remote() == {}

        status = await handle.get_status.remote()
        assert status["active_servers"] == 0

        client = await handle.get_client.remote()
        assert isinstance(client, LLMServerClient)
        # The client routes through the manager's own actor handle.
        assert client._load_balancer == handle
        balancer_status = await client._load_balancer.get_status.remote()
        assert balancer_status["active_servers"] == 0

        ray.kill(handle)

    asyncio.run(scenario())


def test_load_balancer_plane_is_embedded_in_the_manager():
    """The merged balancing plane: add/acquire/release go through the
    manager actor's RPC surface and hit the same embedded balancer state."""
    ray.init(ignore_reinit_error=True)
    manager_cls = _make_manager_cls()

    @ray.remote(num_cpus=0)
    class _StubServer:
        pass

    server_actor = _StubServer.remote()
    manager = manager_cls.create_empty(config=_config())
    ray.get(manager.add_servers.remote({"addr-0": server_actor}))
    server_id, server = ray.get(manager.acquire_server.remote(request_id="req-1"))
    assert server_id == "addr-0"
    # ActorHandle identity survives the acquire round-trip.
    assert server == server_actor
    ray.get(manager.release_server.remote(server_id=server_id, request_id="req-1"))
    assert ray.get(manager.get_total_inflight.remote()) == 0
    assert ray.get(manager.get_all_servers.remote()) == ["addr-0"]
    ray.kill(manager)


def test_create_empty_sync_call_site():
    """Sync driver pattern (auto_await factory + ray.get accessors): the exact
    shape used by the sync trainers."""
    ray.init(ignore_reinit_error=True)
    manager_cls = _make_manager_cls()

    manager = manager_cls.create_empty(config=_config())
    assert ray.get(manager.get_addresses.remote()) == []
    assert ray.get(manager.get_replicas.remote()) == []
    client = ray.get(manager.get_client.remote())
    assert isinstance(client, LLMServerClient)
    ray.kill(manager)


def test_client_cls_kwarg_crosses_the_rpc_boundary():
    """A subclass passed as client_cls is serialized across the RPC and
    constructed inside the actor."""

    class _RecordingClient(LLMServerClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.recorded = True

    ray.init(ignore_reinit_error=True)
    manager_cls = _make_manager_cls()

    manager = manager_cls.create_empty(config=_config())
    client = ray.get(manager.get_client.remote(client_cls=_RecordingClient))
    assert isinstance(client, _RecordingClient)
    assert client.recorded
    ray.kill(manager)
