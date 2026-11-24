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

import pytest
import ray

from verl.workers.rollout.llm_server import LLMServerClient, LLMServerManager
from verl.workers.rollout.router import GlobalRequestLoadBalancer


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


def test_factory_binds_positional_constructor_args():
    """The factories mirror the constructor: positional arguments — including
    ``load_balancer_cls`` — must bind identically to keyword calls."""
    ray.init(ignore_reinit_error=True)
    manager_cls = _make_manager_cls()

    class _TaggedBalancer(GlobalRequestLoadBalancer):
        def get_status(self) -> dict:
            status = super().get_status()
            status["tagged"] = True
            return status

    # create_empty(config, worker_group, rollout_resource_pool, start_rank, load_balancer_cls)
    manager = manager_cls.create_empty(_config(), None, None, 0, _TaggedBalancer)
    assert ray.get(manager.get_status.remote())["tagged"] is True
    ray.kill(manager)


def test_bad_router_config_fails_fast_on_driver():
    """A missing router YAML raises at create_empty, on the caller's node —
    before the actor is launched and before any replica initialization runs."""
    ray.init(ignore_reinit_error=True)
    manager_cls = _make_manager_cls()
    config = _config()
    config.actor_rollout_ref.rollout.router_config_path = "/nonexistent/router.yaml"

    with pytest.raises(FileNotFoundError, match="Router config file not found"):
        manager_cls.create_empty(config=config)


def test_release_omits_undeclared_fields_for_legacy_signatures():
    """The embedded plane must honor the strategy's release-field declaration
    like the client path: legacy release_server(self, server_id) signatures
    keep working through the manager's RPC surface."""
    ray.init(ignore_reinit_error=True)
    manager_cls = _make_manager_cls()

    class _LegacyBalancer(GlobalRequestLoadBalancer):
        def release_server(self, server_id):  # main's single-arg signature
            return super().release_server(server_id)

    @ray.remote(num_cpus=0)
    class _StubServer:
        pass

    manager = manager_cls.create_empty(config=_config(), load_balancer_cls=_LegacyBalancer)
    server_actor = _StubServer.remote()
    ray.get(manager.add_servers.remote({"addr-0": server_actor}))
    server_id, _ = ray.get(manager.acquire_server.remote(request_id="req-1"))
    # Before the fix the wrapper unconditionally passed request_id=None and the
    # single-arg signature raised TypeError inside the actor.
    ray.get(manager.release_server.remote(server_id=server_id, request_id="req-1"))
    assert ray.get(manager.get_total_inflight.remote()) == 0
    ray.kill(manager)


def test_release_forwards_request_id_when_declared():
    """Content-aware strategies declaring ``request_id`` still receive it
    through the embedded plane."""
    ray.init(ignore_reinit_error=True)
    manager_cls = _make_manager_cls()

    class _DemandingBalancer(GlobalRequestLoadBalancer):
        def require_release_fields(self) -> list[str]:
            return ["request_id"]

        def release_server(self, server_id, request_id=None):
            if request_id is None:
                raise ValueError("content-aware balancer requires request_id")
            return super().release_server(server_id, request_id)

    @ray.remote(num_cpus=0)
    class _StubServer:
        pass

    manager = manager_cls.create_empty(config=_config(), load_balancer_cls=_DemandingBalancer)
    server_actor = _StubServer.remote()
    ray.get(manager.add_servers.remote({"addr-0": server_actor}))
    server_id, _ = ray.get(manager.acquire_server.remote(request_id="req-1"))
    # A dropped request_id would surface as a RayTaskError(ValueError) here.
    ray.get(manager.release_server.remote(server_id=server_id, request_id="req-1"))
    assert ray.get(manager.get_total_inflight.remote()) == 0
    ray.kill(manager)
