import httpx
import pytest
import ray

from tests.experimental.agent_gateway.support import (
    FakeTokenizer,
    QueuedBackend,
    RecordingLoadBalancer,
    RecordingRolloutServer,
)


@pytest.fixture
def ray_runtime():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.mark.asyncio
async def test_async_llm_server_manager_owns_gateway_lifecycle_and_session_runtime(ray_runtime):
    from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager

    manager = AsyncLLMServerManager(
        config=None,
        servers=[],
        load_balancer_handle=None,
        gateway_count=1,
        gateway_actor_kwargs={
            "tokenizer": FakeTokenizer(),
            "backend": QueuedBackend(["OWNER"]),
            "host": "127.0.0.1",
        },
    )

    session = await manager.create_session("session-owner")
    wait_task = manager.wait_for_completion("session-owner", timeout=2.0)

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{session.base_url}/chat/completions",
            json={"model": "dummy-model", "messages": [{"role": "user", "content": "owner path"}]},
        )
        assert response.status_code == 200

        complete = await client.post(
            f"{session.base_url.removesuffix('/v1')}/complete",
            json={"reward_info": {"score": 0.5, "label": "owner"}},
        )
        assert complete.status_code == 200

    await wait_task
    trajectories = await manager.finalize_session("session-owner")
    await manager.shutdown()

    assert len(trajectories) == 1
    assert trajectories[0].reward_info == {"score": 0.5, "label": "owner"}


def test_async_llm_server_manager_gateway_count_zero_disables_session_runtime(ray_runtime):
    from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager

    manager = AsyncLLMServerManager(
        config=None,
        servers=[],
        load_balancer_handle=None,
        gateway_count=0,
    )

    assert manager.gateway_manager is None
    assert manager.owned_gateway_actors == []

    with pytest.raises(RuntimeError, match="gateway_count=0"):
        manager.create_session("disabled")


def test_async_llm_server_manager_keeps_gateway_ownership_local(ray_runtime):
    from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager

    manager = AsyncLLMServerManager(
        config=None,
        servers=[],
        load_balancer_handle=None,
        gateway_count=2,
        gateway_actor_kwargs={
            "tokenizer": FakeTokenizer(),
            "backend": QueuedBackend(["A", "B"]),
            "host": "127.0.0.1",
        },
    )

    assert len(manager.owned_gateway_actors) == 2
    assert manager.gateway_manager is not None
    assert manager.gateway_manager.gateways == manager.owned_gateway_actors
    assert manager.gateway_manager.gateway_count == 2

    manager.shutdown()


@pytest.mark.asyncio
async def test_async_llm_server_manager_injects_manager_owned_gateway_backend(ray_runtime):
    from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager

    rollout_server = RecordingRolloutServer.remote("MANAGED")
    load_balancer = RecordingLoadBalancer.remote("server-0")
    manager = AsyncLLMServerManager(
        config=None,
        servers=[("server-0", rollout_server)],
        load_balancer_handle=load_balancer,
        gateway_count=1,
        gateway_actor_kwargs={
            "tokenizer": FakeTokenizer(),
            "host": "127.0.0.1",
        },
    )

    session = await manager.create_session("session-managed-backend")
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{session.base_url}/chat/completions",
            json={"model": "dummy-model", "messages": [{"role": "user", "content": "managed path"}]},
        )
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "MANAGED"

    trajectories = await manager.finalize_session("session-managed-backend")
    await manager.shutdown()

    stats = ray.get(load_balancer.stats.remote())
    calls = ray.get(rollout_server.get_calls.remote())

    assert len(trajectories) == 1
    assert len(stats["acquire_calls"]) == 1
    assert stats["release_calls"] == ["server-0"]
    assert len(calls) == 1
    assert calls[0]["prompt_ids"]
