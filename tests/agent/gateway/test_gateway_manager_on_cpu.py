import httpx
import pytest
import ray

from tests.agent.support import FakeTokenizer, QueuedBackend, TrackingGatewayActor


@pytest.fixture
def ray_runtime():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.mark.asyncio
async def test_gateway_manager_routes_sessions_stickily(ray_runtime):
    from verl.agent.gateway.gateway import GatewayActor
    from verl.agent.gateway.manager import GatewayManager

    gateways = [
        GatewayActor.remote(tokenizer=FakeTokenizer(), backend=QueuedBackend(["A"]), host="127.0.0.1"),
        GatewayActor.remote(tokenizer=FakeTokenizer(), backend=QueuedBackend(["B"]), host="127.0.0.1"),
    ]
    ray.get([gateway.start.remote() for gateway in gateways])

    manager = GatewayManager(gateways)
    session_a = await manager.create_session("session-a")
    session_b = await manager.create_session("session-b")

    assert manager.gateway_count == 2
    assert session_a.base_url != session_b.base_url

    async with httpx.AsyncClient(timeout=5.0) as client:
        first = await client.post(
            f"{session_a.base_url}/chat/completions",
            json={"model": "dummy-model", "messages": [{"role": "user", "content": "route a"}]},
        )
        second = await client.post(
            f"{session_b.base_url}/chat/completions",
            json={"model": "dummy-model", "messages": [{"role": "user", "content": "route b"}]},
        )
        assert first.status_code == 200
        assert second.status_code == 200

    trajectories_a = await manager.finalize_session("session-a")
    trajectories_b = await manager.finalize_session("session-b")

    assert len(trajectories_a) == 1
    assert len(trajectories_b) == 1

    ray.get([gateway.shutdown.remote() for gateway in gateways])


@pytest.mark.asyncio
async def test_gateway_manager_uses_least_active_sessions_routing(ray_runtime):
    from verl.agent.gateway.manager import GatewayManager

    gateways = [
        TrackingGatewayActor.remote("gw-0"),
        TrackingGatewayActor.remote("gw-1"),
    ]
    ray.get([gateway.start.remote() for gateway in gateways])

    manager = GatewayManager(gateways)
    session_a = await manager.create_session("session-a")
    session_b = await manager.create_session("session-b")
    session_c = await manager.create_session("session-c")

    assert manager.active_sessions_per_gateway == [2, 1]
    assert session_a.base_url.startswith("http://gw-0/")
    assert session_b.base_url.startswith("http://gw-1/")
    assert session_c.base_url.startswith("http://gw-0/")

    await manager.finalize_session("session-a")
    assert manager.active_sessions_per_gateway == [1, 1]

    session_d = await manager.create_session("session-d")
    assert session_d.base_url.startswith("http://gw-0/")
    assert manager.active_sessions_per_gateway == [2, 1]

    ray.get([gateway.shutdown.remote() for gateway in gateways])


@pytest.mark.asyncio
async def test_gateway_manager_wait_for_completion_delegates_to_session_owner(ray_runtime):
    from verl.agent.gateway.manager import GatewayManager

    gateways = [
        TrackingGatewayActor.remote("gw-0"),
        TrackingGatewayActor.remote("gw-1"),
    ]
    ray.get([gateway.start.remote() for gateway in gateways])

    manager = GatewayManager(gateways)
    await manager.create_session("session-a")
    await manager.create_session("session-b")

    await manager.wait_for_completion("session-a", timeout=1.5)

    stats_0 = ray.get(gateways[0].stats.remote())
    stats_1 = ray.get(gateways[1].stats.remote())

    assert stats_0["waited"] == [("session-a", 1.5)]
    assert stats_1["waited"] == []

    ray.get([gateway.shutdown.remote() for gateway in gateways])
