from __future__ import annotations

import httpx
import pytest
import ray

from tests.agent.support import FakeTokenizer, RecordingLLMClient


@pytest.fixture
def ray_runtime():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.mark.asyncio
async def test_gateway_serving_runtime_owns_gateway_lifecycle_and_session_runtime(ray_runtime):
    from verl.agent.gateway.runtime import GatewayServingRuntime

    llm_client = RecordingLLMClient("OWNER")
    runtime = GatewayServingRuntime(
        llm_client=llm_client,
        gateway_count=1,
        gateway_actor_kwargs={
            "tokenizer": FakeTokenizer(),
        },
    )

    session = await runtime.create_session("session-owner")
    wait_task = runtime.wait_for_completion("session-owner", timeout=2.0)

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{session.base_url}/chat/completions",
            json={"model": "dummy-model", "messages": [{"role": "user", "content": "owner path"}]},
        )
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "OWNER"

        complete = await client.post(
            f"{session.base_url.removesuffix('/v1')}/complete",
            json={"reward_info": {"score": 0.5, "label": "owner"}},
        )
        assert complete.status_code == 200

    await wait_task
    trajectories = await runtime.finalize_session("session-owner")
    await runtime.shutdown()

    assert len(trajectories) == 1
    assert trajectories[0].reward_info == {"score": 0.5, "label": "owner"}


@pytest.mark.asyncio
async def test_gateway_serving_runtime_delegates_generate_to_llm_client(ray_runtime):
    from verl.agent.gateway.runtime import GatewayServingRuntime

    llm_client = RecordingLLMClient("DELEGATED")
    runtime = GatewayServingRuntime(llm_client=llm_client, gateway_count=0)

    output = await runtime.generate(
        "request-direct",
        prompt_ids=[4, 5, 6],
        sampling_params={"temperature": 0.2},
        image_data=["image://direct.png"],
    )

    await runtime.shutdown()

    assert output.token_ids == [ord(char) for char in "DELEGATED"]
    assert llm_client.calls == [
        {
            "request_id": "request-direct",
            "prompt_ids": [4, 5, 6],
            "sampling_params": {"temperature": 0.2},
            "image_data": ["image://direct.png"],
            "video_data": None,
            "kwargs": {},
        }
    ]


@pytest.mark.asyncio
async def test_gateway_serving_runtime_round_robins_actors_across_alive_nodes(ray_runtime, monkeypatch):
    """gateway_count > 1 should distribute actors across alive CPU nodes round-robin."""
    from verl.agent.gateway.runtime import GatewayServingRuntime

    fake_nodes = [
        {"NodeID": "a" * 56, "Alive": True, "Resources": {"CPU": 8.0}},
        {"NodeID": "b" * 56, "Alive": True, "Resources": {"CPU": 8.0}},
        {"NodeID": "c" * 56, "Alive": False, "Resources": {"CPU": 8.0}},
        {"NodeID": "d" * 56, "Alive": True, "Resources": {"GPU": 1.0}},
    ]
    monkeypatch.setattr("verl.agent.gateway.runtime.ray.nodes", lambda: fake_nodes)

    captured_node_ids = []

    class _StubStartHandle:
        @staticmethod
        def remote():
            return ray.put(None)

    class _StubActorHandle:
        start = _StubStartHandle()

    class _RecordingActor:
        @classmethod
        def options(cls, *, scheduling_strategy):
            captured_node_ids.append(scheduling_strategy.node_id)
            return cls

        @classmethod
        def remote(cls, **kwargs):
            return _StubActorHandle()

    monkeypatch.setattr("verl.agent.gateway.gateway.GatewayActor", _RecordingActor)

    runtime = GatewayServingRuntime(
        llm_client=object(),
        gateway_count=5,
        gateway_actor_kwargs={"tokenizer": object()},
    )

    assert captured_node_ids == ["a" * 56, "b" * 56, "a" * 56, "b" * 56, "a" * 56], captured_node_ids
    assert len(runtime.owned_gateway_actors) == 5
    # No shutdown call needed: stub actors have no real Ray state.
