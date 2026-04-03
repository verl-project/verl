import asyncio

import httpx
import pytest
import ray

from tests.experimental.agent_gateway.support import (
    FailingBackend,
    FakeTokenizer,
    QueuedBackend,
    RejectConcurrentSessionBackend,
    RejectToolsSamplingParamsBackend,
    SequencedBackend,
)


@pytest.fixture
def ray_runtime():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_normalize_request_context_preserves_structured_fields():
    from verl.experimental.agent_gateway.gateway import _normalize_request_context

    context = _normalize_request_context(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {"type": "image_url", "image_url": {"url": "file://image.png"}},
                    ],
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "search", "arguments": "{\"query\": \"weather\"}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": [{"type": "text", "text": "sunny"}],
                },
            ],
            "tools": [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}],
        }
    )

    assert context["tools"][0]["function"]["name"] == "search"
    assert context["messages"][0]["content"][1]["type"] == "image_url"
    assert context["messages"][1]["tool_calls"][0]["id"] == "call-1"
    assert context["messages"][2]["tool_call_id"] == "call-1"


def test_normalize_request_context_rejects_unsupported_name_field():
    from verl.experimental.agent_gateway.gateway import _normalize_request_context

    with pytest.raises(ValueError, match="name"):
        _normalize_request_context(
            {
                "messages": [
                    {
                        "role": "user",
                        "name": "alice",
                        "content": "hello",
                    }
                ]
            }
        )


def test_normalize_request_context_canonicalizes_tool_argument_strings():
    from verl.experimental.agent_gateway.gateway import _normalize_request_context

    context = _normalize_request_context(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": '{"b": 2, "a": 1}',
                            },
                        }
                    ],
                }
            ]
        }
    )

    assert context["messages"][0]["tool_calls"][0]["function"]["arguments"] == {"a": 1, "b": 2}


def test_normalize_request_context_rejects_non_object_tool_arguments():
    from verl.experimental.agent_gateway.gateway import _normalize_request_context

    with pytest.raises(ValueError, match="JSON object"):
        _normalize_request_context(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '["bad"]',
                                },
                            }
                        ],
                    }
                ]
            }
        )


@pytest.mark.asyncio
async def test_gateway_actor_complete_wait_and_finalize(ray_runtime):
    from verl.experimental.agent_gateway.gateway import GatewayActor

    actor = GatewayActor.remote(tokenizer=FakeTokenizer(), backend=QueuedBackend(["ANSWER: A"]), host="127.0.0.1")
    ray.get(actor.start.remote())

    session = ray.get(actor.create_session.remote("session-0"))
    wait_ref = actor.wait_for_completion.remote("session-0", timeout=2.0)

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "messages": [{"role": "user", "content": "Pick label A"}],
            },
        )
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "ANSWER: A"

        complete = await client.post(
            f"{session.base_url.removesuffix('/v1')}/complete",
            json={"reward_info": {"score": 1.0, "label": "A"}},
        )
        assert complete.status_code == 200

    ray.get(wait_ref)
    trajectories = ray.get(actor.finalize_session.remote("session-0"))
    ray.get(actor.shutdown.remote())

    assert len(trajectories) == 1
    assert trajectories[0].reward_info == {"score": 1.0, "label": "A"}
    assert trajectories[0].trajectory_id == 0
    assert trajectories[0].response_ids
    assert all(mask == 1 for mask in trajectories[0].response_mask)


@pytest.mark.asyncio
async def test_gateway_actor_prefix_mismatch_splits_trajectories(ray_runtime):
    from verl.experimental.agent_gateway.gateway import GatewayActor

    actor = GatewayActor.remote(tokenizer=FakeTokenizer(), backend=QueuedBackend(["FIRST", "SECOND"]), host="127.0.0.1")
    ray.get(actor.start.remote())

    session = ray.get(actor.create_session.remote("session-1"))

    async with httpx.AsyncClient(timeout=5.0) as client:
        first = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "messages": [{"role": "user", "content": "first turn"}],
            },
        )
        assert first.status_code == 200

        second = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "messages": [{"role": "user", "content": "replacement context"}],
            },
        )
        assert second.status_code == 200

    trajectories = ray.get(actor.finalize_session.remote("session-1"))
    ray.get(actor.shutdown.remote())

    assert len(trajectories) == 2
    assert [trajectory.trajectory_id for trajectory in trajectories] == [0, 1]
    assert trajectories[0].prompt_ids != trajectories[1].prompt_ids


@pytest.mark.asyncio
async def test_gateway_actor_tool_context_change_splits_trajectory(ray_runtime):
    from verl.experimental.agent_gateway.gateway import GatewayActor

    actor = GatewayActor.remote(tokenizer=FakeTokenizer(), backend=QueuedBackend(["FIRST", "SECOND"]), host="127.0.0.1")
    ray.get(actor.start.remote())

    session = ray.get(actor.create_session.remote("session-tools"))

    async with httpx.AsyncClient(timeout=5.0) as client:
        first = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "tools": [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}],
                "messages": [{"role": "user", "content": "first turn"}],
            },
        )
        assert first.status_code == 200

        second = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
                "messages": [
                    {"role": "user", "content": "first turn"},
                    {"role": "assistant", "content": "FIRST"},
                    {"role": "user", "content": "follow up"},
                ],
            },
        )
        assert second.status_code == 200

    trajectories = ray.get(actor.finalize_session.remote("session-tools"))
    ray.get(actor.shutdown.remote())

    assert len(trajectories) == 2


@pytest.mark.asyncio
async def test_gateway_actor_does_not_forward_tools_in_sampling_params(ray_runtime):
    from verl.experimental.agent_gateway.gateway import GatewayActor

    actor = GatewayActor.remote(
        tokenizer=FakeTokenizer(),
        backend=RejectToolsSamplingParamsBackend("SAFE"),
        host="127.0.0.1",
    )
    ray.get(actor.start.remote())
    session = ray.get(actor.create_session.remote("session-no-tools-sampling"))

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "tools": [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}],
                "messages": [{"role": "user", "content": "first turn"}],
            },
        )

    ray.get(actor.shutdown.remote())

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_gateway_actor_continuation_preserves_prompt_and_generation_masks(ray_runtime):
    from verl.experimental.agent_gateway.gateway import GatewayActor

    actor = GatewayActor.remote(tokenizer=FakeTokenizer(), backend=QueuedBackend(["FIRST", "SECOND"]), host="127.0.0.1")
    ray.get(actor.start.remote())
    session = ray.get(actor.create_session.remote("session-continuation-mask"))

    async with httpx.AsyncClient(timeout=5.0) as client:
        first = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "messages": [
                    {
                        "role": "user",
                        "content": "first turn",
                    }
                ],
            },
        )
        assert first.status_code == 200

        second = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "messages": [
                    {"role": "user", "content": "first turn"},
                    {"role": "assistant", "content": "FIRST"},
                    {"role": "user", "content": "follow up"},
                ],
            },
        )
        assert second.status_code == 200

    trajectories = ray.get(actor.finalize_session.remote("session-continuation-mask"))
    ray.get(actor.shutdown.remote())

    assert len(trajectories) == 1
    assert 0 in trajectories[0].response_mask
    assert trajectories[0].response_mask[-len("SECOND") :] == [1] * len("SECOND")


def test_gateway_actor_abort_discards_session(ray_runtime):
    from verl.experimental.agent_gateway.gateway import GatewayActor

    actor = GatewayActor.remote(tokenizer=FakeTokenizer(), backend=QueuedBackend(["unused"]), host="127.0.0.1")
    ray.get(actor.start.remote())
    ray.get(actor.create_session.remote("session-abort"))

    ray.get(actor.abort_session.remote("session-abort"))

    with pytest.raises(ray.exceptions.RayTaskError, match="session-abort"):
        ray.get(actor.finalize_session.remote("session-abort"))

    ray.get(actor.shutdown.remote())


def test_gateway_actor_rejects_duplicate_session_creation(ray_runtime):
    from verl.experimental.agent_gateway.gateway import GatewayActor

    actor = GatewayActor.remote(tokenizer=FakeTokenizer(), backend=QueuedBackend(["unused"]), host="127.0.0.1")
    ray.get(actor.start.remote())
    ray.get(actor.create_session.remote("session-dup"))

    with pytest.raises(ray.exceptions.RayTaskError, match="session-dup"):
        ray.get(actor.create_session.remote("session-dup"))

    ray.get(actor.shutdown.remote())


@pytest.mark.asyncio
async def test_gateway_actor_serializes_same_session_concurrent_requests(ray_runtime):
    from verl.experimental.agent_gateway.gateway import GatewayActor

    actor = GatewayActor.remote(
        tokenizer=FakeTokenizer(),
        backend=RejectConcurrentSessionBackend(["FIRST", "SECOND"]),
        host="127.0.0.1",
    )
    ray.get(actor.start.remote())
    session = ray.get(actor.create_session.remote("session-concurrent"))

    async with httpx.AsyncClient(timeout=5.0) as client:
        async def send_request():
            return await client.post(
                f"{session.base_url}/chat/completions",
                json={
                    "model": "dummy-model",
                    "messages": [{"role": "user", "content": "same session prompt"}],
                },
            )

        first, second = await asyncio.gather(send_request(), send_request())

    trajectories = ray.get(actor.finalize_session.remote("session-concurrent"))
    ray.get(actor.shutdown.remote())

    assert first.status_code == 200
    assert second.status_code == 200
    assert len(trajectories) == 2
    assert trajectories[0].response_ids == [ord(char) for char in "FIRST"]
    assert trajectories[1].response_ids == [ord(char) for char in "SECOND"]
    assert trajectories[0].response_mask == [1] * len("FIRST")
    assert trajectories[1].response_mask == [1] * len("SECOND")


@pytest.mark.asyncio
async def test_gateway_actor_wait_for_completion_times_out_for_active_session(ray_runtime):
    from verl.experimental.agent_gateway.gateway import GatewayActor

    actor = GatewayActor.remote(tokenizer=FakeTokenizer(), backend=QueuedBackend(["DONE"]), host="127.0.0.1")
    ray.get(actor.start.remote())
    ray.get(actor.create_session.remote("session-timeout"))

    with pytest.raises(ray.exceptions.RayTaskError):
        ray.get(actor.wait_for_completion.remote("session-timeout", timeout=0.01))

    ray.get(actor.shutdown.remote())


def test_gateway_actor_wait_for_completion_fails_after_abort(ray_runtime):
    from verl.experimental.agent_gateway.gateway import GatewayActor

    actor = GatewayActor.remote(tokenizer=FakeTokenizer(), backend=QueuedBackend(["DONE"]), host="127.0.0.1")
    ray.get(actor.start.remote())
    ray.get(actor.create_session.remote("session-aborted-wait"))
    ray.get(actor.abort_session.remote("session-aborted-wait"))

    with pytest.raises(ray.exceptions.RayTaskError, match="aborted"):
        ray.get(actor.wait_for_completion.remote("session-aborted-wait", timeout=0.1))

    ray.get(actor.shutdown.remote())


@pytest.mark.asyncio
async def test_gateway_actor_rejects_chat_after_complete(ray_runtime):
    from verl.experimental.agent_gateway.gateway import GatewayActor

    actor = GatewayActor.remote(tokenizer=FakeTokenizer(), backend=QueuedBackend(["DONE"]), host="127.0.0.1")
    ray.get(actor.start.remote())
    session = ray.get(actor.create_session.remote("session-completed-chat"))
    ray.get(actor.complete_session.remote("session-completed-chat"))

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{session.base_url}/chat/completions",
            json={"model": "dummy-model", "messages": [{"role": "user", "content": "after complete"}]},
        )

    ray.get(actor.shutdown.remote())

    assert response.status_code == 409


@pytest.mark.asyncio
async def test_gateway_actor_distinguishes_malformed_and_unsupported_requests(ray_runtime):
    from verl.experimental.agent_gateway.gateway import GatewayActor

    actor = GatewayActor.remote(tokenizer=FakeTokenizer(), backend=QueuedBackend(["DONE"]), host="127.0.0.1")
    ray.get(actor.start.remote())
    session = ray.get(actor.create_session.remote("session-validation"))

    async with httpx.AsyncClient(timeout=5.0) as client:
        malformed = await client.post(
            f"{session.base_url}/chat/completions",
            json={"model": "dummy-model", "messages": []},
        )
        unsupported = await client.post(
            f"{session.base_url}/chat/completions",
            json={"model": "dummy-model", "messages": [{"role": "user", "name": "alice", "content": "hello"}]},
        )

    ray.get(actor.shutdown.remote())

    assert malformed.status_code == 400
    assert unsupported.status_code == 422


@pytest.mark.asyncio
async def test_gateway_actor_backend_failure_does_not_commit_partial_state(ray_runtime):
    from verl.experimental.agent_gateway.gateway import GatewayActor

    actor = GatewayActor.remote(tokenizer=FakeTokenizer(), backend=FailingBackend("boom"), host="127.0.0.1")
    ray.get(actor.start.remote())
    session = ray.get(actor.create_session.remote("session-backend-failure"))

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{session.base_url}/chat/completions",
            json={"model": "dummy-model", "messages": [{"role": "user", "content": "first turn"}]},
        )

    state = ray.get(actor.get_session_state.remote("session-backend-failure"))
    ray.get(actor.shutdown.remote())

    assert response.status_code == 500
    assert state["num_trajectories"] == 0
    assert state["has_active_trajectory"] is False


@pytest.mark.asyncio
async def test_gateway_actor_backend_failure_after_tool_mismatch_does_not_split(ray_runtime):
    from verl.experimental.agent_gateway.gateway import GatewayActor

    actor = GatewayActor.remote(
        tokenizer=FakeTokenizer(),
        backend=SequencedBackend(["FIRST", RuntimeError("boom")]),
        host="127.0.0.1",
    )
    ray.get(actor.start.remote())
    session = ray.get(actor.create_session.remote("session-failure-mismatch"))

    async with httpx.AsyncClient(timeout=5.0) as client:
        first = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "tools": [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}],
                "messages": [{"role": "user", "content": "first turn"}],
            },
        )
        assert first.status_code == 200

    async with httpx.AsyncClient(timeout=5.0) as client:
        second = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
                "messages": [
                    {"role": "user", "content": "first turn"},
                    {"role": "assistant", "content": "FIRST"},
                    {"role": "user", "content": "follow up"},
                ],
            },
        )
        assert second.status_code == 500

    state = ray.get(actor.get_session_state.remote("session-failure-mismatch"))
    trajectories = ray.get(actor.finalize_session.remote("session-failure-mismatch"))
    ray.get(actor.shutdown.remote())

    assert state["num_trajectories"] == 0
    assert len(trajectories) == 1
    assert trajectories[0].response_ids == [ord(char) for char in "FIRST"]


@pytest.mark.asyncio
async def test_gateway_actor_complete_does_not_materialize_trajectory(ray_runtime):
    from verl.experimental.agent_gateway.gateway import GatewayActor

    actor = GatewayActor.remote(tokenizer=FakeTokenizer(), backend=QueuedBackend(["DONE"]), host="127.0.0.1")
    ray.get(actor.start.remote())
    session = ray.get(actor.create_session.remote("session-complete-no-materialize"))

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{session.base_url}/chat/completions",
            json={"model": "dummy-model", "messages": [{"role": "user", "content": "track state"}]},
        )
        assert response.status_code == 200

        complete = await client.post(
            f"{session.base_url.removesuffix('/v1')}/complete",
            json={"reward_info": {"score": 1.0}},
        )
        assert complete.status_code == 200

    state = ray.get(actor.get_session_state.remote("session-complete-no-materialize"))
    trajectories = ray.get(actor.finalize_session.remote("session-complete-no-materialize"))
    ray.get(actor.shutdown.remote())

    assert state["num_trajectories"] == 0
    assert state["has_active_trajectory"] is True
    assert len(trajectories) == 1


def test_gateway_actor_rejects_finalize_after_abort(ray_runtime):
    from verl.experimental.agent_gateway.gateway import GatewayActor

    actor = GatewayActor.remote(tokenizer=FakeTokenizer(), backend=QueuedBackend(["DONE"]), host="127.0.0.1")
    ray.get(actor.start.remote())
    ray.get(actor.create_session.remote("session-finalize-after-abort"))
    ray.get(actor.abort_session.remote("session-finalize-after-abort"))

    with pytest.raises(ray.exceptions.RayTaskError, match="aborted"):
        ray.get(actor.finalize_session.remote("session-finalize-after-abort"))

    ray.get(actor.shutdown.remote())


def test_gateway_actor_rejects_complete_after_abort(ray_runtime):
    from verl.experimental.agent_gateway.gateway import GatewayActor

    actor = GatewayActor.remote(tokenizer=FakeTokenizer(), backend=QueuedBackend(["DONE"]), host="127.0.0.1")
    ray.get(actor.start.remote())
    ray.get(actor.create_session.remote("session-complete-after-abort"))
    ray.get(actor.abort_session.remote("session-complete-after-abort"))

    with pytest.raises(ray.exceptions.RayTaskError, match="aborted"):
        ray.get(actor.complete_session.remote("session-complete-after-abort"))

    ray.get(actor.shutdown.remote())


@pytest.mark.asyncio
async def test_gateway_actor_session_state_tracks_metadata_flags_and_timestamps(ray_runtime):
    from verl.experimental.agent_gateway.gateway import GatewayActor

    actor = GatewayActor.remote(tokenizer=FakeTokenizer(), backend=QueuedBackend(["DONE"]), host="127.0.0.1")
    ray.get(actor.start.remote())

    session = ray.get(actor.create_session.remote("session-state", metadata={"uid": "sample-7", "split": "train"}))
    created_state = ray.get(actor.get_session_state.remote("session-state"))

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{session.base_url}/chat/completions",
            json={"model": "dummy-model", "messages": [{"role": "user", "content": "track state"}]},
        )
        assert response.status_code == 200

        completed = await client.post(
            f"{session.base_url.removesuffix('/v1')}/complete",
            json={"reward_info": {"score": 1.0}},
        )
        assert completed.status_code == 200

    completed_state = ray.get(actor.get_session_state.remote("session-state"))
    trajectories = ray.get(actor.finalize_session.remote("session-state"))
    ray.get(actor.shutdown.remote())

    assert created_state["metadata"] == {"uid": "sample-7", "split": "train"}
    assert created_state["completed_flag"] is False
    assert created_state["aborted_flag"] is False
    assert created_state["created_at"] <= created_state["updated_at"]
    assert completed_state["completed_flag"] is True
    assert completed_state["aborted_flag"] is False
    assert completed_state["updated_at"] >= created_state["updated_at"]
    assert len(trajectories) == 1
