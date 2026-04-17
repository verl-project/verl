import asyncio
import copy
import json

import httpx
import pytest
import ray

from tests.agent.support import (
    FailingBackend,
    FakeTokenizer,
    QueuedBackend,
    RejectConcurrentSessionBackend,
    RejectRequestEnvelopeBackend,
    RejectToolsSamplingParamsBackend,
    SequencedBackend,
)


@pytest.fixture
def ray_runtime():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_normalize_request_context_preserves_structured_fields():
    from verl.agent.gateway.gateway import _normalize_request_context

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


@pytest.mark.asyncio
async def test_gateway_actor_complete_wait_and_finalize(ray_runtime):
    from verl.agent.gateway.gateway import GatewayActor

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
    from verl.agent.gateway.gateway import GatewayActor

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
    from verl.agent.gateway.gateway import GatewayActor

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
    from verl.agent.gateway.gateway import GatewayActor

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
async def test_gateway_actor_strips_request_envelope_but_keeps_sampling_params(ray_runtime):
    from verl.agent.gateway.gateway import GatewayActor

    actor = GatewayActor.remote(
        tokenizer=FakeTokenizer(),
        backend=RejectRequestEnvelopeBackend("SAFE"),
        host="127.0.0.1",
    )
    ray.get(actor.start.remote())
    session = ray.get(actor.create_session.remote("session-envelope-boundary"))

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "temperature": 0.25,
                "tools": [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}],
                "messages": [{"role": "user", "content": "first turn"}],
            },
        )

    ray.get(actor.shutdown.remote())

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_gateway_actor_continuation_preserves_prompt_and_generation_masks(ray_runtime):
    from verl.agent.gateway.gateway import GatewayActor

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


@pytest.mark.asyncio
async def test_gateway_actor_tool_argument_format_drift_splits_after_valid_continuation(ray_runtime):
    from verl.agent.gateway.gateway import GatewayActor

    tool_call_text = '<tool_call>\n{"name": "search", "arguments": {"b": 2, "a": 1}}\n</tool_call>'
    actor = GatewayActor.remote(
        tokenizer=FakeTokenizer(),
        backend=QueuedBackend([tool_call_text, "SECOND", "THIRD"]),
        host="127.0.0.1",
        tool_parser_name="hermes",
    )
    ray.get(actor.start.remote())
    session = ray.get(actor.create_session.remote("session-tool-arg-drift"))

    async with httpx.AsyncClient(timeout=5.0) as client:
        first = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "tools": [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}],
                "messages": [{"role": "user", "content": "what is the weather?"}],
            },
        )
        assert first.status_code == 200
        assistant_tool_message = first.json()["choices"][0]["message"]

        second = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "tools": [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}],
                "messages": [
                    {"role": "user", "content": "what is the weather?"},
                    assistant_tool_message,
                    {"role": "tool", "tool_call_id": assistant_tool_message["tool_calls"][0]["id"], "content": "sunny"},
                ],
            },
        )
        assert second.status_code == 200

        drifted_tool_message = copy.deepcopy(assistant_tool_message)
        drifted_tool_message["tool_calls"][0]["function"]["arguments"] = json.dumps({"a": 1, "b": 2})
        third = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "tools": [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}],
                "messages": [
                    {"role": "user", "content": "what is the weather?"},
                    drifted_tool_message,
                    {"role": "tool", "tool_call_id": assistant_tool_message["tool_calls"][0]["id"], "content": "sunny"},
                    {"role": "assistant", "content": "SECOND"},
                    {"role": "user", "content": "follow up"},
                ],
            },
        )
        assert third.status_code == 200

    trajectories = ray.get(actor.finalize_session.remote("session-tool-arg-drift"))
    ray.get(actor.shutdown.remote())

    assert len(trajectories) == 2
    assert trajectories[0].trajectory_id == 0
    assert trajectories[1].trajectory_id == 1
    assert 0 in trajectories[0].response_mask
    assert trajectories[1].response_mask == [1] * len("THIRD")


@pytest.mark.asyncio
async def test_gateway_actor_serializes_same_session_concurrent_requests(ray_runtime):
    from verl.agent.gateway.gateway import GatewayActor

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
async def test_gateway_actor_rejects_chat_after_complete(ray_runtime):
    from verl.agent.gateway.gateway import GatewayActor

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
async def test_gateway_actor_finalizes_without_complete(ray_runtime):
    from verl.agent.gateway.gateway import GatewayActor

    actor = GatewayActor.remote(tokenizer=FakeTokenizer(), backend=QueuedBackend(["DONE"]), host="127.0.0.1")
    ray.get(actor.start.remote())
    session = ray.get(actor.create_session.remote("session-finalize-without-complete"))

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{session.base_url}/chat/completions",
            json={"model": "dummy-model", "messages": [{"role": "user", "content": "finish directly"}]},
        )
        assert response.status_code == 200

    trajectories = ray.get(actor.finalize_session.remote("session-finalize-without-complete"))
    ray.get(actor.shutdown.remote())

    assert len(trajectories) == 1
    assert trajectories[0].trajectory_id == 0
    assert trajectories[0].reward_info == {}


@pytest.mark.parametrize(
    ("payload", "detail_fragment"),
    [
        ({"model": "dummy-model", "messages": []}, "messages must be non-empty"),
        (
            {"model": "dummy-model", "messages": [{"role": "user", "name": "alice", "content": "hello"}]},
            "message.name is not supported",
        ),
        (
            {"model": "dummy-model", "messages": [{"role": "user", "content": 123}]},
            "Unsupported content type",
        ),
        (
            {
                "model": "dummy-model",
                "messages": [{"role": "assistant", "content": "", "tool_calls": {"id": "call-1"}}],
            },
            "tool_calls must be a list",
        ),
        (
            {
                "model": "dummy-model",
                "tools": {"type": "function"},
                "messages": [{"role": "user", "content": "hello"}],
            },
            "tools must be a list",
        ),
    ],
)
@pytest.mark.asyncio
async def test_gateway_actor_rejects_malformed_requests_with_bad_request(ray_runtime, payload, detail_fragment):
    from verl.agent.gateway.gateway import GatewayActor

    actor = GatewayActor.remote(tokenizer=FakeTokenizer(), backend=QueuedBackend(["DONE"]), host="127.0.0.1")
    ray.get(actor.start.remote())
    session = ray.get(actor.create_session.remote("session-validation"))

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{session.base_url}/chat/completions",
            json=payload,
        )

    ray.get(actor.shutdown.remote())

    assert response.status_code == 400
    assert detail_fragment in response.text


@pytest.mark.asyncio
async def test_gateway_actor_backend_failure_does_not_commit_partial_state(ray_runtime):
    from verl.agent.gateway.gateway import GatewayActor

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
    from verl.agent.gateway.gateway import GatewayActor

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
async def test_gateway_actor_tool_call_decode_returns_openai_format(ray_runtime):
    """When tool_parser_name is set and model outputs tool call tokens,
    the HTTP response should contain tool_calls in OpenAI format."""
    from verl.agent.gateway.gateway import GatewayActor

    tool_call_text = '<tool_call>\n{"name": "search", "arguments": {"query": "weather"}}\n</tool_call>'
    actor = GatewayActor.remote(
        tokenizer=FakeTokenizer(),
        backend=QueuedBackend([tool_call_text, "sunny today"]),
        host="127.0.0.1",
        tool_parser_name="hermes",
    )
    ray.get(actor.start.remote())
    session = ray.get(actor.create_session.remote("session-tool-call"))

    async with httpx.AsyncClient(timeout=5.0) as client:
        # First request: model returns a tool call
        first = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "tools": [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}],
                "messages": [{"role": "user", "content": "what is the weather?"}],
            },
        )
        assert first.status_code == 200
        first_data = first.json()
        assert first_data["choices"][0]["finish_reason"] == "tool_calls"
        tool_calls = first_data["choices"][0]["message"].get("tool_calls")
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "search"
        assert tool_calls[0]["type"] == "function"
        assert "id" in tool_calls[0]
        # HTTP response arguments should be a JSON string (OpenAI compatible)
        assert isinstance(tool_calls[0]["function"]["arguments"], str)

        # Second request: agent sends back tool result as continuation
        second = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "tools": [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}],
                "messages": [
                    {"role": "user", "content": "what is the weather?"},
                    {"role": "assistant", "content": None, "tool_calls": tool_calls},
                    {"role": "tool", "tool_call_id": tool_calls[0]["id"], "content": "sunny and warm"},
                ],
            },
        )
        assert second.status_code == 200
        assert second.json()["choices"][0]["message"]["content"] == "sunny today"

    trajectories = ray.get(actor.finalize_session.remote("session-tool-call"))
    ray.get(actor.shutdown.remote())

    assert len(trajectories) == 1
    # Should have both mask=0 (incremental) and mask=1 (model output) tokens
    assert 0 in trajectories[0].response_mask
    assert 1 in trajectories[0].response_mask
