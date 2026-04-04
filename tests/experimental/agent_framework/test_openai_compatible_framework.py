import httpx
import numpy as np
import pytest
import ray

from tests.experimental.agent_gateway.support import FakeTokenizer, QueuedBackend
from verl.protocol import DataProto


class _RecordingSessionRuntime:
    def __init__(self, delegate):
        self.delegate = delegate
        self.created_sessions = []
        self.waited_sessions = []
        self.finalized_sessions = []
        self.aborted_sessions = []

    async def create_session(self, session_id: str, **kwargs):
        self.created_sessions.append(session_id)
        return await self.delegate.create_session(session_id, **kwargs)

    async def finalize_session(self, session_id: str):
        self.finalized_sessions.append(session_id)
        return await self.delegate.finalize_session(session_id)

    async def abort_session(self, session_id: str) -> None:
        self.aborted_sessions.append(session_id)
        await self.delegate.abort_session(session_id)

    async def wait_for_completion(self, session_id: str, timeout: float | None = None) -> None:
        self.waited_sessions.append(session_id)
        await self.delegate.wait_for_completion(session_id, timeout=timeout)


@pytest.fixture
def ray_runtime():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.mark.asyncio
async def test_openai_compatible_framework_runs_minimal_remote_style_path(ray_runtime):
    from verl.experimental.agent_framework.openai_compatible_framework import OpenAICompatibleAgentFramework
    from verl.experimental.agent_framework.types import TrajectoryRewardContext
    from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager

    manager = AsyncLLMServerManager(
        config=None,
        servers=[],
        load_balancer_handle=None,
        gateway_count=1,
        gateway_actor_kwargs={
            "tokenizer": FakeTokenizer(),
            "backend": QueuedBackend(["ANSWER: A"]),
            "host": "127.0.0.1",
        },
    )
    session_runtime = _RecordingSessionRuntime(manager)
    seen_base_urls = []

    async def mock_agent(*, raw_prompt, session, sample_index):
        seen_base_urls.append(session.base_url)
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{session.base_url}/chat/completions",
                json={"model": "dummy-model", "messages": list(raw_prompt)},
            )
            assert response.status_code == 200

            complete = await client.post(
                f"{session.base_url.removesuffix('/v1')}/complete",
                json={"reward_info": {"score": 1.0, "label": f"sample-{sample_index}"}},
            )
            assert complete.status_code == 200

    def reward_fn(contexts: list[TrajectoryRewardContext]) -> list[float]:
        # Extract score from reward_info injected by the agent via /complete
        return [float(ctx.trajectory.reward_info.get("score", 0.0)) for ctx in contexts]

    framework = OpenAICompatibleAgentFramework(
        session_runtime=session_runtime,
        agent_runner=mock_agent,
        reward_fn=reward_fn,
    )

    prompts = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array(
                [
                    [{"role": "user", "content": "Return label A"}],
                ],
                dtype=object,
            ),
        }
    )

    output = await framework.generate_sequences(prompts)
    await manager.shutdown()

    assert len(session_runtime.created_sessions) == 1
    assert len(session_runtime.waited_sessions) == 0
    assert len(session_runtime.finalized_sessions) == 1
    assert seen_base_urls and seen_base_urls[0].endswith("/v1")
    assert tuple(output.batch["responses"].shape) == (1, output.batch["responses"].shape[1])
    assert tuple(output.batch["rm_scores"].shape) == tuple(output.batch["responses"].shape)
    assert output.non_tensor_batch["label"].tolist() == ["sample-0"]


@pytest.mark.asyncio
async def test_openai_compatible_framework_does_not_require_complete_signal(ray_runtime):
    from verl.experimental.agent_framework.openai_compatible_framework import OpenAICompatibleAgentFramework
    from verl.experimental.agent_framework.types import TrajectoryRewardContext
    from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager

    manager = AsyncLLMServerManager(
        config=None,
        servers=[],
        load_balancer_handle=None,
        gateway_count=1,
        gateway_actor_kwargs={
            "tokenizer": FakeTokenizer(),
            "backend": QueuedBackend(["ANSWER: B"]),
            "host": "127.0.0.1",
        },
    )
    session_runtime = _RecordingSessionRuntime(manager)

    async def mock_agent(*, raw_prompt, session, sample_index):
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{session.base_url}/chat/completions",
                json={"model": "dummy-model", "messages": list(raw_prompt)},
            )
            assert response.status_code == 200

    def reward_fn(contexts: list[TrajectoryRewardContext]) -> list[float]:
        # Agent did not call /complete, so reward_info is empty; return a default score.
        return [0.0 for _ in contexts]

    framework = OpenAICompatibleAgentFramework(
        session_runtime=session_runtime,
        agent_runner=mock_agent,
        reward_fn=reward_fn,
    )

    prompts = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array(
                [
                    [{"role": "user", "content": "Return label B"}],
                ],
                dtype=object,
            ),
        }
    )

    output = await framework.generate_sequences(prompts)
    await manager.shutdown()

    assert len(session_runtime.waited_sessions) == 0
    assert len(session_runtime.finalized_sessions) == 1
    # reward_fn always runs, so rm_scores is always present
    assert "rm_scores" in output.batch.keys()
