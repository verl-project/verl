from __future__ import annotations

import numpy as np
import pytest
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu


class _FakeSessionRuntime:
    def __init__(self, finalized_by_session_id: dict[str, list]):
        self.finalized_by_session_id = finalized_by_session_id
        self.created_sessions: list[str] = []
        self.waited_sessions: list[tuple[str, float | None]] = []
        self.finalized_sessions: list[str] = []
        self.aborted_sessions: list[str] = []

    async def create_session(self, session_id: str, **kwargs):
        from verl.agent.framework.types import SessionHandle

        self.created_sessions.append(session_id)
        return SessionHandle(session_id=session_id, base_url=f"http://fake/{session_id}/v1")

    async def finalize_session(self, session_id: str):
        self.finalized_sessions.append(session_id)
        return self.finalized_by_session_id[session_id]

    async def abort_session(self, session_id: str) -> None:
        self.aborted_sessions.append(session_id)

    async def wait_for_completion(self, session_id: str, timeout: float | None = None) -> None:
        self.waited_sessions.append((session_id, timeout))


def _build_prompts(**non_tensor_batch) -> TensorDict:
    return tu.get_tensordict(
        tensor_dict={
            key: list(values)
            for key, values in non_tensor_batch.items()
        }
    )


def _build_trajectory(
    *,
    uid: str,
    session_id: str,
    trajectory_id: int,
    prompt_ids: list[int],
    response_ids: list[int],
    response_mask: list[int],
    response_logprobs: list[float] | None = None,
    reward_info: dict | None = None,
    reward_score: float | None = None,
    num_turns: int = 2,
):
    from verl.agent.framework.types import Trajectory

    return Trajectory(
        uid=uid,
        session_id=session_id,
        trajectory_id=trajectory_id,
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=response_mask,
        response_logprobs=response_logprobs,
        reward_info=reward_info or {},
        reward_score=reward_score,
        num_turns=num_turns,
    )


@pytest.mark.asyncio
async def test_openai_compatible_framework_runs_against_fake_session_runtime():
    from verl.agent.framework.framework import OpenAICompatibleAgentFramework
    from verl.agent.framework.types import SessionRewardContext

    prompts = _build_prompts(
        raw_prompt=[[{"role": "user", "content": "Return label A"}]],
        uid=["sample-uid"],
        data_source=["openai/gsm8k"],
    )
    session_id = "session-0-fixed"
    runtime = _FakeSessionRuntime(
        {
            session_id: [
                _build_trajectory(
                    uid="sample-uid",
                    session_id=session_id,
                    trajectory_id=0,
                    prompt_ids=[10, 11],
                    response_ids=[20, 21],
                    response_mask=[1, 1],
                    response_logprobs=[-0.1, -0.2],
                    reward_info={"score": 1.0, "label": "sample-0"},
                )
            ]
        }
    )

    async def agent_runner(*, raw_prompt, session, sample_index):
        assert raw_prompt == [{"role": "user", "content": "Return label A"}]
        assert session.base_url == f"http://fake/{session_id}/v1"
        assert sample_index == 0

    def reward_fn(ctx: SessionRewardContext) -> list[float]:
        return [float(traj.reward_info["score"]) for traj in ctx.trajectories]

    framework = OpenAICompatibleAgentFramework(
        session_runtime=runtime,
        agent_runner=agent_runner,
        reward_fn=reward_fn,
    )
    framework._build_session_id = lambda prompts, sample_index: session_id

    output = await framework.generate_sequences(prompts)

    assert isinstance(output, TensorDict)
    assert runtime.created_sessions == [session_id]
    assert runtime.finalized_sessions == [session_id]
    assert runtime.waited_sessions == []
    assert runtime.aborted_sessions == []
    assert tu.get(output, "label") == ["sample-0"]
    assert tu.get(output, "uid") == ["sample-uid"]
    assert tu.get(output, "data_source") == ["openai/gsm8k"]
    assert tuple(output["responses"].shape) == (1, 2)
    assert tuple(output["rm_scores"].shape) == (1, 2)


@pytest.mark.asyncio
async def test_openai_compatible_framework_waits_for_completion_when_configured():
    from verl.agent.framework.framework import OpenAICompatibleAgentFramework
    from verl.agent.framework.types import SessionRewardContext

    prompts = _build_prompts(
        raw_prompt=[[{"role": "user", "content": "Wait for completion"}]],
    )
    session_id = "session-1-fixed"
    runtime = _FakeSessionRuntime(
        {
            session_id: [
                _build_trajectory(
                    uid="sample-uid",
                    session_id=session_id,
                    trajectory_id=0,
                    prompt_ids=[1],
                    response_ids=[2],
                    response_mask=[1],
                ),
            ]
        }
    )

    async def agent_runner(*, raw_prompt, session, sample_index):
        assert raw_prompt == [{"role": "user", "content": "Wait for completion"}]
        assert sample_index == 0
        assert session.session_id == session_id

    def reward_fn(ctx: SessionRewardContext) -> list[float]:
        return [1.0]

    framework = OpenAICompatibleAgentFramework(
        session_runtime=runtime,
        agent_runner=agent_runner,
        reward_fn=reward_fn,
        wait_for_completion_after_agent_run=True,
        completion_timeout=12.5,
    )
    framework._build_session_id = lambda prompts, sample_index: session_id

    output = await framework.generate_sequences(prompts)

    assert runtime.waited_sessions == [(session_id, 12.5)]
    assert runtime.finalized_sessions == [session_id]
    assert len(output) == 1


@pytest.mark.asyncio
async def test_openai_compatible_framework_broadcasts_sample_fields_to_multiple_trajectories():
    from verl.agent.framework.framework import OpenAICompatibleAgentFramework
    from verl.agent.framework.types import SessionRewardContext

    prompts = _build_prompts(
        raw_prompt=[[{"role": "user", "content": "Return two trajectories"}]],
        uid=["sample-uid"],
        extra_info=[{"split": "train"}],
    )
    session_id = "session-1b-fixed"
    runtime = _FakeSessionRuntime(
        {
            session_id: [
                _build_trajectory(
                    uid="sample-uid",
                    session_id=session_id,
                    trajectory_id=0,
                    prompt_ids=[1],
                    response_ids=[2],
                    response_mask=[1],
                    reward_info={"label": "first"},
                ),
                _build_trajectory(
                    uid="sample-uid",
                    session_id=session_id,
                    trajectory_id=1,
                    prompt_ids=[3],
                    response_ids=[4, 5],
                    response_mask=[1, 1],
                    reward_info={"label": "second"},
                ),
            ]
        }
    )

    async def agent_runner(*, raw_prompt, session, sample_index):
        assert sample_index == 0
        assert session.session_id == session_id

    def reward_fn(ctx: SessionRewardContext) -> list[float]:
        assert ctx.sample_fields["uid"] == "sample-uid"
        assert ctx.sample_fields["extra_info"] == {"split": "train"}
        return [1.0 + index for index, _ in enumerate(ctx.trajectories)]

    framework = OpenAICompatibleAgentFramework(
        session_runtime=runtime,
        agent_runner=agent_runner,
        reward_fn=reward_fn,
    )
    framework._build_session_id = lambda prompts, sample_index: session_id

    output = await framework.generate_sequences(prompts)

    assert runtime.waited_sessions == []
    assert tu.get(output, "uid") == ["sample-uid", "sample-uid"]
    assert tu.get(output, "extra_info") == [{"split": "train"}, {"split": "train"}]
    assert tu.get(output, "label") == ["first", "second"]
    assert len(output) == 2


@pytest.mark.asyncio
async def test_openai_compatible_framework_aborts_session_on_agent_error():
    from verl.agent.framework.framework import OpenAICompatibleAgentFramework
    from verl.agent.framework.types import SessionRewardContext

    prompts = _build_prompts(raw_prompt=[[{"role": "user", "content": "boom"}]])
    session_id = "session-2-fixed"
    runtime = _FakeSessionRuntime({session_id: []})

    async def agent_runner(*, raw_prompt, session, sample_index):
        raise RuntimeError("agent failed")

    def reward_fn(ctx: SessionRewardContext) -> list[float]:
        return []

    framework = OpenAICompatibleAgentFramework(
        session_runtime=runtime,
        agent_runner=agent_runner,
        reward_fn=reward_fn,
    )
    framework._build_session_id = lambda prompts, sample_index: session_id

    with pytest.raises(RuntimeError, match="agent failed"):
        await framework.generate_sequences(prompts)

    assert runtime.created_sessions == [session_id]
    assert runtime.aborted_sessions == [session_id]
    assert runtime.finalized_sessions == []


@pytest.mark.asyncio
async def test_openai_compatible_framework_omits_rollout_log_probs_when_missing():
    from verl.agent.framework.framework import OpenAICompatibleAgentFramework
    from verl.agent.framework.types import SessionRewardContext

    prompts = _build_prompts(raw_prompt=[[{"role": "user", "content": "No logprobs expected"}]])
    session_id = "session-3-fixed"
    runtime = _FakeSessionRuntime(
        {
            session_id: [
                _build_trajectory(
                    uid="sample-uid",
                    session_id=session_id,
                    trajectory_id=0,
                    prompt_ids=[7],
                    response_ids=[8, 9],
                    response_mask=[1, 1],
                    response_logprobs=None,
                )
            ]
        }
    )

    async def agent_runner(*, raw_prompt, session, sample_index):
        return None

    def reward_fn(ctx: SessionRewardContext) -> list[float]:
        return [1.0]

    framework = OpenAICompatibleAgentFramework(
        session_runtime=runtime,
        agent_runner=agent_runner,
        reward_fn=reward_fn,
    )
    framework._build_session_id = lambda prompts, sample_index: session_id

    output = await framework.generate_sequences(prompts)

    assert "rollout_log_probs" not in output.keys()
