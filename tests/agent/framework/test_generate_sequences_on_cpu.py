from __future__ import annotations

import pytest

from verl.agent.framework.types import SessionHandle, Trajectory
from verl.utils import tensordict_utils as tu


class _FakeTransferQueue:
    def __init__(self):
        self.puts = []
        self.batch_puts = []

    async def async_kv_put(self, *, key, partition_id, tag):
        self.puts.append({"key": key, "partition_id": partition_id, "tag": dict(tag)})

    async def async_kv_batch_put(self, *, keys, fields, tags, partition_id):
        self.batch_puts.append(
            {
                "keys": list(keys),
                "fields": fields,
                "tags": [dict(tag) for tag in tags],
                "partition_id": partition_id,
            }
        )


class _FakeSessionRuntime:
    def __init__(self, finalized_by_session_id: dict[str, list[Trajectory]]):
        self.finalized_by_session_id = finalized_by_session_id
        self.created_sessions = []
        self.finalized_sessions = []
        self.aborted_sessions = []

    async def create_session(self, session_id: str, **kwargs):
        self.created_sessions.append(session_id)
        return SessionHandle(session_id=session_id, base_url=f"http://fake/{session_id}/v1")

    async def finalize_session(self, session_id: str):
        self.finalized_sessions.append(session_id)
        return self.finalized_by_session_id[session_id]

    async def abort_session(self, session_id: str) -> None:
        self.aborted_sessions.append(session_id)

    async def wait_for_completion(self, session_id: str, timeout: float | None = None) -> None:
        return None


def _build_prompts(count: int = 2):
    return tu.get_tensordict(
        tensor_dict={
            "raw_prompt": [[{"role": "user", "content": f"sample {i}"}] for i in range(count)],
            "uid": [f"uid-{i}" for i in range(count)],
            "data_source": ["deepeyes"] * count,
            "reward_model": [{"ground_truth": f"answer-{i}"} for i in range(count)],
            "extra_info": [{"index": i} for i in range(count)],
            "tools_kwargs": [{"tool": i} for i in range(count)],
            "agent_name": ["deepeyes"] * count,
        }
    )


def _trajectory(
    *,
    uid: str,
    session_id: str,
    trajectory_id: int = 0,
    prompt_ids: list[int] | None = None,
    response_ids: list[int] | None = None,
    response_logprobs: list[float] | None = None,
    num_turns: int = 2,
):
    prompt_ids = prompt_ids or [10, 11]
    response_ids = response_ids or [20, 21]
    return Trajectory(
        uid=uid,
        session_id=session_id,
        trajectory_id=trajectory_id,
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=[1] * len(response_ids),
        response_logprobs=response_logprobs,
        reward_score=None,
        num_turns=num_turns,
        multi_modal_data={"images": ["raw-image-should-not-be-written"]},
    )


@pytest.mark.asyncio
async def test_generate_sequences_writes_tq_schema_for_each_session(monkeypatch):
    from verl.agent.framework import framework as framework_module
    from verl.agent.framework.framework import OpenAICompatibleAgentFramework

    fake_tq = _FakeTransferQueue()
    monkeypatch.setattr(framework_module, "tq", fake_tq)

    runtime = _FakeSessionRuntime(
        {
            "session-0-0": [_trajectory(uid="uid-0", session_id="session-0-0", response_logprobs=[-0.1, -0.2])],
            "session-0-1": [_trajectory(uid="uid-0", session_id="session-0-1", response_logprobs=[-0.3, -0.4])],
            "session-1-0": [_trajectory(uid="uid-1", session_id="session-1-0", response_logprobs=[-0.5, -0.6])],
            "session-1-1": [_trajectory(uid="uid-1", session_id="session-1-1", response_logprobs=[-0.7, -0.8])],
        }
    )

    async def agent_runner(*, raw_prompt, session, sample_index, tools_kwargs):
        assert raw_prompt == [{"role": "user", "content": f"sample {sample_index}"}]
        assert tools_kwargs == {"tool": sample_index}

    def reward_fn(ctx):
        return [float(ctx.sample_fields["extra_info"]["index"]) + 0.25 for _ in ctx.trajectories]

    framework = OpenAICompatibleAgentFramework(
        session_runtime=runtime,
        agent_runner=agent_runner,
        reward_fn=reward_fn,
    )
    framework._build_session_id = lambda prompts, sample_index, session_index=0: f"session-{sample_index}-{session_index}"

    stats = await framework.generate_sequences(
        _build_prompts(),
        global_steps=7,
        partition_id="train",
        num_sessions=2,
    )

    assert stats["num_input_prompts"] == 2
    assert stats["num_success_sessions"] == 4
    assert stats["num_failed_sessions"] == 0
    assert stats["num_success_outputs"] == 4
    assert stats["num_failed_uids"] == 0
    assert fake_tq.batch_puts[0]["keys"] == ["uid-0_0_0"]
    assert fake_tq.batch_puts[1]["keys"] == ["uid-0_1_0"]
    assert fake_tq.batch_puts[2]["keys"] == ["uid-1_0_0"]
    assert fake_tq.batch_puts[3]["keys"] == ["uid-1_1_0"]
    assert fake_tq.puts == [
        {"key": "uid-0", "partition_id": "train", "tag": {"status": "finished"}},
        {"key": "uid-1", "partition_id": "train", "tag": {"status": "finished"}},
    ]

    first = fake_tq.batch_puts[0]
    fields = first["fields"]
    assert first["partition_id"] == "train"
    assert first["tags"] == [
        {"global_steps": 7, "status": "success", "prompt_len": 2, "response_len": 2, "seq_len": 4}
    ]
    assert fields["input_ids"].is_nested
    assert fields["response_mask"].is_nested
    assert fields["position_ids"].is_nested
    assert fields["prompts"][0].tolist() == [10, 11]
    assert fields["responses"][0].tolist() == [20, 21]
    assert fields["response_mask"][0].tolist() == [1, 1]
    assert fields["loss_mask"][0].tolist() == [1, 1]
    assert fields["input_ids"][0].tolist() == [10, 11, 20, 21]
    assert fields["attention_mask"][0].tolist() == [1, 1, 1, 1]
    assert fields["position_ids"][0].tolist() == [0, 1, 2, 3]
    assert fields["rollout_log_probs"][0].tolist() == pytest.approx([-0.1, -0.2])
    assert fields["rm_scores"][0].tolist() == [0.0, 0.25]
    assert tu.get(fields, "multi_modal_inputs") == [{}]
    assert tu.get(fields, "uid") == ["uid-0"]
    assert tu.get(fields, "raw_prompt") == [[{"role": "user", "content": "sample 0"}]]
    assert tu.get(fields, "data_source") == ["deepeyes"]
    assert tu.get(fields, "reward_model") == [{"ground_truth": "answer-0"}]
    assert tu.get(fields, "extra_info") == [{"index": 0}]
    assert tu.get(fields, "tools_kwargs") == [{"tool": 0}]
    assert tu.get(fields, "agent_name") == ["deepeyes"]
    assert tu.get(fields, "session_id") == [0]
    assert tu.get(fields, "global_steps") == [7]
    assert fields["num_turns"].tolist() == [2]
    assert "multi_modal_data" not in fields.keys()


@pytest.mark.asyncio
async def test_generate_sequences_keeps_successful_sessions_when_one_session_fails(monkeypatch):
    from verl.agent.framework import framework as framework_module
    from verl.agent.framework.framework import OpenAICompatibleAgentFramework

    fake_tq = _FakeTransferQueue()
    monkeypatch.setattr(framework_module, "tq", fake_tq)
    runtime = _FakeSessionRuntime(
        {
            "session-0-0": [_trajectory(uid="uid-0", session_id="session-0-0")],
            "session-0-1": [_trajectory(uid="uid-0", session_id="session-0-1")],
        }
    )

    async def agent_runner(*, raw_prompt, session, sample_index, tools_kwargs):
        if session.session_id == "session-0-1":
            raise RuntimeError("gateway failed once")

    framework = OpenAICompatibleAgentFramework(
        session_runtime=runtime,
        agent_runner=agent_runner,
        reward_fn=lambda ctx: [1.0 for _ in ctx.trajectories],
    )
    framework._build_session_id = lambda prompts, sample_index, session_index=0: f"session-{sample_index}-{session_index}"

    stats = await framework.generate_sequences(
        _build_prompts(count=1),
        global_steps=8,
        partition_id="train",
        num_sessions=2,
    )

    assert stats["num_success_sessions"] == 1
    assert stats["num_failed_sessions"] == 1
    assert stats["num_success_outputs"] == 1
    assert stats["num_failed_uids"] == 0
    assert "gateway failed once" in stats["failure_reasons"][0]
    assert fake_tq.batch_puts[0]["keys"] == ["uid-0_0_0"]
    assert fake_tq.puts == [{"key": "uid-0", "partition_id": "train", "tag": {"status": "finished"}}]
    assert runtime.aborted_sessions == ["session-0-1"]


@pytest.mark.asyncio
async def test_generate_sequences_marks_prompt_failure_when_all_sessions_fail(monkeypatch):
    from verl.agent.framework import framework as framework_module
    from verl.agent.framework.framework import OpenAICompatibleAgentFramework

    fake_tq = _FakeTransferQueue()
    monkeypatch.setattr(framework_module, "tq", fake_tq)
    runtime = _FakeSessionRuntime({"session-0-0": [], "session-0-1": []})

    async def agent_runner(*, raw_prompt, session, sample_index, tools_kwargs):
        raise RuntimeError(f"failed {session.session_id}")

    framework = OpenAICompatibleAgentFramework(
        session_runtime=runtime,
        agent_runner=agent_runner,
        reward_fn=lambda ctx: [1.0 for _ in ctx.trajectories],
    )
    framework._build_session_id = lambda prompts, sample_index, session_index=0: f"session-{sample_index}-{session_index}"

    stats = await framework.generate_sequences(
        _build_prompts(count=1),
        global_steps=9,
        partition_id="val",
        num_sessions=2,
    )

    assert stats["num_success_sessions"] == 0
    assert stats["num_failed_sessions"] == 2
    assert stats["num_success_outputs"] == 0
    assert stats["num_failed_uids"] == 1
    assert fake_tq.batch_puts == []
    assert fake_tq.puts == [{"key": "uid-0", "partition_id": "val", "tag": {"status": "failure"}}]


@pytest.mark.asyncio
async def test_generate_sequences_omits_rm_scores_when_reward_fn_is_none(monkeypatch):
    from verl.agent.framework import framework as framework_module
    from verl.agent.framework.framework import OpenAICompatibleAgentFramework

    fake_tq = _FakeTransferQueue()
    monkeypatch.setattr(framework_module, "tq", fake_tq)
    runtime = _FakeSessionRuntime({"session-0-0": [_trajectory(uid="uid-0", session_id="session-0-0")]})

    async def agent_runner(*, raw_prompt, session, sample_index, tools_kwargs):
        return None

    framework = OpenAICompatibleAgentFramework(
        session_runtime=runtime,
        agent_runner=agent_runner,
        reward_fn=None,
    )
    framework._build_session_id = lambda prompts, sample_index, session_index=0: f"session-{sample_index}-{session_index}"

    await framework.generate_sequences(
        _build_prompts(count=1),
        global_steps=10,
        partition_id="train",
    )

    assert "rm_scores" not in fake_tq.batch_puts[0]["fields"].keys()


@pytest.mark.asyncio
async def test_generate_sequences_keeps_other_prompts_when_prompt_task_raises(monkeypatch):
    from verl.agent.framework.framework import OpenAICompatibleAgentFramework

    runtime = _FakeSessionRuntime(
        {
            "session-1-0": [_trajectory(uid="uid-1", session_id="session-1-0")],
        }
    )

    framework = OpenAICompatibleAgentFramework(
        session_runtime=runtime,
        agent_runner=lambda **_: None,
        reward_fn=lambda ctx: [1.0 for _ in ctx.trajectories],
    )

    async def fake_run_prompt_to_replay_buffer(*, sample_index, **kwargs):
        if sample_index == 0:
            raise RuntimeError("prompt 0 exploded")
        return {
            "num_success_sessions": 1,
            "num_failed_sessions": 0,
            "num_success_outputs": 1,
            "num_failed_uids": 0,
            "failure_reasons": [],
        }

    monkeypatch.setattr(framework, "_run_prompt_to_replay_buffer", fake_run_prompt_to_replay_buffer)

    stats = await framework.generate_sequences(
        _build_prompts(count=2),
        global_steps=11,
        partition_id="train",
        num_sessions=1,
    )

    assert stats["num_input_prompts"] == 2
    assert stats["num_success_sessions"] == 1
    assert stats["num_failed_sessions"] == 1
    assert stats["num_success_outputs"] == 1
    assert stats["num_failed_uids"] == 1
    assert "prompt 0 exploded" in stats["failure_reasons"][0]
