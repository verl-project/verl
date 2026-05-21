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


class _FakeReplayBuffer:
    def __init__(self):
        self.adds = []

    def add(self, partition_id, items):
        self.adds.append({"partition_id": partition_id, "items": dict(items)})


class _FakeSessionRuntime:
    """Fake runtime that matches session IDs by prefix (``session-{sample}-{session}``)
    to support the real uuid-suffixed IDs produced by the framework."""

    def __init__(self, finalized_by_session_prefix: dict[str, list[Trajectory]]):
        self._finalized_by_prefix = finalized_by_session_prefix
        self.created_sessions = []
        self.finalized_sessions = []
        self.aborted_sessions = []

    def _lookup(self, session_id: str) -> list[Trajectory]:
        for prefix, trajectories in self._finalized_by_prefix.items():
            if session_id.startswith(prefix):
                return trajectories
        raise KeyError(f"No prefix match for session_id={session_id}")

    async def create_session(self, session_id: str, **kwargs):
        self.created_sessions.append(session_id)
        return SessionHandle(session_id=session_id, base_url=f"http://fake/{session_id}/v1")

    async def finalize_session(self, session_id: str):
        self.finalized_sessions.append(session_id)
        return self._lookup(session_id)

    async def abort_session(self, session_id: str) -> None:
        self.aborted_sessions.append(session_id)

    async def wait_for_completion(self, session_id: str, timeout: float | None = None) -> None:
        return None


def _build_prompts(count: int = 2, *, global_steps: int = 7, validate: bool = False):
    non_tensor_dict = {"global_steps": global_steps}
    if validate:
        non_tensor_dict["validate"] = True
    return tu.get_tensordict(
        tensor_dict={
            "raw_prompt": [[{"role": "user", "content": f"sample {i}"}] for i in range(count)],
            "uid": [f"uid-{i}" for i in range(count)],
            "data_source": ["deepeyes"] * count,
            "reward_model": [{"ground_truth": f"answer-{i}"} for i in range(count)],
            "extra_info": [{"index": i} for i in range(count)],
            "tools_kwargs": [{"tool": i} for i in range(count)],
            "agent_name": ["deepeyes"] * count,
        },
        non_tensor_dict=non_tensor_dict,
    )


def _trajectory(
    *,
    prompt_ids: list[int] | None = None,
    response_ids: list[int] | None = None,
    response_logprobs: list[float] | None = None,
    num_turns: int = 2,
):
    prompt_ids = prompt_ids or [10, 11]
    response_ids = response_ids or [20, 21]
    return Trajectory(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=[1] * len(response_ids),
        response_logprobs=response_logprobs,
        reward_score=None,
        num_turns=num_turns,
        multi_modal_data={"images": ["raw-image-should-not-be-written"]},
    )


def _install_fake_score(monkeypatch, *, score_from_sample_fields=None, default_score=1.0):
    """Replace OpenAICompatibleAgentFramework._score_trajectories with a fake.

    Mirrors the production "score-last + broadcast" behavior: returns the same
    (score, extra_info) for every trajectory in the session. The score is
    derived from sample_fields if a callable is provided; otherwise default_score.
    """
    from verl.agent.framework.framework import OpenAICompatibleAgentFramework

    async def fake_score(self, trajectories, sample_fields):
        if score_from_sample_fields is not None:
            score = float(score_from_sample_fields(sample_fields))
        else:
            score = float(default_score)
        return [(score, {})] * len(trajectories)

    monkeypatch.setattr(OpenAICompatibleAgentFramework, "_score_trajectories", fake_score)


@pytest.mark.asyncio
async def test_generate_sequences_writes_tq_schema_for_each_session(monkeypatch):
    from verl.agent.framework import framework as framework_module
    from verl.agent.framework.framework import OpenAICompatibleAgentFramework

    fake_tq = _FakeTransferQueue()
    replay_buffer = _FakeReplayBuffer()
    monkeypatch.setattr(framework_module, "tq", fake_tq)

    runtime = _FakeSessionRuntime(
        {
            "session-0-0": [_trajectory(response_logprobs=[-0.1, -0.2])],
            "session-0-1": [_trajectory(response_logprobs=[-0.3, -0.4])],
            "session-1-0": [_trajectory(response_logprobs=[-0.5, -0.6])],
            "session-1-1": [_trajectory(response_logprobs=[-0.7, -0.8])],
        }
    )

    async def agent_runner(*, raw_prompt, session, sample_index, tools_kwargs, **kwargs):
        assert raw_prompt == [{"role": "user", "content": f"sample {sample_index}"}]
        assert tools_kwargs == {"tool": sample_index}

    # Score derived from sample_fields["extra_info"]["index"] + 0.25 (same as legacy lambda)
    _install_fake_score(
        monkeypatch,
        score_from_sample_fields=lambda sf: sf["extra_info"]["index"] + 0.25,
    )

    framework = OpenAICompatibleAgentFramework(
        session_runtime=runtime,
        agent_runner=agent_runner,
        reward_loop_worker_handles=["sentinel"],
        replay_buffer=replay_buffer,
        rollout_config={"n": 2, "val_kwargs": {"n": 2}},
    )


    await framework.generate_sequences(_build_prompts(global_steps=7))

    assert replay_buffer.adds == [
        {
            "partition_id": "train",
            "items": {
                "uid-0": {"global_steps": 7, "status": "running"},
                "uid-1": {"global_steps": 7, "status": "running"},
            },
        }
    ]
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
    replay_buffer = _FakeReplayBuffer()
    monkeypatch.setattr(framework_module, "tq", fake_tq)
    runtime = _FakeSessionRuntime(
        {
            "session-0-0": [_trajectory()],
            "session-0-1": [_trajectory()],
        }
    )

    async def agent_runner(*, raw_prompt, session, sample_index, tools_kwargs, **kwargs):
        if session.session_id.startswith("session-0-1-"):
            raise RuntimeError("gateway failed once")

    _install_fake_score(monkeypatch, default_score=1.0)

    framework = OpenAICompatibleAgentFramework(
        session_runtime=runtime,
        agent_runner=agent_runner,
        reward_loop_worker_handles=["sentinel"],
        replay_buffer=replay_buffer,
        rollout_config={"n": 2, "val_kwargs": {"n": 2}},
    )


    await framework.generate_sequences(_build_prompts(count=1, global_steps=8))

    assert replay_buffer.adds == [
        {"partition_id": "train", "items": {"uid-0": {"global_steps": 8, "status": "running"}}}
    ]
    assert fake_tq.batch_puts[0]["keys"] == ["uid-0_0_0"]
    assert fake_tq.puts == [{"key": "uid-0", "partition_id": "train", "tag": {"status": "finished"}}]
    assert len(runtime.aborted_sessions) == 1
    assert runtime.aborted_sessions[0].startswith("session-0-1-")


@pytest.mark.asyncio
async def test_generate_sequences_marks_prompt_failure_when_all_sessions_fail(monkeypatch):
    from verl.agent.framework import framework as framework_module
    from verl.agent.framework.framework import OpenAICompatibleAgentFramework

    fake_tq = _FakeTransferQueue()
    replay_buffer = _FakeReplayBuffer()
    monkeypatch.setattr(framework_module, "tq", fake_tq)
    runtime = _FakeSessionRuntime({"session-0-0": [], "session-0-1": []})

    async def agent_runner(*, raw_prompt, session, sample_index, tools_kwargs, **kwargs):
        raise RuntimeError(f"failed {session.session_id}")

    _install_fake_score(monkeypatch, default_score=1.0)

    framework = OpenAICompatibleAgentFramework(
        session_runtime=runtime,
        agent_runner=agent_runner,
        reward_loop_worker_handles=["sentinel"],
        replay_buffer=replay_buffer,
        rollout_config={"n": 1, "val_kwargs": {"n": 2}},
    )


    with pytest.raises(RuntimeError, match="All rollouts failed at global_steps=9"):
        await framework.generate_sequences(_build_prompts(count=1, global_steps=9, validate=True))

    assert replay_buffer.adds == [
        {"partition_id": "val", "items": {"uid-0": {"global_steps": 9, "status": "running"}}}
    ]
    assert fake_tq.batch_puts == []
    assert fake_tq.puts == [{"key": "uid-0", "partition_id": "val", "tag": {"status": "failure"}}]


@pytest.mark.asyncio
async def test_generate_sequences_zero_fills_rm_scores_when_no_reward_handles(monkeypatch):
    from verl.agent.framework import framework as framework_module
    from verl.agent.framework.framework import OpenAICompatibleAgentFramework

    fake_tq = _FakeTransferQueue()
    replay_buffer = _FakeReplayBuffer()
    monkeypatch.setattr(framework_module, "tq", fake_tq)
    runtime = _FakeSessionRuntime({"session-0-0": [_trajectory()]})

    async def agent_runner(*, raw_prompt, session, sample_index, tools_kwargs, **kwargs):
        return None

    framework = OpenAICompatibleAgentFramework(
        session_runtime=runtime,
        agent_runner=agent_runner,
        replay_buffer=replay_buffer,
        rollout_config={"n": 1, "val_kwargs": {"n": 1}},
    )


    await framework.generate_sequences(_build_prompts(count=1, global_steps=10))

    # rm_scores is always written (zero-filled when no reward) so the trainer's
    # KVBatchMeta select_fields never hits a missing field across the batch.
    rm_scores = fake_tq.batch_puts[0]["fields"]["rm_scores"]
    assert rm_scores[0].tolist() == [0.0, 0.0]


@pytest.mark.asyncio
async def test_generate_sequences_keeps_other_prompts_when_prompt_task_raises(monkeypatch, caplog):
    from verl.agent.framework.framework import OpenAICompatibleAgentFramework

    replay_buffer = _FakeReplayBuffer()
    runtime = _FakeSessionRuntime(
        {
            "session-1-0": [_trajectory()],
        }
    )

    _install_fake_score(monkeypatch, default_score=1.0)

    framework = OpenAICompatibleAgentFramework(
        session_runtime=runtime,
        agent_runner=lambda **_: None,
        reward_loop_worker_handles=["sentinel"],
        replay_buffer=replay_buffer,
        rollout_config={"n": 1, "val_kwargs": {"n": 1}},
    )

    async def fake_run_prompt_sessions_to_tq(*, sample_index, **kwargs):
        if sample_index == 0:
            raise RuntimeError("prompt 0 exploded")
        return {
            "num_success_sessions": 1,
            "num_failed_sessions": 0,
            "num_success_outputs": 1,
            "num_failed_uids": 0,
            "failure_reasons": [],
        }

    monkeypatch.setattr(framework, "_run_prompt_sessions_to_tq", fake_run_prompt_sessions_to_tq)

    caplog.set_level("INFO")
    await framework.generate_sequences(_build_prompts(count=2, global_steps=11))

    assert replay_buffer.adds == [
        {
            "partition_id": "train",
            "items": {
                "uid-0": {"global_steps": 11, "status": "running"},
                "uid-1": {"global_steps": 11, "status": "running"},
            },
        }
    ]
    assert "num_failed_uids=1" in caplog.text
    assert "prompt 0 exploded" in caplog.text


@pytest.mark.asyncio
async def test_generate_sequences_zero_fills_rollout_log_probs_when_missing(monkeypatch):
    from verl.agent.framework import framework as framework_module
    from verl.agent.framework.framework import OpenAICompatibleAgentFramework

    fake_tq = _FakeTransferQueue()
    replay_buffer = _FakeReplayBuffer()
    monkeypatch.setattr(framework_module, "tq", fake_tq)
    # Trajectory without response_logprobs (e.g. backend returned no logprobs).
    runtime = _FakeSessionRuntime({"session-0-0": [_trajectory(response_logprobs=None)]})

    async def agent_runner(*, raw_prompt, session, sample_index, tools_kwargs, **kwargs):
        return None

    framework = OpenAICompatibleAgentFramework(
        session_runtime=runtime,
        agent_runner=agent_runner,
        replay_buffer=replay_buffer,
        rollout_config={"n": 1, "val_kwargs": {"n": 1}},
    )


    await framework.generate_sequences(_build_prompts(count=1, global_steps=10))

    # rollout_log_probs is zero-filled rather than omitted so the trainer's
    # bypass-mode select_fields(["rollout_log_probs"]) never KeyErrors.
    rollout_log_probs = fake_tq.batch_puts[0]["fields"]["rollout_log_probs"]
    assert rollout_log_probs[0].tolist() == [0.0, 0.0]


@pytest.mark.asyncio
async def test_max_concurrent_sessions_caps_in_flight_sessions(monkeypatch):
    import asyncio

    from verl.agent.framework import framework as framework_module
    from verl.agent.framework.framework import OpenAICompatibleAgentFramework

    fake_tq = _FakeTransferQueue()
    replay_buffer = _FakeReplayBuffer()
    monkeypatch.setattr(framework_module, "tq", fake_tq)
    runtime = _FakeSessionRuntime(
        {f"session-{i}-0": [_trajectory()] for i in range(4)}
    )

    in_flight = 0
    max_observed = 0

    async def agent_runner(*, raw_prompt, session, sample_index, tools_kwargs, **kwargs):
        nonlocal in_flight, max_observed
        in_flight += 1
        max_observed = max(max_observed, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return None

    _install_fake_score(monkeypatch, default_score=1.0)
    framework = OpenAICompatibleAgentFramework(
        session_runtime=runtime,
        agent_runner=agent_runner,
        replay_buffer=replay_buffer,
        rollout_config={"n": 1, "val_kwargs": {"n": 1}},
        max_concurrent_sessions=2,
    )


    await framework.generate_sequences(_build_prompts(count=4, global_steps=10))

    assert max_observed <= 2


# ---------------------------------------------------------------------------
# _score_trajectories method-level tests
# ---------------------------------------------------------------------------


@pytest.fixture
def ray_runtime():
    import ray
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.mark.asyncio
async def test_score_trajectories_dispatches_only_final_trajectory_and_broadcasts(ray_runtime):
    """_score_trajectories scores trajectories[-1] only, broadcasts to all (matches AgentLoopWorkerTQ)."""
    import ray as ray_module
    from verl.agent.framework.framework import OpenAICompatibleAgentFramework
    from verl.agent.framework.types import Trajectory

    @ray_module.remote
    class _StubWorker:
        def __init__(self):
            self.calls = []

        def compute_score(self, data):
            self.calls.append(data)
            return {"reward_score": 0.42, "reward_extra_info": {"acc": 1.0, "format": 0.8}}

        def get_call_count(self):
            return len(self.calls)

    worker = _StubWorker.remote()

    runtime = _FakeSessionRuntime({})  # not used in this test
    framework = OpenAICompatibleAgentFramework(
        session_runtime=runtime,
        agent_runner=lambda **_: None,
        reward_loop_worker_handles=[worker],
        replay_buffer=_FakeReplayBuffer(),
        rollout_config={"n": 1, "val_kwargs": {"n": 1}},
    )

    trajectories = [
        Trajectory(prompt_ids=[1, 2], response_ids=[3, 4], response_mask=[1, 1], num_turns=1),
        Trajectory(prompt_ids=[5, 6], response_ids=[7, 8], response_mask=[1, 1], num_turns=2),
        Trajectory(prompt_ids=[9, 10], response_ids=[11, 12], response_mask=[1, 1], num_turns=3),
    ]
    sample_fields = {"data_source": "test", "raw_prompt": [{"role": "user", "content": "hi"}]}
    annotations = await framework._score_trajectories(trajectories, sample_fields)

    # Score-last + broadcast: 3 trajectories, but only 1 worker call
    assert ray_module.get(worker.get_call_count.remote()) == 1
    # All 3 trajectories get the same score and extra_info
    assert annotations == [
        (0.42, {"acc": 1.0, "format": 0.8}),
        (0.42, {"acc": 1.0, "format": 0.8}),
        (0.42, {"acc": 1.0, "format": 0.8}),
    ]
