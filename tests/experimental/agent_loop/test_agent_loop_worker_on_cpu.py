# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import asyncio

import pytest
import torch
from omegaconf import OmegaConf

import verl.experimental.agent_loop.agent_loop as agent_loop_module
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopMetrics,
    AgentLoopOutput,
    AgentLoopWorker,
    _settle_session_tasks,
    apply_greedy_sampling_params,
)


def test_settle_session_tasks_waits_for_siblings_after_failure():
    async def run():
        settled = asyncio.Event()

        async def fail():
            raise RuntimeError("session failed")

        async def finish_later():
            await asyncio.sleep(0.01)
            settled.set()

        tasks = [asyncio.create_task(fail()), asyncio.create_task(finish_later())]
        errors = await _settle_session_tasks(tasks)

        assert settled.is_set()
        assert all(task.done() for task in tasks)
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)

    asyncio.run(run())


class _DummyWorker:
    _compute_multi_modal_inputs = AgentLoopWorker._compute_multi_modal_inputs
    _compute_position_ids = AgentLoopWorker._compute_position_ids
    _compute_score = AgentLoopWorker._compute_score
    _compute_teacher_logprobs = AgentLoopWorker._compute_teacher_logprobs
    distillation_enabled = False

    def __init__(self, topk_log_probs: int):
        self.rollout_config = OmegaConf.create({"topk_log_probs": topk_log_probs})
        self.processor = None
        self.reward_loop_worker_handles = None


def _output(extra_fields: dict) -> AgentLoopOutput:
    return AgentLoopOutput(
        prompt_ids=[101, 102, 103],
        response_ids=[11, 12],
        response_mask=[1, 1],
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields=extra_fields,
    )


async def _postprocess(monkeypatch, worker: _DummyWorker, output: AgentLoopOutput, validate: bool):
    captured = {}

    async def async_kv_batch_put(*, keys, fields, tags, partition_id):
        captured["fields"] = fields

    monkeypatch.setattr(agent_loop_module.tq, "async_kv_batch_put", async_kv_batch_put)
    await AgentLoopWorker._agent_loop_postprocess(
        worker, output, validate, uid="u0", session_id=0, global_steps=1, raw_prompt=[{"role": "user", "content": "hi"}]
    )
    return captured["fields"]


@pytest.mark.asyncio
async def test_agent_loop_postprocess_stores_unpadded_rollout_topk(monkeypatch):
    output = _output(
        {"response_topk_ids": [[11, 13], [12, 14]], "response_topk_log_probs": [[-0.1, -2.0], [-0.2, -1.5]]}
    )

    fields = await _postprocess(monkeypatch, _DummyWorker(topk_log_probs=2), output, validate=False)

    ids, log_probs = fields["rollout_topk_ids"][0], fields["rollout_topk_log_probs"][0]
    assert ids.shape == (5, 2) and ids.dtype == torch.int32
    assert log_probs.shape == (5, 2) and log_probs.dtype == torch.float32
    assert ids[2].tolist() == [11, 13] and ids[3].tolist() == [12, 14]
    assert log_probs[2].tolist() == pytest.approx([-0.1, -2.0]) and log_probs[3].tolist() == pytest.approx([-0.2, -1.5])
    assert ids[0].tolist() == [0, 1] and ids[4].tolist() == [0, 1]
    assert "response_topk_ids" not in fields["extra_fields"][0]
    assert fields["rollout_support_complete"][0].tolist() == [True, True, False, False, True]


@pytest.mark.asyncio
async def test_agent_loop_postprocess_stores_nested_provider_coverage(monkeypatch):
    output = _output(
        {
            "response_topk_ids": [[11, 13], [12, 14]],
            "response_topk_log_probs": [[0.0, float("-inf")], [-0.2, -1.5]],
            "response_support_complete": [True, False],
        }
    )
    short_output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[11],
        response_mask=[1],
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields={
            "response_topk_ids": [[11, 13]],
            "response_topk_log_probs": [[-0.1, -2.0]],
            "response_support_complete": [False],
        },
    )
    fields = await _postprocess(monkeypatch, _DummyWorker(topk_log_probs=2), [output, short_output], validate=False)
    complete = fields["rollout_support_complete"]
    assert complete.is_nested and complete.dtype == torch.bool
    assert complete[0].tolist() == [True, True, True, False, True]
    assert complete[1].tolist() == [True, False, True]


def test_greedy_reward_baseline_does_not_request_score_centering_heads():
    params = {"top_p": 0.9, "top_k": 64, "temperature": 1.0, "topk_log_probs": 128, "logprobs": True}
    apply_greedy_sampling_params(params)
    assert params == {"top_p": 1.0, "top_k": -1, "temperature": 0, "logprobs": True}


@pytest.mark.asyncio
async def test_agent_loop_postprocess_skips_rollout_topk_on_validate(monkeypatch):
    fields = await _postprocess(monkeypatch, _DummyWorker(topk_log_probs=2), _output({}), validate=True)

    assert "rollout_topk_ids" not in fields.keys()
    assert "rollout_topk_log_probs" not in fields.keys()


@pytest.mark.asyncio
@pytest.mark.parametrize("validate,n", [(False, 2), (True, 1)])
async def test_generate_sequences_publishes_sessions_to_tq(monkeypatch, validate, n):
    from tensordict import NonTensorData

    from verl.utils.tensordict_utils import list_of_dict_to_tensordict

    worker = object.__new__(AgentLoopWorker)
    worker.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "n": 2,
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "top_k": -1,
                    "calculate_log_probs": True,
                    "topk_log_probs": 2,
                    "val_kwargs": {"n": 1, "temperature": 0.0, "top_p": 1.0, "top_k": -1},
                    "agent": {"default_agent_loop": "single_turn_agent"},
                }
            }
        }
    )
    worker.background_tasks = set()
    events, sessions = [], []

    async def put(*, key, partition_id, tag):
        events.append((key, partition_id, tag["status"]))

    async def run(sampling_params, *, trajectory, trace, session_id, **prompt):
        await asyncio.sleep(0)
        sessions.append((prompt["uid"], session_id, sampling_params.copy(), prompt["agent_name"]))
        events.append((prompt["uid"], "val" if validate else "train", "published"))

    monkeypatch.setattr(agent_loop_module.tq, "async_kv_put", put)
    worker._run_agent_loop = run
    batch = list_of_dict_to_tensordict(
        [{"uid": uid, "index": i, "global_steps": 3} for i, uid in enumerate(["a", "b"])]
    )
    batch["validate"] = NonTensorData(validate)
    assert await worker.generate_sequences(batch) is None
    await asyncio.gather(*worker.background_tasks)
    assert {(uid, sid) for uid, sid, _, _ in sessions} == {(uid, sid) for uid in ["a", "b"] for sid in range(n)}
    for uid in ["a", "b"]:
        assert [status for key, _, status in events if key == uid] == ["running"] + ["published"] * n + ["finished"]
    assert all(partition == ("val" if validate else "train") for _, partition, _ in events)
    for _, _, params, agent_name in sessions:
        assert agent_name == "single_turn_agent"
        assert params["temperature"] == (0.0 if validate else 0.8)
        assert ("topk_log_probs" in params) == (not validate)


@pytest.mark.asyncio
async def test_failed_prompt_waits_for_remaining_session_publication(monkeypatch):
    worker = object.__new__(AgentLoopWorker)
    worker.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"n": 2}}})
    events = []

    async def put(*, key, partition_id, tag):
        events.append(tag["status"])

    async def run(*args, session_id, **kwargs):
        if session_id == 0:
            raise RuntimeError("session failed")
        await asyncio.sleep(0.01)
        events.append("published")

    monkeypatch.setattr(agent_loop_module.tq, "async_kv_put", put)
    worker._run_agent_loop = run
    await worker._run_prompt({"uid": "a"}, {}, {"validate": False})
    assert events == ["running", "published", "failure"]


def test_manager_dispatches_tensordict_without_waiting_for_trajectories(monkeypatch):
    from types import SimpleNamespace

    from verl.experimental.agent_loop import AgentLoopManager
    from verl.utils.tensordict_utils import list_of_dict_to_tensordict

    chunks = []
    manager = object.__new__(AgentLoopManager)
    manager.agent_loop_workers = [
        SimpleNamespace(generate_sequences=SimpleNamespace(remote=chunks.append)) for _ in range(4)
    ]
    monkeypatch.setattr(agent_loop_module.ray, "get", lambda refs: None)
    batch = list_of_dict_to_tensordict([{"uid": "a"}, {"uid": "b"}])
    assert manager.generate_sequences(batch) is None
    assert sum(len(chunk) for chunk in chunks) == len(batch)
    assert [chunk["uid"][0] for chunk in chunks] == ["a", "b"]


@pytest.mark.asyncio
async def test_postprocess_publishes_unpadded_multi_output_session(monkeypatch):
    worker = _DummyWorker(topk_log_probs=0)
    outputs = [
        _output({"min_global_steps": 1, "max_global_steps": 2}),
        AgentLoopOutput(
            prompt_ids=[101],
            response_ids=[11],
            response_mask=[1],
            num_turns=2,
            metrics=AgentLoopMetrics(),
            extra_fields={},
        ),
    ]
    captured = {}

    async def score(outputs, kwargs):
        outputs[-1].reward_score = 0.5
        outputs[-1].extra_fields["reward_extra_info"] = {"acc": 0.5}

    async def put(**kwargs):
        captured.update(kwargs)

    worker._compute_score = score
    monkeypatch.setattr(agent_loop_module.tq, "async_kv_batch_put", put)
    await AgentLoopWorker._agent_loop_postprocess(
        worker,
        outputs,
        False,
        uid="u",
        session_id=2,
        global_steps=3,
        multi_modal_data={"images": []},
    )
    assert captured["keys"] == ["u_2_0", "u_2_1"]
    assert captured["partition_id"] == "train"
    fields = captured["fields"]
    assert "multi_modal_data" not in fields
    for i, expected_length in enumerate([5, 2]):
        assert fields["input_ids"][i].numel() == expected_length
        assert fields["position_ids"][i].numel() == expected_length
        torch.testing.assert_close(fields["loss_mask"][i], fields["response_mask"][i])
        assert fields["rm_scores"][i].sum().item() == 0.5
    assert captured["tags"][0] == {
        "status": "success",
        "prompt_len": 3,
        "response_len": 2,
        "seq_len": 5,
        "global_steps": 3,
        "min_global_steps": 1,
        "max_global_steps": 2,
    }
