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

import verl.trainer.ppo.v1.agent_loop_tq as agent_loop_tq_module
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ, _settle_session_tasks


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


_WorkerTQ = AgentLoopWorkerTQ.__ray_metadata__.modified_class


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

    monkeypatch.setattr(agent_loop_tq_module.tq, "async_kv_batch_put", async_kv_batch_put)
    await _WorkerTQ._agent_loop_postprocess(
        worker, output, validate, uid="u0", session_id=0, global_steps=1, raw_prompt=[{"role": "user", "content": "hi"}]
    )
    return captured["fields"]


@pytest.mark.asyncio
async def test_agent_loop_tq_postprocess_stores_unpadded_rollout_topk(monkeypatch):
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


@pytest.mark.asyncio
async def test_agent_loop_tq_postprocess_skips_rollout_topk_on_validate(monkeypatch):
    fields = await _postprocess(monkeypatch, _DummyWorker(topk_log_probs=2), _output({}), validate=True)

    assert "rollout_topk_ids" not in fields.keys()
    assert "rollout_topk_log_probs" not in fields.keys()


def test_prompt_dispatch_restores_each_session_and_releases_live_states(monkeypatch):
    from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient

    async def fake_put(**kwargs):
        pass

    monkeypatch.setattr("verl.trainer.ppo.v1.agent_loop_tq.tq.async_kv_put", fake_put)
    worker_cls = AgentLoopWorkerTQ.__ray_actor_class__
    worker = worker_cls.__new__(worker_cls)
    worker.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"n": 2}}})
    worker.llm_client = FullyAsyncLLMServerClient(config=worker.config)
    worker.partial_rollout_states = {}
    received = {}

    async def run_session(params, *, session_id, **kwargs):
        received[session_id] = kwargs["partial_rollout_state"]
        kwargs["partial_rollout_checkpoint_callback"]({"session": session_id})
        assert worker.partial_rollout_states["prompt"][str(session_id)] == {"session": session_id}

    worker._run_agent_loop = run_session
    asyncio.run(
        worker._run_prompt(
            {"uid": "prompt", "agent_name": "single_turn_agent", "partial_rollout_states": {"0": {"prefix": [7]}}},
            {},
            {"validate": False},
        )
    )
    assert received == {0: {"prefix": [7]}, 1: None}
    assert worker.partial_rollout_states == {}
