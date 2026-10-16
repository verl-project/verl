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
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

import verl.trainer.ppo.v1.agent_loop_tq as agent_loop_tq
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ, AgentLoopWorkerTQ, _settle_session_tasks


def _recording_worker(worker_id, dispatched):
    def remote(chunk):
        dispatched.append((worker_id, chunk["index"].tolist()))

    return SimpleNamespace(generate_sequences=SimpleNamespace(remote=remote))


def test_manager_rotates_workers_across_small_refills(monkeypatch):
    dispatched = []
    manager = object.__new__(AgentLoopManagerTQ)
    manager.agent_loop_workers = [_recording_worker(i, dispatched) for i in range(3)]
    manager._next_worker = 0
    monkeypatch.setattr(agent_loop_tq.ray, "get", lambda refs: refs)

    for start in range(4):
        manager.generate_sequences(TensorDict({"index": torch.tensor([start])}, batch_size=[1]))
    manager.generate_sequences(TensorDict({"index": torch.tensor([10, 11, 12])}, batch_size=[3]))

    assert dispatched == [(0, [0]), (1, [1]), (2, [2]), (0, [3]), (1, [10]), (2, [11]), (0, [12])]


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


def test_failed_session_writes_terminal_marker(monkeypatch):
    async def run():
        worker_cls = AgentLoopWorkerTQ.__ray_metadata__.modified_class
        worker = object.__new__(worker_cls)

        async def fail(*_args, **_kwargs):
            raise RuntimeError("session failed")

        puts = []

        async def put(**kwargs):
            puts.append(kwargs)

        worker._run_agent_loop = fail
        monkeypatch.setattr(agent_loop_tq.tq, "async_kv_put", put)

        with pytest.raises(RuntimeError, match="session failed"):
            await worker._run_session(
                uid="prompt",
                session_id=2,
                partition_id="train",
                sampling_params={},
                trajectory={},
                trace=False,
                prompt={},
            )

        assert puts == [
            {
                "key": "prompt_2",
                "partition_id": "train",
                "tag": {"is_session": True, "status": "running", "session_id": 2},
            },
            {
                "key": "prompt_2",
                "partition_id": "train",
                "tag": {"is_session": True, "status": "failure", "session_id": 2},
            },
        ]

    asyncio.run(run())


def test_successful_session_writes_metadata_after_trajectories(monkeypatch):
    async def run():
        worker_cls = AgentLoopWorkerTQ.__ray_metadata__.modified_class
        worker = object.__new__(worker_cls)
        order = []

        async def succeed(*_args, **_kwargs):
            order.append("trajectory")

        async def put(**kwargs):
            order.append(kwargs)

        worker._run_agent_loop = succeed
        monkeypatch.setattr(agent_loop_tq.tq, "async_kv_put", put)

        await worker._run_session(
            uid="prompt",
            session_id=1,
            partition_id="train",
            sampling_params={},
            trajectory={},
            trace=False,
            prompt={},
        )

        assert order == [
            {
                "key": "prompt_1",
                "partition_id": "train",
                "tag": {"is_session": True, "status": "running", "session_id": 1},
            },
            "trajectory",
            {
                "key": "prompt_1",
                "partition_id": "train",
                "tag": {"is_session": True, "status": "finished", "session_id": 1},
            },
        ]

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

    monkeypatch.setattr(agent_loop_tq.tq, "async_kv_batch_put", async_kv_batch_put)
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
