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

from verl.trainer.ppo.v1.agent_loop_tq import _settle_session_tasks


def test_settle_session_tasks_waits_for_siblings_after_failure():
    async def run():
        settled = asyncio.Event()

        async def fail():
            raise RuntimeError("session failed")

        async def finish_later():
            await asyncio.sleep(0.01)
            settled.set()

        tasks = [asyncio.create_task(fail()), asyncio.create_task(finish_later())]
        errors, written = await _settle_session_tasks(tasks)

        assert settled.is_set()
        assert all(task.done() for task in tasks)
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
        assert written == 0

    asyncio.run(run())


def test_settle_session_tasks_counts_written_trajectories():
    """Sessions return how many trajectories they wrote; empty-output sessions return 0 and are not errors."""

    async def run():
        async def wrote(n):
            return n

        tasks = [asyncio.create_task(wrote(2)), asyncio.create_task(wrote(0)), asyncio.create_task(wrote(1))]
        errors, written = await _settle_session_tasks(tasks)
        assert errors == []
        assert written == 3

    asyncio.run(run())


def _run_prompt_status(session_results):
    """Drive AgentLoopWorkerTQ._run_prompt with stubbed sessions; return the terminal group status it published."""
    from unittest import mock

    from verl.trainer.ppo.v1 import agent_loop_tq as mod

    published = []

    async def fake_kv_put(key, partition_id, tag):
        published.append((key, partition_id, dict(tag)))

    results = list(session_results)

    async def fake_run_agent_loop(self, *args, **kwargs):
        r = results[kwargs["session_id"]]
        if isinstance(r, BaseException):
            raise r
        return r

    worker = mod.AgentLoopWorkerTQ.__new__(mod.AgentLoopWorkerTQ)
    worker.config = mock.Mock()
    worker.config.actor_rollout_ref.rollout.n = len(results)
    prompt = {"uid": "u1"}
    trajectory = {"validate": False}
    with (
        mock.patch.object(mod.tq, "async_kv_put", fake_kv_put),
        mock.patch.object(mod.AgentLoopWorkerTQ, "_run_agent_loop", fake_run_agent_loop),
    ):
        asyncio.run(worker._run_prompt(prompt, {}, trajectory=trajectory))
    terminal = [t for k, p, t in published if k == "u1" and t["status"] != "running"]
    assert len(terminal) == 1
    return terminal[0]["status"]


def test_run_prompt_group_with_data_is_finished():
    assert _run_prompt_status([1, 1, 0, 1]) == "finished"


def test_run_prompt_group_where_every_session_returned_nothing_is_failed():
    """DAPO filter_groups reads the group metric off finished groups; a finished group with zero
    trajectories has none and would raise. Such a group is a failed group: cleared and refilled."""
    assert _run_prompt_status([0, 0, 0, 0]) == "failure"


def test_run_prompt_group_with_session_error_is_failed():
    assert _run_prompt_status([1, RuntimeError("boom"), 1]) == "failure"
