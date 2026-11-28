# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""ALFWorld integration lifecycle and measured tail selection without downloading environments."""

import asyncio
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from examples.group_delivery import alfworld_agent as agent
from examples.group_delivery import alfworld_server
from examples.group_delivery.alfworld_server import create_app
from examples.group_delivery.prepare_alfworld import group_durations, select_long_tail, tail_report
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop


class Environment:
    """A stateful two-action environment, deliberately not resettable per tool call."""

    def __init__(self, fail=False):
        self.steps, self.closed, self.fail = 0, False, fail

    def reset(self):
        return ["initial"], {"admissible_commands": [["look"]]}

    def step(self, actions):
        if self.fail:
            raise RuntimeError("environment crashed")
        self.steps += 1
        return (
            [f"step {self.steps}"],
            [0.0],
            [self.steps == 2],
            {"won": [self.steps == 2], "admissible_commands": [["look"]]},
        )

    def close(self):
        self.closed = True


@pytest.mark.parametrize("fail", [False, True])
def test_episode_owns_environment_across_tool_calls_and_closes_on_error(tmp_path, monkeypatch, fail):
    (tmp_path / "game.tw-pddl").write_text("{}")
    environments = []

    def factory(*args):
        env = Environment(fail)
        environments.append(env)
        return env

    with TestClient(create_app(tmp_path, environment_factory=factory)) as client:

        async def request(method, path, **kwargs):
            response = client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()

        async def parent_run(self, sampling_params, **kwargs):
            tool = agent.AlfworldTool()
            data = SimpleNamespace(extra_fields={})
            for _ in range(2):
                session_id, _ = await tool.create(**kwargs["tools_kwargs"]["alfworld"])
                await tool.execute(session_id, {"action": "look"}, agent_data=data)
                await tool.release(session_id)
                assert not environments[0].closed
            return AgentLoopOutput(
                prompt_ids=[1],
                response_ids=[2, 3, 4],
                response_mask=[1, 0, 1],
                num_turns=4,
                metrics={},
                extra_fields=data.extra_fields,
            )

        monkeypatch.setattr(agent, "request", request)
        monkeypatch.setattr(ToolAgentLoop, "run", parent_run)
        loop = object.__new__(agent.AlfworldAgentLoop)
        call = loop.run({}, raw_prompt=[], extra_info={"game_file": "game.tw-pddl"}, uid="a", session_id=0)
        if fail:
            with pytest.raises(RuntimeError, match="environment crashed"):
                asyncio.run(call)
        else:
            output = asyncio.run(call)
            assert output.reward_score == 1
            assert output.response_mask == [1, 0, 1]
            assert output.extra_fields["reward_extra_info"]["steps"] == 2
        assert len(environments) == 1 and environments[0].closed


def test_sessions_are_isolated_and_cannot_open_paths_outside_data_root(tmp_path):
    (tmp_path / "game.tw-pddl").write_text("{}")
    with TestClient(create_app(tmp_path, environment_factory=lambda *args: Environment())) as client:
        ids = [client.post("/sessions", json={"game_file": "game.tw-pddl"}).json()["session_id"] for _ in range(2)]
        client.post(f"/sessions/{ids[0]}/step", json={"action": "look"})
        assert client.get(f"/sessions/{ids[1]}").json()["steps"] == 0
        assert client.post("/sessions", json={"game_file": "../game.tw-pddl"}).status_code == 400
        for session_id in ids:
            client.delete(f"/sessions/{session_id}")
            assert client.get(f"/sessions/{session_id}").status_code == 404


def test_expired_sessions_are_evicted_on_the_next_open(tmp_path, monkeypatch):
    (tmp_path / "game.tw-pddl").write_text("{}")
    environments = []

    def factory(*args):
        env = Environment()
        environments.append(env)
        return env

    monkeypatch.setattr(alfworld_server, "SESSION_TTL_S", 0.2)
    with TestClient(alfworld_server.create_app(tmp_path, environment_factory=factory)) as client:
        first = client.post("/sessions", json={"game_file": "game.tw-pddl"}).json()["session_id"]
        time.sleep(0.3)
        second = client.post("/sessions", json={"game_file": "game.tw-pddl"}).json()["session_id"]
        assert client.get(f"/sessions/{first}").status_code == 404
        assert client.get(f"/sessions/{second}").status_code == 200
    assert environments[0].closed


def test_tail_selection_requires_measured_latency_and_freezes_real_games():
    rows = [{"extra_info": {"game_file": f"game{i}"}} for i in range(100)]
    completed = [(f"game{i}", 10.0 if i < 90 else 150.0) for i in range(100)]
    selected, report = select_long_tail(rows, completed, count=100, seed=42)
    assert len(selected) == 100
    assert report["p95_over_p50"] == 15.0
    assert all(row in rows for row in selected)
    with pytest.raises(ValueError, match="not sufficiently long-tailed"):
        select_long_tail(rows, [(f"game{i}", 10.0) for i in range(100)], count=100, seed=42)


def test_group_tail_uses_slowest_session_and_rejects_incomplete_groups():
    records = [
        dict(uid="a", session=0, game_file="game1", duration_s=10),
        dict(uid="a", session=1, game_file="game1", duration_s=30),
        dict(uid="b", session=0, game_file="game2", duration_s=100),
    ]
    completed = group_durations(records + [records[0]])
    assert completed == [("game1", 30)]
    assert tail_report(completed)["p50_s"] == 30
