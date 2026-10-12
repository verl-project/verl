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
"""Fail-fast guard for unschedulable rollout server actors.

A colocated rollout server actor is hard-pinned to its training worker's node; when that
node has no free slot the actor stays PENDING_CREATION and the following ``.remote()`` call
would block forever. ``wait_rollout_servers_scheduled`` bounds that wait.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from verl.workers.rollout import replica


def _fake_server(hex_id: str):
    return SimpleNamespace(_actor_id=SimpleNamespace(hex=lambda: hex_id))


@pytest.mark.asyncio
async def test_returns_once_actor_is_alive(monkeypatch):
    monkeypatch.setattr(replica, "get_actor", lambda _id: {"state": "ALIVE"})
    await replica.wait_rollout_servers_scheduled([_fake_server("a")], timeout_s=5.0, poll_interval_s=0.01)


@pytest.mark.asyncio
async def test_waits_through_transient_pending(monkeypatch):
    states = iter(["PENDING_CREATION", "PENDING_CREATION", "ALIVE"])
    monkeypatch.setattr(replica, "get_actor", lambda _id: {"state": next(states)})
    await replica.wait_rollout_servers_scheduled([_fake_server("a")], timeout_s=5.0, poll_interval_s=0.01)


@pytest.mark.asyncio
async def test_raises_with_diagnostic_when_unschedulable(monkeypatch):
    info = {
        "name": "vllm_server_0_0",
        "state": "PENDING_CREATION",
        "node_id": "node0",
        "required_resources": {"GPU": 1.0},
        "actor_id": "a",
    }
    monkeypatch.setattr(replica, "get_actor", lambda _id: info)
    with pytest.raises(RuntimeError) as excinfo:
        await replica.wait_rollout_servers_scheduled([_fake_server("a")], timeout_s=0.05, poll_interval_s=0.01)
    msg = str(excinfo.value)
    assert "vllm_server_0_0" in msg
    assert "PENDING_CREATION" in msg
    assert "required_resources" in msg


@pytest.mark.asyncio
async def test_treats_missing_actor_as_unschedulable(monkeypatch):
    monkeypatch.setattr(replica, "get_actor", lambda _id: None)
    with pytest.raises(RuntimeError):
        await replica.wait_rollout_servers_scheduled([_fake_server("a")], timeout_s=0.05, poll_interval_s=0.01)
