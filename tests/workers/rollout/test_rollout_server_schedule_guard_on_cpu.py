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

import asyncio
from types import SimpleNamespace

import pytest

from verl.workers.rollout import replica


def _fake_server(hex_id: str, ready_after_s: float | None):
    async def ready():
        if ready_after_s is None:
            await asyncio.Event().wait()
        await asyncio.sleep(ready_after_s)

    return SimpleNamespace(
        _actor_id=SimpleNamespace(hex=lambda: hex_id),
        __ray_ready__=SimpleNamespace(remote=ready),
    )


def _state_api_must_not_be_called(*_args, **_kwargs):
    raise AssertionError("State API queried although every actor was ready")


@pytest.mark.asyncio
async def test_returns_once_every_actor_is_ready(monkeypatch):
    monkeypatch.setattr(replica, "get_actor", _state_api_must_not_be_called)
    servers = [_fake_server("a", 0.0), _fake_server("b", 0.02)]
    await replica.wait_rollout_servers_scheduled(servers, ["n0", "n1"], timeout_s=5.0)


@pytest.mark.asyncio
async def test_raises_with_state_api_diagnostic_when_unschedulable(monkeypatch):
    info = {
        "name": "vllm_server_0_0",
        "state": "PENDING_CREATION",
        "node_id": "",
        "required_resources": {"GPU": 1.0},
    }
    monkeypatch.setattr(replica, "get_actor", lambda _id, timeout: info)
    servers = [_fake_server("a", 0.0), _fake_server("b", None)]
    with pytest.raises(RuntimeError) as excinfo:
        await replica.wait_rollout_servers_scheduled(servers, ["n0", "n1"], timeout_s=0.05)
    msg = str(excinfo.value)
    assert "vllm_server_0_0 target_node=n1 state=PENDING_CREATION" in msg
    assert "required_resources={'GPU': 1.0}" in msg


@pytest.mark.asyncio
async def test_raises_with_actor_id_when_state_api_is_unavailable(monkeypatch):
    def no_dashboard(_id, timeout):
        raise ConnectionError("Could not read 'dashboard' from GCS")

    monkeypatch.setattr(replica, "get_actor", no_dashboard)
    with pytest.raises(RuntimeError, match="b target_node=n0 state=unknown"):
        await replica.wait_rollout_servers_scheduled([_fake_server("b", None)], ["n0"], timeout_s=0.05)


@pytest.mark.asyncio
async def test_raises_init_failure_without_waiting_for_pending_actors(monkeypatch):
    monkeypatch.setattr(replica, "get_actor", _state_api_must_not_be_called)

    async def failing_ready():
        raise ValueError("bad rollout config")

    failing = SimpleNamespace(
        _actor_id=SimpleNamespace(hex=lambda: "a"),
        __ray_ready__=SimpleNamespace(remote=failing_ready),
    )
    loop = asyncio.get_running_loop()
    start = loop.time()
    with pytest.raises(ValueError, match="bad rollout config"):
        await replica.wait_rollout_servers_scheduled([failing, _fake_server("b", None)], ["n0", "n1"], timeout_s=30.0)
    assert loop.time() - start < 5.0
