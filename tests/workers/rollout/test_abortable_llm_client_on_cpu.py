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
"""CPU unit tests for AbortableLLMServerClient: per-request abort + in-flight cleanup.

These mock the Ray load balancer and server handles, so no GPU/vLLM is required.
"""

import asyncio
from types import SimpleNamespace

import pytest

from verl.workers.rollout.llm_server import AbortableLLMServerClient
from verl.workers.rollout.replica import TokenOutput


class _FakeRemoteMethod:
    """Mimics a Ray actor method: ``handle.method.remote(...)``."""

    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _FakeServer:
    """Fake vLLMHttpServer handle.

    ``generate`` blocks on ``release`` so the test can observe the in-flight state and
    issue an abort while the request is mid-flight. ``abort_request`` records its arg.
    """

    def __init__(self, stop_reason="aborted"):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.aborted = []
        self._stop_reason = stop_reason

        async def _generate(**kwargs):
            self.started.set()
            await self.release.wait()
            return TokenOutput(token_ids=[1, 2, 3], stop_reason=self._stop_reason, extra_fields={})

        def _abort(request_id):
            # Record synchronously (like a Ray .remote() dispatch) so fire-and-forget callers
            # that never await still register the abort. Return a resolved future so awaiting
            # callers (abort()) work too.
            self.aborted.append(request_id)
            fut = asyncio.get_running_loop().create_future()
            fut.set_result({"aborted": True})
            return fut

        self.generate = _FakeRemoteMethod(_generate)
        self.abort_request = _FakeRemoteMethod(_abort)


class _FakeLoadBalancer:
    def __init__(self, server):
        async def _acquire(request_id):
            return ("server-0", server)

        def _release(server_id):
            # Fire-and-forget in the client; return a non-awaitable to avoid "coroutine
            # never awaited" warnings.
            return None

        self.acquire_server = _FakeRemoteMethod(_acquire)
        self.release_server = _FakeRemoteMethod(_release)


def _make_client(server):
    # config is only touched by generate() on the priority!=0 path, which we never hit.
    return AbortableLLMServerClient(config=SimpleNamespace(), load_balancer_handle=_FakeLoadBalancer(server))


@pytest.mark.asyncio
async def test_record_abort_and_cleanup():
    """Dispatch records the request, abort targets the inner id, completion clears it."""
    server = _FakeServer(stop_reason="aborted")
    client = _make_client(server)

    task = asyncio.create_task(client.generate(request_id="req-1", prompt_ids=[1, 2], sampling_params={}))

    # Wait until the request is actually dispatched to the (fake) server.
    await asyncio.wait_for(server.started.wait(), timeout=1.0)
    assert "req-1" in client._inflight
    inner_request_id, recorded_server = client._inflight["req-1"]
    assert recorded_server is server

    # Abort the specific request; it must hit the inner (vLLM) request id, not the outer one.
    await client.abort("req-1")
    assert server.aborted == [inner_request_id]
    assert inner_request_id != "req-1"

    # Entry is only cleared once generate() returns through its finally clause.
    server.release.set()
    output = await asyncio.wait_for(task, timeout=1.0)
    assert output.stop_reason == "aborted"
    assert "req-1" not in client._inflight


@pytest.mark.asyncio
async def test_inflight_cleared_on_normal_completion():
    """A request that finishes normally is also removed from the in-flight table."""
    server = _FakeServer(stop_reason="stop")
    client = _make_client(server)

    task = asyncio.create_task(client.generate(request_id="req-2", prompt_ids=[1], sampling_params={}))
    await asyncio.wait_for(server.started.wait(), timeout=1.0)
    assert "req-2" in client._inflight

    server.release.set()
    await asyncio.wait_for(task, timeout=1.0)
    assert "req-2" not in client._inflight


@pytest.mark.asyncio
async def test_abort_unknown_request_is_noop():
    """Aborting an unknown/finished request neither raises nor calls the server."""
    server = _FakeServer()
    client = _make_client(server)

    await client.abort("does-not-exist")
    assert server.aborted == []


@pytest.mark.asyncio
async def test_cancellation_aborts_server_and_clears_inflight():
    """Cancelling generate() (e.g. timeout) aborts the server-side request, no leak."""
    server = _FakeServer()
    client = _make_client(server)

    task = asyncio.create_task(client.generate(request_id="req-3", prompt_ids=[1], sampling_params={}))
    await asyncio.wait_for(server.started.wait(), timeout=1.0)
    inner_request_id, _ = client._inflight["req-3"]

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # _on_cancel fired the abort to the server, and _on_complete cleared the entry.
    assert server.aborted == [inner_request_id]
    assert "req-3" not in client._inflight
