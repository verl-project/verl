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

"""Exercise reward HTTP timeouts against a real local server, without models."""

import asyncio

import aiohttp
import pytest
import pytest_asyncio
from aiohttp import web
from omegaconf import OmegaConf

from verl.experimental.reward_loop.reward_loop import RewardLoopWorker


@pytest_asyncio.fixture
async def reward_server():
    """Serve healthy, stalled, and failing reward responses on an ephemeral port."""
    state = {"calls": 0, "behavior": "healthy", "headers_only": False}
    started = asyncio.Event()
    release = asyncio.Event()

    async def handle(request):
        """Emulate a reward backend that can stall before headers or in its body."""
        state["calls"] += 1
        behavior = state["behavior"]
        if behavior == "hang" or (behavior == "hang_once" and state["calls"] == 1):
            if state["headers_only"]:
                response = web.StreamResponse(headers={"Content-Type": "application/json"})
                await response.prepare(request)
            started.set()
            await release.wait()
        if behavior == "client_error":
            raise web.HTTPBadRequest()
        if behavior == "server_error_once" and state["calls"] == 1:
            raise web.HTTPServiceUnavailable()
        return web.json_response({"score": 1.0})

    app = web.Application()
    app.router.add_post("/classify", handle)
    runner = web.AppRunner(app, shutdown_timeout=0.1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0]
    worker = object.__new__(RewardLoopWorker)
    worker.config = OmegaConf.create({"reward": {"reward_model": {"request_timeout": 0.1}}})
    worker.reward_router_address = f"{host}:{port}"
    try:
        yield worker, state, started
    finally:
        release.set()
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("headers_only", [False, True])
async def test_stalled_request_exhausts_timeout(reward_server, headers_only):
    """Both a missing response and a stalled JSON body must reach the retry limit."""
    worker, state, _ = reward_server
    state.update(behavior="hang", headers_only=headers_only)
    task = asyncio.create_task(worker._post_request({}, "classify", max_retries=1))
    try:
        done, _ = await asyncio.wait({task}, timeout=2)
        assert task in done, "reward request never reached its configured timeout"
        with pytest.raises(asyncio.TimeoutError):
            await task
        assert state["calls"] == 1
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("behavior", ["hang_once", "server_error_once"])
async def test_request_recovers_on_retry(reward_server, behavior):
    """A transient stalled backend or HTTP 503 must be retried successfully."""
    worker, state, _ = reward_server
    state["behavior"] = behavior
    result = await asyncio.wait_for(worker._post_request({}, "classify", max_retries=2), timeout=4)
    assert result == {"score": 1.0}
    assert state["calls"] == 2


@pytest.mark.asyncio
async def test_client_error_is_not_retried(reward_server):
    """Permanent client errors must still propagate after one attempt."""
    worker, state, _ = reward_server
    state["behavior"] = "client_error"
    with pytest.raises(aiohttp.ClientResponseError, match="400"):
        await worker._post_request({}, "classify", max_retries=2)
    assert state["calls"] == 1


@pytest.mark.asyncio
async def test_cancellation_is_not_retried(reward_server):
    """Cancelling an in-flight reward request must propagate without a retry."""
    worker, state, started = reward_server
    state["behavior"] = "hang"
    worker.config.reward.reward_model.request_timeout = 10
    task = asyncio.create_task(worker._post_request({}, "classify", max_retries=2))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert state["calls"] == 1
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
async def test_invalid_timeout_fails_before_request(reward_server, timeout):
    """Reject timeout values that would disable the deadline or make it invalid."""
    worker, state, _ = reward_server
    worker.config.reward.reward_model.request_timeout = timeout
    with pytest.raises(ValueError, match="request_timeout"):
        await worker._post_request({}, "classify", max_retries=1)
    assert state["calls"] == 0
