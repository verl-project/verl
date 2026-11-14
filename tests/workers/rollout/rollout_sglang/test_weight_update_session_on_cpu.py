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

"""SGLang weight-update session calls: sent once, and skipped on servers that lack them."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("sglang")

import aiohttp  # noqa: E402

from verl.workers.rollout.sglang_rollout import http_server_engine  # noqa: E402
from verl.workers.rollout.sglang_rollout.http_server_engine import AsyncHttpServerAdapter  # noqa: E402
from verl.workers.rollout.sglang_rollout.sglang_rollout import ServerAdapter  # noqa: E402


def _http_adapter():
    adapter = object.__new__(AsyncHttpServerAdapter)
    adapter.node_rank = 0
    adapter.max_attempts = 3
    adapter.retry_delay = 0.0
    adapter.server_args = SimpleNamespace(host="127.0.0.1", port=1)
    return adapter


@pytest.mark.parametrize(
    ("call", "endpoint"),
    [
        (lambda adapter: adapter.begin_weight_update(), "begin_weight_update"),
        (lambda adapter: adapter.end_weight_update(), "end_weight_update"),
    ],
)
def test_session_calls_are_sent_once_with_their_own_timeout(call, endpoint):
    adapter = _http_adapter()
    adapter._make_async_request = AsyncMock(return_value={"success": True})

    asyncio.run(call(adapter))

    request = adapter._make_async_request.await_args
    assert request.args[0] == endpoint
    assert request.kwargs["max_attempts"] == 1
    assert request.kwargs["timeout"] == http_server_engine.WEIGHT_UPDATE_SESSION_TIMEOUT


def test_single_attempt_request_is_not_resent_after_a_timeout():
    adapter = _http_adapter()
    opened = []

    @asynccontextmanager
    async def timing_out_session():
        opened.append(1)
        raise asyncio.TimeoutError
        yield  # pragma: no cover

    adapter._get_session = timing_out_session

    with pytest.raises(RuntimeError, match="after 1 attempts"):
        asyncio.run(adapter._make_async_request("end_weight_update", max_attempts=1))
    assert len(opened) == 1


def _http_error(status):
    return aiohttp.ClientResponseError(request_info=Mock(), history=(), status=status)


def _rollout(engine):
    return SimpleNamespace(_engine=engine, _is_server_tp_leader=lambda: True, _weight_update_sessions_supported=True)


async def _run_session(rollout, events):
    async with ServerAdapter._weight_update_session(rollout):
        events.append("bucket")


def test_session_wraps_the_refit_when_the_server_supports_it():
    events = []
    engine = SimpleNamespace(
        begin_weight_update=AsyncMock(side_effect=lambda: events.append("begin")),
        end_weight_update=AsyncMock(side_effect=lambda: events.append("end")),
    )

    asyncio.run(_run_session(_rollout(engine), events))

    assert events == ["begin", "bucket", "end"]


def test_session_falls_back_when_the_server_has_no_session_endpoints():
    events = []
    engine = SimpleNamespace(begin_weight_update=AsyncMock(side_effect=_http_error(404)), end_weight_update=AsyncMock())
    rollout = _rollout(engine)

    asyncio.run(_run_session(rollout, events))
    asyncio.run(_run_session(rollout, events))

    # The buckets still go out, without a session and without probing the server again.
    assert events == ["bucket", "bucket"]
    assert rollout._weight_update_sessions_supported is False
    assert engine.begin_weight_update.await_count == 1
    engine.end_weight_update.assert_not_awaited()


def test_session_rejection_is_not_mistaken_for_a_missing_endpoint():
    events = []
    engine = SimpleNamespace(begin_weight_update=AsyncMock(side_effect=_http_error(400)), end_weight_update=AsyncMock())

    with pytest.raises(aiohttp.ClientResponseError):
        asyncio.run(_run_session(_rollout(engine), events))
    assert events == []
