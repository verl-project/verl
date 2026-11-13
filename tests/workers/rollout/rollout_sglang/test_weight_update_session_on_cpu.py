# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""Exercise real rollout session boundaries without a server or accelerator."""

import asyncio
import importlib.util
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import aiohttp
import pytest
import requests

if importlib.util.find_spec("sglang") is None:
    pytest.skip("SGLang is optional in the CPU test environment", allow_module_level=True)

from verl.workers.rollout.sglang_rollout import http_server_engine as http_engine  # noqa: E402
from verl.workers.rollout.sglang_rollout import sglang_rollout as rollout  # noqa: E402
from verl.workers.rollout.sglang_rollout.http_server_engine import (  # noqa: E402
    AsyncHttpServerAdapter,
    HttpServerAdapter,
)


@pytest.fixture
def supports_sessions(request, monkeypatch):
    """Model the optional upstream session API independently of installed SGLang."""
    # Model the upstream EngineBase contract, independently of installed SGLang.
    base = type("EngineBase", (), {})
    enabled = getattr(request, "param", True)
    if enabled:
        base.begin_weight_update = lambda self: None
        base.end_weight_update = lambda self: None
    monkeypatch.setattr(rollout, "EngineBase", base)
    return enabled


def test_http_session_methods():
    """Check both concrete adapters and their session request arguments."""
    assert not HttpServerAdapter.__abstractmethods__
    assert not AsyncHttpServerAdapter.__abstractmethods__
    sync = HttpServerAdapter.__new__(HttpServerAdapter)
    sync._make_request = Mock(return_value={"success": True})
    sync.begin_weight_update("target")
    sync.end_weight_update()
    assert sync._make_request.call_args_list == [
        (("begin_weight_update", {"selector": "target"}), {"max_attempts": 1}),
        (("end_weight_update", {}),),
    ]

    async def run():
        """Execute the asynchronous operation under test."""
        adapter = AsyncHttpServerAdapter.__new__(AsyncHttpServerAdapter)
        adapter._make_async_request = AsyncMock(return_value={"success": True})
        await adapter.begin_weight_update()
        await adapter.end_weight_update()
        assert adapter._make_async_request.call_args_list == [
            (("begin_weight_update", {"selector": "all"}), {"max_attempts": 1}),
            (("end_weight_update", {}),),
        ]

    asyncio.run(run())


@pytest.mark.parametrize(
    "supports_sessions,quantized,has_engine,leader,fail",
    [
        pytest.param(False, False, True, True, False, id="legacy"),
        pytest.param(True, False, True, True, False, id="sessions"),
        pytest.param(True, True, True, True, False, id="quantized-finalize"),
        pytest.param(True, False, True, True, True, id="second-bucket-failure"),
        pytest.param(True, False, False, False, False, id="no-engine"),
        pytest.param(True, False, True, False, False, id="non-leader-with-engine"),
    ],
    indirect=["supports_sessions"],
)
def test_all_buckets_share_session(monkeypatch, leader, fail, quantized, has_engine, supports_sessions):
    """Check session boundaries, finalization, errors, and non-leader consumption."""
    events = []
    adapter = rollout.ServerAdapter.__new__(rollout.ServerAdapter)
    adapter._init_server_adapter = AsyncMock()
    adapter._is_server_tp_leader = lambda: leader
    adapter.device_mesh = object()
    adapter.config = SimpleNamespace(checkpoint_engine=SimpleNamespace(update_weights_bucket_megabytes=1))
    adapter.model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3"))

    async def record(event):
        """Record the order of session and cache operations."""
        events.append(event)

    engine = SimpleNamespace(
        begin_weight_update=lambda: record("begin"),
        end_weight_update=lambda: record("end"),
        flush_cache=lambda: record("flush"),
    )
    adapter._engine = engine if has_engine else None
    monkeypatch.setattr(rollout, "_uses_quantized_refit", lambda *args: quantized)
    monkeypatch.setattr(rollout, "finalize_tensors", lambda: [("finalize", 3)])
    monkeypatch.setattr(rollout, "_to_ipc_device", lambda x: x)

    async def buckets(weights, *args, **kwargs):
        """Yield each input bucket so every sending rank consumes the iterator."""
        for batch in weights:
            yield batch

    async def update(**kwargs):
        """Record the mocked transfer and exercise the requested failure path."""
        events.append(kwargs["params_batch"][0][0])
        if fail and kwargs["params_batch"][0][0] == "bucket2":
            raise RuntimeError("transfer failed")

    monkeypatch.setattr(rollout, "get_named_tensor_buckets", buckets)
    monkeypatch.setattr(rollout, "sgl_update_weights", update)
    weights = iter([[("bucket1", 1)], [("bucket2", 2)]])
    work = adapter.update_weights(weights)
    if fail:
        with pytest.raises(RuntimeError, match="transfer failed"):
            asyncio.run(work)
        assert events == ["begin", "bucket1", "bucket2", "end"]
    else:
        asyncio.run(work)
        expected = ["bucket1", "bucket2"] + (["finalize"] if quantized else [])
        if leader and supports_sessions:
            expected = ["begin", *expected, "end"]
        if leader:
            expected.append("flush")
        assert events == expected
    # Every rank must consume the generator, including ranks without an engine.
    assert list(weights) == []


def test_failed_begin_does_not_end_unopened_session(supports_sessions):
    """Do not transfer or close an unconfirmed session after begin fails."""
    adapter = rollout.ServerAdapter.__new__(rollout.ServerAdapter)
    adapter._is_server_tp_leader = lambda: True
    adapter._engine = SimpleNamespace(
        begin_weight_update=AsyncMock(side_effect=RuntimeError("begin failed")),
        end_weight_update=AsyncMock(),
    )

    async def run():
        """Execute the asynchronous operation under test."""
        async with adapter._weight_update_session():
            pytest.fail("must not transfer without a session")

    with pytest.raises(RuntimeError, match="begin failed"):
        asyncio.run(run())
    adapter._engine.end_weight_update.assert_not_awaited()


@pytest.mark.parametrize("supports_sessions", [False, True], indirect=True, ids=["legacy", "sessions"])
def test_delta_flushes_share_session(supports_sessions):
    """Bracket delta flushes only when the upstream session API exists."""
    events = []
    adapter = rollout.ServerAdapter.__new__(rollout.ServerAdapter)
    adapter._pd_role = None
    adapter._is_server_tp_leader = lambda: True

    async def record(event):
        """Record the order of session and cache operations."""
        events.append(event)

    adapter._engine = SimpleNamespace(
        begin_weight_update=lambda: record("begin"), end_weight_update=lambda: record("end")
    )

    async def update(named, flush_cache):
        """Record the mocked transfer and exercise the requested failure path."""
        events.append((named, flush_cache))

    adapter._update_weights_delta_flush = update
    asyncio.run(adapter._update_weights_delta(iter([("first", False), ("last", True)])))
    expected = [("first", False), ("last", True)]
    assert events == (["begin", *expected, "end"] if supports_sessions else expected)


@pytest.mark.parametrize("transfer_fails", [True, False], ids=["preserve-transfer-error", "propagate-end-error"])
def test_end_failure(supports_sessions, transfer_fails, caplog):
    """Preserve a transfer error, but propagate a standalone finalization error."""
    adapter = rollout.ServerAdapter.__new__(rollout.ServerAdapter)
    adapter._is_server_tp_leader = lambda: True
    transfer_error = RuntimeError("transfer failed")
    end_error = RuntimeError("end failed")
    adapter._engine = SimpleNamespace(
        begin_weight_update=AsyncMock(), end_weight_update=AsyncMock(side_effect=end_error)
    )

    async def run():
        """Execute the asynchronous operation under test."""
        async with adapter._weight_update_session():
            if transfer_fails:
                raise transfer_error

    with pytest.raises(RuntimeError) as raised:
        asyncio.run(run())
    assert raised.value is (transfer_error if transfer_fails else end_error)
    adapter._engine.begin_weight_update.assert_awaited_once()
    adapter._engine.end_weight_update.assert_awaited_once()
    cleanup_logs = [record for record in caplog.records if "Failed to end SGLang" in record.getMessage()]
    if transfer_fails:
        assert len(cleanup_logs) == 1
        assert cleanup_logs[0].exc_info[1] is end_error
    else:
        assert not cleanup_logs


def test_lora_adapter_update_does_not_open_session(monkeypatch, supports_sessions):
    """Keep LoRA adapter loads outside base-weight update sessions."""
    adapter = rollout.ServerAdapter.__new__(rollout.ServerAdapter)
    adapter._init_server_adapter = AsyncMock()
    adapter._is_server_tp_leader = lambda: True
    adapter.device_mesh = {"infer_tp": SimpleNamespace(get_local_rank=lambda: 0)}
    adapter.config = SimpleNamespace()
    adapter.model_config = SimpleNamespace()
    adapter.wrap_lora_params = Mock(return_value=({}, []))
    adapter._engine = SimpleNamespace(
        begin_weight_update=AsyncMock(),
        end_weight_update=AsyncMock(),
        available_models=AsyncMock(return_value={"data": []}),
        load_lora_adapter_from_tensor=AsyncMock(),
        flush_cache=AsyncMock(),
    )
    monkeypatch.setattr(rollout, "_uses_quantized_refit", lambda *args: False)
    asyncio.run(adapter.update_weights(iter([]), peft_config={"r": 8}, base_sync_done=True))
    adapter._engine.load_lora_adapter_from_tensor.assert_awaited_once()
    adapter._engine.begin_weight_update.assert_not_awaited()
    adapter._engine.end_weight_update.assert_not_awaited()


@pytest.mark.parametrize("begin", [True, False], ids=["begin-once", "ordinary-retry"])
@pytest.mark.parametrize("error_type", [requests.Timeout, requests.ConnectionError], ids=["timeout", "connection"])
def test_sync_request_attempts(monkeypatch, begin, error_type):
    """Run the real sync helper to check begin failures and ordinary retries."""
    adapter = HttpServerAdapter.__new__(HttpServerAdapter)
    adapter.node_rank = 0
    adapter.server_args = SimpleNamespace(host="localhost", port=8000)
    adapter.timeout, adapter.retry_delay, adapter.max_attempts = 10, 1, 3
    error = error_type("response unavailable")
    post = Mock(side_effect=[error, Mock(status_code=204)])
    sleep = Mock()
    monkeypatch.setattr(http_engine.requests, "post", post)
    monkeypatch.setattr(http_engine.time, "sleep", sleep)

    if begin:
        with pytest.raises(error_type) as raised:
            adapter.begin_weight_update()
        assert raised.value is error
        assert post.call_count == 1
        sleep.assert_not_called()
    else:
        assert adapter._make_request("flush_cache") == {}
        assert post.call_count == 2
        sleep.assert_called_once_with(1)
    assert adapter.max_attempts == 3


@pytest.mark.parametrize("begin", [True, False], ids=["begin-once", "ordinary-retry"])
@pytest.mark.parametrize("connection_error", [False, True], ids=["timeout", "connection"])
def test_async_request_attempts(monkeypatch, begin, connection_error):
    """Run the real async helper without sharing mutable retry configuration."""
    adapter = AsyncHttpServerAdapter.__new__(AsyncHttpServerAdapter)
    adapter.node_rank = 0
    adapter.server_args = SimpleNamespace(host="localhost", port=8000)
    adapter.retry_delay, adapter.max_attempts = 1, 3
    error = (
        aiohttp.ClientConnectorError(SimpleNamespace(host="localhost", port=8000, ssl=False), OSError("disconnected"))
        if connection_error
        else asyncio.TimeoutError("response unavailable")
    )
    response_context = AsyncMock()
    response_context.__aenter__.return_value = SimpleNamespace(status=204)
    post = Mock(side_effect=[error, response_context])
    session_context = AsyncMock()
    session_context.__aenter__.return_value = SimpleNamespace(post=post)
    adapter._get_session = Mock(return_value=session_context)
    sleep = AsyncMock()
    monkeypatch.setattr(http_engine.asyncio, "sleep", sleep)

    async def run():
        """Exercise a non-idempotent begin or an ordinary retried request."""
        if begin:
            with pytest.raises(type(error)) as raised:
                await adapter.begin_weight_update()
            assert raised.value is error
            assert post.call_count == 1
            sleep.assert_not_awaited()
        else:
            assert await adapter._make_async_request("flush_cache") == {}
            assert post.call_count == 2
            sleep.assert_awaited_once_with(1)
        assert adapter.max_attempts == 3

    asyncio.run(run())
