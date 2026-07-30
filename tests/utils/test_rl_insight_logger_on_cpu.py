# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from verl.utils.tracking import RLInsightLogger


@pytest.fixture(autouse=True)
def _reset_rl_insight_logger():
    RLInsightLogger._init_done = False
    RLInsightLogger._rl_insight_module = None
    RLInsightLogger._registered_metrics.clear()
    yield
    RLInsightLogger._init_done = False
    RLInsightLogger._rl_insight_module = None
    RLInsightLogger._registered_metrics.clear()


@pytest.fixture
def mock_rl_insight(monkeypatch):
    module = MagicMock()
    module.metric_gauge = MagicMock()

    @contextmanager
    def _trace_state(*args, **kwargs):
        yield

    module.trace_state.side_effect = _trace_state
    monkeypatch.setattr(RLInsightLogger, "_get_rl_insight", classmethod(lambda cls: module))
    return module


def test_apis_are_noops_when_env_disabled(monkeypatch, mock_rl_insight):
    monkeypatch.delenv(RLInsightLogger.ENABLE_ENV, raising=False)

    RLInsightLogger.init(
        project_name="p", experiment_name="e", config={"transfer_queue": {"metrics": {"enabled": True}}}
    )
    RLInsightLogger.log({"reward/mean": 1.0}, step=1)
    with RLInsightLogger.trace_state("rollout", state_lane_id="replica_0"):
        pass
    RLInsightLogger.register_metrics(["127.0.0.1:8000"], "vllm", [{"replica": 0}])
    RLInsightLogger.finish()

    mock_rl_insight.init.assert_not_called()
    mock_rl_insight.metric_gauge.assert_not_called()
    mock_rl_insight.trace_state.assert_not_called()
    mock_rl_insight.update_prometheus_config.assert_not_called()
    mock_rl_insight.finish.assert_not_called()
    assert RLInsightLogger._init_done is False


def test_init_registers_transfer_queue_after_client_init(monkeypatch, mock_rl_insight):
    monkeypatch.setenv(RLInsightLogger.ENABLE_ENV, "1")
    config = {
        "trainer": {"rl_insight": {"server": {"url": "http://127.0.0.1:18080"}}},
        "transfer_queue": {"metrics": {"enabled": True}},
    }
    fake_tq = MagicMock()
    fake_tq.get_metrics_endpoint.return_value = "127.0.0.1:9000"
    monkeypatch.setitem(__import__("sys").modules, "transfer_queue", fake_tq)

    RLInsightLogger.init(project_name="proj", experiment_name="exp", config=config)

    mock_rl_insight.init.assert_called_once_with(
        project="proj",
        experiment_name="exp",
        config=config["trainer"]["rl_insight"],
    )
    mock_rl_insight.update_prometheus_config.assert_called_once_with(["127.0.0.1:9000"], "transfer_queue", None)
    assert RLInsightLogger._init_done is True


def test_log_lazy_inits_and_writes_gauge(monkeypatch, mock_rl_insight):
    monkeypatch.setenv(RLInsightLogger.ENABLE_ENV, "1")

    RLInsightLogger.log({"reward/mean": 1.25, "bad": object()}, step=3)

    mock_rl_insight.init.assert_called_once_with()
    mock_rl_insight.metric_gauge.assert_called_once_with("reward_mean", 1.25)
    assert RLInsightLogger._init_done is True


def test_trace_state_lazy_inits_once_under_concurrent_async_tasks(monkeypatch, mock_rl_insight):
    monkeypatch.setenv(RLInsightLogger.ENABLE_ENV, "1")

    async def _one(i: int):
        with RLInsightLogger.trace_state(f"state_{i}", state_lane_id=f"lane_{i}", step=i):
            await asyncio.sleep(0)

    async def _run():
        await asyncio.gather(*[_one(i) for i in range(16)])

    asyncio.run(_run())

    mock_rl_insight.init.assert_called_once_with()
    assert mock_rl_insight.trace_state.call_count == 16
    assert RLInsightLogger._init_done is True


def test_register_metrics_dedups_and_finish_resets(monkeypatch, mock_rl_insight):
    monkeypatch.setenv(RLInsightLogger.ENABLE_ENV, "1")
    addresses = ["127.0.0.1:8000", "127.0.0.1:8001"]
    labels = [{"replica": 0}, {"replica": 1}]

    RLInsightLogger.register_metrics(addresses, "vllm", labels)
    RLInsightLogger.register_metrics(addresses, "vllm", labels)
    mock_rl_insight.update_prometheus_config.assert_called_once_with(addresses, "vllm", labels)

    RLInsightLogger._init_done = True
    RLInsightLogger.finish()
    mock_rl_insight.finish.assert_called_once_with()
    assert RLInsightLogger._init_done is False
    assert not RLInsightLogger._registered_metrics
