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

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("ray")
pytest.importorskip("vllm")

from vllm.v1.metrics.loggers import PrometheusStatLogger

from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer


class _Metric:
    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value

    def collect(self):
        return [SimpleNamespace(samples=[SimpleNamespace(name=self.name, value=self.value)])]


def test_vllm_http_server_snapshot_reads_scheduler_gauges():
    server = object.__new__(vLLMHttpServer)
    prometheus_logger = object.__new__(PrometheusStatLogger)
    prometheus_logger.gauge_kv_cache_usage = {
        0: _Metric("vllm:kv_cache_usage_perc", 0.25),
        1: _Metric("vllm:kv_cache_usage_perc", 0.75),
    }
    prometheus_logger.gauge_scheduler_waiting = {
        0: _Metric("vllm:num_requests_waiting", 1),
        1: _Metric("vllm:num_requests_waiting", 2),
    }
    prometheus_logger.gauge_scheduler_running = {
        0: _Metric("vllm:num_requests_running", 2),
        1: _Metric("vllm:num_requests_running", 3),
    }
    server.engine = SimpleNamespace(
        logger_manager=SimpleNamespace(stat_loggers=[prometheus_logger]),
    )

    snapshot = asyncio.run(server.snapshot())

    assert snapshot["kv_cache_usage"] == 0.75
    assert snapshot["num_waiting_requests"] == 3
    assert snapshot["num_running_requests"] == 5


def test_vllm_http_server_finds_current_logger_manager_shape():
    prometheus_logger = object.__new__(PrometheusStatLogger)
    server = object.__new__(vLLMHttpServer)
    server.engine = SimpleNamespace(logger_manager=SimpleNamespace(stat_loggers=[prometheus_logger]))

    assert server._prometheus_logger is prometheus_logger
