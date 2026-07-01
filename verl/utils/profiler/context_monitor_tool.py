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

import logging
from contextlib import contextmanager
from typing import Any

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


class ContextMonitorTool:
    @contextmanager
    def annotate_context_record(self, **kwargs: Any):
        yield

    def add_prometheus_server_addresses(
        self,
        config: Any,
        server_addresses: list[str],
        job_name: str | None = None,
        labels: list[dict[str, Any] | None] | None = None,
    ) -> None:
        return

    def register_transfer_queue_metrics(self, config: Any) -> None:
        return


class RLInsightContextMonitor(ContextMonitorTool):
    _init_done = False

    @classmethod
    @contextmanager
    def annotate_context_record(cls, **kwargs: Any):
        try:
            from rl_insight import init, trace_state

            if not RLInsightContextMonitor._init_done:
                init()
                RLInsightContextMonitor._init_done = True
            name = kwargs.pop("name", None)
            rank_id = kwargs.pop("rank_id", None)
            with trace_state(name, state_lane_id=rank_id, **kwargs):
                yield
        except Exception:
            yield

    def add_prometheus_server_addresses(
        self,
        config: Any,
        server_addresses: list[str],
        job_name: str | None = None,
        labels: list[dict[str, Any] | None] | None = None,
    ) -> None:
        try:
            from rl_insight import update_prometheus_config

            update_prometheus_config(config, server_addresses, job_name, labels=labels)
        except Exception:
            pass

    def register_transfer_queue_metrics(self, config: Any) -> None:
        try:
            import transfer_queue as tq
            from rl_insight import update_prometheus_config

            endpoint = tq.get_metrics_endpoint()
            print(f"tmc debug endpoint: {endpoint}")
            if not endpoint:
                return
            update_prometheus_config(config, [endpoint], job_name="transfer_queue")
        except Exception:
            logger.exception("Failed to register transfer queue metrics")


# Supported online context monitor tools. Add new backends here.
SUPPORTED_CONTEXT_MONITOR_TOOLS: dict[str, type[ContextMonitorTool]] = {
    "rl_insight": RLInsightContextMonitor,
}


def build_context_monitors(context_monitor_tool: list[str] | str | None) -> list[ContextMonitorTool]:
    """Instantiate monitors listed in ``context_monitor_tool``."""
    if not context_monitor_tool:
        return []
    if isinstance(context_monitor_tool, str):
        context_monitor_tool = [context_monitor_tool]
    return [
        SUPPORTED_CONTEXT_MONITOR_TOOLS[name]()
        for name in context_monitor_tool
        if name in SUPPORTED_CONTEXT_MONITOR_TOOLS
    ]


def notify_transfer_queue_initialized(config: Any) -> None:
    """Notify context monitors after ``transfer_queue.init()`` to register scrape targets."""
    if not OmegaConf.select(config, "transfer_queue.metrics.enabled", default=False):
        return
    for monitor in build_context_monitors(OmegaConf.select(config, "trainer.logger", default=[])):
        monitor.register_transfer_queue_metrics(config)
