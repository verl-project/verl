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

from contextlib import contextmanager
from typing import Any


class ContextMonitorTool:
    @contextmanager
    def annotate_context_record(self, **kwargs: Any):
        yield

    def add_prometheus_server_addresses(
        self,
        config: Any,
        server_addresses: list[str],
        job_name: str | None = None,
        replica_ranks: list[int] | None = None,
    ) -> None:
        return


class RLInsightContextMonitor(ContextMonitorTool):
    _init_done = False

    @classmethod
    @contextmanager
    def annotate_context_record(cls, name: str, **kwargs: Any):
        try:
            from rl_insight import init, trace_state

            if not RLInsightContextMonitor._init_done:
                init()
                RLInsightContextMonitor._init_done = True
            lane_id = kwargs.pop("lane_id", None)
            if lane_id is None:
                with trace_state(name, **kwargs):
                    yield
            else:
                with trace_state(name, state_lane_id=lane_id, **kwargs):
                    yield
        except Exception:
            yield

    def add_prometheus_server_addresses(
        self,
        config: Any,
        server_addresses: list[str],
        job_name: str | None = None,
        replica_ranks: list[int] | None = None,
    ) -> None:
        try:
            from rl_insight import update_prometheus_config

            update_prometheus_config(config, server_addresses, job_name)
        except Exception:
            pass


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
