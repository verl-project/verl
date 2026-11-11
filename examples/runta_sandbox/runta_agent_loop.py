# Copyright 2026 Individual Contributor
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
"""``ToolAgentLoop`` variant that cleans up per-trajectory sandboxes.

``ToolAgentLoop`` releases a tool after every call, which is too early for a sandbox that must
keep its files across the turns of a trajectory. This loop tells such tools when the trajectory
is over, whether it finished, failed or was cancelled.
"""

import logging
import os
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.schemas import ToolResponse

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("runta_tool_agent")
class RuntaToolAgentLoop(ToolAgentLoop):
    """Tool agent loop that deletes the trajectory's Runta runtime when ``run`` returns.

    Any tool exposing ``async release_trajectory(trajectory_id)`` (see ``RuntaSandboxTool``) is
    notified with the ``request_id`` of the trajectory.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Request ids seen by tool calls. An agent loop instance serves one trajectory, so this
        # holds at most one id; it is only known inside `run`, hence the bookkeeping here.
        self._trajectory_ids: set[str] = set()

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        try:
            return await super().run(sampling_params, **kwargs)
        finally:
            await self._release_trajectory_tools()

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict]:
        self._trajectory_ids.add(agent_data.request_id)
        return await super()._call_tool(tool_call, tools_kwargs, agent_data)

    async def _release_trajectory_tools(self) -> None:
        while self._trajectory_ids:
            trajectory_id = self._trajectory_ids.pop()
            for tool in self.tools.values():
                release_trajectory = getattr(tool, "release_trajectory", None)
                if release_trajectory is None:
                    continue
                try:
                    await release_trajectory(trajectory_id)
                except Exception as e:
                    # Never let cleanup replace the trajectory's own result or error.
                    logger.warning(f"Failed to release tool '{tool.name}' for trajectory {trajectory_id}: {e}")
