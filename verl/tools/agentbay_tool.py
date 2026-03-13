# Copyright 2025 verl contributors
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

"""AgentBay sandbox tool for verl.

Provides secure cloud-based code execution via AgentBay (https://github.com/agentbay-ai/wuying-agentbay-sdk).
Requires: pip install wuying-agentbay-sdk
Environment variable: AGENTBAY_API_KEY
"""

import logging
import os
import re
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentBayTool(BaseTool):
    """Code execution tool backed by AgentBay cloud sandbox.

    Each tool instance (trajectory) gets its own isolated cloud session.
    Sessions are created on `create()` and destroyed on `release()`.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._sessions = {}
        self._agent_bay = None
        self.default_language = config.get("default_language", "python")
        self.image_id = config.get("image_id", "code_latest")
        self.code_pattern = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL)

    def _get_agent_bay(self):
        """Lazy init to avoid import errors when SDK is not installed."""
        if self._agent_bay is None:
            from agentbay import AsyncAgentBay

            api_key = self.config.get("api_key") or os.getenv("AGENTBAY_API_KEY")
            self._agent_bay = AsyncAgentBay(api_key=api_key)
        return self._agent_bay

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())

        from agentbay import CreateSessionParams

        agent_bay = self._get_agent_bay()
        result = await agent_bay.create(CreateSessionParams(image_id=self.image_id))
        if not result.success:
            logger.error(f"Failed to create AgentBay session for {instance_id}: {result}")
            self._sessions[instance_id] = None
            return instance_id, ToolResponse(text="Failed to create sandbox session.")

        self._sessions[instance_id] = result.session
        logger.info(f"Created AgentBay session for {instance_id}")
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        session = self._sessions.get(instance_id)
        if session is None:
            return ToolResponse(text="Error: sandbox session not available."), None, None

        code = parameters.get("code", "")
        language = parameters.get("language", self.default_language)
        if not isinstance(code, str):
            code = str(code)

        matches = self.code_pattern.findall(code)
        if matches:
            code = matches[0].strip()

        try:
            result = await session.code.run_code(code, language)
            if result.success:
                text = result.result or ""
            else:
                text = getattr(result, "error", None) or getattr(result, "result", "") or "Execution failed."
            logger.debug(f"AgentBay execution result for {instance_id}: {text[:200]}")
            return ToolResponse(text=text), None, None
        except Exception as e:
            logger.warning(f"AgentBay execution error for {instance_id}: {e}")
            return ToolResponse(text=f"Execution error: {e}"), None, None

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        session = self._sessions.pop(instance_id, None)
        if session is not None:
            try:
                agent_bay = self._get_agent_bay()
                await agent_bay.delete(session)
                logger.info(f"Released AgentBay session for {instance_id}")
            except Exception as e:
                logger.warning(f"Error releasing AgentBay session for {instance_id}: {e}")
