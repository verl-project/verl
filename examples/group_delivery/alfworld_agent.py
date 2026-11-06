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
"""Real ALFWorld episodes using the existing tool-agent tokenization and masking."""

import json
import logging
import os
import time

import httpx

from verl.experimental.agent_loop.agent_loop import ToolListWrap
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def server_url():
    """Use a separately installed CPU environment service on the same machine."""
    return os.environ.get("ALFWORLD_SERVER_URL", "http://127.0.0.1:8765").rstrip("/")


async def request(method, path, **kwargs):
    """Fail transport errors instead of converting environment failures into rewards."""
    # The synthetic slow-task tail can hold a session-open request for minutes;
    # keep a generous timeout so it is not mistaken for an environment failure.
    timeout = float(os.environ.get("ALFWORLD_HTTP_TIMEOUT", "600"))
    async with httpx.AsyncClient(base_url=server_url(), timeout=timeout, trust_env=False) as client:
        response = await client.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json()


def observation(state):
    """Expose observations and legal actions, never the expert plan."""
    return state["observation"] + "\nAvailable actions: " + "; ".join(state["commands"])


class AlfworldTool(BaseTool):
    """A tool call borrows a session; the agent loop owns its complete lifetime."""

    def __init__(self):
        schema = OpenAIFunctionToolSchema.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "alfworld",
                    "description": "Execute one ALFWorld action.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": "An available action, copied exactly from the observation.",
                            }
                        },
                        "required": ["action"],
                    },
                },
            }
        )
        super().__init__({}, schema)

    async def create(self, **kwargs):
        """Reuse this trajectory's environment instead of resetting it per action."""
        return kwargs["create_kwargs"]["session_id"], ToolResponse()

    async def execute(self, instance_id, parameters, **kwargs):
        """Step the environment and retain terminal status for the agent loop."""
        agent_data = kwargs["agent_data"]
        action = parameters.get("action")
        if not isinstance(action, str) or not action:
            return ToolResponse(text="Provide a nonempty action string."), 0.0, {}
        try:
            state = await request("POST", f"/sessions/{instance_id}/step", json={"action": action})
        except Exception:
            agent_data.extra_fields["alfworld_environment_error"] = True
            raise
        agent_data.extra_fields["alfworld_done"] = state["done"]
        return ToolResponse(text=observation(state)), float(state["won"]), {}

    async def release(self, instance_id, **kwargs):
        """Per-call release must not destroy the episode owned by run()."""


class AlfworldAgentLoop(ToolAgentLoop):
    """One model-driven multi-turn episode, one final trajectory, binary environment reward."""

    def __init__(self, *args, tools=None, **kwargs):
        super().__init__(*args, tools=ToolListWrap([AlfworldTool()]), **kwargs)
        if self.max_parallel_calls != 1:
            raise ValueError("ALFWorld requires max_parallel_calls=1")

    async def _handle_processing_tools_state(self, agent_data):
        state = await super()._handle_processing_tools_state(agent_data)
        if agent_data.extra_fields.get("alfworld_environment_error"):
            raise RuntimeError("ALFWorld environment failed; discard the entire RL batch")
        return AgentState.TERMINATED if agent_data.extra_fields.get("alfworld_done") else state

    async def run(self, sampling_params, priority=0, **kwargs):
        """Keep environment state alive through all tool calls and close it on failure."""
        started = time.monotonic()
        game_file = kwargs["extra_info"]["game_file"]
        state = await request("POST", "/sessions", json={"game_file": game_file})
        session_id = state["session_id"]
        try:
            kwargs["raw_prompt"] = list(kwargs["raw_prompt"]) + [{"role": "user", "content": observation(state)}]
            kwargs["tools_kwargs"] = {"alfworld": {"create_kwargs": {"session_id": session_id}}}
            output = await super().run(sampling_params, priority=priority, **kwargs)
            state = await request("GET", f"/sessions/{session_id}")
            if state.get("error") or output.extra_fields.get("alfworld_environment_error"):
                raise RuntimeError("ALFWorld environment failed")
            output.reward_score = float(state["won"])
            record = dict(
                game_file=game_file,
                uid=kwargs.get("uid"),
                session=kwargs.get("session_id"),
                duration_s=time.monotonic() - started,
                steps=state["steps"],
                won=state["won"],
                capped=state["steps"] >= state["max_steps"],
                response_tokens=len(output.response_ids),
                generated_tokens=sum(output.response_mask),
            )
            output.extra_fields["reward_extra_info"] = {"acc": output.reward_score, **record}
            # Ray forwards worker stdout to the driver log; a bare logger.info can be
            # dropped when the worker root handler level is WARNING. The pilot tooling
            # parses these lines, so emit them on stdout as well.
            record_line = "alfworld_episode " + json.dumps(record)
            logger.info("%s", record_line)
            print(record_line, flush=True)
            return output
        finally:
            await request("DELETE", f"/sessions/{session_id}")
