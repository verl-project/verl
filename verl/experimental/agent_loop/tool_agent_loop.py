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
import asyncio
import json
import logging
import os
from enum import Enum
from typing import Any
from uuid import uuid4

import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"


class AgentData:
    """Encapsulates all state variables for the agent loop. AgentData is passed to tool calling in case that
    tool may need to access full history state. User can store any tool session data in `extra_fields`."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Image.Image],
        video_data: list[tuple[torch.Tensor, dict[str, Any]]],
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
    ):
        self.messages = messages
        self.image_data = image_data
        self.video_data = video_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []

        self.routed_experts = None

        # Extra fields for dynamic addition, e.g., tool session data
        self.extra_fields: dict[str, Any] = {}


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize tools from config file
        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self.max_parallel_calls = self.rollout_config.multi_turn.max_parallel_calls
        self.max_tool_response_length = self.rollout_config.multi_turn.max_tool_response_length
        self.tool_response_truncate_side = self.rollout_config.multi_turn.tool_response_truncate_side
        tool_config_path = self.rollout_config.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer)
        self.tool_parser_name = self.rollout_config.multi_turn.format

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
        )

        # Per-sample tool selection: filter global tools by extra_info.tool_selection
        extra_info = kwargs.get("extra_info", {}) or {}
        tool_selection = extra_info.get("tool_selection")
        if tool_selection and self.tools:
            selected = {name: self.tools[name] for name in tool_selection if name in self.tools}
            agent_data._active_tools = selected
            agent_data._active_tool_schemas = [
                t.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for t in selected.values()
            ]
        else:
            agent_data._active_tools = self.tools
            agent_data._active_tool_schemas = self.tool_schemas

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=agent_data.routed_experts,
            extra_fields=agent_data.extra_fields,
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=schemas,
            images=agent_data.image_data,
            videos=agent_data.video_data,
        )
        agent_data.prompt_ids = prompt_ids
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        with simple_timer("generate_sequences", agent_data.metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )
        # first time to set num_preempted
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        # then add num_preempted to the metrics
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output.extra_fields)
        else:
            # Multi-round calls, only update the maximum max_global_steps.
            max_global_steps = output.extra_fields.get("max_global_steps", None)
            if max_global_steps:
                agent_data.extra_fields["max_global_steps"] = max_global_steps

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # Check termination conditions
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # Extract tool calls (use per-sample tools if routed)
        active_tools = getattr(agent_data, "_active_tools", self.tools)
        tools = [tool.tool_schema for tool in active_tools.values()]
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)

        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        else:
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        for tool_response, tool_reward, _ in responses:
            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            # Handle image data
            if tool_response.image:
                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        agent_data.messages.extend(add_messages)

        if self.tool_parser_name == "gpt-oss":
            logger.info("manually format tool responses for gpt-oss")
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:
            # Note that we have to pass None to the images and videos if there are no new images / videos
            # to stay compatible with downstream image processing logic!
            images = new_images_this_turn if new_images_this_turn else None
            videos = None
            response_ids = await self.apply_chat_template(
                add_messages,
                images=images,
                videos=videos,
                remove_system_prompt=True,
            )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        active_tools = getattr(agent_data, "_active_tools", self.tools)
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = active_tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, tool_reward, res = await tool.execute(
                instance_id, tool_args, agent_data=agent_data
            )
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return (
                ToolResponse(
                    text=f"Error when executing tool: {e}",
                ),
                0.0,
                {},
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res
