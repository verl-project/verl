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
from abc import ABC, abstractmethod
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register
from verl.experimental.agent_loop.context_manager import (
    ContextManager,
    ContextState,
    SlidingWindowContextManager,
    SummarizerContextManager,
)
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentLoopWithContextManagement(AgentLoopBase, ABC):
    """Abstract base class for custom agent loops with pluggable context management."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_length = self.rollout_config.response_length
        self.context_manager: Optional[ContextManager] = None

    def _build_output_from_state(self, state: ContextState) -> AgentLoopOutput:
        response_length = len(state.response_mask)
        prompt_ids = state.trajectory_ids[:-response_length] if response_length > 0 else list(state.trajectory_ids)
        response_ids = state.trajectory_ids[-response_length:] if response_length > 0 else []

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=state.response_mask[: self.response_length],
            response_logprobs=state.response_logprobs[: self.response_length] if state.response_logprobs else None,
            routed_experts=state.routed_experts,
            multi_modal_data=state.multi_modal_data or None,
            reward_score=state.reward_score,
            num_turns=state.num_turns,
            metrics=state.metrics,
            extra_fields=dict(state.extra_fields),
        )
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        return output

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> list[AgentLoopOutput]:
        raise NotImplementedError


@register("naive_summarizer_agent")
class SummarizerAgentLoop(AgentLoopWithContextManagement):
    """Naive agent loop of multi-trajectory that uses model-generated summaries for context compression."""

    def __init__(self, *args, max_context_compressions: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        if max_context_compressions < 0:
            raise ValueError("max_context_compressions must be non-negative.")

        self.max_context_compressions = max_context_compressions
        self.context_manager = SummarizerContextManager(
            tokenizer=self.tokenizer,
            apply_chat_template_kwargs=self.apply_chat_template_kwargs,
        )

    async def _generate_next_state(
        self,
        *,
        state: ContextState,
        request_id: str,
        sampling_params: dict[str, Any],
        images,
        videos,
    ) -> ContextState:
        metrics = {}
        prompt_ids = state.trajectory_ids

        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )
        metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        response_text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
        messages = list(state.messages)
        messages.append({"role": "assistant", "content": response_text})

        response_mask = list(state.response_mask) + [1] * len(output.token_ids)
        if state.response_logprobs or output.log_probs:
            prefix_logprobs = (
                list(state.response_logprobs) if state.response_logprobs else [0.0] * len(state.response_mask)
            )
            current_logprobs = output.log_probs if output.log_probs is not None else [0.0] * len(output.token_ids)
            response_logprobs = prefix_logprobs + current_logprobs
        else:
            response_logprobs = []

        return ContextState(
            messages=messages,
            trajectory_ids=list(state.trajectory_ids) + output.token_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            multi_modal_data=dict(state.multi_modal_data),
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            reward_score=state.reward_score,
            num_turns=sum(1 for message in messages if message.get("role") != "system"),
            metrics=AgentLoopMetrics(**metrics),
            extra_fields=dict(output.extra_fields),
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> list[AgentLoopOutput]:
        messages = list(kwargs["raw_prompt"])

        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        prompt_ids = await self.apply_chat_template(messages, images=images, videos=videos)

        state = ContextState(
            messages=messages,
            trajectory_ids=prompt_ids,
            multi_modal_data=multi_modal_data,
            num_turns=sum(1 for message in messages if message.get("role") != "system"),
            metrics=AgentLoopMetrics(),
        )

        outputs = []
        request_id = uuid4().hex
        compression_count = 0
        while True:
            state = await self._generate_next_state(
                state=state,
                request_id=request_id,
                sampling_params=sampling_params,
                images=images,
                videos=videos,
            )
            outputs.append(self._build_output_from_state(state))

            if compression_count >= self.max_context_compressions:
                break

            next_state, compressed = await self.context_manager.check_and_compress(state)
            if not compressed:
                break

            state = next_state
            compression_count += 1

        return outputs


@register("tool_sliding_window_agent")
class ToolSlidingWindowAgentLoop(AgentLoopWithContextManagement):
    """Text-only tool agent loop with sliding-window context compression.

    Targets coder-style text tools: no multi-modal tool returns and no user interaction loop.
    """

    def __init__(
        self,
        *args,
        max_context_compressions: int = 4,
        compress_when_m_observations: int = 16,
        keep_last_n_observations: int = 2,
        replacing_text: str = "[Compressed]",
        tool_response_pattern: str = r"(<tool_response>)(.*?)(</tool_response>)",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if max_context_compressions < 0:
            raise ValueError("max_context_compressions must be non-negative.")

        mt = self.rollout_config.multi_turn
        if mt.interaction_config_path:
            raise ValueError("ToolSlidingWindowAgentLoop does not support interaction_config_path.")

        self.max_context_compressions = max_context_compressions
        self.max_user_turns = mt.max_user_turns
        self.max_assistant_turns = mt.max_assistant_turns
        self.max_parallel_calls = mt.max_parallel_calls
        self.max_tool_response_length = mt.max_tool_response_length
        self.tool_response_truncate_side = mt.tool_response_truncate_side

        # Tool infrastructure: parser, schemas, and tool instances (same config as ToolAgentLoop)
        tool_config_path = mt.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        self.tool_parser_schemas = [tool.tool_schema for tool in tool_list]
        self.tool_parser = ToolParser.get_tool_parser(mt.format, self.tokenizer)
        self.tool_parser_name = mt.format

        # Sliding window compressor: replaces old <tool_response> blocks with placeholder text
        self.context_manager = SlidingWindowContextManager(
            compress_when_m_observations=compress_when_m_observations,
            keep_last_n_observations=keep_last_n_observations,
            replacing_text=replacing_text,
            tool_response_pattern=tool_response_pattern,
            tokenizer=self.tokenizer,
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> list[AgentLoopOutput]:
        # Input validation: text-only, no multi-modal, no interaction
        messages = [dict(message) for message in list(kwargs["raw_prompt"])]
        self._validate_text_messages(messages)

        multi_modal_data = await self.process_vision_info(messages)
        if multi_modal_data:
            raise ValueError("ToolSlidingWindowAgentLoop only supports text prompts.")

        prompt_ids = await self.apply_chat_template(messages, tools=self.tool_schemas)
        state = ContextState(
            messages=messages,
            trajectory_ids=prompt_ids,
            num_turns=sum(1 for message in messages if message.get("role") != "system"),
            metrics=AgentLoopMetrics(),
        )

        # Session-level bookkeeping
        outputs: list[AgentLoopOutput] = []
        request_id = uuid4().hex
        compression_count = 0
        assistant_turns = 0
        tool_turns = 0
        session_tool_rewards: list[float] = []  # accumulated across all trajectories
        trajectory_tool_rewards: list[float] = []  # reset on each compression boundary

        def build_output(current_state: ContextState) -> AgentLoopOutput:
            """Closure that captures the current reward/compression counters."""
            return self._build_tool_output_from_state(
                state=current_state,
                trajectory_tool_rewards=trajectory_tool_rewards,
                session_tool_rewards=session_tool_rewards,
                compression_count=compression_count,
            )

        # Main generate-tool-compress loop
        while True:
            # 1. LLM generation: append assistant tokens (mask=1) to state
            state, assistant_response_ids = await self._generate_next_state(
                state=state,
                request_id=request_id,
                sampling_params=sampling_params,
            )
            assistant_turns += 1

            # 2. Check termination (response budget / turn budget)
            if self._should_terminate_after_generation(state, assistant_turns, tool_turns):
                outputs.append(build_output(state))
                break

            # 3. Extract tool calls from the latest assistant response
            _, tool_calls = await self.tool_parser.extract_tool_calls(assistant_response_ids, self.tool_parser_schemas)
            if not tool_calls:
                # No tool call → final answer, terminate
                outputs.append(build_output(state))
                break

            # 4. Execute tools and append observation tokens (mask=0) to state
            state, current_tool_rewards = await self._append_tool_responses(
                state=state,
                tool_calls=tool_calls,
                tools_kwargs=kwargs.get("tools_kwargs", {}),
            )
            tool_turns += 1
            session_tool_rewards.extend(current_tool_rewards)
            trajectory_tool_rewards.extend(current_tool_rewards)

            # 5. Check response budget after tool response append
            if len(state.response_mask) >= self.response_length:
                outputs.append(build_output(state))
                break

            # 6. Sliding window compression: emit trajectory and start new one
            if compression_count < self.max_context_compressions:
                next_state, compressed = await self.context_manager.check_and_compress(state)
                if compressed:
                    outputs.append(build_output(state))
                    state = next_state
                    compression_count += 1
                    trajectory_tool_rewards = []  # reset for the new trajectory

        return outputs

    async def _generate_next_state(
        self,
        *,
        state: ContextState,
        request_id: str,
        sampling_params: dict[str, Any],
    ) -> tuple[ContextState, list[int]]:
        """Call the LLM and return (updated state, raw response token ids).

        The raw response ids are returned separately so the caller can pass them
        directly to ``extract_tool_calls`` without re-slicing from state.
        Metrics (generate_sequences timing, num_preempted) are accumulated across turns.
        """
        # Carry forward metrics from previous turns
        metrics = state.metrics.model_dump()
        prompt_ids = state.trajectory_ids

        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )

        # Accumulate num_preempted across turns (first turn initializes, subsequent turns add)
        num_preempted = output.num_preempted if output.num_preempted is not None else -1
        if metrics.get("num_preempted", -1) < 0:
            metrics["num_preempted"] = num_preempted
        elif num_preempted > 0:
            metrics["num_preempted"] += num_preempted

        response_text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
        messages = [dict(message) for message in state.messages]
        messages.append({"role": "assistant", "content": response_text})

        # All generated tokens are mask=1 (model-generated, contribute to gradient)
        response_mask = list(state.response_mask) + [1] * len(output.token_ids)
        if state.response_logprobs or output.log_probs:
            prefix_logprobs = (
                list(state.response_logprobs) if state.response_logprobs else [0.0] * len(state.response_mask)
            )
            current_logprobs = output.log_probs if output.log_probs is not None else [0.0] * len(output.token_ids)
            response_logprobs = prefix_logprobs + current_logprobs
        else:
            response_logprobs = []

        # Preserve extra_fields across turns; only update max_global_steps from new output
        extra_fields = dict(state.extra_fields)
        if not extra_fields:
            extra_fields.update(output.extra_fields)
        else:
            max_global_steps = output.extra_fields.get("max_global_steps")
            if max_global_steps:
                extra_fields["max_global_steps"] = max_global_steps

        next_state = ContextState(
            messages=messages,
            trajectory_ids=list(state.trajectory_ids) + output.token_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else state.routed_experts
            ),
            reward_score=state.reward_score,
            num_turns=sum(1 for message in messages if message.get("role") != "system"),
            metrics=AgentLoopMetrics(**metrics),
            extra_fields=extra_fields,
        )
        return next_state, output.token_ids

    async def _append_tool_responses(
        self,
        *,
        state: ContextState,
        tool_calls,
        tools_kwargs: dict[str, Any],
    ) -> tuple[ContextState, list[float]]:
        """Execute tool calls in parallel and append observation tokens (mask=0) to the state.

        Returns (updated_state, tool_rewards) where tool_rewards is a list of per-tool
        reward floats for this round of tool execution.
        """
        metrics = state.metrics.model_dump()
        tasks = []
        tool_call_names = []
        for tool_call in tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_text_tool(tool_call, tools_kwargs))
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", metrics):
            responses = await asyncio.gather(*tasks)

        # Collect text responses and rewards; reject any multi-modal returns
        add_messages: list[dict[str, Any]] = []
        tool_rewards: list[float] = []
        for tool_response, tool_reward in responses:
            if tool_response.image or tool_response.video:
                raise ValueError("ToolSlidingWindowAgentLoop only supports text tool responses.")
            add_messages.append({"role": "tool", "content": tool_response.text or ""})
            if tool_reward is not None:
                tool_rewards.append(tool_reward)

        # Tokenize tool response messages into token ids
        response_ids = await self._encode_tool_response_messages(add_messages, tool_call_names)

        # Tool response tokens are mask=0 (environment observation, no gradient)
        response_mask = list(state.response_mask) + [0] * len(response_ids)
        response_logprobs = list(state.response_logprobs) + [0.0] * len(response_ids) if state.response_logprobs else []
        messages = [dict(message) for message in state.messages] + add_messages

        next_state = ContextState(
            messages=messages,
            trajectory_ids=list(state.trajectory_ids) + response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            routed_experts=state.routed_experts,
            reward_score=state.reward_score,
            num_turns=sum(1 for message in messages if message.get("role") != "system"),
            metrics=AgentLoopMetrics(**metrics),
            extra_fields=dict(state.extra_fields),
        )
        return next_state, tool_rewards

    async def _call_text_tool(self, tool_call, tools_kwargs: dict[str, Any]) -> tuple[ToolResponse, Optional[float]]:
        """Execute a single tool call. Returns (response, reward).

        On failure, returns an error message as the response text with reward=0.
        The caller is responsible for rejecting multi-modal responses.
        """
        tool, instance_id = None, None
        try:
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, tool_reward, _ = await tool.execute(instance_id, tool_args)
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return ToolResponse(text=f"Error when executing tool: {e}"), 0.0
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = self._truncate_tool_response_text(tool_execution_response.text)

        # Propagate image/video attrs so the caller can detect and reject multi-modal returns
        tool_response_kwargs = {"text": tool_response_text}
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward

    async def _encode_tool_response_messages(
        self, add_messages: list[dict[str, Any]], tool_call_names: list[str]
    ) -> list[int]:
        """Tokenize tool response messages into token ids, handling gpt-oss format specially."""
        if self.tool_parser_name == "gpt-oss":
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            return await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        return await self.apply_chat_template(add_messages, remove_system_prompt=True)

    def _truncate_tool_response_text(self, text: Optional[str]) -> Optional[str]:
        """Truncate tool response text to max_tool_response_length if exceeded."""
        if not text or len(text) <= self.max_tool_response_length:
            return text

        if self.tool_response_truncate_side == "left":
            return text[: self.max_tool_response_length] + "...(truncated)"
        if self.tool_response_truncate_side == "right":
            return "(truncated)..." + text[-self.max_tool_response_length :]

        length = self.max_tool_response_length // 2
        return text[:length] + "...(truncated)..." + text[-length:]

    def _build_tool_output_from_state(
        self,
        *,
        state: ContextState,
        trajectory_tool_rewards: list[float],
        session_tool_rewards: list[float],
        compression_count: int,
    ) -> AgentLoopOutput:
        """Build output with tool-specific extra_fields for downstream metrics."""
        output = self._build_output_from_state(state)
        output.extra_fields.update(
            {
                "turn_scores": [],
                "tool_rewards": list(trajectory_tool_rewards),
                "session_tool_rewards": list(session_tool_rewards),
                "context_compression_count": compression_count,
                "agent_loop_impl": "ToolSlidingWindowAgentLoop",
            }
        )
        return output

    def _should_terminate_after_generation(
        self,
        state: ContextState,
        assistant_turns: int,
        tool_turns: int,
    ) -> bool:
        """Check whether the loop should stop after an LLM generation step."""
        if len(state.response_mask) >= self.response_length:
            return True
        if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
            return True
        # tool_turns maps to max_user_turns (each tool round is one "user" turn in agent loop semantics)
        return bool(self.max_user_turns and tool_turns >= self.max_user_turns)

    @staticmethod
    def _validate_text_messages(messages: list[dict[str, Any]]) -> None:
        """Reject messages with non-string content (e.g. multi-modal structured content)."""
        for message in messages:
            content = message.get("content", "")
            if not isinstance(content, str):
                raise ValueError("ToolSlidingWindowAgentLoop only supports string message content.")
