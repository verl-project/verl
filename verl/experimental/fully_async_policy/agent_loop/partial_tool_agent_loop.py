# Copyright 2025 Meituan Ltd. and/or its affiliates
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
import copy
import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("async_partial_tool_agent")
class AsyncPartialToolAgentLoop(ToolAgentLoop):
    """
    Support for partial rollout with multiple tool invocations in Agent Loop

    """

    def __init__(self, trainer_config, **kwargs):
        super().__init__(trainer_config, **kwargs)
        self.enable_partial_rollout = trainer_config.config.async_training.get("partial_rollout", False)

    # async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
    async def run(
        self, sampling_params: dict[str, Any], *, cancellation_event: asyncio.Event = None, **kwargs
    ) -> AgentLoopOutput:
        """
        Main entrance, supports interruption/recovery

        Args:
            sampling_params: Sampling parameters
            cancellation_event: cancellationn sginal
            **kwargs: Contains output (for recovery), raw_prompt, param_version, etc.

        Returns:
            AgentLoopOutput: Include the is_cancel flag
        """
        param_version = kwargs.get("param_version", 0)
        agent_data = None
        state = None

        # 1. check whether is the partial task
        output: Optional[AgentLoopOutput] = kwargs.get("output", None)
        if output and output.extra_fields.get("is_cancel", False):
            agent_data, state = self._restore_from_output(output)

            logger.info(f"[PartialToolAgent] Resuming from {state.value}")
        else:
            if output and not output.extra_fields.get("is_cancel", False):
                # Completed, return directly
                return output

            agent_data = await self._init_agent_data(kwargs, param_version)
            state = AgentState.PENDING
            logger.info("[PartialToolAgent] Start from scratch")
        # 2. run state machine
        state = await self._run_state_machine(agent_data, state, sampling_params, cancellation_event)

        # 3. bulid output
        if state == AgentState.TERMINATED:
            return self._build_completed_output(agent_data, param_version)
        else:
            # build cancelled output
            return self._build_cancelled_output(agent_data, state)

    async def _init_agent_data(self, kwargs: dict, param_version: int) -> AgentData:
        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        video_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("video", None))
        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)
        # Create AgentData instance to encapsulate all state
        agent_data = AgentData(
            messages=messages,
            image_data=image_data,
            video_data=video_data,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )

        # additional param version record
        agent_data.extra_fields["param_version_start"] = param_version
        agent_data.extra_fields["param_version_end"] = param_version

        return agent_data

    def _restore_from_output(self, output: AgentLoopOutput) -> tuple[AgentData, AgentState]:
        """restore AgentState and AgentData from output"""
        agent_data = output.extra_fields.get("agent_data", None)
        agent_state = output.extra_fields.get("agent_state", None)
        if agent_data is None or agent_state is None:
            raise ValueError(f"Unexpected situation: agent_data is {agent_data}, agent_state is {agent_state}")
        return agent_data, agent_state

    async def _run_state_machine(
        self,
        agent_data: AgentData,
        state: AgentState,
        sampling_params: dict[str, Any],
        cancellation_event: asyncio.Event = None,
    ) -> AgentState:
        """
        State machine.
        Currently, interruptions are only supported to occur in the GENERATING state or other states have ended.
        """
        # State machine loop
        while state != AgentState.TERMINATED:
            if cancellation_event and cancellation_event.is_set():
                logger.info(f"[PartialToolAgent] Cancellation detected. Interrupted before/at state: {state.value}")
                return state
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state_partial(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                logger.error(f"[PartialToolAgent] Invalid state: {state}")
                return AgentState.TERMINATED

        return AgentState.TERMINATED

    async def _handle_generating_state_partial(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """
        Handle GENERATING state, support partial rollout
        """
        add_messages: list[dict[str, Any]] = []

        with simple_timer("generate_sequences", agent_data.metrics):
            # partial interface
            if self.enable_partial_rollout:
                response_ids, log_probs, is_cancel = await self.server_manager.generate_for_partial(
                    request_id=agent_data.request_id,
                    prompt_ids=agent_data.prompt_ids,
                    sampling_params=sampling_params,
                    image_data=agent_data.image_data,
                )

                if is_cancel:
                    # Save the generated parts
                    agent_data.response_ids = response_ids
                    agent_data.prompt_ids += agent_data.response_ids
                    agent_data.response_mask += [1] * len(response_ids)
                    if log_probs:
                        agent_data.response_logprobs += log_probs
                    if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
                        # If response_length has reached the limit,
                        # it is considered to have ended normally.
                        agent_data.assistant_turns += 1
                        return AgentState.TERMINATED
                    return AgentState.GENERATING
            else:
                # original generate interface
                output = await self.server_manager.generate(
                    request_id=agent_data.request_id,
                    prompt_ids=agent_data.prompt_ids,
                    sampling_params=sampling_params,
                    image_data=agent_data.image_data,
                )
                response_ids = output.token_ids
                log_probs = output.log_probs

        agent_data.assistant_turns += 1
        agent_data.response_ids = response_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if log_probs:
            agent_data.response_logprobs += log_probs

        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # Extract tool calls
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)

        # Handle interaction if needed
        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            add_messages.append({"role": "assistant", "content": assistant_message})
            agent_data.messages.extend(add_messages)

        # Determine next state
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            return AgentState.INTERACTING
        else:
            return AgentState.TERMINATED

    def _build_completed_output(self, agent_data: AgentData, param_version: int) -> AgentLoopOutput:
        """build completed output"""
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {"image": agent_data.image_data} if agent_data.image_data is not None else {}
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields={},
        )
        output.extra_fields.update(
            {
                "turn_scores": agent_data.turn_scores,
                "tool_rewards": agent_data.tool_rewards,
                "is_cancel": False,
                "param_version_start": agent_data.extra_fields["param_version_start"],
                "param_version_end": param_version,
            }
        )
        return output

    def _build_cancelled_output(self, agent_data: AgentData, state: AgentState) -> AgentLoopOutput:
        """build cancelled output"""
        return AgentLoopOutput(
            prompt_ids=[],
            response_ids=[],
            response_mask=[],
            multi_modal_data={},
            response_logprobs=None,
            num_turns=0,
            metrics=agent_data.metrics,
            extra_fields={
                "is_cancel": True,
                "agent_data": agent_data,
                "agent_state": state,
            },
        )
