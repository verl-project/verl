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

from abc import ABC, abstractmethod
from typing import Any, Optional

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, AgentLoopMetrics
from verl.experimental.agent_loop.context_manager import ContextManager, ContextState


class AgentLoopWithContextManagement(AgentLoopBase, ABC):
    """Minimal scaffold for loops that may emit multiple trajectories."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_length = self.rollout_config.response_length
        self.context_manager: Optional[ContextManager] = None

    def set_context_manager(self, context_manager: Optional[ContextManager]) -> None:
        self.context_manager = context_manager

    async def create_initial_context(self, **kwargs) -> ContextState:
        messages = list(kwargs["raw_prompt"])
        multi_modal_data = await self.process_vision_info(messages)
        return ContextState(
            messages=messages,
            multi_modal_data=multi_modal_data,
            metrics=AgentLoopMetrics(),
        )

    async def check_and_compress_context(self, state: ContextState) -> tuple[ContextState, bool]:
        if self.context_manager is None:
            return state, False
        return await self.context_manager.check_and_compress(state)

    def build_output(
        self,
        *,
        state: ContextState,
        extra_fields: Optional[dict[str, Any]] = None,
    ) -> AgentLoopOutput:
        response_length = len(state.response_mask)
        if response_length == 0:
            prompt_ids = list(state.trajectory_ids)
            response_ids = []
        else:
            prompt_ids = state.trajectory_ids[:-response_length]
            response_ids = state.trajectory_ids[-response_length:]

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
        if extra_fields:
            output.extra_fields.update(extra_fields)
        return output

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> list[AgentLoopOutput]:
        raise NotImplementedError
