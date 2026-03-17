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

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.experimental.agent_loop.context_manager import ContextManager, ContextState

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentLoopWithContextManagement(AgentLoopBase, ABC):
    """Minimal scaffold for loops that may emit multiple trajectories."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.context_manager: Optional[ContextManager] = None

    def set_context_manager(self, context_manager: Optional[ContextManager]) -> None:
        self.context_manager = context_manager

    async def create_initial_context(self, **kwargs) -> ContextState:
        messages = list(kwargs["raw_prompt"])
        multi_modal_data = await self.process_vision_info(messages)
        return ContextState(messages=messages, multi_modal_data=multi_modal_data)

    async def check_and_compress_context(self, state: ContextState) -> ContextState:
        if self.context_manager is None:
            return state
        return await self.context_manager.check_and_compress(state)

    def build_output(
        self,
        *,
        state: ContextState,
        prompt_ids: list[int],
        response_ids: list[int],
        response_mask: list[int],
        metrics: dict[str, Any],
        response_logprobs: Optional[list[float]] = None,
        routed_experts: Optional[Any] = None,
        num_turns: int = 0,
        extra_fields: Optional[dict[str, Any]] = None,
    ) -> AgentLoopOutput:
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            routed_experts=routed_experts,
            multi_modal_data=state.multi_modal_data or None,
            num_turns=num_turns,
            metrics=metrics,
            extra_fields=dict(state.metadata),
        )
        if extra_fields:
            output.extra_fields.update(extra_fields)
        return output

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> list[AgentLoopOutput]:
        raise NotImplementedError
