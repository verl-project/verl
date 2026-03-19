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

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics


@dataclass
class ContextState:
    """State boundary shared by agent loops and context managers."""

    messages: list[dict[str, Any]]
    trajectory_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    response_logprobs: list[float] = field(default_factory=list)
    multi_modal_data: dict[str, Any] = field(default_factory=dict)
    routed_experts: Optional[Any] = None
    reward_score: Optional[float] = None
    num_turns: int = 0
    metrics: AgentLoopMetrics = field(default_factory=AgentLoopMetrics)
    extra_fields: dict[str, Any] = field(default_factory=dict)


class ContextManager(ABC):
    """Plugin interface for context management."""

    async def check_and_compress(self, state: ContextState) -> tuple[ContextState, bool]:
        if not await self.should_compress(state):
            return state, False
        compressed_state = await self.compress(state)
        return compressed_state, compressed_state != state

    @abstractmethod
    async def should_compress(self, state: ContextState) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def compress(self, state: ContextState) -> ContextState:
        raise NotImplementedError


class SlidingWindowContextManager(ContextManager):
    """Rule-based compressor to keep the last N tool responses/observations when having M of them."""

    def __init__(
        self,
        compress_when_m_observations: int = 16,
        keep_last_n_observations: int = 0,
        replacing_text: str = "[Compressed]",
        tool_response_pattern: str = r"(<tool_response>)(.*?)(</tool_response>)",
        *,
        tokenizer: Any,
    ):
        if compress_when_m_observations <= 0 or keep_last_n_observations < 0:
            raise ValueError(
                "compress_when_m_observations must be positive and keep_last_n_observations must be non-negative."
            )
        if keep_last_n_observations >= compress_when_m_observations:
            raise ValueError("keep_last_n_observations must be less than compress_when_m_observations.")
        if tokenizer is None:
            raise ValueError("tokenizer must be provided for SlidingWindowContextManager.")

        self.compress_when_m_observations = compress_when_m_observations
        self.keep_last_n_observations = keep_last_n_observations
        self.replacing_text = replacing_text
        self.tokenizer = tokenizer
        self.tool_response_pattern = re.compile(tool_response_pattern, re.DOTALL)

    async def should_compress(self, state: ContextState) -> bool:
        observation_count = self._count_remaining_observations(state)
        return observation_count >= self.compress_when_m_observations

    async def compress(self, state: ContextState) -> ContextState:
        response_length = len(state.response_mask)

        # 'response_length' won't be zero as it has been checked by should_compress()
        prompt_ids = state.trajectory_ids[:-response_length]
        response_ids = state.trajectory_ids[-response_length:]
            
        compressed_response_ids, removed_num_obs_from_ids = self._compress_token_ids(response_ids)
        compressed_messages, removed_num_obs_from_messages = self._compress_messages(state.messages)
        
        if removed_num_obs_from_ids != removed_num_obs_from_messages:
            raise ValueError(
                "_compress_token_ids and _compress_messages must remove the same number of observations."
            )
        if removed_num_obs_from_ids == 0:
            return state

        # Reconstruct the context state
        compressed_trajectory_ids = prompt_ids + compressed_response_ids
        response_mask = [0] * len(compressed_response_ids)
        response_logprobs = [0.0] * len(compressed_response_ids) if state.response_logprobs else []

        return ContextState(
            messages=compressed_messages,
            trajectory_ids=compressed_trajectory_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            multi_modal_data=dict(state.multi_modal_data),
            routed_experts=None,
            reward_score=state.reward_score,
            num_turns=state.num_turns,
            metrics=state.metrics.model_copy(deep=True),
            extra_fields=dict(state.extra_fields),
        )

    def _count_remaining_observations(self, state: ContextState) -> int:
        response_length = len(state.response_mask)
        if response_length == 0:
            return 0
        response_ids = state.trajectory_ids[-response_length:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)

        count = 0
        for _, body, _ in self.tool_response_pattern.findall(response_text):
            # Skip those tool responses that have been compressed/replaced already.
            if body.strip() != self.replacing_text:
                count += 1

        return count

    def _compress_messages(self, messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        """Compress earlier observations in messages through message struct and keep the last N unchanged."""
        compressed_messages = [dict(message) for message in messages]
        removed_num_obs = 0

        tool_message_indices = [index for index, message in enumerate(messages) if message.get("role") == "tool"]
        num_to_compress = len(tool_message_indices) - self.keep_last_n_observations
        for message_index in tool_message_indices[:num_to_compress]:
            content = messages[message_index].get("content")
            already_compressed = False

            # For Multi-modal messages, we will replace them entirely.
            if isinstance(content, str):
                already_compressed = content.strip() == self.replacing_text

            if not already_compressed:
                removed_num_obs += 1
            compressed_messages[message_index]["content"] = self.replacing_text

        return compressed_messages, removed_num_obs

    def _compress_token_ids(self, token_ids: list[int]) -> tuple[list[int], int]:
        """Compress earlier observations in token ids through regex matching and keep the last N unchanged."""
        text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        matches = list(self.tool_response_pattern.finditer(text))
        num_to_compress = len(matches) - self.keep_last_n_observations

        compressed_parts = []
        last_end = 0
        removed_num_obs = 0
        for index, match in enumerate(matches):
            compressed_parts.append(text[last_end : match.start()])
            if index < num_to_compress:
                start_tag, _, end_tag = match.groups()
                # Previously compressed wouldn't be counted as 'removed_num_obs' this time
                if match.group(2).strip() != self.replacing_text:
                    removed_num_obs += 1
                compressed_parts.append(f"{start_tag}{self.replacing_text}{end_tag}")
            else:
                compressed_parts.append(match.group(0))
            last_end = match.end()
        compressed_parts.append(text[last_end:])
        compressed_text = "".join(compressed_parts)
        return self.tokenizer.encode(compressed_text, add_special_tokens=False), removed_num_obs


class SummarizerContextManager(ContextManager):
    """Skeleton for model-based summarization compression."""

    def __init__(
        self,
        summary_prompt: Optional[str] = None,
        summary_model: Optional[str] = None,
    ):
        self.summary_prompt = summary_prompt
        self.summary_model = summary_model

    async def should_compress(self, state: ContextState) -> bool:
        raise NotImplementedError("SummarizerContextManager.should_compress is recipe-specific.")

    async def compress(self, state: ContextState) -> ContextState:
        raise NotImplementedError("SummarizerContextManager.compress is recipe-specific.")


class HybridContextManager(ContextManager):
    """Compose multiple managers without coupling them to a specific loop."""

    def __init__(self, managers: Optional[Sequence[ContextManager]] = None):
        self.managers = list(managers) if managers is not None else []

    async def should_compress(self, state: ContextState) -> bool:
        for manager in self.managers:
            if await manager.should_compress(state):
                return True
        return False

    async def compress(self, state: ContextState) -> ContextState:
        next_state = state
        for manager in self.managers:
            if await manager.should_compress(next_state):
                next_state = await manager.compress(next_state)
        return next_state
