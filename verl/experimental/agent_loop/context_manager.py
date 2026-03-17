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
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence


@dataclass
class ContextState:
    """State boundary shared by agent loops and context managers."""

    messages: list[dict[str, Any]]
    prompt_ids: list[int] = field(default_factory=list)
    response_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    response_logprobs: list[float] = field(default_factory=list)
    multi_modal_data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ContextManager(ABC):
    """Plugin interface for context management."""

    async def check_and_compress(self, state: ContextState) -> ContextState:
        if not await self.should_compress(state):
            return state
        return await self.compress(state)

    @abstractmethod
    async def should_compress(self, state: ContextState) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def compress(self, state: ContextState) -> ContextState:
        raise NotImplementedError


class SlidingWindowContextManager(ContextManager):
    """Skeleton for rule-based sliding-window compression."""

    def __init__(
        self,
        max_messages: Optional[int] = None,
        keep_last_n_messages: Optional[int] = None,
    ):
        self.max_messages = max_messages
        self.keep_last_n_messages = keep_last_n_messages

    async def should_compress(self, state: ContextState) -> bool:
        raise NotImplementedError("SlidingWindowContextManager.should_compress is recipe-specific.")

    async def compress(self, state: ContextState) -> ContextState:
        raise NotImplementedError("SlidingWindowContextManager.compress is recipe-specific.")


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
