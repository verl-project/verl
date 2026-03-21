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
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register
from verl.experimental.agent_loop.context_manager import ContextManager, ContextState, SummarizerContextManager
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput


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
        self.context_manager = SummarizerContextManager(tokenizer=self.tokenizer)

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
