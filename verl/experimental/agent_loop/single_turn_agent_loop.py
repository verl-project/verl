# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
from typing import Any
from uuid import uuid4

import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.diffusion_agent_loop import DiffusionAgentLoopOutput
from verl.utils.chat_template import apply_chat_template
from verl.utils.profiler import simple_timer
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # 1. extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # 2. apply chat template and tokenize
        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
        )

        # 3. generate sequences
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        response_mask = [1] * len(output.token_ids)

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )

        # keeping the schema consistent with tool_agent_loop
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})

        return output


@register("diffusion_single_turn_agent")
class DiffusionSingleTurnAgentLoop(AgentLoopBase):
    """Agent loop for diffusion model serving."""

    # Keys from non_tensor_batch that are pipeline/dataset metadata and must
    # NOT be forwarded to server_manager.generate() (which passes **kwargs
    # down to the vllm-omni server that has a fixed signature).
    _KEYS_EXCLUDED_FROM_GENERATE = frozenset(
        {
            "raw_prompt",
            "raw_negative_prompt",
            "data_source",
            "reward_model",
            "index",
        }
    )

    async def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        images: list[Image.Image] | None = None,
        videos: list[tuple[torch.Tensor, dict]] | None = None,
        remove_system_prompt: bool = False,
    ) -> list[int]:
        """Tokenize on the asyncio thread for fast tokenizers when no processor is used.

        Rust-backed fast tokenizers are not reliably safe across ``run_in_executor`` thread
        boundaries with recent transformers (``RuntimeError: Already borrowed``). The diffusion
        path is tokenizer-only for Qwen-Image-style models; keep tokenization on the event-loop
        thread in that case.
        """
        if self.processor is not None:
            return await super().apply_chat_template(
                messages,
                tools=tools,
                images=images,
                videos=videos,
                remove_system_prompt=remove_system_prompt,
            )
        if getattr(self.tokenizer, "is_fast", False):
            tokenized_prompt = apply_chat_template(
                self.tokenizer,
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            )
            prompt_ids = normalize_token_ids(tokenized_prompt)
            if remove_system_prompt:
                prompt_ids = prompt_ids[len(self.system_prompt) :]
            return prompt_ids
        return await super().apply_chat_template(
            messages,
            tools=tools,
            images=images,
            videos=videos,
            remove_system_prompt=remove_system_prompt,
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> DiffusionAgentLoopOutput:
        raw_prompt = kwargs.pop("raw_prompt")
        raw_negative_prompt = kwargs.pop("raw_negative_prompt", None)
        for key in self._KEYS_EXCLUDED_FROM_GENERATE:
            kwargs.pop(key, None)

        # 1. extract images and videos from messages
        multi_modal_data = await self.process_vision_info(raw_prompt)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # 2. apply chat template and tokenize
        prompt_ids = await self.apply_chat_template(raw_prompt, images=images, videos=videos)

        if raw_negative_prompt is not None:
            negative_prompt_ids = await self.apply_chat_template(raw_negative_prompt, images=images, videos=videos)
        else:
            negative_prompt_ids = None

        # 3. generate sequences
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
                negative_prompt_ids=negative_prompt_ids,
                **kwargs,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        output = DiffusionAgentLoopOutput(
            prompt_ids=prompt_ids,
            response_diffusion_output=output.diffusion_output,
            response_logprobs=output.log_probs,
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )
        return output
