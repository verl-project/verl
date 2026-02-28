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
"""
Test agent loop that returns AgentLoopGroupOutput with multiple trajectories.

This agent wraps the SingleTurnAgentLoop and produces a group of 2 trajectories:
1. The original end-to-end trajectory (role="final")
2. An "intermediate" trajectory with a shorter prompt/response (simulating a
   context-modifying agent that produces step-level training samples)

Both trajectories share the same reward from the end-to-end trajectory.
This is used to test the multi-trajectory group pipeline end-to-end.
"""

import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopGroupOutput,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("multi_traj_test_agent")
class MultiTrajTestAgentLoop(AgentLoopBase):
    """Test agent loop that returns AgentLoopGroupOutput with 2 trajectories.

    Trajectory 1 (final): full prompt + full response from LLM
    Trajectory 2 (intermediate): shortened prompt + same response (simulates a
    context-compressing agent step)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopGroupOutput:
        messages = list(kwargs["raw_prompt"])

        # 1. extract images and videos
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # 2. apply chat template and tokenize
        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
        )

        # 3. generate sequences (one LLM call)
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        response_ids = output.token_ids[: self.response_length]
        response_mask = [1] * len(response_ids)
        response_logprobs = output.log_probs[: self.response_length] if output.log_probs else None

        agent_metrics = AgentLoopMetrics(**metrics)

        # --- Trajectory 1: the full end-to-end trajectory ---
        traj_final = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=agent_metrics,
            extra_fields={"turn_scores": [], "tool_rewards": [], "trajectory_role": "final"},
        )

        # --- Trajectory 2: a "compressed context" intermediate trajectory ---
        # Simulate a context-modifying agent that uses a shorter prompt
        # (e.g., the last half of the prompt) but the same response.
        # This creates a genuinely different token sequence for training.
        half = max(len(prompt_ids) // 2, 1)
        short_prompt_ids = prompt_ids[half:]

        traj_intermediate = AgentLoopOutput(
            prompt_ids=short_prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            multi_modal_data=multi_modal_data,
            num_turns=1,
            metrics=agent_metrics,
            extra_fields={"turn_scores": [], "tool_rewards": [], "trajectory_role": "intermediate"},
        )

        # Return group — shared_reward=None means framework uses the final trajectory's reward
        return AgentLoopGroupOutput(
            trajectories=[traj_final, traj_intermediate],
            shared_reward=None,
        )
