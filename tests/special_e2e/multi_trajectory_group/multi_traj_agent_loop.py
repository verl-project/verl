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
Test agent loop that returns AgentLoopGroupOutput with 3 trajectories.

This agent simulates a context-compressing agent (like MemAgent):

Step 1: Prompt the model normally with the original question.
        → response_1 (the model's initial reasoning)

Step 2: Ask the model to compress response_1 into a shorter summary.
        prompt = original prompt + response_1 + "Compress your reasoning above."
        → response_2 (the compressed summary)

Step 3: Use the compressed summary as context and ask for a final answer.
        prompt = original prompt + response_2 + "Give a final concise answer."
        → response_3 (the final concise answer)

This produces 3 on-policy trajectories, each from a separate LLM call:

  Trajectory 1 ("intermediate"): original prompt → response_1
  Trajectory 2 ("intermediate"): (original prompt + response_1 + compress instruction) → response_2
  Trajectory 3 ("final"):        (original prompt + response_2 + final instruction) → response_3

The reward comes from trajectory 3 (the final answer) and is shared to all 3.
All trajectories have different prompt/response tokens and are on-policy.
"""

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


@register("multi_traj_test_agent")
class MultiTrajTestAgentLoop(AgentLoopBase):
    """Test agent that returns AgentLoopGroupOutput with 3 on-policy trajectories."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopGroupOutput:
        messages = list(kwargs["raw_prompt"])

        # Extract images and videos
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # Tokenize the original prompt
        prompt_ids = await self.apply_chat_template(messages, images=images, videos=videos)

        request_id = uuid4().hex
        metrics = {}

        # ================================================================
        # Step 1: Prompt the model normally → response_1
        # ================================================================
        with simple_timer("generate_sequences", metrics):
            output_1 = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output_1.num_preempted if output_1.num_preempted is not None else -1

        response_1_ids = output_1.token_ids[: self.response_length]
        response_1_mask = [1] * len(response_1_ids)
        response_1_logprobs = output_1.log_probs[: len(response_1_ids)] if output_1.log_probs else None

        agent_metrics = AgentLoopMetrics(**metrics)

        # --- Trajectory 1: original prompt → response_1 ---
        traj_step1 = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_1_ids,
            response_mask=response_1_mask,
            response_logprobs=response_1_logprobs,
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=agent_metrics,
            extra_fields={"turn_scores": [], "tool_rewards": [], "trajectory_role": "intermediate"},
        )

        # ================================================================
        # Step 2: Ask the model to compress its response_1
        #   prompt = original prompt + response_1 + compress instruction
        #   → response_2 (compressed summary)
        # ================================================================
        compress_instruction = "\n\nNow compress your reasoning above into a short summary."
        compress_ids = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.encode(compress_instruction, add_special_tokens=False)
        )

        step2_prompt_ids = prompt_ids + response_1_ids + compress_ids
        if len(step2_prompt_ids) > self.prompt_length:
            step2_prompt_ids = step2_prompt_ids[-self.prompt_length :]

        with simple_timer("generate_sequences", metrics):
            output_2 = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=step2_prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )

        response_2_ids = output_2.token_ids[: self.response_length]
        response_2_logprobs = output_2.log_probs[: len(response_2_ids)] if output_2.log_probs else None

        # Build trajectory 2's full prompt+response for training.
        # The prompt includes (original + response_1 + compress_instruction), all masked as prompt.
        # The response is response_2, generated by the model.
        # We encode response_1 + compress_instruction as non-trainable prompt context (mask=0).
        step2_full_response_ids = response_1_ids + compress_ids + response_2_ids
        step2_full_response_mask = [0] * len(response_1_ids) + [0] * len(compress_ids) + [1] * len(response_2_ids)
        step2_full_logprobs = None
        if response_2_logprobs is not None:
            step2_full_logprobs = [0.0] * len(response_1_ids) + [0.0] * len(compress_ids) + response_2_logprobs

        step2_full_response_ids = step2_full_response_ids[: self.response_length]
        step2_full_response_mask = step2_full_response_mask[: self.response_length]
        if step2_full_logprobs is not None:
            step2_full_logprobs = step2_full_logprobs[: self.response_length]

        # --- Trajectory 2: compress step ---
        traj_step2 = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=step2_full_response_ids,
            response_mask=step2_full_response_mask,
            response_logprobs=step2_full_logprobs,
            multi_modal_data=multi_modal_data,
            num_turns=3,
            metrics=agent_metrics,
            extra_fields={"turn_scores": [], "tool_rewards": [], "trajectory_role": "intermediate"},
        )

        # ================================================================
        # Step 3: Use compressed summary (response_2) and ask for final answer
        #   prompt = original prompt + response_2 + final instruction
        #   → response_3 (final concise answer)
        # ================================================================
        final_instruction = "\n\nBased on the summary above, give a final concise answer."
        final_instr_ids = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.encode(final_instruction, add_special_tokens=False)
        )

        step3_prompt_ids = prompt_ids + response_2_ids + final_instr_ids
        if len(step3_prompt_ids) > self.prompt_length:
            step3_prompt_ids = step3_prompt_ids[-self.prompt_length :]

        with simple_timer("generate_sequences", metrics):
            output_3 = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=step3_prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )

        response_3_ids = output_3.token_ids[: self.response_length]
        response_3_logprobs = output_3.log_probs[: len(response_3_ids)] if output_3.log_probs else None

        # Build trajectory 3's response: response_2 + final_instruction + response_3
        # response_2 and final_instruction are non-trainable context (mask=0).
        step3_full_response_ids = response_2_ids + final_instr_ids + response_3_ids
        step3_full_response_mask = [0] * len(response_2_ids) + [0] * len(final_instr_ids) + [1] * len(response_3_ids)
        step3_full_logprobs = None
        if response_3_logprobs is not None:
            step3_full_logprobs = [0.0] * len(response_2_ids) + [0.0] * len(final_instr_ids) + response_3_logprobs

        step3_full_response_ids = step3_full_response_ids[: self.response_length]
        step3_full_response_mask = step3_full_response_mask[: self.response_length]
        if step3_full_logprobs is not None:
            step3_full_logprobs = step3_full_logprobs[: self.response_length]

        # --- Trajectory 3 (final): compressed context → final answer ---
        traj_final = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=step3_full_response_ids,
            response_mask=step3_full_response_mask,
            response_logprobs=step3_full_logprobs,
            multi_modal_data=multi_modal_data,
            num_turns=4,
            metrics=agent_metrics,
            extra_fields={"turn_scores": [], "tool_rewards": [], "trajectory_role": "final"},
        )

        # ---- Detailed trajectory logging ----
        def _decode_preview(ids, max_chars=200):
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            return text.replace("\n", "\\n")

        trainable_1 = sum(response_1_mask)
        trainable_2 = sum(step2_full_response_mask)
        trainable_3 = sum(step3_full_response_mask)

        print("=" * 80)
        print("[MultiTrajTestAgent] Returning group with 3 trajectories")
        print("-" * 80)
        print("  Trajectory 1 (intermediate - step1):")
        print(f"    prompt_len={len(prompt_ids)}, response_len={len(response_1_ids)}, trainable_tokens={trainable_1}")
        print(f"    prompt: {_decode_preview(prompt_ids)}")
        print(f"    response: {_decode_preview(response_1_ids)}")
        print("-" * 80)
        print("  Trajectory 2 (intermediate - compress):")
        print(
            f"    prompt_len={len(prompt_ids)}, response_len={len(step2_full_response_ids)}, "
            f"trainable_tokens={trainable_2} (response_1={len(response_1_ids)} mask=0, "
            f"compress_instr={len(compress_ids)} mask=0, response_2={len(response_2_ids)} mask=1)"
        )
        print(f"    prompt: {_decode_preview(prompt_ids)}")
        print(f"    response (trainable part / response_2): {_decode_preview(response_2_ids)}")
        print("-" * 80)
        print("  Trajectory 3 (final):")
        print(
            f"    prompt_len={len(prompt_ids)}, response_len={len(step3_full_response_ids)}, "
            f"trainable_tokens={trainable_3} (response_2={len(response_2_ids)} mask=0, "
            f"final_instr={len(final_instr_ids)} mask=0, response_3={len(response_3_ids)} mask=1)"
        )
        print(f"    prompt: {_decode_preview(prompt_ids)}")
        print(f"    response (trainable part / response_3): {_decode_preview(response_3_ids)}")
        print("=" * 80)

        return AgentLoopGroupOutput(
            trajectories=[traj_final, traj_step1, traj_step2],
            shared_reward=None,
        )
