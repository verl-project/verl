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
"""IFGym multi-turn agent loop.

This fork has no upstream-style ``BaseInteraction`` / ``interaction_config_path``
mechanism, so the swiss-ai/if-gym multi-turn setup (a scripted user simulator
that emits one pre-defined user turn at a time, each with its own
instruction-following constraints) is implemented here as a dedicated agent loop.

Per sample, the conversation is fully scripted by the data: ``extra_info``
carries ``interaction_kwargs.turns_json``, a JSON list of turns, each with a
``prompt`` (the user message) and ``active_constraints``. For every turn the
loop generates one assistant response, scores it against that turn's
constraints with the vendored IFGym checker, then injects the next user prompt
and continues. There are no tool calls.

The per-turn scores are exposed two ways:
  * ``turn_scores`` in ``extra_fields`` — one score per assistant turn, consumed
    by the ``ifgym_per_turn_grpo`` / ``ifgym_per_turn_rloo`` advantage
    estimators (and placed at each turn's last token in ``rm_scores`` by
    ``agent_loop._postprocess``).
  * ``reward_score`` — the trajectory-level mean, used for metrics/validation
    and as the credit signal for trajectory-level estimators (grpo, rloo, ...).

Registered as ``ifgym_agent`` (see ifgym_agent.yaml).
"""

import json
import logging
import os
from typing import Any

from apertus.ifgym.ifgym_instructions.instructions_registry import (
    HISTORY_AWARE_INSTRUCTIONS,
    INSTRUCTION_DICT,
)
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    get_generation_request_id,
    register,
)
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("ifgym_agent")
class IfgymAgentLoop(AgentLoopBase):
    """Scripted-user-turn instruction-following loop with per-turn scoring."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        extra_info = kwargs.get("extra_info", {}) or {}
        interaction_kwargs = extra_info.get("interaction_kwargs", {}) or {}

        turns = []
        turns_json = interaction_kwargs.get("turns_json")
        if turns_json:
            try:
                turns = json.loads(turns_json)
            except (json.JSONDecodeError, TypeError):
                turns = []

        # Cap the number of scripted user turns we run if requested.
        if self.max_user_turns:
            turns = turns[: self.max_user_turns]

        metrics: dict[str, Any] = {}
        request_id = get_generation_request_id(self.rollout_config, kwargs)

        # The dataset's first prompt is turns[0]; ``messages`` already contains it.
        prompt_ids = await self.apply_chat_template(messages)

        response_mask: list[int] = []
        response_logprobs: list[float] = []
        has_logprobs = False
        turn_scores: list[float] = []
        prev_assistant: str | None = None
        assistant_turns = 0

        for idx in range(len(turns)):
            # 1. Generate one assistant turn.
            with simple_timer("generate_sequences", metrics):
                output: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=None,
                    video_data=None,
                    audio_data=None,
                    mm_processor_kwargs=None,
                )
            if metrics.get("num_preempted") is None:
                metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

            token_ids = output.token_ids
            prompt_ids += token_ids
            response_mask += [1] * len(token_ids)
            if output.log_probs:
                has_logprobs = True
                response_logprobs += output.log_probs
            elif has_logprobs:
                response_logprobs += [0.0] * len(token_ids)
            assistant_turns += 1

            # 2. Score this assistant turn against its active constraints.
            assistant_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            constraints = turns[idx].get("active_constraints", [])
            turn_scores.append(self._score(assistant_text, constraints, prev_assistant))
            prev_assistant = assistant_text

            # 3. Stop if we've run out of response budget.
            if len(response_mask) >= self.response_length:
                break

            # 4. Inject the next scripted user turn (masked out of the loss).
            if idx + 1 < len(turns):
                next_prompt = turns[idx + 1].get("prompt", "")
                user_ids = await self.apply_chat_template(
                    [{"role": "user", "content": next_prompt}],
                    remove_system_prompt=True,
                )
                if len(response_mask) + len(user_ids) >= self.response_length:
                    break
                prompt_ids += user_ids
                response_mask += [0] * len(user_ids)
                if has_logprobs:
                    response_logprobs += [0.0] * len(user_ids)

        # Finalize: split prompt prefix from the generated+injected response region.
        response_ids = prompt_ids[-len(response_mask) :] if response_mask else []
        prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]
        reward_score = sum(turn_scores) / len(turn_scores) if turn_scores else 0.0

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if has_logprobs else None,
            num_turns=2 * assistant_turns + 1,
            reward_score=reward_score,
            metrics=metrics,
            extra_fields={},
        )
        output.extra_fields.update({"turn_scores": turn_scores, "tool_rewards": []})
        return output

    @staticmethod
    def _score(response: str, constraints: list[dict], prev: str | None) -> float:
        """Fraction of this turn's constraints the response satisfies."""
        if not constraints:
            return 0.0
        n_checked = n_pass = 0
        for c in constraints:
            cid = c.get("constraint_id")
            kw = dict(c.get("kwargs") or {})
            if cid not in INSTRUCTION_DICT:
                continue
            if cid in HISTORY_AWARE_INSTRUCTIONS:
                kw["previous_response"] = prev
            try:
                inst = INSTRUCTION_DICT[cid](cid)
                inst.build_description(**kw)
                n_checked += 1
                if response and response.strip() and inst.check_following(response):
                    n_pass += 1
            except Exception:
                continue
        return (n_pass / n_checked) if n_checked else 0.0
