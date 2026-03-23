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
import asyncio
import json
import logging
import os
import re
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


class AgentData:
    """Encapsulates all state variables for the agent loop. AgentData is passed to tool calling in case that
    tool may need to access full history state. User can store any tool session data in `extra_fields`."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Image.Image],
        video_data: list[tuple[torch.Tensor, dict[str, Any]]],
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.video_data = video_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []

        self.routed_experts = None

        # Extra fields for dynamic addition, e.g., tool session data
        self.extra_fields: dict[str, Any] = {}


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize tools from config file
        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self.max_parallel_calls = self.rollout_config.multi_turn.max_parallel_calls
        self.max_tool_response_length = self.rollout_config.multi_turn.max_tool_response_length
        self.tool_response_truncate_side = self.rollout_config.multi_turn.tool_response_truncate_side
        tool_config_path = self.rollout_config.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer)
        self.tool_parser_name = self.rollout_config.multi_turn.format

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

        # Initialize interactions from config file
        self.interaction_config_file = self.rollout_config.multi_turn.interaction_config_path
        if self.interaction_config_file:
            self.interaction_map: dict[str, BaseInteraction] = self._initialize_interactions(
                self.interaction_config_file
            )

        triage_cfg = self._cfg_get(self.config, "actor_rollout_ref.rollout.multi_turn.triage", {}) or {}
        self.enable_difficulty_triage = bool(self._cfg_get(triage_cfg, "enable", False))
        self.enable_online_escalation = bool(self._cfg_get(triage_cfg, "online_escalation", True))
        self.tool_budget_by_difficulty = {
            "easy": {
                "max_search": int(self._cfg_get(triage_cfg, "budget.easy.max_search", 1)),
                "max_check": int(self._cfg_get(triage_cfg, "budget.easy.max_check", 0)),
                "max_turn": int(self._cfg_get(triage_cfg, "budget.easy.max_turn", 2)),
            },
            "medium": {
                "max_search": int(self._cfg_get(triage_cfg, "budget.medium.max_search", 2)),
                "max_check": int(self._cfg_get(triage_cfg, "budget.medium.max_check", 1)),
                "max_turn": int(self._cfg_get(triage_cfg, "budget.medium.max_turn", 4)),
            },
            "hard": {
                "max_search": int(self._cfg_get(triage_cfg, "budget.hard.max_search", 4)),
                "max_check": int(self._cfg_get(triage_cfg, "budget.hard.max_check", 2)),
                "max_turn": int(self._cfg_get(triage_cfg, "budget.hard.max_turn", 6)),
            },
        }
        self.triage_easy_threshold = float(self._cfg_get(triage_cfg, "heuristic.easy_threshold", 0.35))
        self.triage_hard_threshold = float(self._cfg_get(triage_cfg, "heuristic.hard_threshold", 0.65))
        self.triage_long_words = int(self._cfg_get(triage_cfg, "heuristic.long_words", 120))
        self.triage_long_chars = int(self._cfg_get(triage_cfg, "heuristic.long_chars", 700))
        self.escalate_on_search_error = bool(self._cfg_get(triage_cfg, "escalation.on_search_error", True))
        self.escalate_on_empty_search = bool(self._cfg_get(triage_cfg, "escalation.on_empty_search", True))
        self.escalate_on_checker_http_error = bool(
            self._cfg_get(triage_cfg, "escalation.on_checker_http_error", False)
        )
        self.escalate_contradiction_threshold = float(
            self._cfg_get(triage_cfg, "escalation.contradiction_threshold", 0.30)
        )
        self.escalate_support_threshold = float(
            self._cfg_get(triage_cfg, "escalation.support_threshold", 0.40)
        )
        self.escalate_reset_counters_on_search = bool(
            self._cfg_get(triage_cfg, "escalation.reset_counters_on_search_error", True)
        )
        self.escalate_reset_counters_on_checker = bool(
            self._cfg_get(triage_cfg, "escalation.reset_counters_on_checker", True)
        )
        self.auto_check_enabled = bool(self._cfg_get(triage_cfg, "auto_check.enable", True))
        self.auto_check_require_search = bool(self._cfg_get(triage_cfg, "auto_check.require_search", True))
        self.auto_check_allow_plain_answer = bool(self._cfg_get(triage_cfg, "auto_check.allow_plain_answer", False))
        self.auto_check_min_answer_chars = int(self._cfg_get(triage_cfg, "auto_check.min_answer_chars", 80))

    @staticmethod
    def _cfg_get(root: Any, path: str, default: Any = None) -> Any:
        cur = root
        for key in path.split("."):
            if cur is None:
                return default
            if isinstance(cur, dict):
                cur = cur.get(key)
                continue
            if hasattr(cur, key):
                cur = getattr(cur, key)
                continue
            try:
                cur = cur[key]  # type: ignore[index]
            except Exception:
                return default
        return default if cur is None else cur

    @staticmethod
    def _extract_plain_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        chunks.append(str(text))
            return "\n".join(chunks)
        return ""

    def _extract_user_question(self, messages: list[dict[str, Any]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return self._extract_plain_text(msg.get("content", ""))
        return ""

    @staticmethod
    def _strip_xml_block(text: str, tag: str) -> str:
        return re.sub(rf"<{tag}>(.*?)</{tag}>", "", text, flags=re.DOTALL | re.IGNORECASE)

    def _extract_candidate_answer_for_checker(self, text: str) -> str:
        if not text:
            return ""

        answer_match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip()

        if not self.auto_check_allow_plain_answer:
            return ""

        cleaned = text
        cleaned = self._strip_xml_block(cleaned, "think")
        cleaned = self._strip_xml_block(cleaned, "search")
        cleaned = self._strip_xml_block(cleaned, "check")
        cleaned = re.sub(r"</?answer>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    async def _maybe_inject_checker_call(self, agent_data: AgentData) -> bool:
        if not self.auto_check_enabled:
            return False
        if "check" not in self.tools:
            return False
        if self.auto_check_require_search and int(agent_data.extra_fields.get("search_used", 0)) <= 0:
            return False
        if int(agent_data.extra_fields.get("check_used", 0)) > 0:
            return False

        response_text = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
        )
        candidate_answer = self._extract_candidate_answer_for_checker(response_text)
        if not candidate_answer:
            return False
        if len(candidate_answer) < self.auto_check_min_answer_chars:
            return False

        should_skip, _ = self.tools["check"]._should_skip_check_answer(candidate_answer)
        if should_skip:
            return False

        agent_data.tool_calls = [
            FunctionCall(name="check", arguments=json.dumps({"answer": candidate_answer}, ensure_ascii=False))
        ]
        logger.warning(
            "[AUTO_CHECK] request=%s injected checker call after search with answer=%s",
            agent_data.request_id,
            candidate_answer[:200],
        )
        return True

    def _classify_by_faithfulness(self, faithfulness_score: float) -> str:
        if faithfulness_score >= self.triage_hard_threshold:
            return "easy"
        if faithfulness_score >= self.triage_easy_threshold:
            return "medium"
        return "hard"

    def _classify_difficulty_by_rules(self, question: str) -> str:
        q = (question or "").lower()
        if not q.strip():
            return "medium"
        n_chars = len(q)
        n_words = len(q.split())
        has_multique = int(q.count("?") >= 2)
        has_multihop = int(any(k in q for k in ["why", "how", "because", "differential", "vs", "versus"]))
        clinical_signal_count = sum(
            1
            for k in ["history", "medication", "dose", "lab", "wbc", "mri", "ct", "diagnosis", "symptom"]
            if k in q
        )
        has_many_conditions = int(clinical_signal_count >= 3)
        has_bullets = int(bool(re.search(r"(^|\n)\s*[-*]\s+", q)))
        long_text = int(n_words >= self.triage_long_words or n_chars >= self.triage_long_chars)
        score = (
            0.30 * long_text
            + 0.20 * has_multihop
            + 0.20 * has_many_conditions
            + 0.15 * has_multique
            + 0.15 * has_bullets
        )
        if score < self.triage_easy_threshold:
            return "easy"
        if score < self.triage_hard_threshold:
            return "medium"
        return "hard"

    def _init_difficulty_state(self, agent_data: AgentData, extra_info: Any) -> None:
        if not self.enable_difficulty_triage:
            return
        difficulty = None
        source = "heuristic"
        if isinstance(extra_info, dict):
            explicit = extra_info.get("difficulty")
            if explicit in ("easy", "medium", "hard"):
                difficulty = explicit
                source = "explicit"
            if difficulty is None:
                faith = extra_info.get("faithfulness_score")
                if faith is not None:
                    try:
                        difficulty = self._classify_by_faithfulness(float(faith))
                        source = "faithfulness_score"
                    except (TypeError, ValueError):
                        pass
        if difficulty not in ("easy", "medium", "hard"):
            difficulty = self._classify_difficulty_by_rules(self._extract_user_question(agent_data.messages))
        budget = dict(self.tool_budget_by_difficulty.get(difficulty, self.tool_budget_by_difficulty["medium"]))
        agent_data.extra_fields["difficulty_initial"] = difficulty
        agent_data.extra_fields["difficulty_current"] = difficulty
        agent_data.extra_fields["difficulty_escalation_count"] = 0
        agent_data.extra_fields["difficulty_source"] = source
        agent_data.extra_fields["tool_budget"] = budget
        agent_data.extra_fields["search_used"] = 0
        agent_data.extra_fields["check_used"] = 0
        logger.warning(
            "[TRIAGE] request=%s init difficulty=%s source=%s budget=%s",
            agent_data.request_id,
            difficulty,
            source,
            budget,
        )

    def _check_tool_budget(self, tool_name: str, agent_data: AgentData) -> tuple[bool, str]:
        if not self.enable_difficulty_triage:
            return True, ""
        budget = agent_data.extra_fields.get("tool_budget", {})
        if not isinstance(budget, dict):
            return True, ""
        if tool_name == "search":
            used = int(agent_data.extra_fields.get("search_used", 0))
            max_allowed = int(budget.get("max_search", 99))
            if used >= max_allowed:
                return False, f"search budget exceeded ({used}/{max_allowed})"
        if tool_name == "check":
            used = int(agent_data.extra_fields.get("check_used", 0))
            max_allowed = int(budget.get("max_check", 99))
            if used >= max_allowed:
                return False, f"check budget exceeded ({used}/{max_allowed})"
        return True, ""

    def _update_tool_usage_counters(self, tool_name: str, agent_data: AgentData) -> None:
        if tool_name == "search":
            agent_data.extra_fields["search_used"] = int(agent_data.extra_fields.get("search_used", 0)) + 1
        if tool_name == "check":
            agent_data.extra_fields["check_used"] = int(agent_data.extra_fields.get("check_used", 0)) + 1

    def _escalate_difficulty(self, agent_data: AgentData, reason: str, reset_counters: bool = False) -> None:
        old = agent_data.extra_fields.get("difficulty_current", "medium")
        nxt = {"easy": "medium", "medium": "hard", "hard": "hard"}.get(old, "hard")
        if nxt == old:
            return
        agent_data.extra_fields["difficulty_current"] = nxt
        agent_data.extra_fields["difficulty_escalation_count"] = (
            int(agent_data.extra_fields.get("difficulty_escalation_count", 0)) + 1
        )
        agent_data.extra_fields["tool_budget"] = dict(self.tool_budget_by_difficulty[nxt])
        if reset_counters:
            agent_data.extra_fields["search_used"] = 0
            agent_data.extra_fields["check_used"] = 0
        logger.warning(
            "[TRIAGE] request=%s escalate %s -> %s reason=%s budget=%s",
            agent_data.request_id,
            old,
            nxt,
            reason,
            agent_data.extra_fields["tool_budget"],
        )

    def _maybe_escalate_after_tool(self, tool_name: str, tool_metrics: Any, agent_data: AgentData) -> None:
        if not self.enable_difficulty_triage or not self.enable_online_escalation:
            return
        if not isinstance(tool_metrics, dict):
            return
        if tool_name == "search":
            api_error = tool_metrics.get("api_request_error")
            total_results = int(tool_metrics.get("total_results", 0) or 0)
            if (
                (self.escalate_on_search_error and api_error)
                or (self.escalate_on_empty_search and total_results == 0)
            ):
                self._escalate_difficulty(
                    agent_data,
                    reason="weak_or_empty_search",
                    reset_counters=self.escalate_reset_counters_on_search,
                )
            return
        if tool_name == "check":
            if self.escalate_on_checker_http_error and (tool_metrics.get("http_error") or tool_metrics.get("error")):
                self._escalate_difficulty(agent_data, reason="checker_http_error", reset_counters=False)
                return
            verification_results = tool_metrics.get("verification_results")
            if not isinstance(verification_results, list) or not verification_results:
                return
            total = len(verification_results)
            contradicted = sum(
                1 for x in verification_results if isinstance(x, dict) and x.get("label") == "contradict"
            )
            supported = sum(
                1 for x in verification_results if isinstance(x, dict) and x.get("label") == "entail"
            )
            contradiction_rate = contradicted / total if total else 0.0
            support_rate = supported / total if total else 0.0
            should_escalate = (
                contradiction_rate >= self.escalate_contradiction_threshold
                or support_rate < self.escalate_support_threshold
            )
            logger.warning(
                "[TRIAGE] request=%s checker_result contradiction_rate=%.2f support_rate=%.2f threshold=(c>=%.2f or s<%.2f) escalate=%s",
                agent_data.request_id,
                contradiction_rate,
                support_rate,
                self.escalate_contradiction_threshold,
                self.escalate_support_threshold,
                should_escalate,
            )
            if (
                should_escalate
            ):
                self._escalate_difficulty(
                    agent_data,
                    reason=f"checker_conflict(c={contradiction_rate:.2f},s={support_rate:.2f})",
                    reset_counters=self.escalate_reset_counters_on_checker,
                )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

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
            image_data=images,
            video_data=videos,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )
        self._init_difficulty_state(agent_data, kwargs.get("extra_info"))

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=agent_data.routed_experts,
            extra_fields=agent_data.extra_fields,
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        output.extra_fields.update(
            {
                "difficulty_initial": agent_data.extra_fields.get("difficulty_initial"),
                "difficulty_current": agent_data.extra_fields.get("difficulty_current"),
                "difficulty_escalation_count": agent_data.extra_fields.get("difficulty_escalation_count", 0),
                "difficulty_source": agent_data.extra_fields.get("difficulty_source", "heuristic"),
                "search_used": agent_data.extra_fields.get("search_used", 0),
                "check_used": agent_data.extra_fields.get("check_used", 0),
                "tool_budget": agent_data.extra_fields.get("tool_budget", {}),
            }
        )
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=self.tool_schemas,
            images=agent_data.image_data,
            videos=agent_data.video_data,
        )
        agent_data.prompt_ids = prompt_ids
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        add_messages: list[dict[str, Any]] = []

        with simple_timer("generate_sequences", agent_data.metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )
        # first time to set num_preempted
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        # then add num_preempted to the metrics
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output.extra_fields)
        else:
            # Multi-round calls, only update the maximum max_global_steps.
            max_global_steps = output.extra_fields.get("max_global_steps", None)
            if max_global_steps:
                agent_data.extra_fields["max_global_steps"] = max_global_steps

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # Check termination conditions
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        budget = agent_data.extra_fields.get("tool_budget", {})
        if isinstance(budget, dict):
            max_turn = int(budget.get("max_turn", 0) or 0)
            if max_turn > 0 and agent_data.assistant_turns >= max_turn:
                return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # Extract tool calls
        tools = [tool.tool_schema for tool in self.tools.values()]
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)

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
        elif await self._maybe_inject_checker_call(agent_data):
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            return AgentState.INTERACTING
        else:
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        for tool_response, tool_reward, _ in responses:
            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            # Handle image data
            if tool_response.image:
                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        agent_data.messages.extend(add_messages)

        if self.tool_parser_name == "gpt-oss":
            logger.info("manually format tool responses for gpt-oss")
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:
            # Note that we have to pass None to the images and videos if there are no new images / videos
            # to stay compatible with downstream image processing logic!
            images = new_images_this_turn if new_images_this_turn else None
            videos = None
            response_ids = await self.apply_chat_template(
                add_messages,
                images=images,
                videos=videos,
                remove_system_prompt=True,
            )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """Handle the interacting state: get user input from interaction."""
        (
            should_terminate_sequence,
            interaction_responses,
            reward,
            metrics,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )
        agent_data.user_turns += 1

        add_messages: list[dict[str, Any]] = [{"role": "user", "content": interaction_responses}]
        agent_data.messages.extend(add_messages)

        if reward is not None:
            agent_data.turn_scores.append(reward)

        # Update prompt with user responses (similar to _handle_processing_tools_state)
        response_ids = await self.apply_chat_template(
            add_messages,
            remove_system_prompt=True,
        )

        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        # double check prompt
        # Check termination condition
        if should_terminate_sequence:
            return AgentState.TERMINATED
        else:
            return AgentState.GENERATING

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            allowed, deny_reason = self._check_tool_budget(tool_name, agent_data)
            if not allowed:
                logger.warning(
                    "[TRIAGE] request=%s tool=%s denied reason=%s current=%s budget=%s search_used=%s check_used=%s",
                    agent_data.request_id,
                    tool_name,
                    deny_reason,
                    agent_data.extra_fields.get("difficulty_current"),
                    agent_data.extra_fields.get("tool_budget", {}),
                    agent_data.extra_fields.get("search_used", 0),
                    agent_data.extra_fields.get("check_used", 0),
                )
                return (
                    ToolResponse(
                        text=(
                            f"[Budget exhausted] Tool '{tool_name}' is not available: {deny_reason}. "
                            "Please answer based on the information gathered so far."
                        )
                    ),
                    0.0,
                    {"triage_denied": True, "triage_reason": deny_reason},
                )
            self._update_tool_usage_counters(tool_name, agent_data)
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, tool_reward, res = await tool.execute(
                instance_id, tool_args, agent_data=agent_data
            )
            self._maybe_escalate_after_tool(tool_name, res, agent_data)
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return (
                ToolResponse(
                    text=f"Error when executing tool: {e}",
                ),
                0.0,
                {},
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res

    def _initialize_interactions(self, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        return interaction_map
