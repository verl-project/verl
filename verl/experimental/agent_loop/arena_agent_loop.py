# Copyright 2025 Individual Contributor: albert-lv
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
OpenAgora (Arena) agent loop: sandboxed agent execution for verl rollouts.

This module provides an ``AgentLoopBase`` implementation that delegates agent
execution to the OpenAgora sandbox infrastructure
(https://github.com/albert-lv/OpenAgora) instead of calling the LLM server
in-process:

1. The prompt messages are rendered with the tokenizer chat template and
   tokenized (left-truncated to ``rollout.prompt_length``).
2. The task is submitted to the OpenAgora server, which runs the agent inside
   a sandbox (e.g. Docker). All LLM calls made by the agent are transparently
   proxied through the OpenAgora LLM proxy, which can be pointed at verl's
   vLLM/SGLang inference server.
3. The loop waits for the rollout to finish and reads the reward computed by
   OpenAgora's independent verification plane (not by the agent itself).
4. The captured trajectory is converted back into verl's ``AgentLoopOutput``
   format: response text and per-token logprobs are extracted from the
   recorded proxy requests/responses and tokenized (right-truncated to
   ``rollout.response_length``).

Usage in the training config::

    actor_rollout_ref.rollout.agent.default_agent_loop=arena_agent

Required environment variables:

- ``ARENA_ENDPOINT``: gRPC endpoint of the OpenAgora server (default: localhost:9090).
- ``ARENA_AGENT_IMAGE``: sandbox image for the agent (default: openagora-agent-minimal:latest).
- ``ARENA_LLM_BACKEND``: URL of the LLM backend the proxy forwards to
  (default: http://localhost:8000/v1).

Optional environment variables:

- ``ARENA_VERIFY_COMMAND``: fallback verification command (default: "true").
  Per-sample ``extra_info.openagora_verify`` takes precedence over this.
- ``ARENA_TIMEOUT_SECONDS``: rollout timeout in seconds (default: 3600).

The integration depends on the external ``openagora-sdk`` package, which is
imported lazily so that verl works without it when the arena agent loop is
not used. Install it with ``pip install openagora-sdk`` or from source at
https://github.com/albert-lv/OpenAgora/tree/main/python/openagora-sdk
"""

import json
import logging
import os
import time
from typing import Any, Optional

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_OPENAGORA_SDK_HINT = (
    "ArenaAgentLoop requires the `openagora-sdk` package, which is not installed. "
    "Install it with `pip install openagora-sdk`, or from source: "
    "https://github.com/albert-lv/OpenAgora/tree/main/python/openagora-sdk"
)


def _extract_response_text(trajectory: list[dict[str, Any]]) -> str:
    """Extract the agent's final response text from the Arena trajectory.

    Trajectory steps contain raw HTTP request/response bodies recorded by the
    OpenAgora proxy. Parse each step's response choices and concatenate the
    assistant messages. Handles both raw ``choices`` arrays and full OpenAI
    response JSON.

    Args:
        trajectory: List of trajectory step dicts from OpenAgora.

    Returns:
        Concatenated assistant response text.
    """
    texts = []
    for step in trajectory:
        resp = step.get("response") or {}
        choices_json = resp.get("choices_json") or resp.get("choices")
        if not choices_json:
            continue
        try:
            if isinstance(choices_json, bytes):
                choices_json = choices_json.decode("utf-8")
            data = json.loads(choices_json)
            # choices_json may be the full OpenAI response dict or just the choices list.
            if isinstance(data, dict):
                choices = data.get("choices", [])
            elif isinstance(data, list):
                choices = data
            else:
                continue
            if isinstance(choices, list) and len(choices) > 0:
                choice = choices[0]
                msg = choice.get("message", {})
                content = msg.get("content", "")
                if content:
                    texts.append(content)
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.debug("Failed to parse choices JSON in trajectory step")
            continue
    return "\n".join(texts)


def _extract_logprobs(trajectory: list[dict[str, Any]], response_length: int) -> Optional[list[float]]:
    """Extract per-token logprobs from the trajectory if available.

    Expects the OpenAI-compatible logprobs format recorded by the proxy::

        {
            "content": [
                {"token": "...", "logprob": -0.123, "top_logprobs": [...]},
                ...
            ]
        }

    Args:
        trajectory: List of trajectory step dicts from OpenAgora.
        response_length: Expected number of response tokens (for padding/truncation).

    Returns:
        A flat list of logprob floats aligned with the response tokens, or None if unavailable.
    """
    logprobs: list[float] = []
    for step in trajectory:
        resp = step.get("response") or {}
        lp_raw = resp.get("logprobs_json")
        if lp_raw:
            try:
                if isinstance(lp_raw, bytes):
                    lp_raw = lp_raw.decode("utf-8")
                # strict=False allows stray control characters in token strings.
                lp_data = json.loads(lp_raw, strict=False)
                content = lp_data.get("content") or lp_data.get("text")
                if isinstance(content, list):
                    for item in content:
                        lp = item.get("logprob")
                        if lp is not None:
                            logprobs.append(float(lp))
            except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
                continue
    if not logprobs:
        return None
    # Pad or truncate to response_length.
    if len(logprobs) < response_length:
        logprobs.extend([0.0] * (response_length - len(logprobs)))
    return logprobs[:response_length]


def _step_role(step: dict[str, Any]) -> str:
    """Infer the role of a trajectory step from request/response messages.

    Returns one of: ``assistant``, ``tool``, ``observation``, ``unknown``.
    """
    req = step.get("request") or {}
    messages_json = req.get("messages_json")
    if messages_json:
        try:
            if isinstance(messages_json, bytes):
                messages_json = messages_json.decode("utf-8")
            data = json.loads(messages_json)
            msgs = data.get("messages", [])
            if msgs:
                last_role = msgs[-1].get("role", "unknown")
                if last_role in ("user", "system"):
                    return "assistant"  # LLM is generating a response to user/system
                if last_role == "assistant":
                    return "tool"  # Next call is likely tool execution
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError, TypeError):
            pass
    # Fallback: inspect response content.
    resp = step.get("response") or {}
    choices_json = resp.get("choices_json") or resp.get("choices")
    if choices_json:
        try:
            if isinstance(choices_json, bytes):
                choices_json = choices_json.decode("utf-8")
            choices = json.loads(choices_json)
            if isinstance(choices, list) and len(choices) > 0:
                msg = choices[0].get("message", {})
                if msg.get("tool_calls"):
                    return "tool"
                if msg.get("role") == "assistant":
                    return "assistant"
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError, TypeError):
            pass
    return "unknown"


def _count_agent_turns(trajectory: list[dict[str, Any]]) -> int:
    """Count the number of assistant/tool turns in the trajectory.

    The initial user prompt is not counted; each assistant response or tool
    call/observation emitted by the agent counts as one turn.
    """
    count = 0
    for step in trajectory:
        if _step_role(step) in ("assistant", "tool", "observation"):
            count += 1
    return max(count, 1)


@register("arena_agent")
class ArenaAgentLoop(AgentLoopBase):
    """OpenAgora-backed agent loop: run the agent in a sandbox, score via the verification plane.

    The agent runs inside an OpenAgora sandbox (e.g. Docker). All LLM calls
    made by the agent are transparently proxied through OpenAgora's LLM proxy,
    which can be pointed at verl's vLLM/SGLang inference server. The reward is
    computed by OpenAgora's independent verification plane and returned as
    ``AgentLoopOutput.reward_score``.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

        # Imported lazily so verl does not require openagora-sdk unless this
        # agent loop is actually instantiated.
        try:
            from openagora_sdk.client import ArenaClient
        except ImportError as e:
            raise ImportError(_OPENAGORA_SDK_HINT) from e

        arena_endpoint = os.environ.get("ARENA_ENDPOINT", "localhost:9090")
        self._arena = ArenaClient(arena_endpoint)

        self._agent_image = os.environ.get("ARENA_AGENT_IMAGE", "openagora-agent-minimal:latest")
        self._llm_backend = os.environ.get("ARENA_LLM_BACKEND", "http://localhost:8000/v1")
        self._verify_command = os.environ.get("ARENA_VERIFY_COMMAND", "true")
        self._timeout_seconds = int(os.environ.get("ARENA_TIMEOUT_SECONDS", "3600"))

        logger.info(
            "ArenaAgentLoop initialized: endpoint=%s image=%s backend=%s",
            arena_endpoint,
            self._agent_image,
            self._llm_backend,
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        """Run one OpenAgora sandboxed rollout and return tokenized results.

        Args:
            sampling_params: LLM sampling params forwarded to the sandboxed agent
                (temperature, top_p, seed are propagated through the OpenAgora proxy).
            **kwargs: dataset fields from ``verl.utils.dataset.RLHFDataset``. Requires
                ``raw_prompt``; honors ``extra_info`` (dict or JSON string) with optional
                ``task_file`` and ``openagora_verify`` keys, ``index``, and ``global_steps``.

        Returns:
            AgentLoopOutput with prompt/response token ids, per-token logprobs (if the
            proxy recorded them), the reward from OpenAgora verification, and Arena
            metadata in ``extra_fields``.
        """
        messages: list[dict[str, Any]] = list(kwargs.get("raw_prompt", []))
        if not messages:
            raise ValueError("ArenaAgentLoop requires 'raw_prompt' in kwargs")

        total_start = time.time()

        # 1. Render the prompt text (sent to the sandbox as the task), then
        # tokenize the prompt. ``apply_chat_template`` left-truncates to
        # ``rollout.prompt_length``.
        prompt_text = self._render_prompt_text(messages)
        prompt_ids = await self.apply_chat_template(messages)

        # 2. Build the task payload. Allow a per-sample task file override via
        # extra_info (e.g. coding-competition problems).
        extra = kwargs.get("extra_info", {})
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except json.JSONDecodeError:
                extra = {}
        if not isinstance(extra, dict):
            extra = {}

        custom_task_file = extra.get("task_file")
        if custom_task_file:
            if isinstance(custom_task_file, str):
                task_payload = custom_task_file.encode("utf-8")
            else:
                task_payload = custom_task_file
        else:
            task_payload = json.dumps(
                {
                    "task_id": kwargs.get("index", "0"),
                    "prompt": prompt_text,
                    "messages": messages,
                }
            ).encode("utf-8")

        sampling_cfg = {
            "temperature": sampling_params.get("temperature", 1.0),
            "top_p": sampling_params.get("top_p", 1.0),
            "seed": sampling_params.get("seed", 0),
        }

        # Per-sample verify command takes precedence over the environment fallback.
        verify_cmd = extra.get("openagora_verify", self._verify_command)

        # 3. Submit the task to the OpenAgora server.
        rollout_create_start = time.time()
        rollout_info = self._arena.create_rollout(
            task_id=f"verl-{kwargs.get('index', '0')}",
            image=self._agent_image,
            llm_backend=self._llm_backend,
            sampling=sampling_cfg,
            verify={"command": verify_cmd} if verify_cmd else None,
            task_file=task_payload,
            timeout_seconds=self._timeout_seconds,
        )
        rollout_create_time = time.time() - rollout_create_start
        rollout_id = rollout_info["rollout_id"]
        logger.info("Arena rollout created: %s (%.2fs)", rollout_id, rollout_create_time)

        # 4. Wait for the sandboxed agent to finish and read the reward from
        # the verification plane.
        wait_start = time.time()
        result = self._arena.wait(rollout_id, timeout=self._timeout_seconds)
        wait_time = time.time() - wait_start
        status = result.get("status", "unknown")
        reward_score = float(result.get("reward", 0.0))
        logger.info(
            "Arena rollout %s finished: status=%s reward=%s (waited %.2fs)",
            rollout_id,
            status,
            reward_score,
            wait_time,
        )

        # 5. Fetch the recorded trajectory and extract the response text.
        trajectory = self._arena.get_trajectory(rollout_id)
        response_text = _extract_response_text(trajectory)

        # Handle empty response (e.g. agent never replied).
        if not response_text or not response_text.strip():
            response_text = "I could not generate a response."

        # 6. Tokenize the response and truncate to the response budget.
        response_ids = self._encode_text(response_text)
        if len(response_ids) > self.response_length:
            logger.warning(
                "Response truncated from %d to %d tokens",
                len(response_ids),
                self.response_length,
            )
            response_ids = response_ids[: self.response_length]

        response_mask = [1] * len(response_ids)

        # 7. Align per-token logprobs with the (possibly truncated) response.
        response_logprobs = _extract_logprobs(trajectory, len(response_ids))

        # 8. Count agent turns from the trajectory.
        num_turns = _count_agent_turns(trajectory)

        total_time = time.time() - total_start
        metrics = AgentLoopMetrics(
            generate_sequences=total_time,
            tool_calls=float(num_turns),
            compute_score=wait_time,
        )

        # Policy-version tags enable verl's staleness tracking and off-policy
        # detection in async trainers. For single-shot Arena rollouts the
        # policy version does not change between request and response, so both
        # are equal to the current global_steps when the caller provides it.
        global_steps = kwargs.get("global_steps", None)

        extra_fields = {
            "arena_rollout_id": rollout_id,
            "arena_status": status,
            "trajectory_steps": len(trajectory),
            "verification_report": result.get("verification_report"),
        }
        if global_steps is not None:
            extra_fields["min_global_steps"] = global_steps
            extra_fields["max_global_steps"] = global_steps

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            reward_score=reward_score,
            num_turns=num_turns,
            metrics=metrics,
            extra_fields=extra_fields,
        )

    def _render_prompt_text(self, messages: list[dict[str, Any]]) -> str:
        """Render messages to a single text string for the Arena task payload."""
        processing_class = self.processor if self.processor is not None else self.tokenizer
        if processing_class is None:
            raise RuntimeError("ArenaAgentLoop requires a tokenizer or processor")

        if hasattr(processing_class, "apply_chat_template"):
            try:
                return processing_class.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                )
            except Exception:
                # chat_template not usable on this tokenizer; fall back below.
                logger.debug("apply_chat_template failed; using naive message concatenation", exc_info=True)
        # Fallback: naive concatenation.
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<{role}>\n{content}\n</{role}>")
        return "\n".join(parts)

    def _encode_text(self, text: str) -> list[int]:
        """Encode text to token ids without special tokens."""
        processing_class = self.processor if self.processor is not None else self.tokenizer
        if processing_class is None:
            raise RuntimeError("ArenaAgentLoop requires a tokenizer or processor")

        if hasattr(processing_class, "encode"):
            return processing_class.encode(text, add_special_tokens=False)
        # Fallback for HF tokenizers.
        return processing_class(text, add_special_tokens=False)["input_ids"]
