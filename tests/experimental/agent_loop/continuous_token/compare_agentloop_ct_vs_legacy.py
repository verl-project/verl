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
"""Compare agent-loop Continuous Token (CT) and legacy token end to end.

The trajectories are messages-level fixtures. The test builds a deterministic
rollout server for each case, plus deterministic tools for ToolAgentLoop cases.
For each assistant turn, the mock generation is obtained from the tokenizer's
own chat template:

1. render ``prefix_messages`` with ``add_generation_prompt=True``;
2. render ``prefix_messages + [assistant_message]`` with
   ``add_generation_prompt=False``;
3. return the token-id suffix of the full render after the prompt render;

This avoids hard-coding a Hermes-style raw ``<tool_call>`` string as the model
output and lets each tokenizer render structured ``assistant.tool_calls`` in
its own canonical format.

In order to mimic the model's generation behavior, if the suffix contains EOS,
truncate after the final EOS token to mimic rollout stopping at EOS instead of
returning template whitespace that follows it. Missing EOS is allowed only for GLM;
GLM tool-call turns append the assistant/tool boundary token ``<|observation|>``.

CT and legacy are then compared by running the real
agent loop twice with the same deterministic server outputs: legacy constructs
the loop with CT disabled, while CT constructs it with the real
builder enabled. ToolAgentLoop cases execute hardcoded ``FunctionTool``
instances that return the trajectory's expected tool response.

``--dump-sequences`` writes CT/legacy token ids, decoded text, and
loss masks/logprobs.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from scripts.chat_template_mock_trajectories import (
    TRAJECTORY_BY_NAME,
    MockTrajectory,
    SingleTurnTrajectory,
    ToolAgentTrajectory,
    get_trajectory,
)
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, DictConfigWrap, ToolListWrap
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.tools.function_tool import FunctionTool
from verl.utils.tokenizer import normalize_token_ids
from verl.utils.tokenizer.chat_template import apply_chat_template
from verl.workers.rollout.replica import TokenOutput

DEFAULT_E2E_TRAJECTORIES = ("singleturnchat", "multiturnsingletool", "multiturnmultitool")


DEFAULT_MODELS = [
    # Qwen family.
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-4B-Thinking-2507",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3.5-0.8B",
    # GLM family.
    "zai-org/GLM-4.7",
    "zai-org/GLM-4.7-Flash",
    "zai-org/GLM-5",
    "zai-org/GLM-5.1",
    # Moonshot Kimi.
    "moonshotai/Kimi-K2-Instruct",
    "moonshotai/Kimi-K2.5",
    "moonshotai/Kimi-K2.6",
    # ByteDance Seed.
    "ByteDance-Seed/Seed-OSS-36B-Instruct",
    # MiniMax/MiMo/Memo-adjacent tokenizer families.
    "MiniMaxAI/MiniMax-M2",
    "MiniMaxAI/MiniMax-M2.5",
    "MiniMaxAI/MiniMax-M2.7",
    "XiaomiMiMo/MiMo-7B-SFT",
    "XiaomiMiMo/MiMo-7B-RL",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    # Google Gemma-4 (non-unified ``gemma4`` variant; ``gemma4_unified`` such as
    # gemma-4-12B-it is unsupported here since its Gemma4UnifiedProcessor is not
    # shipped by the installed transformers).
    "google/gemma-4-E4B-it",
    # OpenAI GPT-OSS.
    "openai/gpt-oss-20b",
]


@dataclass(frozen=True)
class PreparedAssistantOutput:
    token_ids: list[int]
    log_probs: list[float]


@dataclass(frozen=True)
class E2ERunOutput:
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    response_logprobs: list[float] | None
    generation_prompt_ids: list[list[int]]
    num_turns: int


class _NoopDataset:
    pass


class _DeterministicServer:
    """Deterministic replacement for rollout server generation."""

    def __init__(self, assistant_outputs: list[PreparedAssistantOutput]):
        self.assistant_outputs = list(assistant_outputs)
        self.prompt_ids_by_call: list[list[int]] = []
        self._next_index = 0

    async def generate(self, *, prompt_ids: list[int], **kwargs) -> TokenOutput:
        del kwargs
        self.prompt_ids_by_call.append(list(prompt_ids))
        if self._next_index >= len(self.assistant_outputs):
            raise RuntimeError(
                f"Deterministic server received unexpected generate call #{self._next_index + 1}; "
                f"only {len(self.assistant_outputs)} assistant outputs were prepared"
            )
        output = self.assistant_outputs[self._next_index]
        self._next_index += 1
        return TokenOutput(
            token_ids=list(output.token_ids),
            log_probs=list(output.log_probs),
            extra_fields={"mock_generation_turn": self._next_index},
        )

    @property
    def num_generate_calls(self) -> int:
        return self._next_index


def _set_pad_token(tokenizer) -> None:
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token


def _load_tokenizer(model: str, *, local_files_only: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        model,
        trust_remote_code=True,
        local_files_only=local_files_only,
        use_fast=True,
    )
    _set_pad_token(tokenizer)
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("tokenizer has no chat_template")
    return tokenizer


def _render_chat_tokens(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    add_generation_prompt: bool,
) -> list[int]:
    tokenized = apply_chat_template(
        tokenizer,
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
    )
    return normalize_token_ids(tokenized)


def _render_assistant_output_ids(
    tokenizer,
    prefix_messages: list[dict[str, Any]],
    assistant_message: dict[str, Any],
    *,
    tools: list[dict[str, Any]] | None,
) -> list[int]:
    prompt_ids = _render_chat_tokens(
        tokenizer,
        prefix_messages,
        tools=tools,
        add_generation_prompt=True,
    )
    full_ids = _render_chat_tokens(
        tokenizer,
        prefix_messages + [assistant_message],
        tools=tools,
        add_generation_prompt=False,
    )
    prompt_ids = _trim_generation_think_prefix(tokenizer, prompt_ids, full_ids)
    prompt_ids = _replace_glm_generation_think_open_with_close(tokenizer, prompt_ids, full_ids)
    prompt_ids = _replace_nemotron_generation_think_open_with_empty_block(tokenizer, prompt_ids, full_ids)
    prompt_ids = _replace_qwen_generation_think_open_with_empty_block(tokenizer, prompt_ids, full_ids)
    prompt_ids = _trim_gemma_generation_thought_channel(tokenizer, prompt_ids, full_ids)
    if _is_gptoss_tokenizer(tokenizer):
        # GPT-OSS needs two special handlings that make the whole-list re-render
        # unusable, so the assistant output is constructed directly instead:
        #   1) harmony intentionally drops prior tool-call turns' analysis (CoT)
        #      once a later final answer exists, so re-rendering is not a token-prefix
        #      of the generation prompt (RL rollout appends tokens and never re-drops);
        #   2) this checkpoint's template renders tool calls in a layout verl's
        #      GptOssToolParser cannot parse, which would stall the tool-agent flow.
        # The append-only truth is ``prompt (keeps prior CoT) + this turn's own
        # generation`` where the generation uses the model's real harmony layout.
        full_ids = prompt_ids + _gptoss_assistant_generation_ids(tokenizer, assistant_message, tools=tools)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Assistant output token-id suffix diff failed")
    return _truncate_after_final_eos(
        tokenizer,
        full_ids[len(prompt_ids) :],
        assistant_message=assistant_message,
    )


def _trim_generation_think_prefix(tokenizer, prompt_ids: list[int], full_ids: list[int]) -> list[int]:
    if full_ids[: len(prompt_ids)] == prompt_ids:
        return prompt_ids
    tokenizer_name = _tokenizer_name(tokenizer).lower()
    compact_name = "".join(char for char in tokenizer_name if char.isalnum())
    if "minimax" not in compact_name:
        return prompt_ids
    think_prefix_ids = tokenizer.encode("<think>\n", add_special_tokens=False)
    if think_prefix_ids and prompt_ids[-len(think_prefix_ids) :] == think_prefix_ids:
        trimmed_prompt_ids = prompt_ids[: -len(think_prefix_ids)]
        if full_ids[: len(trimmed_prompt_ids)] == trimmed_prompt_ids:
            return trimmed_prompt_ids
    return prompt_ids


def _replace_glm_generation_think_open_with_close(tokenizer, prompt_ids: list[int], full_ids: list[int]) -> list[int]:
    if full_ids[: len(prompt_ids)] == prompt_ids:
        return prompt_ids
    tokenizer_name = _tokenizer_name(tokenizer).lower()
    compact_name = "".join(char for char in tokenizer_name if char.isalnum())
    if not (
        any(marker in tokenizer_name for marker in ("glm-4.7", "glm_4.7", "glm-5", "glm_5"))
        or any(marker in compact_name for marker in ("glm47", "glm5"))
    ):
        return prompt_ids

    think_open_ids = tokenizer.encode("<think>", add_special_tokens=False)
    think_close_ids = tokenizer.encode("</think>", add_special_tokens=False)
    if not think_open_ids or not think_close_ids:
        return prompt_ids
    if prompt_ids[-len(think_open_ids) :] != think_open_ids:
        return prompt_ids

    replaced_prompt_ids = prompt_ids[: -len(think_open_ids)] + think_close_ids
    if full_ids[: len(replaced_prompt_ids)] == replaced_prompt_ids:
        return replaced_prompt_ids
    return prompt_ids


def _replace_nemotron_generation_think_open_with_empty_block(
    tokenizer, prompt_ids: list[int], full_ids: list[int]
) -> list[int]:
    if full_ids[: len(prompt_ids)] == prompt_ids:
        return prompt_ids
    tokenizer_name = _tokenizer_name(tokenizer).lower()
    if "nemotron" not in tokenizer_name:
        return prompt_ids

    think_open_ids = tokenizer.encode("<think>\n", add_special_tokens=False)
    think_empty_block_ids = tokenizer.encode("<think></think>", add_special_tokens=False)
    if not think_open_ids or not think_empty_block_ids:
        return prompt_ids
    if prompt_ids[-len(think_open_ids) :] != think_open_ids:
        return prompt_ids

    replaced_prompt_ids = prompt_ids[: -len(think_open_ids)] + think_empty_block_ids
    if full_ids[: len(replaced_prompt_ids)] == replaced_prompt_ids:
        return replaced_prompt_ids
    return prompt_ids


def _replace_qwen_generation_think_open_with_empty_block(
    tokenizer, prompt_ids: list[int], full_ids: list[int]
) -> list[int]:
    """Reconcile Qwen-family thinking templates on the prompt side.

    Qwen3-*-Thinking (e.g. ``Qwen3-4B-Thinking-2507``) append ``<think>\\n`` at the
    generation prompt, but a plain (reasoning-free) assistant message renders an
    *empty* block ``<think>\\n\\n</think>\\n\\n``. The prompt's ``\\n`` and the block's
    ``\\n\\n`` tokenize to different ids, so the plain-assistant suffix diff cannot
    align. Fold the empty block into the prompt (exactly like the Nemotron helper)
    so the assistant output is extracted correctly, without injecting any synthetic
    ``<think>`` content into the mock message.
    """
    if full_ids[: len(prompt_ids)] == prompt_ids:
        return prompt_ids
    name = _tokenizer_name(tokenizer).lower()
    if "qwen" not in name and "mimo" not in name:
        return prompt_ids
    think_open_ids = tokenizer.encode("<think>\n", add_special_tokens=False)
    think_empty_block_ids = tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
    if not think_open_ids or not think_empty_block_ids:
        return prompt_ids
    if prompt_ids[-len(think_open_ids) :] != think_open_ids:
        return prompt_ids
    replaced_prompt_ids = prompt_ids[: -len(think_open_ids)] + think_empty_block_ids
    if full_ids[: len(replaced_prompt_ids)] == replaced_prompt_ids:
        return replaced_prompt_ids
    return prompt_ids


def _trim_gemma_generation_thought_channel(tokenizer, prompt_ids: list[int], full_ids: list[int]) -> list[int]:
    """Reconcile Gemma-4's thought-channel generation prompt.

    Gemma-4 opens a reasoning channel at the generation prompt
    (``<|channel>thought\\n<channel|>``), but a plain (reasoning-free) assistant
    message renders no channel at all -- it goes straight from ``model\\n`` to the
    content. The opener has no counterpart in the full render, so the plain-assistant
    suffix diff cannot align. Trim the trailing opener from the prompt so the assistant
    output is extracted correctly, without injecting any synthetic reasoning content.
    """
    if full_ids[: len(prompt_ids)] == prompt_ids:
        return prompt_ids
    if "gemma" not in _tokenizer_name(tokenizer).lower():
        return prompt_ids
    opener_ids = tokenizer.encode("<|channel>thought\n<channel|>", add_special_tokens=False)
    if not opener_ids:
        return prompt_ids
    if prompt_ids[-len(opener_ids) :] != opener_ids:
        return prompt_ids
    trimmed_prompt_ids = prompt_ids[: -len(opener_ids)]
    if full_ids[: len(trimmed_prompt_ids)] == trimmed_prompt_ids:
        return trimmed_prompt_ids
    return prompt_ids


def _tokenizer_name(tokenizer) -> str:
    name = getattr(tokenizer, "name_or_path", None)
    if name:
        return str(name)
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, dict) and init_kwargs.get("name_or_path"):
        return str(init_kwargs["name_or_path"])
    return ""


def _tokenizer_eos_token_ids(tokenizer) -> set[int]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, int):
        return {eos_token_id}
    if isinstance(eos_token_id, list | tuple | set):
        return {int(token_id) for token_id in eos_token_id if token_id is not None}
    raise TypeError(f"Unsupported eos_token_id type: {type(eos_token_id)!r}")


def _truncate_after_final_eos(
    tokenizer,
    token_ids: list[int],
    *,
    assistant_message: dict[str, Any],
) -> list[int]:
    eos_token_ids = _tokenizer_eos_token_ids(tokenizer)
    eos_token_ids.update(_kimi_assistant_end_token_ids(tokenizer))
    eos_token_ids.update(_gemma_assistant_end_token_ids(tokenizer))
    eos_token_ids.update(_gptoss_assistant_end_token_ids(tokenizer))
    if not token_ids:
        raise ValueError("Assistant output token-id suffix is empty")
    if eos_token_ids:
        if token_ids[-1] in eos_token_ids:
            return token_ids
        for index in range(len(token_ids) - 1, -1, -1):
            if token_ids[index] in eos_token_ids:
                return token_ids[: index + 1]

    if _is_glm_tokenizer(tokenizer):
        if assistant_message.get("tool_calls"):
            return token_ids + [_require_single_token_id(tokenizer, "<|observation|>")]
        return token_ids

    if "gemma" in _tokenizer_name(tokenizer).lower():
        # Gemma-4 renders an assistant tool-call turn as
        # ``<|tool_call>...<tool_call|>`` + content + ``<|tool_response>`` -- the
        # trailing ``<|tool_response>`` is the opener of the following tool message,
        # emitted as scaffolding. The model stops before it and the loop appends the
        # tool response separately, so drop that trailing opener.
        tool_response_open_id = tokenizer.convert_tokens_to_ids("<|tool_response>")
        if (
            isinstance(tool_response_open_id, int)
            and tool_response_open_id >= 0
            and token_ids[-1] == tool_response_open_id
        ):
            return token_ids[:-1]

    tail = token_ids[-16:]
    raise ValueError(
        f"Assistant output token-id suffix does not contain eos_token_id {sorted(eos_token_ids)}; tail={tail}"
    )


def _is_glm_tokenizer(tokenizer) -> bool:
    tokenizer_name = _tokenizer_name(tokenizer).lower()
    compact_name = "".join(char for char in tokenizer_name if char.isalnum())
    return any(marker in tokenizer_name for marker in ("glm-4.7", "glm_4.7", "glm-5", "glm_5")) or any(
        marker in compact_name for marker in ("glm47", "glm5")
    )


def _kimi_assistant_end_token_ids(tokenizer) -> set[int]:
    tokenizer_name = _tokenizer_name(tokenizer).lower()
    if "kimi" not in tokenizer_name:
        return set()
    token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(token_id, int) and token_id >= 0:
        return {token_id}
    return set()


def _gemma_assistant_end_token_ids(tokenizer) -> set[int]:
    """Gemma-4 ends an assistant turn with ``<turn|>`` (not the tokenizer eos)."""
    if "gemma" not in _tokenizer_name(tokenizer).lower():
        return set()
    token_id = tokenizer.convert_tokens_to_ids("<turn|>")
    if isinstance(token_id, int) and token_id >= 0:
        return {token_id}
    return set()


def _is_gptoss_tokenizer(tokenizer) -> bool:
    tokenizer_name = _tokenizer_name(tokenizer).lower()
    compact_name = "".join(char for char in tokenizer_name if char.isalnum())
    return "gpt-oss" in tokenizer_name or "gptoss" in compact_name


def _gptoss_assistant_generation_ids(
    tokenizer,
    assistant_message: dict[str, Any],
    *,
    tools: list[dict[str, Any]] | None,
) -> list[int]:
    """Return the GPT-OSS assistant turn's *own* generated token ids.

    The harmony template drops prior tool-call turns' analysis channels when a
    later final answer exists (see the template's own "CoT is dropped during all
    previous turns" comment), which breaks the whole-list append-only diff. This
    turn's own generation is position-independent, so it can be rendered in
    isolation.

    Tool-call turns are rendered in the model's *real* harmony layout (recipient
    in the channel header + ``<|constrain|>json``) rather than via the chat
    template: this checkpoint's template puts the recipient in the role header and
    omits ``<|constrain|>``, a layout verl's ``GptOssToolParser`` cannot match, so
    the tool-agent flow would stall. Using the real layout (what the model emits
    at rollout and what the parser expects) lets the flow complete.

    Non-tool (final answer) turns have no tool call to reformat, so they are
    rendered via a trivial anchor prompt (where nothing can be dropped) and the
    delta is taken. The special-token boundaries make the content tokenization
    identical to the real context.
    """
    tool_calls = assistant_message.get("tool_calls")
    if tool_calls:
        return _gptoss_tool_call_generation_ids(tokenizer, assistant_message, tool_calls)
    anchor = [{"role": "user", "content": "x"}]
    base_ids = _render_chat_tokens(tokenizer, anchor, tools=tools, add_generation_prompt=True)
    full_ids = _render_chat_tokens(tokenizer, anchor + [assistant_message], tools=tools, add_generation_prompt=False)
    if full_ids[: len(base_ids)] != base_ids:
        raise ValueError("GPT-OSS anchor render is not append-only")
    return full_ids[len(base_ids) :]


def _gptoss_tool_call_generation_ids(
    tokenizer,
    assistant_message: dict[str, Any],
    tool_calls: list[dict[str, Any]],
) -> list[int]:
    """Build a GPT-OSS tool-call turn's generated tokens in the model's real
    harmony layout: an optional analysis (CoT) message followed by one commentary
    tool-call message per call with ``to=functions.NAME`` in the channel header
    and a ``<|constrain|>json`` constraint token, terminated by ``<|call|>``."""
    parts: list[str] = []
    reasoning = assistant_message.get("content")
    if reasoning:
        # The analysis message completes the ``<|start|>assistant`` supplied by the
        # generation prompt; the tool call then opens a fresh assistant message.
        parts.append(f"<|channel|>analysis<|message|>{reasoning}<|end|>")
        first_prefix = "<|start|>assistant"
    else:
        # No CoT: the tool call reuses the generation prompt's ``<|start|>assistant``.
        first_prefix = ""
    for index, tool_call in enumerate(tool_calls):
        function = tool_call.get("function", tool_call)
        name = function["name"]
        arguments = function.get("arguments", {})
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        prefix = first_prefix if index == 0 else "<|start|>assistant"
        parts.append(
            f"{prefix}<|channel|>commentary to=functions.{name} <|constrain|>json<|message|>{arguments}<|call|>"
        )
    text = "".join(parts)
    return normalize_token_ids(tokenizer.encode(text, add_special_tokens=False))


def _gptoss_assistant_end_token_ids(tokenizer) -> set[int]:
    """GPT-OSS (harmony) ends a final answer with ``<|return|>`` (the tokenizer eos)
    but ends an assistant tool-call turn with ``<|call|>``. ``<|call|>`` is the model's
    real stop token for a tool call, so keep it as a valid assistant-end token."""
    tokenizer_name = _tokenizer_name(tokenizer).lower()
    compact_name = "".join(char for char in tokenizer_name if char.isalnum())
    if "gpt-oss" not in tokenizer_name and "gptoss" not in compact_name:
        return set()
    token_id = tokenizer.convert_tokens_to_ids("<|call|>")
    if isinstance(token_id, int) and token_id >= 0:
        return {token_id}
    return set()


def _require_single_token_id(tokenizer, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        raise ValueError(f"Tokenizer does not define required token {token!r}")
    if isinstance(token_id, list):
        if len(token_id) != 1:
            raise ValueError(f"Tokenizer returned multiple ids for required token {token!r}: {token_id!r}")
        token_id = token_id[0]
    if not isinstance(token_id, int) or token_id < 0:
        raise ValueError(f"Tokenizer returned invalid id for required token {token!r}: {token_id!r}")
    return token_id


def _assistant_logprobs(turn_index: int, length: int) -> list[float]:
    return [round(-0.01 * (turn_index + 1) - 0.0001 * ((index % 19) + 1), 7) for index in range(length)]


def _prepare_assistant_outputs(
    tokenizer,
    trajectory: ToolAgentTrajectory,
) -> list[PreparedAssistantOutput]:
    tools = trajectory.tool_schemas()
    canonical_messages = [json.loads(json.dumps(message, ensure_ascii=False)) for message in trajectory.raw_prompt]
    assistant_outputs: list[PreparedAssistantOutput] = []

    for turn_index, step in enumerate(trajectory.steps):
        assistant_message = json.loads(json.dumps(step.assistant, ensure_ascii=False))
        assistant_ids = _render_assistant_output_ids(
            tokenizer,
            canonical_messages,
            assistant_message,
            tools=tools,
        )
        assistant_outputs.append(
            PreparedAssistantOutput(
                token_ids=assistant_ids,
                log_probs=_assistant_logprobs(turn_index, len(assistant_ids)),
            )
        )
        canonical_messages.append(assistant_message)
        canonical_messages.extend(
            json.loads(json.dumps(message, ensure_ascii=False)) for message in step.appended_messages
        )

    return assistant_outputs


def _prepare_single_turn_assistant_output(
    tokenizer,
    trajectory: SingleTurnTrajectory,
) -> PreparedAssistantOutput:
    assistant_ids = _render_assistant_output_ids(
        tokenizer,
        [json.loads(json.dumps(message, ensure_ascii=False)) for message in trajectory.raw_prompt],
        {"role": "assistant", "content": trajectory.assistant_response},
        tools=None,
    )
    return PreparedAssistantOutput(
        token_ids=assistant_ids,
        log_probs=_assistant_logprobs(0, len(assistant_ids)),
    )


def _make_deterministic_tools(trajectory: ToolAgentTrajectory) -> list[FunctionTool]:
    responses_by_tool: dict[str, list[str]] = {}
    responses_by_call: dict[tuple[str, str], list[str]] = {}
    for step in trajectory.steps:
        tool_calls = step.assistant.get("tool_calls") or []
        tool_messages = []
        for message in step.appended_messages:
            if message.get("role") != "tool":
                raise ValueError(
                    f"Trajectory {trajectory.name!r} appends role={message.get('role')!r}; "
                    "the ToolAgentLoop e2e compare currently supports tool-response trajectories only"
                )
            tool_messages.append(message)
            name = message.get("name")
            if not name:
                raise ValueError(f"Trajectory {trajectory.name!r} has a tool message without name: {message!r}")
            responses_by_tool.setdefault(name, []).append(str(message.get("content", "")))
        for tool_call, tool_message in zip(tool_calls, tool_messages, strict=False):
            function_call = tool_call.get("function") or {}
            name = function_call.get("name")
            if not name:
                continue
            arguments_key = _canonical_arguments_key(function_call.get("arguments") or {})
            responses_by_call.setdefault((name, arguments_key), []).append(str(tool_message.get("content", "")))

    deterministic_tools: list[FunctionTool] = []
    for tool in trajectory.tools:
        response_queue = list(responses_by_tool.get(tool.name, []))
        responses_by_args = {
            arguments_key: list(queue)
            for (tool_name, arguments_key), queue in responses_by_call.items()
            if tool_name == tool.name
        }

        def _make_tool_fn(tool_name: str, queue: list[str], by_args: dict[str, list[str]]):
            def _tool_fn(**kwargs):
                arguments_key = _canonical_arguments_key(kwargs)
                if arguments_key in by_args and by_args[arguments_key]:
                    response = by_args[arguments_key].pop(0)
                    if response in queue:
                        queue.remove(response)
                    return response
                if not queue:
                    raise RuntimeError(f"No deterministic response left for tool {tool_name!r}")
                return queue.pop(0)

            return _tool_fn

        deterministic_tools.append(
            FunctionTool(
                name=tool.name,
                fn=_make_tool_fn(tool.name, response_queue, responses_by_args),
                tool_schema=tool.tool_schema,
            )
        )
    return deterministic_tools


def _canonical_arguments_key(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _continuous_token_config(enable_ct: bool) -> dict[str, Any]:
    return {
        "enable": enable_ct,
        "model_family": "auto",
        "custom_builder_module": None,
    }


def _make_tool_agent_config(
    *,
    model: str,
    enable_ct: bool,
    tool_parser: str,
    trajectory: ToolAgentTrajectory,
):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {"path": model, "tokenizer_path": model},
                "rollout": {
                    "prompt_length": 32768,
                    "response_length": 32768,
                    "multi_turn": {
                        "format": tool_parser,
                        "max_user_turns": trajectory.expected_num_turns + 4,
                        "max_assistant_turns": trajectory.expected_generation_turns + 4,
                        "max_parallel_calls": trajectory.max_parallel_calls,
                        "max_tool_response_length": 8192,
                        "tool_response_truncate_side": "right",
                        "continuous_token": _continuous_token_config(enable_ct),
                    },
                },
            }
        }
    )


def _make_single_turn_config(*, model: str, enable_ct: bool):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {"path": model, "tokenizer_path": model},
                "rollout": {
                    "prompt_length": 32768,
                    "response_length": 32768,
                    "multi_turn": {
                        "continuous_token": _continuous_token_config(enable_ct),
                    },
                },
            }
        }
    )


def _infer_tool_parser(model: str, requested_tool_parser: str) -> str:
    if requested_tool_parser != "auto":
        return requested_tool_parser
    normalized = model.lower()
    compact = "".join(char for char in normalized if char.isalnum())
    if any(marker in normalized for marker in ("glm-4.7", "glm_4.7", "glm-5", "glm_5")) or any(
        marker in compact for marker in ("glm47", "glm5")
    ):
        return "glm"
    if "qwen35" in compact or "qwen3coder" in compact:
        return "qwen3_coder"
    if "kimi" in normalized:
        return "kimi"
    if "bytedance-seed" in normalized or "seedoss" in compact or "seedcoder" in compact:
        return "seed"
    if "minimax" in compact:
        return "minimax"
    if "nemotron" in normalized:
        return "qwen3_coder"
    if "gemma" in normalized:
        return "gemma4"
    if "gpt-oss" in normalized or "gptoss" in compact:
        return "gpt-oss"
    return "hermes"


def _data_config(enable_ct: bool):
    return OmegaConf.create(
        {
            "apply_chat_template_kwargs": {},
            "mm_processor_kwargs": {},
            "continuous_token": _continuous_token_config(enable_ct),
        }
    )


async def _run_e2e_path(
    tokenizer,
    model: str,
    trajectory: ToolAgentTrajectory,
    *,
    use_ct: bool,
    tool_parser: str,
) -> E2ERunOutput:
    server = _DeterministicServer(_prepare_assistant_outputs(tokenizer, trajectory))
    loop = ToolAgentLoop(
        trainer_config=DictConfigWrap(
            _make_tool_agent_config(model=model, enable_ct=use_ct, tool_parser=tool_parser, trajectory=trajectory)
        ),
        server_manager=server,
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=_NoopDataset,
        data_config=DictConfigWrap(_data_config(use_ct)),
        tools=ToolListWrap(_make_deterministic_tools(trajectory)),
    )
    output: AgentLoopOutput = await loop.run(
        sampling_params={"logprobs": True},
        raw_prompt=[json.loads(json.dumps(message, ensure_ascii=False)) for message in trajectory.raw_prompt],
    )
    return E2ERunOutput(
        prompt_ids=output.prompt_ids,
        response_ids=output.response_ids,
        response_mask=output.response_mask,
        response_logprobs=output.response_logprobs,
        generation_prompt_ids=server.prompt_ids_by_call,
        num_turns=output.num_turns,
    )


async def _run_single_turn_e2e_path(
    tokenizer,
    model: str,
    trajectory: SingleTurnTrajectory,
    *,
    use_ct: bool,
) -> E2ERunOutput:
    server = _DeterministicServer([_prepare_single_turn_assistant_output(tokenizer, trajectory)])
    loop = SingleTurnAgentLoop(
        trainer_config=DictConfigWrap(_make_single_turn_config(model=model, enable_ct=use_ct)),
        server_manager=server,
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=_NoopDataset,
        data_config=DictConfigWrap(_data_config(use_ct)),
    )
    output: AgentLoopOutput = await loop.run(
        sampling_params={"logprobs": True},
        raw_prompt=[json.loads(json.dumps(message, ensure_ascii=False)) for message in trajectory.raw_prompt],
    )
    return E2ERunOutput(
        prompt_ids=output.prompt_ids,
        response_ids=output.response_ids,
        response_mask=output.response_mask,
        response_logprobs=output.response_logprobs,
        generation_prompt_ids=server.prompt_ids_by_call,
        num_turns=output.num_turns,
    )


def _hash_sequence(seq: list[int] | list[float] | None) -> str | None:
    if seq is None:
        return None
    blob = json.dumps(seq, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _first_mismatch(left: list[Any] | None, right: list[Any] | None) -> int | None:
    if left is None or right is None:
        return None if left is right else 0
    for index, (left_item, right_item) in enumerate(zip(left, right, strict=False)):
        if left_item != right_item:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _compare_sequence(left: list[Any] | None, right: list[Any] | None) -> dict[str, Any]:
    left_len = None if left is None else len(left)
    right_len = None if right is None else len(right)
    first_mismatch = _first_mismatch(left, right)
    return {
        "equal": left == right,
        "legacy_len": left_len,
        "ct_len": right_len,
        "first_mismatch": first_mismatch,
        "legacy_hash": _hash_sequence(left),
        "ct_hash": _hash_sequence(right),
        "legacy_value": (
            None if first_mismatch is None or left is None or first_mismatch >= len(left) else left[first_mismatch]
        ),
        "ct_value": (
            None if first_mismatch is None or right is None or first_mismatch >= len(right) else right[first_mismatch]
        ),
    }


def _full_token_ids(output: E2ERunOutput) -> list[int]:
    return list(output.prompt_ids) + list(output.response_ids)


def _decode_token_ids(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def _compare_generation_prompts(legacy: E2ERunOutput, ct: E2ERunOutput) -> dict[str, Any]:
    comparisons = []
    turn_count = max(len(legacy.generation_prompt_ids), len(ct.generation_prompt_ids))
    for turn_index in range(turn_count):
        legacy_ids = (
            legacy.generation_prompt_ids[turn_index] if turn_index < len(legacy.generation_prompt_ids) else None
        )
        ct_ids = ct.generation_prompt_ids[turn_index] if turn_index < len(ct.generation_prompt_ids) else None
        comparisons.append({"turn": turn_index + 1, **_compare_sequence(legacy_ids, ct_ids)})
    return {
        "equal": all(item["equal"] for item in comparisons),
        "turns": comparisons,
    }


def _summarize_output(output: E2ERunOutput) -> dict[str, Any]:
    response_logprobs_len = None if output.response_logprobs is None else len(output.response_logprobs)
    return {
        "num_turns": output.num_turns,
        "generation_turns": len(output.generation_prompt_ids),
        "generation_prompt_lengths": [len(ids) for ids in output.generation_prompt_ids],
        "token_len": len(_full_token_ids(output)),
        "prompt_len": len(output.prompt_ids),
        "response_len": len(output.response_ids),
        "response_mask_len": len(output.response_mask),
        "response_logprobs_len": response_logprobs_len,
    }


def _run_e2e_path_sync(
    tokenizer,
    model: str,
    trajectory: ToolAgentTrajectory,
    *,
    use_ct: bool,
    tool_parser: str,
) -> E2ERunOutput:
    return asyncio.run(
        _run_e2e_path(
            tokenizer,
            model,
            trajectory,
            use_ct=use_ct,
            tool_parser=tool_parser,
        )
    )


def _run_single_turn_e2e_path_sync(
    tokenizer,
    model: str,
    trajectory: SingleTurnTrajectory,
    *,
    use_ct: bool,
) -> E2ERunOutput:
    return asyncio.run(
        _run_single_turn_e2e_path(
            tokenizer,
            model,
            trajectory,
            use_ct=use_ct,
        )
    )


def _capture_e2e_path(
    tokenizer,
    model: str,
    trajectory: ToolAgentTrajectory,
    *,
    use_ct: bool,
    tool_parser: str,
) -> tuple[E2ERunOutput | None, dict[str, Any] | None]:
    try:
        return (
            _run_e2e_path_sync(
                tokenizer,
                model,
                trajectory,
                use_ct=use_ct,
                tool_parser=tool_parser,
            ),
            None,
        )
    except Exception as exc:
        return (
            None,
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )


def _capture_single_turn_e2e_path(
    tokenizer,
    model: str,
    trajectory: SingleTurnTrajectory,
    *,
    use_ct: bool,
) -> tuple[E2ERunOutput | None, dict[str, Any] | None]:
    try:
        return (
            _run_single_turn_e2e_path_sync(
                tokenizer,
                model,
                trajectory,
                use_ct=use_ct,
            ),
            None,
        )
    except Exception as exc:
        return (
            None,
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )


def _json_safe_path(path: Path) -> str:
    return str(path)


def _artifact_slug(model: str, trajectory_name: str) -> str:
    raw = f"{model}__{trajectory_name}"
    return "".join(char.lower() if char.isalnum() else "_" for char in raw).strip("_")


def _output_dump_payload(tokenizer, output: E2ERunOutput) -> dict[str, Any]:
    token_ids = _full_token_ids(output)
    return {
        "token_ids": token_ids,
        "prompt_ids": output.prompt_ids,
        "response_ids": output.response_ids,
        "response_mask": output.response_mask,
        "loss_mask": output.response_mask,
        "response_logprobs": output.response_logprobs,
        "logprobs": output.response_logprobs,
        "generation_prompt_ids": output.generation_prompt_ids,
        "decoded_text": _decode_token_ids(tokenizer, token_ids),
        "decoded_prompt": _decode_token_ids(tokenizer, output.prompt_ids),
        "decoded_response": _decode_token_ids(tokenizer, output.response_ids),
        "summary": _summarize_output(output),
    }


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _dump_output_artifact(
    *,
    tokenizer,
    dump_dir: Path,
    slug: str,
    side: str,
    output: E2ERunOutput | None,
    error: dict[str, Any] | None,
) -> dict[str, str]:
    json_path = dump_dir / f"{slug}_{side}.json"
    artifact_paths = {f"{side}_json": _json_safe_path(json_path)}
    if output is None:
        _write_json(json_path, {"error": error})
        return artifact_paths

    payload = _output_dump_payload(tokenizer, output)
    _write_json(json_path, payload)
    decoded_path = dump_dir / f"{slug}_{side}_decoded.txt"
    _write_text(decoded_path, payload["decoded_text"])
    artifact_paths[f"{side}_decoded_text"] = _json_safe_path(decoded_path)
    return artifact_paths


def _dump_case_artifacts(
    *,
    tokenizer,
    dump_dir: Path | None,
    model: str,
    trajectory_name: str,
    legacy_output: E2ERunOutput | None,
    legacy_error: dict[str, Any] | None,
    ct_output: E2ERunOutput | None,
    ct_error: dict[str, Any] | None,
) -> dict[str, str]:
    if dump_dir is None:
        return {}
    slug = _artifact_slug(model, trajectory_name)
    artifacts: dict[str, str] = {}
    artifacts.update(
        _dump_output_artifact(
            tokenizer=tokenizer,
            dump_dir=dump_dir,
            slug=slug,
            side="legacy",
            output=legacy_output,
            error=legacy_error,
        )
    )
    artifacts.update(
        _dump_output_artifact(
            tokenizer=tokenizer,
            dump_dir=dump_dir,
            slug=slug,
            side="ct",
            output=ct_output,
            error=ct_error,
        )
    )
    return artifacts


def run_one_case(
    tokenizer,
    model: str,
    trajectory: MockTrajectory,
    *,
    tool_parser: str,
    dump_dir: Path | None = None,
) -> dict[str, Any]:
    loop_type = "single_turn_agent_loop" if isinstance(trajectory, SingleTurnTrajectory) else "tool_agent_loop"
    result: dict[str, Any] = {
        "model": model,
        "trajectory": trajectory.name,
        "loop_type": loop_type,
        "tool_parser": tool_parser if isinstance(trajectory, ToolAgentTrajectory) else None,
    }
    if isinstance(trajectory, SingleTurnTrajectory):
        legacy_output, legacy_error = _capture_single_turn_e2e_path(
            tokenizer,
            model,
            trajectory,
            use_ct=False,
        )
        ct_output, ct_error = _capture_single_turn_e2e_path(
            tokenizer,
            model,
            trajectory,
            use_ct=True,
        )
        expected_generation_turns = 1
    else:
        legacy_output, legacy_error = _capture_e2e_path(
            tokenizer,
            model,
            trajectory,
            use_ct=False,
            tool_parser=tool_parser,
        )
        ct_output, ct_error = _capture_e2e_path(
            tokenizer,
            model,
            trajectory,
            use_ct=True,
            tool_parser=tool_parser,
        )
        expected_generation_turns = trajectory.expected_generation_turns

    artifacts = _dump_case_artifacts(
        tokenizer=tokenizer,
        dump_dir=dump_dir,
        model=model,
        trajectory_name=trajectory.name,
        legacy_output=legacy_output,
        legacy_error=legacy_error,
        ct_output=ct_output,
        ct_error=ct_error,
    )

    if legacy_error or ct_error:
        result.update(
            {
                "status": "error",
                "expected_flow": False,
                "legacy_status": "error" if legacy_error else "pass",
                "ct_status": "error" if ct_error else "pass",
                "legacy": _summarize_output(legacy_output) if legacy_output is not None else None,
                "ct": _summarize_output(ct_output) if ct_output is not None else None,
                "legacy_error": legacy_error,
                "ct_error": ct_error,
                "error_type": (ct_error or legacy_error or {}).get("error_type"),
                "error": (ct_error or legacy_error or {}).get("error"),
                "expected_num_turns": trajectory.expected_num_turns,
                "expected_generation_turns": expected_generation_turns,
                "ct_legacy_match": False,
                "artifacts": artifacts,
            }
        )
        return result

    assert legacy_output is not None
    assert ct_output is not None
    comparisons = {
        "prompt_ids": _compare_sequence(legacy_output.prompt_ids, ct_output.prompt_ids),
        "response_ids": _compare_sequence(legacy_output.response_ids, ct_output.response_ids),
        "loss_mask": _compare_sequence(legacy_output.response_mask, ct_output.response_mask),
        "logprobs": _compare_sequence(legacy_output.response_logprobs, ct_output.response_logprobs),
        "generation_prompt_ids": _compare_generation_prompts(legacy_output, ct_output),
    }
    expected_flow = (
        legacy_output.num_turns == trajectory.expected_num_turns
        and ct_output.num_turns == trajectory.expected_num_turns
        and len(legacy_output.generation_prompt_ids) == expected_generation_turns
        and len(ct_output.generation_prompt_ids) == expected_generation_turns
    )
    exact_match = all(comparisons[name]["equal"] for name in ("prompt_ids", "response_ids", "loss_mask", "logprobs"))
    result.update(
        {
            "status": "pass" if expected_flow and exact_match else "mismatch",
            "expected_flow": expected_flow,
            "exact_output_match": exact_match,
            "ct_legacy_match": exact_match,
            "legacy_status": "pass",
            "ct_status": "pass",
            "legacy": _summarize_output(legacy_output),
            "ct": _summarize_output(ct_output),
            "comparisons": comparisons,
            "expected_num_turns": trajectory.expected_num_turns,
            "expected_generation_turns": expected_generation_turns,
            "artifacts": artifacts,
        }
    )
    return result


def _discover_cached_models() -> list[str]:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    if not hub.exists():
        return []
    models = []
    for path in hub.glob("models--*"):
        if path.is_dir() and not path.name.startswith("."):
            models.append(path.name.removeprefix("models--").replace("--", "/"))
    return sorted(set(models))


def _select_models(args) -> list[str]:
    models = list(args.model or DEFAULT_MODELS)
    if args.discover_cache:
        models.extend(_discover_cached_models())
    seen = set()
    selected = []
    for model in models:
        if model not in seen:
            selected.append(model)
            seen.add(model)
    return selected


def _select_trajectories(args) -> list[MockTrajectory]:
    names = args.trajectory or list(DEFAULT_E2E_TRAJECTORIES)
    return [get_trajectory(name) for name in names]


def _print_flow_details(result: dict[str, Any]) -> None:
    if result.get("expected_flow"):
        return
    legacy = result.get("legacy") or {}
    ct = result.get("ct") or {}
    print(
        "    flow: incomplete "
        f"expected_num_turns={result.get('expected_num_turns')} "
        f"expected_generation_turns={result.get('expected_generation_turns')} "
        f"legacy_num_turns={legacy.get('num_turns')} "
        f"ct_num_turns={ct.get('num_turns')} "
        f"legacy_generation_turns={legacy.get('generation_turns')} "
        f"ct_generation_turns={ct.get('generation_turns')}",
        flush=True,
    )


def _print_sequence_mismatch(name: str, comparison: dict[str, Any]) -> None:
    if comparison.get("equal"):
        return
    print(
        f"    {name}: legacy_len={comparison.get('legacy_len')} "
        f"ct_len={comparison.get('ct_len')} "
        f"first_mismatch={comparison.get('first_mismatch')} "
        f"legacy_value={comparison.get('legacy_value')} "
        f"ct_value={comparison.get('ct_value')}",
        flush=True,
    )


def _format_ct_legacy_state(match: Any) -> str:
    return f"match={match}"


def _format_result_match_state(result: dict[str, Any]) -> str:
    ct_legacy_state = _format_ct_legacy_state(result.get("ct_legacy_match"))
    return f"ct_vs_legacy({ct_legacy_state})"


def _print_mismatch_details(result: dict[str, Any]) -> None:
    _print_flow_details(result)
    comparisons = result.get("comparisons") or {}
    for name in ("prompt_ids", "response_ids", "loss_mask", "logprobs"):
        comparison = comparisons.get(name)
        if comparison:
            _print_sequence_mismatch(name, comparison)

    generation_prompt_comparison = comparisons.get("generation_prompt_ids") or {}
    for turn in generation_prompt_comparison.get("turns", []):
        if turn.get("equal"):
            continue
        print(
            f"    generation_prompt_ids turn={turn.get('turn')}: "
            f"legacy_len={turn.get('legacy_len')} "
            f"ct_len={turn.get('ct_len')} "
            f"first_mismatch={turn.get('first_mismatch')} "
            f"legacy_value={turn.get('legacy_value')} "
            f"ct_value={turn.get('ct_value')}",
            flush=True,
        )

    if result.get("expected_flow") is False and result.get("exact_output_match") is True:
        print(
            "    reason: token outputs match, but the agent loop did not complete the expected flow.",
            flush=True,
        )
    elif result.get("expected_flow") is True and result.get("exact_output_match") is False:
        print(
            "    reason: flow completed, but CT and legacy diverged at the token/metadata positions above.",
            flush=True,
        )


def _print_error_details(result: dict[str, Any]) -> None:
    print(
        f"    status: legacy={result.get('legacy_status', 'error')} ct={result.get('ct_status', 'error')}",
        flush=True,
    )
    printed_side_error = False
    for side in ("legacy", "ct"):
        error = result.get(f"{side}_error")
        if not error:
            continue
        printed_side_error = True
        print(
            f"    {side}_error: {error.get('error_type')}: {error.get('error')}",
            flush=True,
        )
    if not printed_side_error:
        print(f"    error: {result.get('error_type')}: {result.get('error')}", flush=True)
    _print_flow_details(result)


def _resolve_dump_dir(args) -> Path | None:
    if not args.dump_sequences:
        return None
    if args.dump_dir:
        return Path(args.dump_dir)
    if args.ledger:
        return Path(args.ledger).with_suffix("").parent / f"{Path(args.ledger).stem}_artifacts"
    return Path("/private/tmp/verl_ct_vs_legacy_artifacts")


def _run_all(args) -> list[dict[str, Any]]:
    results = []
    models = _select_models(args)
    trajectories = _select_trajectories(args)
    dump_dir = _resolve_dump_dir(args)
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
        print(f"Dump artifacts: {dump_dir}", flush=True)
    total = len(models) * len(trajectories)
    case_index = 0

    for model in models:
        tool_parser = _infer_tool_parser(model, args.tool_parser)
        try:
            tokenizer = _load_tokenizer(model, local_files_only=args.local_files_only)
        except Exception as exc:
            for trajectory in trajectories:
                case_index += 1
                print(f"[{case_index}/{total}] {model} :: {trajectory.name}", flush=True)
                result = {
                    "model": model,
                    "trajectory": trajectory.name,
                    "loop_type": (
                        "single_turn_agent_loop" if isinstance(trajectory, SingleTurnTrajectory) else "tool_agent_loop"
                    ),
                    "requested_tool_parser": args.tool_parser if isinstance(trajectory, ToolAgentTrajectory) else None,
                    "tool_parser": tool_parser if isinstance(trajectory, ToolAgentTrajectory) else None,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "ct_legacy_match": False,
                }
                results.append(result)
                print(
                    f"  ERROR {result['error_type']}: {result['error']} {_format_result_match_state(result)}",
                    flush=True,
                )
                _print_error_details(result)
            continue

        for trajectory in trajectories:
            case_index += 1
            print(f"[{case_index}/{total}] {model} :: {trajectory.name}", flush=True)
            result = run_one_case(
                tokenizer,
                model,
                trajectory,
                tool_parser=tool_parser,
                dump_dir=dump_dir,
            )
            if isinstance(trajectory, ToolAgentTrajectory):
                result["requested_tool_parser"] = args.tool_parser
            results.append(result)
            status = result["status"]
            if status == "pass":
                print(
                    f"  PASS {_format_result_match_state(result)}",
                    flush=True,
                )
            elif status == "mismatch":
                comparisons = result["comparisons"]
                print(
                    f"  MISMATCH {_format_result_match_state(result)} "
                    f"prompt={comparisons['prompt_ids']['equal']} "
                    f"response={comparisons['response_ids']['equal']} "
                    f"mask={comparisons['loss_mask']['equal']} "
                    f"logprobs={comparisons['logprobs']['equal']} "
                    f"flow={result['expected_flow']}",
                    flush=True,
                )
                _print_mismatch_details(result)
            else:
                print(
                    f"  ERROR {result.get('error_type')}: {result.get('error')} {_format_result_match_state(result)}",
                    flush=True,
                )
                _print_error_details(result)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", help="Model/tokenizer id to test. May be repeated.")
    parser.add_argument(
        "--trajectory",
        action="append",
        choices=sorted(TRAJECTORY_BY_NAME),
        help=(
            "Trajectory name to test. May be repeated. Defaults to e2e trajectories: "
            f"{', '.join(DEFAULT_E2E_TRAJECTORIES)}."
        ),
    )
    parser.add_argument(
        "--tool-parser",
        default="auto",
        choices=("auto", "hermes", "qwen3_coder", "gpt-oss", "gemma4", "glm", "kimi", "seed", "minimax"),
        help=(
            "ToolParser format used by ToolAgentLoop. Default: auto, which resolves from model name "
            "(for example Qwen3.5 -> qwen3_coder, GLM -> glm, Kimi -> kimi, "
            "Seed -> seed, MiniMax -> minimax, Nemotron -> qwen3_coder, otherwise hermes)."
        ),
    )
    parser.add_argument("--discover-cache", action="store_true", help="Also test all HF models found in local cache.")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow AutoTokenizer to download missing tokenizers instead of requiring local cache.",
    )
    parser.add_argument(
        "--ledger",
        help="Write the structured per-case report JSON to this path.",
    )
    parser.add_argument(
        "--dump-sequences",
        action="store_true",
        help="Dump CT/legacy token ids, decoded text, masks/logprobs to artifact files.",
    )
    parser.add_argument(
        "--dump-dir",
        help=(
            "Directory for --dump-sequences artifacts. Defaults to <ledger-stem>_artifacts next to --ledger, "
            "or /private/tmp/verl_ct_vs_legacy_artifacts when --ledger is omitted."
        ),
    )
    parser.add_argument("--fail-on-mismatch", action="store_true", help="Return non-zero when any case mismatches.")
    args = parser.parse_args()
    args.local_files_only = not args.allow_download
    return args


def _print_summary(results: list[dict[str, Any]]) -> None:
    total = len(results)
    pass_count = sum(item["status"] == "pass" for item in results)
    mismatch_count = sum(item["status"] == "mismatch" for item in results)
    error_count = sum(item["status"] == "error" for item in results)

    ct_legacy_error_count = sum(item.get("status") == "error" for item in results)
    ct_legacy_match_count = sum(item.get("ct_legacy_match") is True for item in results)
    ct_legacy_mismatch_count = sum(
        item.get("ct_legacy_match") is False and item.get("status") != "error" for item in results
    )

    print(
        f"Summary: total={total} pass={pass_count} mismatch={mismatch_count} error={error_count}",
        flush=True,
    )
    print(
        "  CT vs legacy: "
        f"match={ct_legacy_match_count} "
        f"mismatch={ct_legacy_mismatch_count} "
        f"error={ct_legacy_error_count}",
        flush=True,
    )


def _write_ledger(path: str, results: list[dict[str, Any]]) -> None:
    ledger_path = Path(path)
    _write_json(ledger_path, results)
    print(f"Ledger: {ledger_path}", flush=True)


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    results = _run_all(args)
    if args.ledger:
        _write_ledger(args.ledger, results)
    _print_summary(results)
    if args.fail_on_mismatch and any(item["status"] != "pass" for item in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
