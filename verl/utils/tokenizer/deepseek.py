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
"""Prompt encoding and Continuous Token support for the DeepSeek-V4 series.

DeepSeek-V4 does not use a role/content markup like ChatML. Turns are delimited by dedicated
special tokens, a run of consecutive tool/user messages folds into a single user turn, and tool
calls are written in DSML markup. :func:`encode_messages` is a port of the encoder shipped with the
model, which vLLM reimplements in ``vllm/tokenizers/deepseek_v4_encoding.py``. The matching decoder
lives in :class:`verl.experimental.agent_loop.tool_parser.DeepSeekV4ToolParser`.

The checkpoint ships no usable jinja template, so :class:`DeepSeekV4ContinuousTokenBuilder` renders
through :func:`encode_messages` rather than ``apply_chat_template``. Its root ``model_type`` selects
this builder automatically, removing the need to configure a chat template for rollout and keeping
the prompt byte-exact with the reference encoder -- ``tests/utils/test_deepseek_on_cpu.py`` replays
its golden fixtures.

Length filtering (``data.filter_overlong_prompts=True``) is the one rollout-side path that still
calls ``tokenizer.apply_chat_template`` directly, so leave it off for DeepSeek-V4.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from .continuous_token import ContinuousTokenBuilder, MergeResult, require_token_id
from .tokenizer import normalize_token_ids

BOS_TOKEN = "<｜begin▁of▁sentence｜>"
EOS_TOKEN = "<｜end▁of▁sentence｜>"
USER_TOKEN = "<｜User｜>"
ASSISTANT_TOKEN = "<｜Assistant｜>"
LATEST_REMINDER_TOKEN = "<｜latest_reminder｜>"
THINK_START_TOKEN = "<think>"
THINK_END_TOKEN = "</think>"
DSML_TOKEN = "｜DSML｜"

TOOL_CALLS_START_TOKEN = f"<{DSML_TOKEN}tool_calls>"
TOOL_CALLS_END_TOKEN = f"</{DSML_TOKEN}tool_calls>"

# Quick instruction tokens, selected per message through the optional ``task`` field.
TASK_TOKENS = {
    "action": "<｜action｜>",
    "query": "<｜query｜>",
    "authority": "<｜authority｜>",
    "domain": "<｜domain｜>",
    "title": "<｜title｜>",
    "read_url": "<｜read_url｜>",
}

# Roles that collapse into a single user turn when they appear back to back.
_USER_RUN_ROLES = ("user", "tool")

REASONING_EFFORT_MAX = (
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to "
    "resolve the root cause, rigorously stress-testing your logic against all potential paths, "
    "edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, "
    "considered alternative, and rejected hypothesis to ensure absolutely no assumption is left "
    "unchecked.\n\n"
)

TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by \
writing a "<{dsml}tool_calls>" block like the following:

<{dsml}tool_calls>
<{dsml}invoke name="$TOOL_NAME">
<{dsml}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml}parameter>
...
</{dsml}invoke>
<{dsml}invoke name="$TOOL_NAME2">
...
</{dsml}invoke>
</{dsml}tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, \
booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {think_start}), you MUST output your complete reasoning \
inside {think_start}...{think_end} BEFORE any tool calls or final response.

Otherwise, output directly after {think_end} with tool calls or final response.

### Available Tool Schemas

{schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""

RESPONSE_FORMAT_TEMPLATE = "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{schema}"


def _to_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return json.dumps(value, ensure_ascii=True)


def _text_of(content: Any) -> str:
    """Flatten a message content into text, accepting both plain strings and typed part lists."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if part.get("type") == "text")
    return str(content)


def render_tools(tools: list[dict[str, Any]]) -> str:
    """Render OpenAI-format tool schemas into the DSML tool prompt."""
    schemas = [_to_json(tool.get("function", tool)) for tool in tools]
    return TOOLS_TEMPLATE.format(
        dsml=DSML_TOKEN,
        think_start=THINK_START_TOKEN,
        think_end=THINK_END_TOKEN,
        schemas="\n".join(schemas),
    )


def _render_tool_call(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function", tool_call)
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:
            # Mirrors the reference encoder: unparseable arguments are passed through verbatim.
            arguments = {"arguments": arguments}
    if not isinstance(arguments, dict):
        arguments = {"arguments": arguments}

    params = [
        f'<{DSML_TOKEN}parameter name="{name}" string="{"true" if isinstance(value, str) else "false"}">'
        f"{value if isinstance(value, str) else _to_json(value)}</{DSML_TOKEN}parameter>"
        for name, value in arguments.items()
    ]
    return f'<{DSML_TOKEN}invoke name="{function.get("name")}">\n' + "\n".join(params) + f"\n</{DSML_TOKEN}invoke>"


def encode_messages(
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    *,
    add_generation_prompt: bool = True,
    add_bos_token: bool = True,
    enable_thinking: bool = False,
    drop_thinking: bool = True,
    reasoning_effort: Optional[str] = None,
) -> str:
    """Encode a conversation into a DeepSeek-V4 prompt.

    Args:
        messages: Conversation in OpenAI format. Supported roles are ``system``, ``developer``,
            ``user``, ``tool``, ``assistant`` and ``latest_reminder``.
        tools: OpenAI-format tool schemas, rendered as a leading system message.
        add_generation_prompt: Whether to open an assistant turn at the end.
        add_bos_token: Whether to emit the BOS token. Set to False when rendering a turn that is
            appended to an ongoing sequence.
        enable_thinking: False (chat mode) closes the thinking block right after the assistant
            token so the model answers directly.
        drop_thinking: Drop reasoning from assistant turns before the last user turn. Automatically
            disabled when tools are present, because tool calling needs the full reasoning trace.
        reasoning_effort: ``"max"``/``"xhigh"`` prepend the maximum effort preamble, ``"none"``
            disables thinking.

    Returns:
        The encoded prompt.
    """
    think = enable_thinking and reasoning_effort != "none"
    has_tools = bool(tools) or any(message.get("tools") for message in messages)
    drop = drop_thinking and not has_tools
    last_user = max(
        (i for i, m in enumerate(messages) if m["role"] in (*_USER_RUN_ROLES, "developer")),
        default=-1,
    )

    out: list[str] = []
    if add_bos_token:
        out.append(BOS_TOKEN)
    if think and reasoning_effort in ("max", "xhigh"):
        out.append(REASONING_EFFORT_MAX)
    if tools:
        out.append("\n\n" + render_tools(tools))

    for index, message in enumerate(messages):
        role = message["role"]

        if role in ("system", "developer"):
            # A developer message is a system prompt carried inside a user turn.
            if role == "developer":
                out.append(USER_TOKEN)
            out.append(_text_of(message.get("content")))
            if message.get("tools"):
                out.append("\n\n" + render_tools(message["tools"]))
            if message.get("response_format"):
                out.append("\n\n" + RESPONSE_FORMAT_TEMPLATE.format(schema=_to_json(message["response_format"])))

        elif role == "latest_reminder":
            out.append(LATEST_REMINDER_TOKEN + _text_of(message.get("content")))

        elif role in _USER_RUN_ROLES:
            # A run of consecutive user/tool messages is encoded as one user turn.
            if index == 0 or messages[index - 1]["role"] not in _USER_RUN_ROLES:
                out.append(USER_TOKEN)
            else:
                out.append("\n\n")
            if role == "tool":
                out.append(f"<tool_result>{_text_of(message.get('content'))}</tool_result>")
            else:
                out.append(_text_of(message.get("content")))

        elif role == "assistant":
            # A quick instruction task already opened the assistant turn and forbids reasoning.
            if index == 0 or not messages[index - 1].get("task"):
                out.append(ASSISTANT_TOKEN)
                if think and (not drop or index > last_user):
                    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
                    out.append(THINK_START_TOKEN + reasoning)
                out.append(THINK_END_TOKEN)
            out.append(_text_of(message.get("content")))
            if message.get("tool_calls"):
                calls = "\n".join(_render_tool_call(tool_call) for tool_call in message["tool_calls"])
                out.append(f"\n\n{TOOL_CALLS_START_TOKEN}\n{calls}\n{TOOL_CALLS_END_TOKEN}")
            out.append(EOS_TOKEN)

        else:
            raise ValueError(f"DeepSeek-V4 encoding got an unsupported role: {role}")

        task = message.get("task")
        if task and (index + 1 == len(messages) or messages[index + 1]["role"] in ("assistant", "latest_reminder")):
            if task not in TASK_TOKENS:
                raise ValueError(f"DeepSeek-V4 encoding got an unsupported task: {task}")
            if task == "action":
                out.append(ASSISTANT_TOKEN + (THINK_START_TOKEN if think else THINK_END_TOKEN))
            out.append(TASK_TOKENS[task])

    if add_generation_prompt and not (messages and messages[-1].get("task")):
        out.append(ASSISTANT_TOKEN + (THINK_START_TOKEN if think else THINK_END_TOKEN))

    return "".join(out)


class DeepSeekV4ContinuousTokenBuilder(ContinuousTokenBuilder):
    """DeepSeek-V4 turn boundary handling, rendered by :func:`encode_messages`.

    Unlike the other builders this one does not diff two ``apply_chat_template`` renders. The turn
    markup is emitted directly, so an appended run of tool/user messages is encoded exactly as the
    reference encoder would encode it, and no chat template needs to be configured.

    Assistant turns end with ``<｜end▁of▁sentence｜>``, so that token is inserted when generation
    stopped before emitting it.
    """

    def __init__(self, tokenizer: Any, **kwargs: Any):
        super().__init__(tokenizer, **kwargs)
        self._eos_id = require_token_id(tokenizer, EOS_TOKEN)
        template_kwargs = self.chat_template_kwargs
        self._enable_thinking = bool(template_kwargs.get("enable_thinking") or template_kwargs.get("thinking"))
        self._drop_thinking = bool(template_kwargs.get("drop_thinking", True))
        self._reasoning_effort = template_kwargs.get("reasoning_effort")

    def build_initial_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        audios: list[Any] | None = None,
    ) -> list[int]:
        return self._encode(messages, tools=tools, add_bos_token=True)

    def tokenize_non_assistant_incremental_messages(
        self,
        previous_messages: list[dict[str, Any]],
        updated_messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        self._assert_append_only(previous_messages, updated_messages)
        appended_messages = updated_messages[len(previous_messages) :]
        if not appended_messages:
            return []
        # The committed prefix is intentionally immutable. In particular,
        # ``drop_thinking`` may make a canonical full-history render discard
        # earlier reasoning after a new user turn, but CT appends only the new
        # segment and never adopts that retroactive rewrite.
        # ``tools`` is omitted: the tool preamble already sits in the prompt prefix.
        return self._encode(appended_messages, tools=None, add_bos_token=False)

    def _encode(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        add_bos_token: bool,
    ) -> list[int]:
        text = encode_messages(
            messages,
            tools=tools,
            add_generation_prompt=True,
            add_bos_token=add_bos_token,
            enable_thinking=self._enable_thinking,
            drop_thinking=self._drop_thinking,
            reasoning_effort=self._reasoning_effort,
        )
        return normalize_token_ids(self.tokenizer.encode(text, add_special_tokens=False))

    def _merge_non_assistant_token_ids(
        self, runtime_token_ids: list[int], appended_token_ids: list[int]
    ) -> MergeResult:
        prefix = list(runtime_token_ids)
        inserted_token_ids: list[int] = []
        if appended_token_ids and prefix[-1:] != [self._eos_id]:
            prefix.append(self._eos_id)
            inserted_token_ids.append(self._eos_id)
        return MergeResult(
            token_ids=prefix + list(appended_token_ids),
            appended_token_count=len(appended_token_ids),
            kind="non_assistant",
            inserted_token_ids=inserted_token_ids,
        )
