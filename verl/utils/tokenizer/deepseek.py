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

import torch

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
SYSTEM_TOKEN = "<｜System｜>"

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

REASONING_EFFORT_PROMPTS: dict[str, str] = {
    "low": "",
    "high": (
        "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
        "You MUST be very thorough in your thinking and comprehensively decompose the problem to "
        "resolve the root cause, rigorously stress-testing your logic against all potential paths, "
        "edge cases, and adversarial scenarios.\n"
        "Explicitly write out your entire deliberation process, documenting every intermediate step, "
        "considered alternative, and rejected hypothesis to ensure absolutely no assumption is left "
        "unchecked.\n\n"
    ),
    "max": (
        "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n"
        "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: "
        "exhaustively decompose the problem into its most fundamental components, trace every causal "
        "chain to its root, and resolve the underlying cause rather than any surface symptom.\n"
        "Do not stop reasoning until you have independently verified the solution from multiple angles "
        "and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n"
    ),
}
DEFAULT_REASONING_EFFORT = "low"

# Compatibility aliases preserve the effective prompt used by older DeepSeek-V4
# serving contracts while canonicalizing onto the current low/high/max levels.
_REASONING_EFFORT_ALIASES = {
    "minimal": "low",
    "medium": "low",
    "xhigh": "high",
    "ultra": "max",
}

# vLLM expands one placeholder per image into its image-token span.
IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"


def expand_image_tokens(input_ids, image_grid_hws, image_token_id, downsample_ratio):
    """Expand [sequence] image placeholders, returning [1, sequence] IDs and types."""
    raw_ids = input_ids.reshape(-1).tolist()
    grids = image_grid_hws.tolist()
    if raw_ids.count(image_token_id) != len(grids):
        raise ValueError("DeepSeek-V4.1 requires one unexpanded placeholder per image")
    grids = iter(grids)
    ids, types = [], []
    for token in raw_ids:
        if token != image_token_id:
            ids.append(token)
            types.append(-1)
            continue
        height, width = next(grids)
        height = (height + downsample_ratio - 1) // downsample_ratio
        width = (width + downsample_ratio - 1) // downsample_ratio
        span = [0] + ([1] * width + [2]) * height + [3]
        ids.extend([image_token_id] * len(span))
        types.extend(span)
    return input_ids.new_tensor([ids]), input_ids.new_tensor([types])


def align_deepseek_v41_routes(routes: torch.Tensor, vision_types: list[int]) -> torch.Tensor:
    """Expand compact V4.1 routes into the actor's image-expanded sequence.

    Rollout backends can return one route row for an image placeholder even though
    the Megatron V4.1 actor executes one row for every image-span position.
    The placeholder route cannot be reused for all visual positions because
    V4.1 uses modality-specific router biases.  Consume that compact row and
    leave the expanded visual rows as ignored placeholders; the replay mask
    selects native Megatron routing for those rows.
    """
    expanded_len = len(vision_types)
    route_len = routes.shape[0]
    image_start_count = sum(kind == 0 for kind in vision_types)
    image_position_count = sum(kind >= 0 for kind in vision_types)
    text_count = sum(kind == -1 for kind in vision_types)
    compact_len = text_count + image_start_count
    zero = torch.zeros((1, *routes.shape[1:]), dtype=routes.dtype, device=routes.device)

    # A complete expanded response (or its autoregressive final-row omission)
    # is already position aligned. Keep the actual visual routes in this case;
    # callers may still choose native visual routing through the replay mask.
    if route_len in (expanded_len, expanded_len - 1):
        rows = [routes[index : index + 1] if index < route_len else zero for index in range(expanded_len)]
        return torch.cat(rows, dim=0)

    # vLLM may emit a different number of rows for an image span than the
    # actor's expanded sequence.  Text rows stay position-aligned; image rows
    # are consumed inside the visual span and are ignored by the replay mask.
    # This covers compact placeholders, partial visual expansion, and a missing
    # final autoregressive row without reusing a visual route for text.
    if text_count - 1 <= route_len <= expanded_len:
        if route_len == compact_len:
            text_route_count = text_count
            image_route_count = image_start_count
        elif route_len == compact_len - 1:
            text_route_count = text_count - 1
            image_route_count = image_start_count
        else:
            text_route_count = text_count if route_len >= text_count else text_count - 1
            image_route_count = route_len - text_route_count
        if image_route_count < 0 or image_route_count > image_position_count:
            raise RuntimeError(
                "R3 router replay cannot partition DeepSeek-V4.1 multimodal routes: "
                f"{route_len} route rows for {expanded_len} expanded tokens, "
                f"{text_count} text rows and {image_position_count} image rows"
            )
        rows = []
        cursor = 0
        text_cursor = 0
        image_cursor = 0
        for kind in vision_types:
            if kind == -1:
                if text_cursor < text_route_count:
                    rows.append(routes[cursor : cursor + 1])
                    cursor += 1
                    text_cursor += 1
                else:
                    rows.append(zero)
            elif image_cursor < image_route_count and (route_len > compact_len or kind == 0):
                # Native V4.1 routing owns visual rows; these values are
                # retained only to preserve the returned row count.
                if cursor < route_len:
                    rows.append(routes[cursor : cursor + 1] if route_len > compact_len else zero)
                    cursor += 1
                    image_cursor += 1
                else:
                    rows.append(zero)
            else:
                rows.append(zero)
        if cursor != route_len or image_cursor != image_route_count:
            raise RuntimeError(
                "V4.1 multimodal router replay did not consume all route rows: "
                f"consumed={cursor}, routes={route_len}, expanded={expanded_len}, "
                f"compact={compact_len}, text_routes={text_cursor}, image_routes={image_cursor}"
            )
        return torch.cat(rows, dim=0)

    raise RuntimeError(
        "R3 router replay cannot align DeepSeek-V4.1 multimodal routes: "
        f"{route_len} route rows for {expanded_len} expanded tokens, "
        f"{compact_len} compact tokens ({image_start_count} image spans)"
    )


REASONING_EFFORT_TEMPLATE = (
    "Reasoning Effort: {budget} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
)
REASONING_EFFORT_MAPPINGS = {"low": 25, "high": 50, "xhigh": 75, "max": 100}

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


def _resolve_reasoning_effort(reasoning_effort: Optional[str]) -> Optional[str]:
    """Resolve canonical and compatibility effort names.

    ``None`` keeps checkpoint-level behavior (low). ``none`` is the serving
    compatibility spelling for disabling thinking.
    """
    if reasoning_effort is None:
        return DEFAULT_REASONING_EFFORT
    if reasoning_effort == "none":
        return None
    resolved = _REASONING_EFFORT_ALIASES.get(reasoning_effort, reasoning_effort)
    if resolved not in REASONING_EFFORT_PROMPTS:
        raise ValueError(
            f"Invalid reasoning effort: {reasoning_effort!r}; expected one of "
            f"{[*REASONING_EFFORT_PROMPTS, 'none', *_REASONING_EFFORT_ALIASES]}"
        )
    return resolved


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
    reasoning_effort: Optional[str | int] = None,
    v41_reasoning_effort: bool = False,
    previous_role: Optional[str] = None,
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
        reasoning_effort: Canonical ``"low"``/``"high"``/``"max"`` effort.
            Compatibility aliases are normalized for V4. V4.1 accepts its named
            effort levels or an integer budget in [1, 100]. ``"none"`` disables thinking.
        v41_reasoning_effort: Use the V4.1 numeric budget protocol.
        previous_role: Role immediately preceding ``messages`` when rendering
            an incremental continuation. Consecutive user/tool messages share
            one DeepSeek-V4 user turn across request boundaries.

    Returns:
        The encoded prompt.
    """
    resolved_reasoning_effort = None
    if v41_reasoning_effort:
        think = enable_thinking and reasoning_effort != "none"
    else:
        if isinstance(reasoning_effort, int):
            raise ValueError(f"Invalid DeepSeek-V4 reasoning effort: {reasoning_effort!r}; expected a string")
        resolved_reasoning_effort = _resolve_reasoning_effort(reasoning_effort)
        think = enable_thinking and resolved_reasoning_effort is not None
    has_tools = bool(tools) or any(message.get("tools") for message in messages)
    drop = drop_thinking and not has_tools
    last_user = max(
        (i for i, m in enumerate(messages) if m["role"] in (*_USER_RUN_ROLES, "developer")),
        default=-1,
    )

    out: list[str] = []
    if add_bos_token:
        out.append(BOS_TOKEN)
    # Reasoning effort is a conversation-level prefix. Incremental Continuous
    # Token appends set ``add_bos_token=False`` and must not inject it between
    # assistant and tool turns.
    if add_bos_token and think and v41_reasoning_effort:
        effort = "high" if reasoning_effort is None else reasoning_effort
        if isinstance(effort, str):
            if effort not in REASONING_EFFORT_MAPPINGS:
                raise ValueError(f"Invalid DeepSeek-V4.1 reasoning_effort: {effort!r}")
            effort = REASONING_EFFORT_MAPPINGS[effort]
        if isinstance(effort, bool) or not isinstance(effort, int) or not 1 <= effort <= 100:
            raise ValueError("DeepSeek-V4.1 reasoning_effort must be an int in [1, 100]")
        out.extend((SYSTEM_TOKEN, REASONING_EFFORT_TEMPLATE.format(budget=effort)))
    elif add_bos_token and think and resolved_reasoning_effort is not None:
        out.append(REASONING_EFFORT_PROMPTS[resolved_reasoning_effort])

    # The reference DeepSeek-V4 encoder appends tool schemas to the leading
    # system/developer message. If there is no such message, keep the schemas
    # before the first user turn.
    top_level_tools_target = next(
        (index for index, message in enumerate(messages) if message["role"] in ("system", "developer")),
        None,
    )
    if tools and top_level_tools_target is None:
        out.append("\n\n" + render_tools(tools))

    for index, message in enumerate(messages):
        role = message["role"]

        if role in ("system", "developer"):
            # A developer message is a system prompt carried inside a user turn.
            if role == "developer":
                out.append(USER_TOKEN)
            out.append(_text_of(message.get("content")))
            if tools and index == top_level_tools_target:
                out.append("\n\n" + render_tools(tools))
            if message.get("tools"):
                out.append("\n\n" + render_tools(message["tools"]))
            if message.get("response_format"):
                out.append("\n\n" + RESPONSE_FORMAT_TEMPLATE.format(schema=_to_json(message["response_format"])))

        elif role == "latest_reminder":
            out.append(LATEST_REMINDER_TOKEN + _text_of(message.get("content")))

        elif role in _USER_RUN_ROLES:
            # A run of consecutive user/tool messages is encoded as one user turn.
            preceding_role = messages[index - 1]["role"] if index else previous_role
            if preceding_role not in _USER_RUN_ROLES:
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

    _v41_reasoning_effort = False

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
        mm_inputs_out: dict[str, Any] | None = None,
    ) -> list[int]:
        del images, videos, audios, mm_inputs_out
        return self._encode(messages, tools=tools, add_bos_token=True)

    def tokenize_context_incremental_messages(
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
        if appended_messages[-1].get("role") == "assistant":
            raise ValueError("Continuous Token context incremental messages cannot end with assistant")
        self._assert_prefix_is_stable(previous_messages, tools=tools)
        # ``tools`` is omitted: the tool preamble already sits in the prompt prefix.
        return self._encode(
            appended_messages,
            tools=None,
            add_bos_token=False,
            drop_thinking=False,
            previous_role=previous_messages[-1].get("role") if previous_messages else None,
        )

    def _encode(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        add_bos_token: bool,
        drop_thinking: bool | None = None,
        previous_role: str | None = None,
    ) -> list[int]:
        text = encode_messages(
            messages,
            tools=tools,
            add_generation_prompt=True,
            add_bos_token=add_bos_token,
            enable_thinking=self._enable_thinking,
            drop_thinking=self._drop_thinking if drop_thinking is None else drop_thinking,
            reasoning_effort=self._reasoning_effort,
            previous_role=previous_role,
            v41_reasoning_effort=self._v41_reasoning_effort,
        )
        return normalize_token_ids(self.tokenizer.encode(text, add_special_tokens=False))

    def _assert_prefix_is_stable(
        self,
        previous_messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
    ) -> None:
        """Reject appends whose prefix a full re-render would not reproduce.

        ``drop_thinking`` keeps reasoning only for assistant turns after the last user turn, so
        appending a user turn retroactively strips reasoning from turns already committed to the
        runtime sequence. The encoder disables dropping whenever tools are present, which is the
        case for tool-calling rollouts; thinking mode otherwise requires ``drop_thinking=False``.
        """
        if not (self._enable_thinking and self._drop_thinking):
            return
        if tools or any(message.get("tools") for message in previous_messages):
            return
        if any(message.get("role") == "assistant" for message in previous_messages):
            raise ValueError(
                "DeepSeek-V4 Continuous Token cannot append after an assistant turn when thinking is "
                "enabled and drop_thinking is on, because dropping reasoning would rewrite the "
                "committed prefix. Pass tools, or set drop_thinking=False in chat_template_kwargs."
            )

    def _merge_context_token_ids(
        self,
        runtime_token_ids: list[int],
        appended_token_ids: list[int],
        *,
        previous_messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> MergeResult:
        prefix = list(runtime_token_ids)
        inserted_token_ids: list[int] = []
        previous_role = previous_messages[-1].get("role") if previous_messages else None
        # EOS closes an assistant turn; user/tool runs flow directly into the next assistant turn.
        if appended_token_ids and previous_role == "assistant" and prefix[-1:] != [self._eos_id]:
            prefix.append(self._eos_id)
            inserted_token_ids.append(self._eos_id)
        return MergeResult(
            token_ids=prefix + list(appended_token_ids),
            appended_token_count=len(appended_token_ids),
            kind="context",
            inserted_token_ids=inserted_token_ids,
        )


class DeepSeekV4VLContinuousTokenBuilder(DeepSeekV4ContinuousTokenBuilder):
    """Use DeepSeek text formatting with one placeholder per image."""

    def __init__(
        self,
        tokenizer: Any,
        processor: Any,
        *,
        mm_processor_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Initialize the text builder and retain the multimodal processor and its keyword arguments."""
        super().__init__(tokenizer, **kwargs)
        self.processor = processor
        self.mm_processor_kwargs = mm_processor_kwargs if mm_processor_kwargs is not None else {}

    @classmethod
    def supports_multimodal(cls) -> bool:
        """Report that this builder supports multimodal inputs."""
        return True

    def _encode(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        add_bos_token: bool,
        drop_thinking: bool | None = None,
        previous_role: str | None = None,
    ) -> list[int]:
        """Encode copied messages after replacing image blocks with DeepSeek image placeholders."""
        messages = [dict(message) for message in messages]
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            message["content"] = [
                {"type": "text", "text": IMAGE_PLACEHOLDER}
                if isinstance(block, dict) and block.get("type") in ("image", "image_url")
                else block
                for block in content
            ]
        return super()._encode(
            messages,
            tools=tools,
            add_bos_token=add_bos_token,
            drop_thinking=drop_thinking,
            previous_role=previous_role,
        )


class DeepSeekV41ContinuousTokenBuilder(DeepSeekV4ContinuousTokenBuilder):
    """DeepSeek-V4.1 text protocol, including its numeric reasoning prefix."""

    _v41_reasoning_effort = True


class DeepSeekV41VLContinuousTokenBuilder(DeepSeekV4VLContinuousTokenBuilder):
    """DeepSeek-V4.1 vision-language protocol."""

    _v41_reasoning_effort = True
