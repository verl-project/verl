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
"""Compare agent-loop Continuous Token (CT) and legacy token end to end for
vision-language (VL) models.

This is the multimodal sibling of ``compare_agentloop_ct_vs_legacy.py``. The
difference is that the trajectories embed real images, both in the initial user
prompt and inside tool responses, and every render goes through the model's
multimodal ``processor`` (not the bare tokenizer) so vision placeholders expand
into the same per-image pad-token spans the rollout backend would consume.

The VL trajectories are defined inline in this file (``build_trajectories``);
they are intentionally not persisted to a shared fixtures module. Each case
builds a deterministic rollout server plus deterministic image-returning tools
for ToolAgentLoop cases.

Assistant-turn generation is mocked the same way as the text harness, but
through the processor:

1. render ``prefix_messages`` with ``add_generation_prompt=True`` through the
   processor (with the images available so far);
2. render ``prefix_messages + [assistant_message]`` with
   ``add_generation_prompt=False`` through the processor;
3. take the token-id suffix of the full render after the prompt render;
4. to mimic the model stopping at EOS, truncate after the final EOS token so
   template whitespace that follows EOS (e.g. ``\\n``) is not emitted as part of
   the generated response.

The image placeholders live in the shared prefix of both renders, so they
cancel in the suffix diff; the suffix is the pure assistant text/tool-call
token stream.

CT and legacy are then compared by running the real agent loop twice with the
same deterministic server outputs and the same processor: legacy constructs the
loop with CT disabled (full re-render of history through the processor every
turn), while CT constructs it with the real VL builder enabled (incremental
prefix reuse). ToolAgentLoop cases execute hardcoded ``FunctionTool`` instances
that return the trajectory's expected tool response, including images.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from transformers import AutoProcessor

from scripts.chat_template_mock_trajectories import (
    VLSingleTurnTrajectory,
    VLToolTrajectory,
    build_vl_trajectories,
)
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, DictConfigWrap, ToolListWrap
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.tools.function_tool import FunctionTool
from verl.tools.schemas import ToolResponse
from verl.utils.tokenizer import build_multimodal_processor_inputs, normalize_token_ids
from verl.utils.tokenizer.chat_template import apply_chat_template
from verl.workers.rollout.replica import TokenOutput

DEFAULT_E2E_TRAJECTORIES = ("vl_singleturnchat", "vl_multiturnsingletool", "vl_multiturnmultitool")


# VL processors that load with a bare AutoProcessor and expose a standard
# ``apply_chat_template`` + image-token expansion. DeepSeek-VL2 is intentionally
# excluded: its DeepseekVLV2Processor does not implement apply_chat_template, so
# the legacy agent-loop path cannot render it through the shared harness.
DEFAULT_MODELS = [
    # Qwen VL family.
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-2B-Thinking",
    # MiMo VL (Qwen2.5-VL architecture).
    "XiaomiMiMo/MiMo-VL-7B-RL",
    # GLM vision. GLM-4.6V is fully supported (all trajectories). GLM-4V (4.1V)
    # and GLM-4.5V are single-turn only: their templates mishandle tool-role
    # images, so they cannot run the tool agent loop, but the single-turn agent
    # loop works through the same GLM46VContinuousTokenBuilder.
    "zai-org/GLM-4.1V-9B-Thinking",
    "zai-org/GLM-4.5V",
    "zai-org/GLM-4.6V",
]

# Models whose chat template cannot render tool-role images through the tool
# agent loop, so only ``vl_singleturnchat`` is exercised for them. GLM-4.1V
# (GLM-4V) drops ``role="tool"`` messages entirely (its template has no tool-role
# branch); GLM-4.5V handles the tool role but serializes the tool message's
# multimodal content list as a string instead of expanding the image. In both
# cases the ContinuousTokenBuilder cannot work around the template limitation, so
# only single-turn prompt-image rendering is exercised for them.
#
# NOTE: Gemma-4 is intentionally NOT in this list. Its template renders tool
# responses (text + image) faithfully, but *only* when the tool message follows
# an assistant carrying a *structured* ``tool_calls`` field (forward-scan embed).
# The CT builder satisfies this via its dummy synthetic assistant, and the
# ground-truth builder (:func:`_prepare_tool_assistant_outputs`) mirrors it, so
# Gemma-4 runs the full tool agent loop.
SINGLE_TURN_ONLY_MODEL_MARKERS = ("glm-4.1v", "glm-4v", "glm-4.5v")


def _model_supports_tool_agent_loop(model: str) -> bool:
    lowered = model.lower()
    return not any(marker in lowered for marker in SINGLE_TURN_ONLY_MODEL_MARKERS)


TRAJECTORY_BY_NAME = build_vl_trajectories()


# =============================================================================
# Deterministic rollout server
# =============================================================================


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


class _VLImageDataset:
    """Minimal dataset shim providing multimodal extraction for the agent loop.

    Extracts PIL images directly from OpenAI-style content blocks so the harness
    does not depend on ``qwen_vl_utils``.
    """

    @classmethod
    async def process_multi_modal_info(cls, messages, image_patch_size, config):
        del image_patch_size, config
        images = _extract_images(messages)
        return (images or None, None, None)


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


# =============================================================================
# Processor / tokenizer helpers
# =============================================================================


def _load_processor(model: str, *, local_files_only: bool):
    # NOTE: Gemma-4 must be the non-unified ``gemma4`` variant (e.g.
    # ``google/gemma-4-E4B-it``), whose ``AutoProcessor`` resolves to a real
    # ``Gemma4Processor`` with an image processor. The ``gemma4_unified`` variant
    # (e.g. ``gemma-4-12B-it``) needs ``Gemma4UnifiedProcessor``, which the
    # installed transformers does not ship, so it is not supported here.
    processor = AutoProcessor.from_pretrained(
        model,
        trust_remote_code=True,
        local_files_only=local_files_only,
        use_fast=True,
    )
    if getattr(processor, "image_processor", None) is None:
        raise ValueError(f"{model} did not load a multimodal image processor")
    tokenizer = processor.tokenizer
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(processor, "chat_template", None) and not getattr(tokenizer, "chat_template", None):
        raise ValueError("processor/tokenizer has no chat_template")
    return processor, tokenizer


def _extract_images(messages: list[dict[str, Any]]) -> list[Any]:
    images: list[Any] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image":
                    image = block.get("image")
                    if image is not None:
                        images.append(image)
    return images


def _render_processor_ids(
    processor,
    messages: list[dict[str, Any]],
    images: list[Any],
    *,
    tools: list[dict[str, Any]] | None,
    add_generation_prompt: bool,
) -> list[int]:
    text = apply_chat_template(
        processor,
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )
    output = build_multimodal_processor_inputs(
        processor,
        text=[text],
        images=images if images else None,
    )
    return normalize_token_ids(output["input_ids"])


def _render_assistant_output_ids(
    processor,
    prefix_messages: list[dict[str, Any]],
    assistant_message: dict[str, Any],
    *,
    tools: list[dict[str, Any]] | None,
    has_tool_calls: bool = False,
) -> list[int]:
    # Assistant turns add no images, so both renders share the prefix image set.
    images = _extract_images(prefix_messages)
    prompt_ids = _render_processor_ids(processor, prefix_messages, images, tools=tools, add_generation_prompt=True)
    full_ids = _render_processor_ids(
        processor, prefix_messages + [assistant_message], images, tools=tools, add_generation_prompt=False
    )
    prompt_ids = _trim_generation_think_prefix(processor.tokenizer, prompt_ids, full_ids)
    prompt_ids = _replace_glm_generation_think_open_with_close(processor.tokenizer, prompt_ids, full_ids)
    prompt_ids = _replace_nemotron_generation_think_open_with_empty_block(processor.tokenizer, prompt_ids, full_ids)
    prompt_ids = _replace_qwen_generation_think_open_with_empty_block(processor.tokenizer, prompt_ids, full_ids)
    prompt_ids = _trim_gemma_generation_thought_channel(processor.tokenizer, prompt_ids, full_ids)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("VL assistant output token-id suffix diff failed")
    return _truncate_after_final_eos(
        processor.tokenizer,
        full_ids[len(prompt_ids) :],
        has_tool_calls=has_tool_calls,
    )


def _trim_generation_think_prefix(tokenizer, prompt_ids: list[int], full_ids: list[int]) -> list[int]:
    if full_ids[: len(prompt_ids)] == prompt_ids:
        return prompt_ids
    compact_name = "".join(char for char in _tokenizer_name(tokenizer).lower() if char.isalnum())
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
    if not _is_glm_tokenizer(tokenizer):
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
    if "nemotron" not in _tokenizer_name(tokenizer).lower():
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

    Qwen3-VL-*-Thinking append ``<think>\\n`` at the generation prompt, but a
    plain (reasoning-free) assistant message renders an *empty* block
    ``<think>\\n\\n</think>\\n\\n``. The prompt's ``\\n`` and the block's ``\\n\\n``
    tokenize to different ids, so the plain-assistant suffix diff cannot align.
    Fold the empty block into the prompt (exactly like the Nemotron helper) so
    the assistant output is extracted correctly, without injecting any synthetic
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
    message renders no channel at all. The opener has no counterpart in the full
    render, so the plain-assistant suffix diff cannot align. Trim the trailing
    opener from the prompt so the assistant output is extracted correctly.
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
    has_tool_calls: bool,
) -> list[int]:
    eos_token_ids = _tokenizer_eos_token_ids(tokenizer)
    eos_token_ids.update(_kimi_assistant_end_token_ids(tokenizer))
    eos_token_ids.update(_gemma_assistant_end_token_ids(tokenizer))
    if not token_ids:
        raise ValueError("Assistant output token-id suffix is empty")

    # GLM assistant turns are not terminated by EOS in the template render; the
    # canonical stop after a tool call is <|observation|>, and the template uses
    # the next role token as the turn boundary otherwise. Mirror real GLM rollout
    # by appending <|observation|> for tool-call turns so the assistant/tool
    # boundary token is present (this is what CT's GLM boundary trim operates on).
    if _is_glm_tokenizer(tokenizer):
        if has_tool_calls:
            return token_ids + [_require_single_token_id(tokenizer, "<|observation|>")]
        return token_ids

    if eos_token_ids:
        if token_ids[-1] in eos_token_ids:
            return token_ids
        for index in range(len(token_ids) - 1, -1, -1):
            if token_ids[index] in eos_token_ids:
                return token_ids[: index + 1]

    tail = token_ids[-16:]
    raise ValueError(
        f"Assistant output token-id suffix does not contain eos_token_id {sorted(eos_token_ids)}; tail={tail}"
    )


def _is_glm_tokenizer(tokenizer) -> bool:
    tokenizer_name = _tokenizer_name(tokenizer).lower()
    compact_name = "".join(char for char in tokenizer_name if char.isalnum())
    return any(marker in tokenizer_name for marker in ("glm-4", "glm_4", "glm-5", "glm_5")) or any(
        marker in compact_name for marker in ("glm4", "glm5")
    )


def _kimi_assistant_end_token_ids(tokenizer) -> set[int]:
    if "kimi" not in _tokenizer_name(tokenizer).lower():
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


# =============================================================================
# Assistant-output / tool preparation
# =============================================================================


def _clone(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return copy.deepcopy(messages)


def _gemma_structured_assistant(step) -> dict[str, Any]:
    """Assistant message that mirrors ``ToolAgentLoop._build_assistant_message``.

    Gemma's template only renders the following ``role="tool"`` responses when the
    assistant carries a *structured* ``tool_calls`` field (forward-scan embed). The
    per-turn *output* delta is still computed from the raw-text assistant (what the
    model emits), but the running history must use this structured form so the next
    turn's prefix render expands the tool responses.
    """
    tool_calls = []
    for index, (name, arguments) in enumerate(step.calls):
        tool_calls.append(
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        )
    return {"role": "assistant", "content": step.reasoning, "tool_calls": tool_calls}


def _gemma_stamped_tool_messages(step) -> list[dict[str, Any]]:
    """Standalone ``role="tool"`` messages stamped with ``tool_call_id``/``name``.

    The template resolves the tool name via ``tc.id == follow.tool_call_id`` and
    falls back to ``follow.name``; without either it hits a ``None`` concatenation.
    """
    messages = []
    for index, message in enumerate(copy.deepcopy(step.appended_messages)):
        message["tool_call_id"] = f"call_{index}"
        if not message.get("name") and index < len(step.responses):
            message["name"] = step.responses[index].tool_name
        messages.append(message)
    return messages


def _prepare_tool_assistant_outputs(
    processor,
    trajectory: VLToolTrajectory,
    tool_parser: str,
) -> list[PreparedAssistantOutput]:
    tools = trajectory.tool_schemas
    canonical_messages = _clone(trajectory.raw_prompt)
    assistant_outputs: list[PreparedAssistantOutput] = []
    # Gemma renders tool responses only via forward-scan from a structured
    # assistant; other VL templates render assistant ``content`` verbatim and drop
    # a ``tool_calls`` field, so keep the raw-text form for them.
    is_gemma = "gemma" in _tokenizer_name(processor.tokenizer).lower()

    for turn_index, step in enumerate(trajectory.steps):
        assistant_message = step.assistant_message(tool_parser)
        assistant_ids = _render_assistant_output_ids(
            processor,
            canonical_messages,
            assistant_message,
            tools=tools,
            has_tool_calls=bool(step.calls),
        )
        assistant_outputs.append(
            PreparedAssistantOutput(
                token_ids=assistant_ids,
                log_probs=_assistant_logprobs(turn_index, len(assistant_ids)),
            )
        )
        if is_gemma and step.calls:
            canonical_messages.append(_gemma_structured_assistant(step))
            canonical_messages.extend(_gemma_stamped_tool_messages(step))
        else:
            canonical_messages.append(assistant_message)
            canonical_messages.extend(copy.deepcopy(step.appended_messages))

    return assistant_outputs


def _prepare_single_turn_assistant_output(
    processor,
    trajectory: VLSingleTurnTrajectory,
) -> PreparedAssistantOutput:
    assistant_ids = _render_assistant_output_ids(
        processor,
        _clone(trajectory.raw_prompt),
        trajectory.assistant_message(),
        tools=None,
        has_tool_calls=False,
    )
    return PreparedAssistantOutput(
        token_ids=assistant_ids,
        log_probs=_assistant_logprobs(0, len(assistant_ids)),
    )


def _make_deterministic_tools(trajectory: VLToolTrajectory) -> list[FunctionTool]:
    # The trajectory already carries fully-configured FunctionTool instances that
    # each return a fixed ToolResponse; clone the images per run to avoid any
    # cross-run mutation of shared PIL objects.
    tools: list[FunctionTool] = []
    for tool in trajectory.tools:
        base_response = tool.fn()

        def _make_fn(response: ToolResponse):
            cloned = ToolResponse(
                text=response.text,
                image=[copy.deepcopy(img) for img in response.image] if response.image else None,
            )

            def _fn(query: str = "") -> ToolResponse:
                del query
                return cloned

            return _fn

        tools.append(FunctionTool(name=tool.name, fn=_make_fn(base_response), tool_schema=tool.tool_schema))
    return tools


# =============================================================================
# Agent-loop configuration
# =============================================================================


def _infer_tool_parser(model: str, requested_tool_parser: str) -> str:
    if requested_tool_parser != "auto":
        return requested_tool_parser
    normalized = model.lower()
    compact = "".join(char for char in normalized if char.isalnum())
    if "kimi" in normalized:
        return "kimi"
    if "glm" in compact:
        return "glm"
    if "gemma" in normalized:
        return "gemma4"
    # Qwen VL / MiMo VL and other VL families use the Hermes tool format.
    return "hermes"


def _continuous_token_config(enable_ct: bool) -> dict[str, Any]:
    return {
        "enable": enable_ct,
        "model_family": "auto",
        "custom_builder_module": None,
    }


def _make_tool_agent_config(*, model: str, enable_ct: bool, tool_parser: str, trajectory: VLToolTrajectory):
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


def _data_config(enable_ct: bool):
    return OmegaConf.create(
        {
            "apply_chat_template_kwargs": {},
            "mm_processor_kwargs": {},
            "continuous_token": _continuous_token_config(enable_ct),
        }
    )


# =============================================================================
# Agent-loop execution
# =============================================================================


async def _run_tool_e2e_path(
    processor,
    tokenizer,
    model: str,
    trajectory: VLToolTrajectory,
    *,
    use_ct: bool,
    tool_parser: str,
) -> E2ERunOutput:
    server = _DeterministicServer(_prepare_tool_assistant_outputs(processor, trajectory, tool_parser))
    loop = ToolAgentLoop(
        trainer_config=DictConfigWrap(
            _make_tool_agent_config(model=model, enable_ct=use_ct, tool_parser=tool_parser, trajectory=trajectory)
        ),
        server_manager=server,
        tokenizer=tokenizer,
        processor=processor,
        dataset_cls=_VLImageDataset,
        data_config=DictConfigWrap(_data_config(use_ct)),
        tools=ToolListWrap(_make_deterministic_tools(trajectory)),
    )
    output: AgentLoopOutput = await loop.run(
        sampling_params={"logprobs": True},
        raw_prompt=_clone(trajectory.raw_prompt),
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
    processor,
    tokenizer,
    model: str,
    trajectory: VLSingleTurnTrajectory,
    *,
    use_ct: bool,
) -> E2ERunOutput:
    server = _DeterministicServer([_prepare_single_turn_assistant_output(processor, trajectory)])
    loop = SingleTurnAgentLoop(
        trainer_config=DictConfigWrap(_make_single_turn_config(model=model, enable_ct=use_ct)),
        server_manager=server,
        tokenizer=tokenizer,
        processor=processor,
        dataset_cls=_VLImageDataset,
        data_config=DictConfigWrap(_data_config(use_ct)),
    )
    output: AgentLoopOutput = await loop.run(
        sampling_params={"logprobs": True},
        raw_prompt=_clone(trajectory.raw_prompt),
    )
    return E2ERunOutput(
        prompt_ids=output.prompt_ids,
        response_ids=output.response_ids,
        response_mask=output.response_mask,
        response_logprobs=output.response_logprobs,
        generation_prompt_ids=server.prompt_ids_by_call,
        num_turns=output.num_turns,
    )


# =============================================================================
# Comparison utilities
# =============================================================================


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
        "token_len": len(output.prompt_ids) + len(output.response_ids),
        "prompt_len": len(output.prompt_ids),
        "response_len": len(output.response_ids),
        "response_mask_len": len(output.response_mask),
        "response_logprobs_len": response_logprobs_len,
    }


def _capture(coro_factory) -> tuple[E2ERunOutput | None, dict[str, Any] | None]:
    try:
        return asyncio.run(coro_factory()), None
    except Exception as exc:
        return None, {"error_type": type(exc).__name__, "error": str(exc)}


def run_one_case(
    processor,
    tokenizer,
    model: str,
    trajectory: VLToolTrajectory | VLSingleTurnTrajectory,
    *,
    tool_parser: str,
) -> dict[str, Any]:
    is_single = isinstance(trajectory, VLSingleTurnTrajectory)
    loop_type = "single_turn_agent_loop" if is_single else "tool_agent_loop"
    result: dict[str, Any] = {
        "model": model,
        "trajectory": trajectory.name,
        "loop_type": loop_type,
        "tool_parser": None if is_single else tool_parser,
    }

    if is_single:
        legacy_output, legacy_error = _capture(
            lambda: _run_single_turn_e2e_path(processor, tokenizer, model, trajectory, use_ct=False)
        )
        ct_output, ct_error = _capture(
            lambda: _run_single_turn_e2e_path(processor, tokenizer, model, trajectory, use_ct=True)
        )
    else:
        legacy_output, legacy_error = _capture(
            lambda: _run_tool_e2e_path(processor, tokenizer, model, trajectory, use_ct=False, tool_parser=tool_parser)
        )
        ct_output, ct_error = _capture(
            lambda: _run_tool_e2e_path(processor, tokenizer, model, trajectory, use_ct=True, tool_parser=tool_parser)
        )

    expected_generation_turns = trajectory.expected_generation_turns

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
        }
    )
    return result


# =============================================================================
# CLI / reporting
# =============================================================================


def _select_models(args) -> list[str]:
    models = list(args.model or DEFAULT_MODELS)
    seen = set()
    selected = []
    for model in models:
        if model not in seen:
            selected.append(model)
            seen.add(model)
    return selected


def _select_trajectories(args) -> list[VLToolTrajectory | VLSingleTurnTrajectory]:
    names = args.trajectory or list(DEFAULT_E2E_TRAJECTORIES)
    return [TRAJECTORY_BY_NAME[name] for name in names]


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
            f"legacy_len={turn.get('legacy_len')} ct_len={turn.get('ct_len')} "
            f"first_mismatch={turn.get('first_mismatch')} "
            f"legacy_value={turn.get('legacy_value')} ct_value={turn.get('ct_value')}",
            flush=True,
        )


def _print_error_details(result: dict[str, Any]) -> None:
    print(
        f"    status: legacy={result.get('legacy_status', 'error')} ct={result.get('ct_status', 'error')}",
        flush=True,
    )
    printed = False
    for side in ("legacy", "ct"):
        error = result.get(f"{side}_error")
        if not error:
            continue
        printed = True
        print(f"    {side}_error: {error.get('error_type')}: {error.get('error')}", flush=True)
    if not printed:
        print(f"    error: {result.get('error_type')}: {result.get('error')}", flush=True)


def _trajectories_for_model(model: str, trajectories):
    if _model_supports_tool_agent_loop(model):
        return list(trajectories)
    # Single-turn-only models: skip tool trajectories entirely.
    return [t for t in trajectories if isinstance(t, VLSingleTurnTrajectory)]


def _run_all(args) -> list[dict[str, Any]]:
    results = []
    models = _select_models(args)
    trajectories = _select_trajectories(args)
    per_model_trajectories = {model: _trajectories_for_model(model, trajectories) for model in models}
    total = sum(len(v) for v in per_model_trajectories.values())
    case_index = 0

    for model in models:
        tool_parser = _infer_tool_parser(model, args.tool_parser)
        model_trajectories = per_model_trajectories[model]
        try:
            processor, tokenizer = _load_processor(model, local_files_only=args.local_files_only)
        except Exception as exc:
            for trajectory in model_trajectories:
                case_index += 1
                print(f"[{case_index}/{total}] {model} :: {trajectory.name}", flush=True)
                result = {
                    "model": model,
                    "trajectory": trajectory.name,
                    "loop_type": (
                        "single_turn_agent_loop"
                        if isinstance(trajectory, VLSingleTurnTrajectory)
                        else "tool_agent_loop"
                    ),
                    "tool_parser": None if isinstance(trajectory, VLSingleTurnTrajectory) else tool_parser,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "ct_legacy_match": False,
                }
                results.append(result)
                print(f"  ERROR {result['error_type']}: {result['error']}", flush=True)
                _print_error_details(result)
            continue

        for trajectory in model_trajectories:
            case_index += 1
            print(f"[{case_index}/{total}] {model} :: {trajectory.name}", flush=True)
            result = run_one_case(processor, tokenizer, model, trajectory, tool_parser=tool_parser)
            results.append(result)
            status = result["status"]
            if status == "pass":
                print("  PASS ct_vs_legacy(match=True)", flush=True)
            elif status == "mismatch":
                comparisons = result["comparisons"]
                print(
                    "  MISMATCH ct_vs_legacy(match=False) "
                    f"prompt={comparisons['prompt_ids']['equal']} "
                    f"response={comparisons['response_ids']['equal']} "
                    f"mask={comparisons['loss_mask']['equal']} "
                    f"logprobs={comparisons['logprobs']['equal']} "
                    f"flow={result['expected_flow']}",
                    flush=True,
                )
                _print_mismatch_details(result)
            else:
                print(f"  ERROR {result.get('error_type')}: {result.get('error')}", flush=True)
                _print_error_details(result)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", help="Model/processor id to test. May be repeated.")
    parser.add_argument(
        "--trajectory",
        action="append",
        choices=sorted(TRAJECTORY_BY_NAME),
        help=f"Trajectory name to test. May be repeated. Defaults to: {', '.join(DEFAULT_E2E_TRAJECTORIES)}.",
    )
    parser.add_argument(
        "--tool-parser",
        default="auto",
        choices=("auto", "hermes", "qwen3_coder", "glm", "kimi", "gemma4"),
        help="ToolParser format for ToolAgentLoop. auto: kimi->kimi, glm->glm, gemma->gemma4, else hermes.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow AutoProcessor to download missing files instead of requiring local cache.",
    )
    parser.add_argument("--ledger", help="Write the structured per-case report JSON to this path.")
    parser.add_argument("--mismatch-jsonl", help="Write one JSON line per non-pass case to this path (for triage).")
    parser.add_argument("--fail-on-mismatch", action="store_true", help="Return non-zero when any case is not pass.")
    args = parser.parse_args()
    args.local_files_only = not args.allow_download
    return args


def _print_summary(results: list[dict[str, Any]]) -> None:
    total = len(results)
    pass_count = sum(item["status"] == "pass" for item in results)
    mismatch_count = sum(item["status"] == "mismatch" for item in results)
    error_count = sum(item["status"] == "error" for item in results)
    print(f"Summary: total={total} pass={pass_count} mismatch={mismatch_count} error={error_count}", flush=True)


def _write_json(path: str, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_mismatch_jsonl(path: str, results: list[dict[str, Any]]) -> None:
    lines = [json.dumps(item, ensure_ascii=False) for item in results if item["status"] != "pass"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    results = _run_all(args)
    if args.ledger:
        _write_json(args.ledger, results)
        print(f"Ledger: {args.ledger}", flush=True)
    if args.mismatch_jsonl:
        _write_mismatch_jsonl(args.mismatch_jsonl, results)
        print(f"Mismatch JSONL: {args.mismatch_jsonl}", flush=True)
    _print_summary(results)
    if args.fail_on_mismatch and any(item["status"] != "pass" for item in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
