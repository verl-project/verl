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
"""Reasoning parsers for agent loop post-generation pipeline.

The ``decode -> reasoning_parser -> tool_parser`` pipeline mirrors what
inference engines (vLLM / SGLang) apply when serving chat completions on
reasoning models such as Qwen3-thinking, DeepSeek-R1, and GLM-4. The
reasoning parser strips ``<think>...</think>`` (or model-specific
equivalent) blocks so that tool-call patterns inside the chain-of-thought
are not treated as real tool invocations during training.

Without this step, a Qwen3-thinking model that emits

    <think>I should call <tool_call>{"name": "lookup", ...}</tool_call>
    to verify</think>
    Final answer: ...

would have the *think-block* tool call treated as real, producing a
spurious tool execution that pollutes the conversation. See
https://github.com/verl-project/verl/issues/6424 (and #4757 / #6223 /
#6252) for the original bug reports.

The parser is opt-in: callers that don't set
``rollout_config.multi_turn.reasoning_parser`` get the legacy behavior
(no stripping) and remain byte-for-byte compatible.

Design notes for future contributors:

* The vLLM and SGLang serving stacks ship reasoning parsers with similar
  responsibilities; cross-reference their implementations
  (``vllm/reasoning/`` and ``sglang/srt/reasoning_parser/``) when adding
  model-specific behavior, since the upstream landscape evolves quickly.
* This parser intentionally omits the *serving-side leniency* that
  some serving parsers layer on top -- e.g. vLLM's Qwen3 reasoning
  parser closes an unclosed ``<think>`` implicitly when a
  ``<tool_call>`` arrives. RL training has stricter formatting
  expectations: a model that emits malformed reasoning under training
  signal needs the strict surface here, not the serving stack's
  best-effort recovery.
* The ``enable_thinking`` keyword (mirroring ``chat_template_kwargs``)
  short-circuits stripping when the chat template injected an empty
  ``<think>\\n\\n</think>\\n\\n`` opener and the model is expected to
  emit no think markers at all. Without this, an unrelated literal
  ``<think>`` substring in normal output (code, quoted text) would be
  greedily stripped to end-of-text by the unclosed-tag fallback.
* The base ``ReasoningParser.extract_content`` interface accepts
  ``**kwargs`` so model-specific parameters (Qwen3's
  ``enable_thinking``, hypothetical ``think_effort`` for other
  reasoning surfaces, etc.) can be forwarded verbatim from
  ``chat_template_kwargs`` without the abstract layer growing per-
  model knobs. Each implementation pulls only the keys it understands
  and silently ignores the rest.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__file__)


class ReasoningParser(ABC):
    """Strip the model's chain-of-thought (reasoning) block from a decoded
    response so that downstream tool parsing does not see tool-call
    patterns that occurred *inside* the reasoning block.

    Implementations are stateless: a single instance can be reused across
    requests. Register subclasses with ``@ReasoningParser.register("name")``
    and look them up via ``ReasoningParser.get_reasoning_parser(name)``.
    """

    _registry: dict[str, type[ReasoningParser]] = {}

    @abstractmethod
    def extract_content(self, text: str, **kwargs) -> str:
        """Return ``text`` with reasoning blocks stripped.

        The implementation must be idempotent: calling it twice on the
        same input returns the same output. A response that contains no
        reasoning markers must be returned unchanged.

        Implementations should pull only the keys they understand from
        ``**kwargs`` (e.g. Qwen3 / DeepSeek-R1 / GLM-4 thinking models
        consume ``enable_thinking``; other models may surface a
        different shape such as ``think_effort``). Unknown keys should
        be silently ignored so callers can pass a single
        ``chat_template_kwargs`` blob without per-parser conditioning.

        Args:
            text: The decoded model response, possibly containing
                reasoning markers (e.g. ``<think>...</think>``).
            **kwargs: Model-specific knobs forwarded from
                ``chat_template_kwargs``. Default behavior (no kwargs
                passed) preserves the legacy stripping path so callers
                that have not yet threaded chat-template flags through
                are byte-for-byte unchanged.

        Returns:
            The response text with reasoning blocks removed. Content
            that surrounds the reasoning blocks is preserved verbatim.
        """
        raise NotImplementedError

    @classmethod
    def get_reasoning_parser(cls, name: str) -> ReasoningParser:
        """Look up and instantiate a registered parser by name.

        Args:
            name: A registered parser name (e.g. ``"qwen3"``).

        Returns:
            An instance of the corresponding parser.

        Raises:
            ValueError: If ``name`` is not registered.
        """
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise ValueError(
                f"Unknown reasoning parser: {name!r}. "
                f"Available: {available}. "
                f"Register a custom parser with @ReasoningParser.register('your-name')."
            )
        return cls._registry[name]()

    @classmethod
    def register(cls, name: str):
        """Decorator to register a ``ReasoningParser`` subclass under ``name``."""

        def decorator(subclass: type[ReasoningParser]) -> type[ReasoningParser]:
            if name in cls._registry:
                logger.warning(
                    "ReasoningParser %r is being re-registered (was %r, now %r)",
                    name,
                    cls._registry[name],
                    subclass,
                )
            cls._registry[name] = subclass
            return subclass

        return decorator


@ReasoningParser.register("qwen3")
@ReasoningParser.register("deepseek_r1")
class ThinkBlockReasoningParser(ReasoningParser):
    """Strip ``<think>...</think>`` blocks.

    Covers Qwen3 thinking mode, DeepSeek-R1, GLM-4 with thinking, and any
    other model that follows the ``<think>...</think>`` convention. An
    unclosed ``<think>`` (truncation, mid-stream) is also dropped from
    the opening tag to end-of-text so the unfinished reasoning never
    reaches the tool parser.

    Two response shapes are supported:

    * **Explicit opener.** The decoded response contains an opening
      ``<think>``; the parser walks ``<think>...</think>`` pairs left to
      right with non-overlapping first-opener / first-closer semantics
      (consistent with how the previous regex implementation behaved on
      pathological nested-looking inputs that real Qwen3 / DeepSeek-R1
      outputs do not produce).
    * **Implicit opener.** Qwen3 / DeepSeek-R1 default chat templates
      append ``<think>`` to the prompt itself when
      ``add_generation_prompt=True``, so ``response_ids`` (only the
      generated tokens) decode to ``reasoning…</think>\\n\\nanswer``
      with no leading ``<think>``. The parser treats this as an
      implicit opener at the start of text and strips up to the first
      ``</think>``, matching what vLLM's and SGLang's serving-side
      Qwen3 reasoning parsers do.

    Implementation note: ``str.partition`` (C-level, single early-
    terminating scan) is used instead of a DOTALL regex over the whole
    response. For long reasoning outputs (e.g. 32K-token traces) the
    partition path skips re-scanning the post-``</think>`` tail, matching
    the idiom vLLM and SGLang use in their serving-side reasoning
    parsers.

    **Caveat — literal ``</think>`` in legitimate output.** Under the
    default ``enable_thinking=True`` path, a literal ``</think>``
    substring appearing in normal output (e.g. quoted tag inside a code
    block or a discussion of the parser itself) is treated as the
    closer of an implicit reasoning span and the preceding text is
    dropped. Production Qwen3 / DeepSeek-R1 outputs do not emit
    ``</think>`` outside reasoning closure, so this matches vLLM and
    SGLang's serving-side behavior. Callers that need to preserve such
    substrings must either pass ``enable_thinking=False`` (which
    short-circuits stripping entirely) or register a custom parser.
    """

    def extract_content(self, text: str, **kwargs) -> str:
        # Pull the only key this parser cares about; other models'
        # kwargs (e.g. ``think_effort``) are silently ignored so the
        # caller can forward ``chat_template_kwargs`` verbatim.
        enable_thinking = kwargs.get("enable_thinking", True)
        if not enable_thinking:
            # Chat template injected ``<think>\n\n</think>\n\n`` opener;
            # any literal ``<think>`` in model output is incidental
            # (code, quoted text) and must not trigger greedy stripping.
            return text
        first_open = text.find("<think>")
        first_close = text.find("</think>")
        if first_open < 0 and first_close < 0:
            # No reasoning markers at all -- nothing to strip.
            return text

        parts: list[str] = []
        remainder = text

        # Implicit opener: ``response_ids`` start *inside* the think
        # block because the chat template emitted ``<think>`` as part
        # of the prompt. Detected when ``</think>`` appears before any
        # ``<think>`` (or when there is no ``<think>`` at all). The
        # first ``</think>`` closes that implicit reasoning span; drop
        # everything up to and including it before entering the loop.
        if first_close >= 0 and (first_open < 0 or first_close < first_open):
            _inside, _closer, after_closer = remainder.partition("</think>")
            remainder = after_closer
            # Fall through: any further ``<think>...</think>`` pairs
            # are handled by the regular loop below.

        while True:
            before, opener, after_opener = remainder.partition("<think>")
            if not opener:
                # No more ``<think>``: emit the rest verbatim. If a
                # dangling ``</think>`` is sitting in ``before`` (left
                # over from non-overlapping pair-loop semantics on
                # nested-looking inputs), drop everything up to and
                # including it. Without this, a second call to
                # ``extract_content`` would mis-detect the dangling
                # closer as an implicit opener marker and strip again,
                # violating the documented idempotence contract.
                if "</think>" in before:
                    _inside, _closer, tail = before.partition("</think>")
                    parts.append(tail)
                else:
                    parts.append(before)
                break
            parts.append(before)
            # Look for the matching ``</think>``.
            _inside, closer, after_closer = after_opener.partition("</think>")
            if not closer:
                # Unclosed ``<think>``: drop everything from the opener
                # to end-of-text. Without this, a truncated mid-stream
                # response keeps its tool-call patterns visible to the
                # tool parser.
                break
            remainder = after_closer
        return "".join(parts)
