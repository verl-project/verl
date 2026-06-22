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
"""CPU-only tests for ``verl.experimental.agent_loop.reasoning_parser``.

Pinning the contract that the reasoning parser strips ``<think>...</think>``
content (Qwen3-thinking, DeepSeek-R1, GLM-4 thinking, etc.) so tool-call
regex patterns inside the chain-of-thought are not surfaced to the tool
parser as real calls. Regression for
https://github.com/verl-project/verl/issues/6424 and the three sibling
issues #4757 / #6223 / #6252 that share the same root cause.
"""

from __future__ import annotations

import pytest

from verl.experimental.agent_loop.reasoning_parser import (
    ReasoningParser,
    ThinkBlockReasoningParser,
)


@pytest.fixture
def qwen3_parser() -> ReasoningParser:
    return ReasoningParser.get_reasoning_parser("qwen3")


# --- Core stripping behavior -------------------------------------------------


def test_qwen3_strips_single_think_block(qwen3_parser: ReasoningParser) -> None:
    text = "<think>Let me reason about this</think>The answer is 42."
    assert qwen3_parser.extract_content(text) == "The answer is 42."


def test_qwen3_strips_think_block_with_newlines(qwen3_parser: ReasoningParser) -> None:
    """Multi-line think blocks must be stripped — the partition-based
    implementation has no DOTALL flag to set, so this pins that the loop
    handles ``\\n`` inside ``<think>...</think>`` correctly.
    """
    text = "<think>\nstep 1\nstep 2\n</think>\nFinal: foo"
    assert qwen3_parser.extract_content(text) == "\nFinal: foo"


def test_text_without_think_block_returned_unchanged(
    qwen3_parser: ReasoningParser,
) -> None:
    text = "No reasoning here, just a plain response."
    assert qwen3_parser.extract_content(text) is text or qwen3_parser.extract_content(text) == text


# --- The bug: tool-call inside think block ----------------------------------


def test_tool_call_inside_think_block_not_visible_after_strip(
    qwen3_parser: ReasoningParser,
) -> None:
    """The regression case from issue #6424: a Qwen3-thinking model emits a
    ``<tool_call>`` *inside* a ``<think>`` block as part of its reasoning.

    Before this parser, the regex-based tool parser would treat the inner
    ``<tool_call>`` as a real tool invocation and execute it during
    training, polluting the conversation history (#4757) and creating
    benchmark regressions (#6223, #6252).

    After stripping, only the *real* tool call after ``</think>`` should
    survive.
    """
    text = (
        "<think>I should call <tool_call>"
        '{"name": "lookup", "arguments": {}}'
        "</tool_call> to verify</think>"
        '<tool_call>{"name": "real_call", "arguments": {}}</tool_call>'
    )
    stripped = qwen3_parser.extract_content(text)
    assert "lookup" not in stripped, (
        f"tool call inside <think> block leaked through reasoning parser; got: {stripped!r}"
    )
    assert "real_call" in stripped, f"real tool call after </think> was incorrectly removed; got: {stripped!r}"


def test_multiple_think_blocks_all_stripped(qwen3_parser: ReasoningParser) -> None:
    """A response can contain several short think blocks (e.g. inter-step
    reflection between tool turns). All of them must be stripped, and the
    surrounding content preserved in order.
    """
    text = "<think>first round</think>answer A. <think>second round</think>answer B."
    assert qwen3_parser.extract_content(text) == "answer A. answer B."


def test_unclosed_think_block_truncates_to_eot(qwen3_parser: ReasoningParser) -> None:
    """A truncated mid-stream response may leave an unclosed ``<think>`` tag.
    Without the truncation guard the unfinished reasoning (potentially
    containing tool-call patterns) would still reach the tool parser.
    """
    text = "Some lead-in. <think>partial reasoning <tool_call>{}</tool_call> still going"
    stripped = qwen3_parser.extract_content(text)
    assert stripped == "Some lead-in. "
    assert "tool_call" not in stripped


def test_qwen3_implicit_opener_from_chat_template(
    qwen3_parser: ReasoningParser,
) -> None:
    """Qwen3 / DeepSeek-R1 default chat templates append ``<think>`` to the
    prompt itself when ``add_generation_prompt=True``, so ``response_ids``
    (the model's generated tokens only) decode to text that contains
    ``</think>`` but no leading ``<think>``. The parser must treat the
    start of text as an implicit ``<think>`` opener and strip up to the
    first ``</think>``.

    Regression for the upstream review on PR #6434: the previous fast
    path ``if "<think>" not in text: return text`` returned the response
    verbatim in this default Qwen3 setup, leaving the entire reasoning
    block (including any ``<tool_call>`` patterns inside it) visible to
    the tool parser -- defeating the whole point of the parser for the
    most common production configuration.
    """
    text = "step 1\nstep 2\n</think>\n\nFinal: foo"
    assert qwen3_parser.extract_content(text) == "\n\nFinal: foo"


def test_qwen3_implicit_opener_drops_inner_tool_call(
    qwen3_parser: ReasoningParser,
) -> None:
    """The bug-fixing case for the implicit-opener path: a ``<tool_call>``
    that appears in the chain-of-thought (before ``</think>``) must not
    leak through to the tool parser, even when there is no leading
    ``<think>`` tag in the response.
    """
    text = (
        "Let me think -- maybe I should "
        '<tool_call>{"name": "spurious", "arguments": {}}</tool_call>'
        " first</think>"
        '<tool_call>{"name": "real_call", "arguments": {}}</tool_call>'
    )
    stripped = qwen3_parser.extract_content(text)
    assert "spurious" not in stripped, (
        f"tool call inside implicit-opener think block leaked through reasoning parser; got: {stripped!r}"
    )
    assert "real_call" in stripped, f"real tool call after </think> was incorrectly removed; got: {stripped!r}"


def test_qwen3_implicit_opener_then_explicit_pair(
    qwen3_parser: ReasoningParser,
) -> None:
    """An implicit-opener span followed by additional explicit
    ``<think>...</think>`` pairs (rare for Qwen3 but allowed by the
    grammar) are all stripped: the implicit span is closed at the first
    ``</think>``, then the regular loop handles subsequent pairs.
    """
    text = "implicit reasoning</think>middle<think>more reasoning</think>tail"
    assert qwen3_parser.extract_content(text) == "middletail"


def test_qwen3_lone_close_tag_drops_everything_before(
    qwen3_parser: ReasoningParser,
) -> None:
    """A response consisting entirely of reasoning followed immediately
    by ``</think>`` (no content after the closer) collapses to the empty
    string. Pinning this so an extracted-content branch does not
    accidentally start preserving the reasoning text.
    """
    assert qwen3_parser.extract_content("just reasoning</think>") == ""


def test_idempotent(qwen3_parser: ReasoningParser) -> None:
    """Calling ``extract_content`` twice on the same input returns the same
    output (and on already-stripped text returns it unchanged).
    """
    text = "<think>once</think>final"
    once = qwen3_parser.extract_content(text)
    twice = qwen3_parser.extract_content(once)
    assert once == "final"
    assert twice == once


# --- enable_thinking parameter (chat-template kwargs awareness) -------------


def test_qwen3_short_circuits_when_thinking_disabled(qwen3_parser: ReasoningParser) -> None:
    """When ``enable_thinking=False`` mirrors the chat-template kwarg, the
    chat template has injected an empty ``<think>\\n\\n</think>\\n\\n``
    opener so the model emits no think markers at all. Any literal
    ``<think>`` substring in ``text`` (e.g. from code, quoted prompts,
    or user content) must therefore not trigger stripping.

    Without this short-circuit, a literal ``<think>`` would still be
    matched as a reasoning marker and the surrounding content treated as
    a reasoning block.
    """
    text = "Here's a doc string mentioning <think>...</think> as a marker. Rest of answer."
    # ``enable_thinking=False`` keeps the input verbatim:
    assert qwen3_parser.extract_content(text, enable_thinking=False) == text
    # Default-True path still strips the marker:
    assert qwen3_parser.extract_content(text) == "Here's a doc string mentioning  as a marker. Rest of answer."


def test_qwen3_disabled_thinking_preserves_unclosed_think_substring(
    qwen3_parser: ReasoningParser,
) -> None:
    """Pin the bug from the upstream review: when ``enable_thinking=False``
    the parser must not collapse a literal ``<think>`` substring + every
    character after it to ``""``; that would silently drop legitimate
    output. The unclosed-tag fallback only fires when stripping is on.
    """
    text = "see <think> tag for details"  # literal <think>, no closer
    # ``enable_thinking=False`` returns input unchanged:
    assert qwen3_parser.extract_content(text, enable_thinking=False) == text
    # ``enable_thinking=True`` (default) drops everything from <think> on:
    assert qwen3_parser.extract_content(text) == "see "


def test_qwen3_ignores_unknown_kwargs(qwen3_parser: ReasoningParser) -> None:
    """The base ``extract_content`` interface takes ``**kwargs`` so
    callers can forward ``chat_template_kwargs`` verbatim. An
    implementation must silently ignore keys it does not understand
    (e.g. ``think_effort`` is meaningful for hypothetical non-Qwen3
    parsers, but ``ThinkBlockReasoningParser`` should treat it as a
    no-op rather than ``TypeError`` on an unexpected keyword).
    """
    text = "<think>r</think>final"
    # Single unknown key:
    assert qwen3_parser.extract_content(text, think_effort="low") == "final"
    # Multiple unknown keys mixed with the one we *do* understand:
    assert (
        qwen3_parser.extract_content(
            text,
            enable_thinking=True,
            think_effort="medium",
            reasoning_budget=2048,
        )
        == "final"
    )
    # Unknown keys must not subvert the disabled-thinking short-circuit:
    text_with_loose_think = "see <think> tag"
    assert (
        qwen3_parser.extract_content(
            text_with_loose_think,
            enable_thinking=False,
            unrelated_knob="anything",
        )
        == text_with_loose_think
    )


def test_qwen3_nested_looking_tags_collapse_idempotently(
    qwen3_parser: ReasoningParser,
) -> None:
    """Pin the (idempotent) handling of pathological nested-looking
    inputs.

    For a degenerate input like
    ``a<think>1<think>2</think>b</think>c``, the partition loop walks
    pairs with non-overlapping first-opener / first-closer semantics:
    the outer ``<think>``..first-``</think>`` consumes ``1<think>2``,
    and the trailing ``</think>`` is left dangling. Because a dangling
    ``</think>`` would otherwise re-trigger the implicit-opener branch
    on a second call (violating idempotence), the loop sanitizes it on
    exit by dropping everything from start up to and including the
    dangling closer.

    The result is shorter than what a regex-only implementation would
    produce, but matches the implicit-opener semantics applied
    everywhere else and -- crucially -- is idempotent. Real Qwen3 /
    DeepSeek-R1 outputs do not nest ``<think>`` blocks, so this affects
    only adversarial inputs.
    """
    text = "a<think>1<think>2</think>b</think>c"
    once = qwen3_parser.extract_content(text)
    twice = qwen3_parser.extract_content(once)
    assert once == "ac"
    assert twice == once  # idempotence


def test_qwen3_dangling_close_after_explicit_pair_is_idempotent(
    qwen3_parser: ReasoningParser,
) -> None:
    """A response that contains both a complete ``<think>...</think>``
    pair AND an unmatched trailing ``</think>`` (rare, but generated by
    some malformed sampling traces) must produce the same output on
    every call. The dangling closer is sanitized so the second call
    does not re-fire the implicit-opener branch.
    """
    text = "<think>r1</think>middle</think>tail"
    once = qwen3_parser.extract_content(text)
    twice = qwen3_parser.extract_content(once)
    # The pair strips ``r1``; the dangling ``</think>`` then sanitizes
    # ``middle``; the surviving content is ``tail``.
    assert once == "tail"
    assert twice == once  # idempotence


def test_qwen3_literal_close_in_default_thinking_mode_is_stripped(
    qwen3_parser: ReasoningParser,
) -> None:
    """Document the known caveat: under the default
    ``enable_thinking=True`` path, a literal ``</think>`` substring in
    legitimate output (e.g. a discussion of the parser, quoted tag in a
    code block) is treated as an implicit-opener closer and the
    preceding text is dropped. Production Qwen3 / DeepSeek-R1 outputs
    do not emit ``</think>`` outside reasoning closure, so this matches
    vLLM / SGLang serving-side behavior. Callers that need to preserve
    such substrings must pass ``enable_thinking=False`` or register a
    custom parser.

    Pinning this so the behavior is intentional rather than a silent
    surprise; future contributors who want to lift this caveat must
    update both the parser and this test together.
    """
    text = "Here is a literal closing tag: </think> done"
    # Default: stripped (implicit-opener semantics).
    assert qwen3_parser.extract_content(text) == " done"
    # ``enable_thinking=False`` preserves the literal substring.
    assert qwen3_parser.extract_content(text, enable_thinking=False) == text


def test_qwen3_idempotent_on_implicit_opener_input(
    qwen3_parser: ReasoningParser,
) -> None:
    """The implicit-opener path itself must be idempotent: stripping
    ``reasoning</think>final`` once gives ``final``; stripping
    ``final`` again returns it unchanged.
    """
    text = "reasoning content</think>final answer"
    once = qwen3_parser.extract_content(text)
    twice = qwen3_parser.extract_content(once)
    assert once == "final answer"
    assert twice == once


# --- Registry behavior ------------------------------------------------------


def test_qwen3_and_deepseek_r1_share_implementation() -> None:
    """Both aliases dispatch to ``ThinkBlockReasoningParser`` because Qwen3
    and DeepSeek-R1 follow the same ``<think>...</think>`` convention.
    """
    qwen3 = ReasoningParser.get_reasoning_parser("qwen3")
    deepseek = ReasoningParser.get_reasoning_parser("deepseek_r1")
    assert isinstance(qwen3, ThinkBlockReasoningParser)
    assert isinstance(deepseek, ThinkBlockReasoningParser)


def test_unknown_parser_name_raises_with_available_list() -> None:
    """The error message lists available parser names so callers can
    self-correct without having to read the registry source.
    """
    with pytest.raises(ValueError) as exc_info:
        ReasoningParser.get_reasoning_parser("not-a-real-parser")
    msg = str(exc_info.value)
    assert "not-a-real-parser" in msg
    assert "qwen3" in msg
    assert "deepseek_r1" in msg


def test_custom_register_decorator_extends_registry() -> None:
    """Users can register their own ``ReasoningParser`` for non-built-in
    formats. The decorator must add the entry without mutating other
    registered parsers.
    """

    @ReasoningParser.register("test-custom-parser")
    class _CustomParser(ReasoningParser):
        def extract_content(self, text: str, **kwargs) -> str:
            return text.replace("<reflection>", "").replace("</reflection>", "")

    try:
        parser = ReasoningParser.get_reasoning_parser("test-custom-parser")
        assert parser.extract_content("<reflection>hi</reflection>foo") == "hifoo"
        # Pre-existing parsers still work.
        assert ReasoningParser.get_reasoning_parser("qwen3") is not None
    finally:
        # Clean up the test registration so other tests in the same session
        # see a deterministic registry snapshot.
        ReasoningParser._registry.pop("test-custom-parser", None)


# --- Integration: end-to-end decode → strip → encode → ToolParser ----------


class _ByteTokenizer:
    """Minimal mock tokenizer with perfect roundtrip via UTF-8 bytes.

    Each character maps to its UTF-8 byte sequence; ``decode(encode(x)) == x``
    for any text. Lets us exercise the
    ``decode → reasoning_parser → re-encode → tool_parser`` pipeline in
    ``ToolAgentLoop._handle_generating_state`` without depending on a real
    HuggingFace tokenizer download.
    """

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return bytes(ids).decode("utf-8", errors="replace")


@pytest.mark.asyncio
async def test_integration_think_block_tool_call_ignored_real_tool_call_executed() -> None:
    """End-to-end pipeline check: ``HermesToolParser(reasoning_parser=...)``
    receives the raw response_ids, internally decodes / strips / re-encodes,
    then runs Hermes extraction. The think-block tool call must be dropped
    and only the post-``</think>`` real call surfaces.

    With wzhgba's V5 review applied (verl#6434), the strip layer lives on
    ``ToolParser`` rather than ``ToolAgentLoop._handle_generating_state``,
    so any custom ``AgentLoop`` (``SWEAgentLoop`` / ``GUIAgentLoop`` / etc.)
    inherits reasoning-aware tool extraction without re-implementing the
    decode / strip / re-encode plumbing. This test pins the contract that
    a ``ToolParser`` with a configured ``reasoning_parser`` must produce
    the same tool-call set as the previous out-of-parser strip pipeline.
    """
    from verl.experimental.agent_loop.tool_parser import HermesToolParser

    tokenizer = _ByteTokenizer()
    reasoning_parser = ReasoningParser.get_reasoning_parser("qwen3")
    tool_parser = HermesToolParser(tokenizer, reasoning_parser=reasoning_parser)

    response_text = (
        "<think>"
        "I should call "
        '<tool_call>{"name": "spurious_call", "arguments": {}}</tool_call>'
        " to verify"
        "</think>"
        '<tool_call>{"name": "real_call", "arguments": {}}</tool_call>'
    )
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)

    _, tool_calls = await tool_parser.extract_tool_calls(response_ids, [])

    call_names = [call.name for call in tool_calls]
    assert "spurious_call" not in call_names, (
        "tool call inside <think> block leaked through ToolParser's internal "
        f"reasoning strip; got call_names={call_names!r}. "
        "This is the exact regression #6424 reports."
    )
    assert call_names == ["real_call"], (
        f"real tool call after </think> was lost or duplicated; got call_names={call_names!r}"
    )


@pytest.mark.asyncio
async def test_integration_no_reasoning_parser_keeps_legacy_behavior() -> None:
    """Backwards-compat guard: with ``reasoning_parser=None`` (the default),
    ``ToolParser.extract_tool_calls`` delegates straight to
    ``_extract_tool_calls`` with no decode / strip overhead, so the spurious
    think-block tool call still surfaces.

    Pinning the legacy behavior under ``None`` makes future regressions on
    the opt-in switch obvious: if this test starts failing, someone
    accidentally turned the parser on by default.
    """
    from verl.experimental.agent_loop.tool_parser import HermesToolParser

    tokenizer = _ByteTokenizer()
    tool_parser = HermesToolParser(tokenizer, reasoning_parser=None)

    response_text = (
        "<think>"
        '<tool_call>{"name": "spurious_call", "arguments": {}}</tool_call>'
        "</think>"
        '<tool_call>{"name": "real_call", "arguments": {}}</tool_call>'
    )
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)

    _, tool_calls = await tool_parser.extract_tool_calls(response_ids, [])
    call_names = [call.name for call in tool_calls]

    # Both calls present -- the spurious one is exactly the bug we're
    # fixing for opt-in users; off-by-default keeps the legacy semantics
    # unchanged so existing pipelines don't get an unannounced behavior
    # shift.
    assert call_names == ["spurious_call", "real_call"]


@pytest.mark.asyncio
async def test_tool_parser_skips_strip_when_no_reasoning_markers() -> None:
    """``_maybe_strip_reasoning`` is supposed to short-circuit when the
    decoded text contains no reasoning markers — returning the original
    ``responses_ids`` without re-encoding. This test pins that the no-op
    fast path is actually taken (re-encode would be wasteful on every
    plain response).
    """
    from verl.experimental.agent_loop.tool_parser import HermesToolParser

    tokenizer = _ByteTokenizer()
    reasoning_parser = ReasoningParser.get_reasoning_parser("qwen3")
    tool_parser = HermesToolParser(tokenizer, reasoning_parser=reasoning_parser)

    # No <think> markers anywhere — ``_maybe_strip_reasoning`` must return
    # the original ``responses_ids`` instance.
    response_text = '<tool_call>{"name": "real_call", "arguments": {}}</tool_call>'
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)

    fast_path_ids = await tool_parser._maybe_strip_reasoning(response_ids)
    assert fast_path_ids is response_ids  # identity, not a re-encoded copy

    # And the public path still extracts the real call.
    _, tool_calls = await tool_parser.extract_tool_calls(response_ids, [])
    assert [c.name for c in tool_calls] == ["real_call"]


def test_all_registered_tool_parsers_override_extract_tool_calls() -> None:
    """Each registered ``ToolParser`` subclass must override either
    ``extract_tool_calls`` (legacy) or ``_extract_tool_calls`` (recommended).

    Without this guard a future contributor could register a parser that
    inherits the base ``_extract_tool_calls`` (which raises
    ``NotImplementedError``) and only realize at production-call time that
    no implementation exists. The four in-tree parsers
    (``hermes`` / ``gpt-oss`` / ``qwen3_coder`` / ``gemma4``) all override
    ``_extract_tool_calls`` so they receive the reasoning-strip
    preprocessing the base class applies.
    """
    from verl.experimental.agent_loop.tool_parser import ToolParser

    base_extract = ToolParser._extract_tool_calls
    base_public = ToolParser.extract_tool_calls

    assert ToolParser._registry, "no ToolParser subclasses registered"

    for name, subclass in ToolParser._registry.items():
        overrides_private = subclass._extract_tool_calls is not base_extract
        overrides_public = subclass.extract_tool_calls is not base_public
        assert overrides_private or overrides_public, (
            f"Registered ToolParser {name!r} ({subclass.__name__}) overrides "
            f"neither `_extract_tool_calls` nor `extract_tool_calls`; calling "
            f"it would raise NotImplementedError at runtime."
        )
