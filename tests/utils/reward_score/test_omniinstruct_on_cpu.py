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

from verl.utils.reward_score import default_compute_score, omniinstruct


def test_omniinstruct_exact_match_scores_one():
    result = omniinstruct.compute_score("The man is cooking curry.", "The man is cooking curry.")

    assert result["score"] == 1.0
    assert result["exact_match"] == 1.0


def test_omniinstruct_overlap_reward_is_soft():
    result = omniinstruct.compute_score(
        "A person is cooking in the kitchen.",
        "The man is cooking curry.",
    )

    assert 0.0 < result["score"] < 1.0
    assert result["token_f1"] > 0.0
    assert result["rouge_l_f1"] > 0.0


def test_default_compute_score_routes_omniinstruct():
    result = default_compute_score(
        "m-a-p/OmniInstruct",
        "A person is chopping vegetables while speaking.",
        "A person is chopping vegetables.",
    )

    assert isinstance(result, dict)
    assert "score" in result
    assert result["score"] > 0.0


def test_long_rambling_with_short_ref_gets_no_substring_bonus():
    """A long ramble that happens to embed the reference must not collect the
    substring bonus. Without this guard the policy learns to keep generating
    until it accidentally mentions the answer (observed as length blow-up
    and reward collapse on the qwen3-omni GSPO run)."""
    reference = "The man is cooking curry."
    long_pred = (
        "Honestly I have been thinking about this for a while because the audio "
        "felt ambiguous and I wanted to triple check before answering. "
        * 30
        + " The man is cooking curry. "
        + "Now let me explain why I think so for a few more paragraphs. " * 30
    )

    bounded = omniinstruct.compute_score(long_pred, reference)
    raw = omniinstruct._substring_match(long_pred, [reference])

    assert raw == 1.0, "raw substring detector should still trip"
    assert bounded["substring_match_bounded"] == 0.0
    # Lexical overlap on a 1000-token rambling against a 4-token reference
    # is dominated by the precision penalty, so the resulting score is small.
    assert bounded["score"] < 0.2


def test_bounded_substring_match_grants_partial_credit():
    """A short, on-topic answer that contains the reference still collects the
    softened substring bonus."""
    result = omniinstruct.compute_score(
        "Yes, the man is cooking curry on the stove.",
        "The man is cooking curry.",
    )

    assert result["substring_match_bounded"] == 1.0
    assert result["score"] >= omniinstruct.DEFAULT_SUBSTRING_BONUS


def test_substring_bonus_blocked_for_single_token_reference():
    """Single-token MCQ references like 'B' would otherwise be trivially
    contained in any verbose response. The bounded substring path requires
    at least two reference tokens by default."""
    # Note: 'A' would be removed by ``normalize_answer`` as an article, so we
    # use 'B' which survives normalization and keeps the raw substring helper
    # firing — proving that only the bounded path suppresses the bonus.
    result = omniinstruct.compute_score(
        "After listening carefully I believe the correct answer is B.",
        "B",
    )

    assert result["substring_match_bounded"] == 0.0
    # The raw containment metric still fires for diagnostic purposes.
    assert result["substring_match"] == 1.0


def test_format_bonus_added_when_answer_tag_present():
    no_tag = omniinstruct.compute_score(
        "The man is cooking curry on the stove.",
        "The man is cooking curry.",
    )
    with_tag = omniinstruct.compute_score(
        "<think>let me transcribe</think><answer>The man is cooking curry.</answer>",
        "The man is cooking curry.",
    )

    assert with_tag["format_match"] == 1.0
    assert no_tag["format_match"] == 0.0
    # Exact answers already saturate at 1.0; the bonus only matters when the
    # base score has headroom.
    weak_no_tag = omniinstruct.compute_score(
        "Cooking happens here.",
        "The man is cooking curry.",
    )
    weak_with_tag = omniinstruct.compute_score(
        "<answer>Cooking happens here.</answer>",
        "The man is cooking curry.",
    )
    assert weak_with_tag["score"] > weak_no_tag["score"]


def test_format_bonus_does_not_rescue_zero_score():
    """If the base score is zero, the format bonus must not turn a wrong
    answer into a positive reward."""
    result = omniinstruct.compute_score(
        "<answer>completely unrelated content</answer>",
        "the man is cooking curry",
    )
    assert result["format_match"] == 1.0
    assert result["score"] == 0.0


def test_truncation_penalty_scales_score_down():
    base = omniinstruct.compute_score(
        "The man is cooking curry.",
        "The man is cooking curry.",
    )
    truncated = omniinstruct.compute_score(
        "The man is cooking curry.",
        "The man is cooking curry.",
        extra_info={"truncated": True},
    )

    assert base["score"] == 1.0
    assert truncated["score"] == 0.5
    # The override knob makes it possible to disable the penalty entirely.
    disabled = omniinstruct.compute_score(
        "The man is cooking curry.",
        "The man is cooking curry.",
        extra_info={"truncated": True, "truncation_penalty": 1.0},
    )
    assert disabled["score"] == 1.0


def test_extra_info_overrides_pass_through_default_compute_score():
    """extra_info must reach the omniinstruct scorer when routed through the
    central default_compute_score dispatcher."""
    result = default_compute_score(
        "m-a-p/OmniInstruct",
        "<answer>A person is chopping vegetables.</answer>",
        "A person is chopping vegetables.",
        extra_info={"truncated": True, "truncation_penalty": 0.0},
    )
    assert result["score"] == 0.0


def test_score_bounds_remain_in_unit_interval():
    cases = [
        ("", "anything"),
        ("anything", ""),
        ("anything", None),
        ("anything", []),
        ("anything", ["", None]),
        ("<answer>x</answer>", ["x"]),
        ("a" * 5000, "the man is cooking curry"),
    ]
    for pred, ref in cases:
        out = omniinstruct.compute_score(pred, ref)
        assert 0.0 <= out["score"] <= 1.0, (pred[:30], ref, out["score"])
