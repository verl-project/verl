# Copyright 2026 Alibaba Group
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
"""Unit tests for HarborAgentLoop's pure flatten/merge helpers.

These cover the prefix-aware trajectory merge that converts Harbor's per-turn
rollout_details into the linear (prompt_ids, response_ids, response_mask)
format VeRL's training pipeline expects.
"""

from __future__ import annotations

import pytest

from verl.experimental.agentic_rl_harbor.harbor_agent_loop import (
    _build_step_wise,
    _is_prefix,
    _merge_stepwise,
)


def _rd(prompts, completions, logprobs):
    """Build a single Harbor-style rollout_details segment."""
    return [
        {
            "prompt_token_ids": prompts,
            "completion_token_ids": completions,
            "logprobs": logprobs,
        }
    ]


def test_is_prefix_basic():
    assert _is_prefix([], [1, 2, 3])
    assert _is_prefix([1, 2], [1, 2, 3])
    assert _is_prefix([1, 2, 3], [1, 2, 3])
    assert not _is_prefix([1, 2, 3], [1, 2])
    assert not _is_prefix([1, 4], [1, 2, 3])


def test_build_step_wise_single_turn():
    rd = _rd([[1, 2]], [[10, 11]], [[-0.1, -0.2]])
    turns = _build_step_wise(rd)
    assert len(turns) == 1
    assert turns[0] == {
        "prompt_ids": [1, 2],
        "comp_ids": [10, 11],
        "logprobs": [-0.1, -0.2],
    }


def test_build_step_wise_multi_turn():
    rd = _rd(
        prompts=[[1, 2], [1, 2, 10, 11, 3], [1, 2, 10, 11, 3, 20, 4]],
        completions=[[10, 11], [20], [30, 31]],
        logprobs=[[-0.1, -0.2], [-0.3], [-0.4, -0.5]],
    )
    turns = _build_step_wise(rd)
    assert len(turns) == 3
    assert turns[1]["prompt_ids"] == [1, 2, 10, 11, 3]
    assert turns[2]["comp_ids"] == [30, 31]


def test_build_step_wise_rejects_multiple_segments():
    bad = [
        {"prompt_token_ids": [[1]], "completion_token_ids": [[2]], "logprobs": [[-0.1]]},
        {"prompt_token_ids": [[3]], "completion_token_ids": [[4]], "logprobs": [[-0.2]]},
    ]
    with pytest.raises(AssertionError):
        _build_step_wise(bad)


def test_build_step_wise_rejects_mismatched_lengths():
    bad = _rd([[1, 2]], [[10, 11]], [[-0.1, -0.2, -0.3]])  # logprobs longer than completion
    with pytest.raises(AssertionError):
        _build_step_wise(bad)


def test_merge_stepwise_single_turn_one_group():
    turns = [{"prompt_ids": [1, 2], "comp_ids": [10, 11], "logprobs": [-0.1, -0.2]}]
    groups = _merge_stepwise(turns)
    assert len(groups) == 1
    g = groups[0]
    assert g["prompt_ids"] == [1, 2]
    assert g["response_ids"] == [10, 11]
    assert g["response_mask"] == [1, 1]
    assert g["response_logprobs"] == [-0.1, -0.2]


def test_merge_stepwise_clean_prefix_collapses_to_one_group():
    # Harbor re-rendered prompts each strictly extend prompt[t-1] + comp[t-1]
    # by appending observation tokens (3 between turns 1->2; 4 between 2->3).
    turns = [
        {"prompt_ids": [1, 2], "comp_ids": [10, 11], "logprobs": [-0.1, -0.2]},
        {"prompt_ids": [1, 2, 10, 11, 3], "comp_ids": [20], "logprobs": [-0.3]},
        {"prompt_ids": [1, 2, 10, 11, 3, 20, 4], "comp_ids": [30, 31], "logprobs": [-0.4, -0.5]},
    ]
    groups = _merge_stepwise(turns)
    assert len(groups) == 1
    g = groups[0]
    assert g["prompt_ids"] == [1, 2]
    # comp[0] | obs(3) | comp[1] | obs(4) | comp[2]
    assert g["response_ids"] == [10, 11, 3, 20, 4, 30, 31]
    assert g["response_mask"] == [1, 1, 0, 1, 0, 1, 1]
    # Observation tokens get logprob 0.0 (masked out anyway).
    assert g["response_logprobs"] == [-0.1, -0.2, 0.0, -0.3, 0.0, -0.4, -0.5]
    # Mask sum equals total completion tokens across all turns.
    assert sum(g["response_mask"]) == 5


def test_merge_stepwise_prefix_divergence_flushes_new_group():
    # turn 2's prompt does NOT extend turn 1's prompt+comp (the chat template
    # re-rendered history non-trivially), so a new merge group is started.
    turns = [
        {"prompt_ids": [1, 2], "comp_ids": [10, 11], "logprobs": [-0.1, -0.2]},
        # Diverges at index 2 (5 vs 10).
        {"prompt_ids": [1, 2, 5, 6], "comp_ids": [20, 21], "logprobs": [-0.3, -0.4]},
    ]
    groups = _merge_stepwise(turns)
    assert len(groups) == 2

    g0, g1 = groups
    assert g0["prompt_ids"] == [1, 2]
    assert g0["response_ids"] == [10, 11]
    assert g0["response_mask"] == [1, 1]

    assert g1["prompt_ids"] == [1, 2, 5, 6]
    assert g1["response_ids"] == [20, 21]
    assert g1["response_mask"] == [1, 1]


def test_merge_stepwise_partial_divergence_after_clean_extension():
    # turns 0->1 extend cleanly; turn 2 diverges -> two groups.
    turns = [
        {"prompt_ids": [1, 2], "comp_ids": [10], "logprobs": [-0.1]},
        {"prompt_ids": [1, 2, 10, 3], "comp_ids": [20], "logprobs": [-0.2]},
        # Diverges from g0's cursor [1,2,10,3,20].
        {"prompt_ids": [1, 2, 99], "comp_ids": [30], "logprobs": [-0.3]},
    ]
    groups = _merge_stepwise(turns)
    assert len(groups) == 2
    g0, g1 = groups
    assert g0["prompt_ids"] == [1, 2]
    assert g0["response_ids"] == [10, 3, 20]
    assert g0["response_mask"] == [1, 0, 1]
    assert g1["prompt_ids"] == [1, 2, 99]
    assert g1["response_ids"] == [30]


def test_merge_stepwise_divergence_when_next_prompt_is_shorter_than_cursor():
    # If Harbor returns a re-rendered prompt that is *shorter* than the running
    # cursor (rare, but possible if the template summarised history), the
    # length check alone rejects the prefix and we flush a new group.
    turns = [
        {"prompt_ids": [1, 2, 3, 4], "comp_ids": [10, 11], "logprobs": [-0.1, -0.2]},
        {"prompt_ids": [1, 2], "comp_ids": [20], "logprobs": [-0.3]},
    ]
    groups = _merge_stepwise(turns)
    assert len(groups) == 2
    assert groups[0]["prompt_ids"] == [1, 2, 3, 4]
    assert groups[1]["prompt_ids"] == [1, 2]


def test_merge_stepwise_empty_raises():
    with pytest.raises(AssertionError):
        _merge_stepwise([])


def test_build_then_merge_end_to_end_clean():
    # _build_step_wise + _merge_stepwise on a fully prefix-clean trajectory
    # should be equivalent to gluing all completions into one big group.
    rd = _rd(
        prompts=[[1, 2], [1, 2, 10, 11], [1, 2, 10, 11, 20]],
        completions=[[10, 11], [20], [30]],
        logprobs=[[-0.1, -0.1], [-0.2], [-0.3]],
    )
    turns = _build_step_wise(rd)
    groups = _merge_stepwise(turns)
    assert len(groups) == 1
    g = groups[0]
    assert g["prompt_ids"] == [1, 2]
    assert g["response_ids"] == [10, 11, 20, 30]
    # No observation tokens between turns (next prompt extends by exactly comp).
    assert g["response_mask"] == [1, 1, 1, 1]
    assert sum(g["response_mask"]) == sum(len(c) for c in [[10, 11], [20], [30]])
