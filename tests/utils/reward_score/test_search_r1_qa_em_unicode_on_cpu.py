# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""CPU tests for typographic punctuation in the Search-R1-like exact-match scorer."""

import pytest

from verl.utils.reward_score.search_r1_like_qa_em import compute_score, compute_score_subem, normalize_answer


@pytest.mark.parametrize(
    "answer, gold",
    [
        ("Schindler\u2019s List", "Schindler's List"),  # curly apostrophe
        ("Schindler's List", "Schindler\u2019s List"),  # the gold has it
        ("\u201cHamlet\u201d", "Hamlet"),  # curly quotes
        ("1914\u20131918", "1914-1918"),  # en dash
        ("Jean\u2010Paul Sartre", "Jean-Paul Sartre"),  # Unicode hyphen
        ("Ender\u2019s Game.", "Ender's Game"),
    ],
)
def test_typographic_punctuation_matches(answer, gold):
    assert normalize_answer(answer) == normalize_answer(gold)
    ground_truth = {"target": [gold]}
    assert compute_score(f"<answer>{answer}</answer>", ground_truth) == 1.0
    assert compute_score_subem(f"<answer>{answer}</answer>", ground_truth) == 1.0


@pytest.mark.parametrize(
    "answer, gold, expected",
    [
        ("Schindler's List", "Schindler's List", 1.0),
        ("the Eiffel Tower!", "Eiffel Tower", 1.0),
        ("Hamlet", "Macbeth", 0.0),
        ("$5", "5", 1.0),  # "$" is in string.punctuation, as before
        ("\u00e9t\u00e9", "ete", 0.0),  # accents are letters, not punctuation
    ],
)
def test_existing_behaviour_unchanged(answer, gold, expected):
    assert compute_score(f"<answer>{answer}</answer>", {"target": [gold]}) == expected
