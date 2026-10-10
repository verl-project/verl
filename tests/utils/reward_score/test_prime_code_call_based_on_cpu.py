# Copyright 2024 PRIME team and/or its affiliates
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

import json

import pytest

from verl.utils.reward_score import default_compute_score

correct_completion = "```python\ndef add(a, b):\n    return a + b\n```"
wrong_completion = "```python\ndef add(a, b):\n    return None\n```"

# Call-based tests are published in two layouts:
# - APPS/TACO: each input is already the list of arguments for the call, and each
#   expected output is the value itself;
# - e.g. LiveCodeBench: each input is a string holding one JSON value per line, and
#   each expected output is its JSON encoding.
# A correct solution must get full reward under both layouts (see #8174), and a wrong
# solution must still get zero.
call_based_test_cases = {
    "apps_list_layout": {"fn_name": "add", "inputs": [[1, 2], [3, 4]], "outputs": [3, 7]},
    "json_lines_layout": {"fn_name": "add", "inputs": ["1\n2", "3\n4"], "outputs": ["3", "7"]},
}


@pytest.mark.parametrize("test_cases", list(call_based_test_cases.values()), ids=list(call_based_test_cases.keys()))
def test_call_based_full_reward(test_cases):
    assert default_compute_score("apps", correct_completion, json.dumps(test_cases)) == 1.0
    assert default_compute_score("apps", wrong_completion, json.dumps(test_cases)) == 0.0


def test_call_based_continuous_fraction():
    # with continuous=True a partially correct solution gets the fraction of passed tests
    test_cases = {"fn_name": "add", "inputs": [[1, 2], [3, 4], [5, 999]], "outputs": [3, 7, 999]}
    assert default_compute_score("apps", correct_completion, json.dumps(test_cases)) == pytest.approx(2 / 3)
    assert default_compute_score("apps", wrong_completion, json.dumps(test_cases)) == 0.0
