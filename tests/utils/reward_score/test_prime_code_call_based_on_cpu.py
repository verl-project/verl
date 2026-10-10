# Copyright 2025 Individual Contributor: Prabhu Gopal
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

RIGHT = "```python\ndef add(a, b):\n    return a + b\n```"
WRONG = "```python\ndef add(a, b):\n    return a - b\n```"

# APPS / TACO publish call-based tests with each input as a list of arguments
APPS_LAYOUT = {"fn_name": "add", "inputs": [[1, 2], [3, 4]], "outputs": [3, 7]}
# LiveCodeBench-style: one JSON value per line, outputs JSON-encoded
STRING_LAYOUT = {"fn_name": "add", "inputs": ["1\n2", "3\n4"], "outputs": ["3", "7"]}


@pytest.mark.parametrize("tests", [APPS_LAYOUT, STRING_LAYOUT], ids=["apps_layout", "string_layout"])
def test_call_based_correct_solution_gets_full_reward(tests):
    assert default_compute_score("apps", RIGHT, json.dumps(tests)) == 1.0


@pytest.mark.parametrize("tests", [APPS_LAYOUT, STRING_LAYOUT], ids=["apps_layout", "string_layout"])
def test_call_based_wrong_solution_gets_no_reward(tests):
    assert default_compute_score("apps", WRONG, json.dumps(tests)) == 0.0
