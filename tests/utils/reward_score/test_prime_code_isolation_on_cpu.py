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
"""The generated solution must not be able to read the expected outputs of prime_code tests."""

import json

import pytest

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.prime_code.testing_util import run_test

CALL_GT = json.dumps({"fn_name": "add", "inputs": ["1\n2", "3\n4"], "outputs": ["3", "7"]})
STDIN_GT = json.dumps({"inputs": ["1 2\n", "3 4\n"], "outputs": ["3\n", "7\n"]})

# Walk up the interpreter stack and return the grader's expected output for the current test.
_FIND_EXPECTED = """
import sys
def _expected():
    f = sys._getframe(1)
    while f is not None:
        io, idx = f.f_locals.get("in_outs"), f.f_locals.get("index")
        if isinstance(io, dict) and idx is not None:
            return io["outputs"][idx]
        f = f.f_back
    import gc
    for o in gc.get_objects():
        if isinstance(o, dict) and "outputs" in o and "inputs" in o:
            return o["outputs"][0]
    return None
"""


@pytest.mark.parametrize(
    "ground_truth, solution, expected",
    [
        (CALL_GT, "def add(a, b):\n    return a + b\n", 1.0),
        (CALL_GT, "def add(a, b):\n    return 0\n", 0.0),
        (CALL_GT, _FIND_EXPECTED + "def add(a, b):\n    return _expected()\n", 0.0),
        (STDIN_GT, "a, b = map(int, input().split())\nprint(a + b)\n", 1.0),
        (STDIN_GT, "print(0)\n", 0.0),
        (STDIN_GT, _FIND_EXPECTED + "print(_expected())\n", 0.0),
    ],
    ids=["call_correct", "call_wrong", "call_reads_expected", "stdin_correct", "stdin_wrong", "stdin_reads_expected"],
)
def test_prime_code_solution_cannot_read_expected_outputs(ground_truth, solution, expected):
    score = default_compute_score("codecontests", f"```python\n{solution}```", ground_truth)
    assert float(score) == expected


def test_prime_code_worker_errors():
    gt = {"fn_name": "add", "inputs": ["1\n2"], "outputs": ["3"]}
    res, meta = run_test(json.loads(json.dumps(gt)), test="def add(a, b)\n    return a + b\n", timeout=2)
    assert res == [-2] and "SyntaxError" in meta["error"]
    res, meta = run_test(json.loads(json.dumps(gt)), test="def add(a, b):\n    return 1 // 0\n", timeout=2)
    assert res == [-1] and "ZeroDivisionError" in meta["error"]
    res, _ = run_test(json.loads(json.dumps(gt)), test="def add(a, b):\n    while True:\n        pass\n", timeout=1)
    assert res == [-1]
