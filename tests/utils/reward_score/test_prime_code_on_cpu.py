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

"""CPU tests for prime_code reward scoring and testing_util."""

import json

import pytest

from verl.utils.reward_score.prime_code import compute_score
from verl.utils.reward_score.prime_code.utils import check_correctness


@pytest.fixture
def call_based_add_test_cases():
    return json.dumps(
        {
            "fn_name": "add",
            "inputs": ["1\n2", "-1\n1", "10\n5"],
            "outputs": ["3", "0", "15"],
        }
    )


def test_call_based_correct_solution(call_based_add_test_cases):
    code = "def add(a, b):\n    return a + b\n"
    success, metadata = compute_score(f"```python\n{code}```", call_based_add_test_cases, continuous=True)
    assert success is True


def test_call_based_wrong_solution(call_based_add_test_cases):
    code = "def add(a, b):\n    return a - b\n"
    success, metadata = compute_score(f"```python\n{code}```", call_based_add_test_cases, continuous=True)
    assert success == 0.0


def test_call_based_always_equal_int_subclass_rejected(call_based_add_test_cases):
    # Spoofed class whose __eq__ always returns True
    code = (
        "class _Any(int):\n"
        "    def __eq__(self, other):\n"
        "        return True\n"
        "    __hash__ = int.__hash__\n"
        "def add(a, b):\n"
        "    return _Any(0)\n"
    )
    success, metadata = compute_score(f"```python\n{code}```", call_based_add_test_cases, continuous=True)
    # Only the second test case (-1 + 1 == 0) matches _Any(0) after value extraction; others fail.
    # The overall score must not be True (1.0).
    assert success != 1.0
    assert success is not True


def test_call_based_always_equal_list_subclass_rejected(call_based_add_test_cases):
    code = "class _Any(list):\n    def __eq__(self, other):\n        return True\ndef add(a, b):\n    return _Any()\n"
    success, metadata = compute_score(f"```python\n{code}```", call_based_add_test_cases, continuous=True)
    assert success == 0.0
    assert success is not True


def test_continuous_false_on_failing_answer_does_not_raise(call_based_add_test_cases):
    code = "def add(a, b):\n    return a - b\n"
    success, metadata = compute_score(f"```python\n{code}```", call_based_add_test_cases, continuous=False)
    assert success is False
    assert metadata is None


def test_call_based_tuple_output_matches_list_ground_truth():
    test_cases = json.dumps(
        {
            "fn_name": "pair",
            "inputs": ["1\n2", "3\n4"],
            "outputs": ["[1, 2]", "[3, 4]"],
        }
    )
    code = "def pair(a, b):\n    return (a, b)\n"
    success, metadata = compute_score(f"```python\n{code}```", test_cases, continuous=True)
    assert success is True


def test_check_correctness_direct(call_based_add_test_cases):
    dict_cases = json.loads(call_based_add_test_cases)
    correct_code = "def add(a, b):\n    return a + b\n"
    res, metadata = check_correctness(dict_cases, correct_code, timeout=5)
    assert all(r is True for r in res)

    spoof_code = (
        "class _Any(int):\n"
        "    def __eq__(self, other):\n"
        "        return True\n"
        "    __hash__ = int.__hash__\n"
        "def add(a, b):\n"
        "    return _Any(0)\n"
    )
    res_spoof, _ = check_correctness(dict_cases, spoof_code, timeout=5)
    # Must not all be True
    assert not all(r is True for r in res_spoof)


def test_call_based_dict_integer_keys():
    test_cases = json.dumps(
        {
            "fn_name": "make_map",
            "inputs": ['1\n"a"'],
            "outputs": ['{"1": "a"}'],
        }
    )
    code = "def make_map(k, v):\n    return {k: v}\n"
    success, metadata = compute_score(f"```python\n{code}```", test_cases, continuous=True)
    assert success is True


def test_call_based_unnormalizable_object_rejected():
    test_cases = json.dumps(
        {
            "fn_name": "f",
            "inputs": ["1"],
            "outputs": ["1"],
        }
    )
    code = "class Malicious:\n    def __eq__(self, other):\n        return True\ndef f(x):\n    return Malicious()\n"
    success, metadata = compute_score(f"```python\n{code}```", test_cases, continuous=True)
    assert success == 0.0
    assert success is not True


def test_call_based_always_equal_str_subclass_rejected():
    test_cases = json.dumps(
        {
            "fn_name": "f",
            "inputs": ["1"],
            "outputs": ['"expected"'],
        }
    )
    code = (
        "class SpoofedStr(str):\n"
        "    def __str__(self):\n"
        "        return self\n"
        "    def __eq__(self, other):\n"
        "        return True\n"
        "def f(x):\n"
        "    return SpoofedStr('wrong')\n"
    )
    success, metadata = compute_score(f"```python\n{code}```", test_cases, continuous=True)
    assert success == 0.0
    assert success is not True


def test_call_based_int_subclass_overridden_int_uses_stored_value():
    # Stored value is 5, but overridden __int__ returns 999
    code = (
        "class _OverrideInt(int):\n    def __int__(self):\n        return 999\ndef f(x):\n    return _OverrideInt(5)\n"
    )
    # Matching stored value (5) passes
    test_cases_match = json.dumps({"fn_name": "f", "inputs": ["1"], "outputs": ["5"]})
    success, _ = compute_score(f"```python\n{code}```", test_cases_match, continuous=True)
    assert success is True

    # Matching spoofed override value (999) fails
    test_cases_mismatch = json.dumps({"fn_name": "f", "inputs": ["1"], "outputs": ["999"]})
    success, _ = compute_score(f"```python\n{code}```", test_cases_mismatch, continuous=True)
    assert success == 0.0
    assert success is not True


def test_call_based_int_subclass_tolist_and_item_ignored():
    # Stored value is 5, but tolist() and item() return 999
    code = (
        "class _ToListInt(int):\n"
        "    def tolist(self):\n"
        "        return [999]\n"
        "    def item(self):\n"
        "        return 999\n"
        "def f(x):\n"
        "    return _ToListInt(5)\n"
    )
    test_cases_match = json.dumps({"fn_name": "f", "inputs": ["1"], "outputs": ["5"]})
    success, _ = compute_score(f"```python\n{code}```", test_cases_match, continuous=True)
    assert success is True

    test_cases_mismatch = json.dumps({"fn_name": "f", "inputs": ["1"], "outputs": ["999"]})
    success, _ = compute_score(f"```python\n{code}```", test_cases_mismatch, continuous=True)
    assert success == 0.0
    assert success is not True


def test_call_based_list_subclass_overridden_iter_uses_stored_elements():
    # Stored element is [0], but overridden __iter__ yields iter([3])
    code = (
        "class _OverrideList(list):\n"
        "    def __iter__(self):\n"
        "        return iter([3])\n"
        "def f(x):\n"
        "    l = _OverrideList()\n"
        "    list.append(l, 0)\n"
        "    return l\n"
    )
    test_cases_match = json.dumps({"fn_name": "f", "inputs": ["1"], "outputs": ["[0]"]})
    success, _ = compute_score(f"```python\n{code}```", test_cases_match, continuous=True)
    assert success is True

    test_cases_mismatch = json.dumps({"fn_name": "f", "inputs": ["1"], "outputs": ["[3]"]})
    success, _ = compute_score(f"```python\n{code}```", test_cases_mismatch, continuous=True)
    assert success == 0.0
    assert success is not True


def test_call_based_dict_subclass_overridden_items_uses_stored_entries():
    # Stored dict is {1: 'wrong'}, but items() returns [(1, 'a')]
    code = (
        "class _OverrideDict(dict):\n"
        "    def items(self):\n"
        "        return [(1, 'a')]\n"
        "def f(x):\n"
        "    d = _OverrideDict()\n"
        "    dict.__setitem__(d, 1, 'wrong')\n"
        "    return d\n"
    )
    test_cases_match = json.dumps({"fn_name": "f", "inputs": ["1"], "outputs": ['{"1": "wrong"}']})
    success, _ = compute_score(f"```python\n{code}```", test_cases_match, continuous=True)
    assert success is True

    test_cases_mismatch = json.dumps({"fn_name": "f", "inputs": ["1"], "outputs": ['{"1": "a"}']})
    success, _ = compute_score(f"```python\n{code}```", test_cases_mismatch, continuous=True)
    assert success == 0.0
    assert success is not True
