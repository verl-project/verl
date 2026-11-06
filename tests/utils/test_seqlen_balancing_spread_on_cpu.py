# Copyright 2026 Individual Contributor: Isaac Li
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

"""Exact KK partition regressions, including ties and dtype-specific rounding."""

import importlib.util
import random
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch


def _load_balancer(path=None):
    # Import the complete production module without unrelated DataProto/device setup.
    path = path or Path(__file__).resolve().parents[2] / "verl/utils/seqlen_balancing.py"
    spec = importlib.util.spec_from_file_location("_spread_balancing", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    stubs = {
        "verl.protocol": {"DataProto": object},
        "verl.utils": {"tensordict_utils": ModuleType("tensordict_utils")},
        "verl.utils.device": {"get_device_name": lambda: "cpu"},
    }
    with pytest.MonkeyPatch.context() as monkeypatch:
        for name, attributes in stubs.items():
            stub = ModuleType(name)
            stub.__dict__.update(attributes)
            monkeypatch.setitem(sys.modules, name, stub)
        spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def partition():
    return _load_balancer().karmarkar_karp


# Recorded from the uncached implementation. Compare every index, including its order.
_CASES = [
    ([0] * 8, 4, [[7, 0], [6, 1], [5, 2], [4, 3]], [[7, 0], [6, 1], [5, 2], [4, 3]]),
    ([7] * 8, 4, [[7, 0], [6, 1], [5, 2], [4, 3]], [[7, 0], [6, 1], [5, 2], [4, 3]]),
    (list(range(1, 9)), 4, [[7, 0], [6, 1], [5, 2], [3, 4]], [[7, 0], [6, 1], [5, 2], [4, 3]]),
    ([99, 1, 1, 1, 1, 1, 1, 1], 4, [[0], [5, 4, 1], [7, 2], [6, 3]], [[0, 1], [7, 2], [6, 3], [5, 4]]),
    ([2, 2, 3, 3, 4, 4, 5, 5], 2, [[7, 4, 2, 0], [6, 5, 3, 1]], [[7, 4, 2, 0], [6, 5, 3, 1]]),
]
_CASTS = {
    "int": list,
    "float": lambda values: [float(value) for value in values],
    "numpy32": lambda values: list(np.array(values, dtype=np.float32)),
    "numpy64": lambda values: list(np.array(values, dtype=np.float64)),
    "tensor32": lambda values: list(torch.tensor(values, dtype=torch.float32)),
    "tensor64": lambda values: list(torch.tensor(values, dtype=torch.float64)),
}


@pytest.mark.parametrize("values,k,unequal,equal", _CASES)
@pytest.mark.parametrize("dtype", _CASTS)
@pytest.mark.parametrize("equal_size", [False, True])
def test_exact_partitions_and_tie_breaking(partition, values, k, unequal, equal, dtype, equal_size):
    expected = equal if equal_size else unequal
    for _ in range(3):
        result = partition(_CASTS[dtype](values), k, equal_size)
        assert result == expected
        assert sorted(index for group in result for index in group) == list(range(len(values)))
        if equal_size:
            assert all(len(group) == len(values) // k for group in result)


@pytest.mark.parametrize("dtype", [name for name in _CASTS if name != "int"])
@pytest.mark.parametrize("equal_size", [False, True])
def test_fractional_rounding_preserves_exact_partitions(partition, dtype, equal_size):
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    if dtype.endswith("32"):
        expected = [[7, 0], [5, 2], [6, 1], [4, 3]] if equal_size else [[7, 0], [2, 5], [6, 1], [3, 4]]
    else:
        expected = [[7, 0], [4, 3], [6, 1], [5, 2]] if equal_size else [[3, 4], [0, 7], [6, 1], [5, 2]]
    assert partition(_CASTS[dtype](values), 4, equal_size) == expected


@pytest.mark.parametrize("dtype", _CASTS)
@pytest.mark.parametrize("equal_size", [False, True])
def test_float32_precision_boundary(partition, dtype, equal_size):
    values = [16777216, 1, 2, 3, 16777215, 4, 5, 6]
    if equal_size:
        expected = [[4, 7, 3, 2], [0, 6, 5, 1]]
    elif dtype.endswith("32"):
        expected = [[2, 4, 6, 5], [1, 0, 7, 3]]
    else:
        expected = [[2, 4, 6, 5], [0, 7, 3, 1]]
    assert partition(_CASTS[dtype](values), 2, equal_size) == expected


@pytest.mark.parametrize("equal_size", [False, True])
def test_single_state_and_single_partition(partition, equal_size):
    assert partition([9], 1, equal_size) == [[0]]
    assert partition([4, 3, 2, 1], 4, equal_size) == [[0], [1], [2], [3]]
    assert partition([4, 3, 2, 1], 1, equal_size) == [[0, 1, 2, 3]]


def test_invalid_equal_size_still_raises(partition):
    with pytest.raises(AssertionError, match="3 % 2 != 0"):
        partition([1, 2, 3], 2, True)


# Exact index order recorded from upstream 8718ca30 for both size modes.
_LARGER_CASES = [
    (
        "repeated",
        8,
        [
            [27, 42, 7, 36, 1, 49, 40, 41, 8],
            [11, 63, 25, 56, 48, 21, 28, 62, 14],
            [61, 18, 0, 17, 22, 51, 5],
            [30, 37, 3, 50, 32, 13, 15],
            [59, 4, 39, 29, 23, 33, 6, 38],
            [46, 19, 35, 55, 2, 16, 45, 52],
            [43, 44, 9, 10, 53, 20, 34, 54],
            [31, 26, 60, 47, 12, 24, 57, 58],
        ],
        [
            [41, 1, 45, 61, 48, 25, 60, 0],
            [38, 16, 57, 11, 3, 46, 44, 2],
            [15, 20, 28, 19, 17, 49, 63, 50],
            [5, 21, 13, 30, 53, 23, 42, 10],
            [62, 22, 40, 59, 7, 8, 37, 39],
            [58, 24, 34, 27, 29, 9, 35, 36],
            [54, 32, 6, 43, 47, 31, 18, 56],
            [52, 33, 51, 26, 55, 14, 4, 12],
        ],
    ),
    (
        "skewed",
        16,
        [
            [0],
            [1],
            [2],
            [3],
            [119, 112, 100, 88, 76, 64, 52, 40, 28, 16, 4],
            [118, 113, 101, 89, 77, 65, 53, 41, 29, 17, 5],
            [117, 114, 102, 90, 78, 66, 54, 42, 30, 18, 6],
            [116, 115, 103, 91, 79, 67, 55, 43, 31, 19, 7],
            [127, 104, 92, 80, 68, 56, 44, 32, 20, 8],
            [126, 105, 93, 81, 69, 57, 45, 33, 21, 9],
            [125, 106, 94, 82, 70, 58, 46, 34, 22, 10],
            [124, 107, 95, 83, 71, 59, 47, 35, 23, 11],
            [123, 108, 96, 84, 72, 60, 48, 36, 24, 12],
            [122, 109, 97, 85, 73, 61, 49, 37, 25, 13],
            [121, 110, 98, 86, 74, 62, 50, 38, 26, 14],
            [120, 111, 99, 87, 75, 63, 51, 39, 27, 15],
        ],
        [
            [0, 100, 84, 68, 52, 36, 20, 4],
            [1, 101, 85, 69, 53, 37, 21, 5],
            [2, 102, 86, 70, 54, 38, 22, 6],
            [3, 103, 87, 71, 55, 39, 23, 7],
            [127, 104, 88, 72, 56, 40, 24, 8],
            [126, 105, 89, 73, 57, 41, 25, 9],
            [125, 106, 90, 74, 58, 42, 26, 10],
            [124, 107, 91, 75, 59, 43, 27, 11],
            [123, 108, 92, 76, 60, 44, 28, 12],
            [122, 109, 93, 77, 61, 45, 29, 13],
            [121, 110, 94, 78, 62, 46, 30, 14],
            [120, 111, 95, 79, 63, 47, 31, 15],
            [119, 112, 96, 80, 64, 48, 32, 16],
            [118, 113, 97, 81, 65, 49, 33, 17],
            [117, 114, 98, 82, 66, 50, 34, 18],
            [116, 115, 99, 83, 67, 51, 35, 19],
        ],
    ),
]


@pytest.mark.parametrize("scenario,k,unequal,equal", _LARGER_CASES)
@pytest.mark.parametrize("equal_size", [False, True])
def test_larger_partitions_match_original(partition, scenario, k, unequal, equal, equal_size):
    if scenario == "repeated":
        rng = random.Random(2026)
        values = [rng.randrange(1, 33) for _ in range(64)]
    else:
        values = [32768, 16384, 8192, 4096] + [1] * 124
    assert partition(values, k, equal_size) == (equal if equal_size else unequal)
