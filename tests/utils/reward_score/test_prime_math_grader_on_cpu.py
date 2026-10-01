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

import time

import pytest
from sympy import Expr, Float

from verl.utils.reward_score.prime_math.grader import symbolic_equal


class _SlowExpr(Expr):
    def _eval_evalf(self, prec):
        time.sleep(0.2)
        return Float(1)


@pytest.mark.parametrize(
    ("prediction", "reference"),
    [
        ("x + x", "2 * x"),
        ("1 / 3", "0.3333333333333333"),
    ],
)
def test_symbolic_equal_accepts_equivalent_expressions(prediction, reference):
    assert symbolic_equal(prediction, reference, tolerance=1e-4, timeout=2.0)


def test_symbolic_equal_times_out_numerical_evaluation():
    assert not symbolic_equal(_SlowExpr(), Float(1), tolerance=1e-4, timeout=0.05)
