# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Regression tests for CVE-2026-6878 / GHSA-h57c-v2v3-5v3v.

The prime_math grader used ``eval()`` on model-authored answer text in two places
(``handle_pi`` and the ``math_equal`` matrix branch). Because the reward function runs
in the reward worker's own process, a single poisoned sample could execute arbitrary
code and even monkeypatch grading so every later sample scored full reward. These tests
pin down that:

* the safe evaluator accepts arithmetic / list literals but rejects anything executable,
* neither sink executes an injected payload,
* a poisoned sample can no longer forge rewards for later samples, and
* legitimate arithmetic and list-matrix grading are unchanged.
"""

import math
import time

import pytest

from verl.utils.reward_score.prime_math.grader import (
    handle_pi,
    math_equal,
    safe_arithmetic_eval,
)


class TestSafeArithmeticEval:
    def test_accepts_numbers_and_arithmetic(self):
        assert safe_arithmetic_eval("1*3.14+2") == pytest.approx(1 * 3.14 + 2)
        assert safe_arithmetic_eval("2**3") == 8
        assert safe_arithmetic_eval("-5") == -5
        assert safe_arithmetic_eval("(1 + 2) * 3") == 9
        assert safe_arithmetic_eval("7 / 2") == pytest.approx(3.5)

    def test_accepts_list_and_tuple_literals(self):
        assert safe_arithmetic_eval("[1, 2, 3]") == [1, 2, 3]
        assert safe_arithmetic_eval("[1/2, 3]") == [0.5, 3]
        assert safe_arithmetic_eval("(1, 2)") == (1, 2)

    @pytest.mark.parametrize(
        "expr",
        [
            'exec("import os")',
            '__import__("os").system("echo x")',
            'open("f","w")',
            "os.system('x')",
            "[x for x in range(3)]",
            "lambda: 1",
            "True",  # bool is not a numeric literal we allow
        ],
    )
    def test_rejects_executable_or_non_numeric(self, expr):
        with pytest.raises((ValueError, SyntaxError)):
            safe_arithmetic_eval(expr)


class TestHandlePiIsNotExecutable:
    def test_handle_pi_does_not_execute_payload(self, tmp_path):
        marker = tmp_path / "pwned.txt"
        # Valid-syntax payload: `1*math.pi + exec(...)` was a real RCE via eval().
        payload = r"\pi+" + "exec(" + repr(f"import os as o\no.system('echo x > {marker}')") + ")"
        result = handle_pi(payload, math.pi)
        assert not marker.exists(), "handle_pi executed model-authored code"
        # Non-numeric input falls through unchanged (fail-closed), not raising.
        assert isinstance(result, str)

    def test_handle_pi_still_evaluates_legit_arithmetic(self):
        assert handle_pi(r"2\pi", math.pi) == pytest.approx(2 * math.pi)
        assert handle_pi(r"3\pi+1", math.pi) == pytest.approx(3 * math.pi + 1)
        assert handle_pi("no pi here", math.pi) == "no pi here"


class TestMatrixBranchIsNotExecutable:
    def test_matrix_branch_does_not_execute_payload(self, tmp_path, monkeypatch):
        # Use a relative marker with no underscore so the payload survives the
        # grader's normalize() step and actually reaches the (former) eval() sink,
        # matching the shape of a real attack.
        monkeypatch.chdir(tmp_path)
        marker = tmp_path / "pwned.txt"
        prediction = '[open("pwned.txt","w").write("x"), 2]'
        reference = r"\begin{pmatrix}1&2\end{pmatrix}"
        # Must not raise and must not execute the payload.
        assert math_equal(prediction, reference) is False
        assert not marker.exists(), "matrix branch executed model-authored code"

    def test_matrix_branch_still_grades_list_predictions(self):
        # A list-form prediction still parses and is compared (behavior preserved).
        assert math_equal(r"[1, 2]", r"\begin{pmatrix}9&9\end{pmatrix}") is False


class TestNoRewardForgeryAcrossSamples:
    def test_payload_cannot_monkeypatch_grade_answer(self, tmp_path):
        marker = tmp_path / "pwned.txt"
        from verl.utils.reward_score import prime_math

        original = prime_math.grade_answer
        try:
            # A wrong answer is wrong before the attack.
            assert prime_math.grade_answer("999", "42") is False
            payload_body = (
                "import verl.utils.reward_score.prime_math as p\n"
                "p.grade_answer=lambda *a, **k: True\n"
                f"import os as o; o.system('echo x > {marker}')"
            )
            attack = r"\pi+" + "exec(" + repr(payload_body) + ")"
            handle_pi(attack, math.pi)
            # The attack neither ran nor forged later grading.
            assert not marker.exists()
            assert prime_math.grade_answer is original
            assert prime_math.grade_answer("999", "42") is False
        finally:
            prime_math.grade_answer = original


class TestLegitimateGradingUnchanged:
    def test_boxed_answers(self):
        from verl.utils.reward_score.prime_math import compute_score

        assert compute_score(r"The answer is \boxed{42}", "42")[0] is True
        assert compute_score(r"The answer is \boxed{42}", "7")[0] is False

    def test_pi_and_fraction_equality(self):
        assert math_equal(r"3\pi", "9.42477796") is True
        assert math_equal("0.5", "0.5") is True


class TestResourceBounds:
    """The evaluator must not turn RCE into a CPU/memory denial of service.

    ``safe_arithmetic_eval`` runs on model-authored text in a long-lived reward worker,
    and its callers only guard with ``except Exception`` -- which a hang or an OOM does
    not raise. So an expression that is merely ruinously expensive has to be *rejected*,
    not merely survived.
    """

    @pytest.mark.parametrize(
        "expr",
        [
            "9**9**9**9",  # right-associated tower: the classic int-blowup payload
            "9**9**9",
            "2**999999999",
            "10**100000",
            "(10**1000)**1000",
        ],
    )
    def test_rejects_exponent_blowup(self, expr):
        with pytest.raises(ValueError):
            safe_arithmetic_eval(expr)

    def test_exponent_blowup_is_rejected_promptly(self):
        """Rejection must come from the size check, not from actually computing it."""
        start = time.monotonic()
        with pytest.raises(ValueError):
            safe_arithmetic_eval("9**9**9**9")
        assert time.monotonic() - start < 1.0

    def test_rejects_oversized_intermediate_from_multiplication(self):
        """Bounding only ``**`` would leave repeated multiplication as a way through."""
        with pytest.raises(ValueError):
            safe_arithmetic_eval("(2**4000) * (2**4000)")

    def test_rejects_overlong_expression(self):
        with pytest.raises(ValueError):
            safe_arithmetic_eval("1+" * 20_000 + "1")

    def test_rejects_too_many_nodes(self):
        with pytest.raises(ValueError):
            safe_arithmetic_eval("1+" * 9_000 + "1")

    def test_rejects_deep_nesting(self):
        with pytest.raises(ValueError):
            safe_arithmetic_eval("-" * 100 + "1")

    def test_bounds_do_not_reject_realistic_answers(self):
        """The limits must sit far above anything a real answer contains."""
        assert safe_arithmetic_eval("2**10") == 1024
        assert safe_arithmetic_eval("2**0.5") == pytest.approx(math.sqrt(2))
        assert safe_arithmetic_eval("-2**3") == -8
        assert safe_arithmetic_eval("3.14159*2") == pytest.approx(3.14159 * 2)
        assert safe_arithmetic_eval("[1, 2, 3, 4, 5, 6, 7, 8, 9]") == [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_grader_sinks_survive_a_blowup_payload(self):
        """Both eval() sinks must reject the payload rather than hang.

        ``handle_pi`` substitutes the numeric pi before evaluating and suppresses any
        failure, so a rejected expression leaves the substituted *string* in place. That
        is the intended fallthrough: a non-numeric result simply grades as not correct.
        """
        start = time.monotonic()

        result = handle_pi(r"9**9**9**9\pi", math.pi)
        assert isinstance(result, str)
        assert result == "9**9**9**9*" + repr(math.pi)

        assert math_equal("[9**9**9**9]", r"\begin{pmatrix}1\end{pmatrix}") is False
        assert time.monotonic() - start < 2.0
