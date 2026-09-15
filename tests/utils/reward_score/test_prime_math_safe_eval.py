"""Regression tests for CVE-2026-6878: prime_math grader safe evaluation.

The grader evaluates model-authored strings. Before the fix, both
evaluation sites used builtins eval, so an answer could execute
arbitrary code inside the reward worker. The fixes:

- handle_pi evaluates only whitelisted numeric arithmetic ASTs
- the pmatrix path parses prediction with ast.literal_eval

Behavior on legitimate inputs is unchanged.
"""

import ast
import math

import pytest

from verl.utils.reward_score.prime_math.grader import (
    _safe_arith_eval,
    handle_pi,
)


class TestSafeArithEval:
    def test_numeric_arithmetic_evaluates(self):
        assert _safe_arith_eval("2*3.141592653589793 + 1") == pytest.approx(2 * math.pi + 1)

    def test_unary_and_precedence(self):
        assert _safe_arith_eval("-2*3") == -6
        assert _safe_arith_eval("1+2*3") == 7

    def test_names_rejected(self):
        with pytest.raises(ValueError):
            _safe_arith_eval("pi")

    def test_calls_rejected(self):
        with pytest.raises(ValueError):
            _safe_arith_eval("__import__('os').system('true') or 5")

    def test_attribute_access_rejected(self):
        with pytest.raises(ValueError):
            _safe_arith_eval("(1).__class__")

    def test_subscript_rejected(self):
        with pytest.raises(ValueError):
            _safe_arith_eval("[1, 2][0]")

    def test_comprehension_rejected(self):
        with pytest.raises(ValueError):
            _safe_arith_eval("[x for x in (1, 2)]")


class TestHandlePiAttackSurface:
    def test_import_attack_string_is_not_evaluated(self, tmp_path):
        marker = tmp_path / "pwned"
        attack = f"__import__('os').system('touch {marker}') or 5"
        result = handle_pi(attack, math.pi)
        assert not marker.exists()
        assert isinstance(result, str)

    def test_legitimate_pi_expression_evaluates(self):
        assert handle_pi("2\\pi", math.pi) == pytest.approx(2 * math.pi)
        assert handle_pi("\\pi/2", math.pi) == pytest.approx(math.pi / 2)

    def test_string_without_pi_untouched(self):
        assert handle_pi("1+2*3", math.pi) == "1+2*3"
