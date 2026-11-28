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

import importlib.util
from pathlib import Path

_MATH_REWARD_PATH = Path(__file__).resolve().parents[3] / "verl" / "utils" / "reward_score" / "math_reward.py"
_SPEC = importlib.util.spec_from_file_location("verl_math_reward_under_test", _MATH_REWARD_PATH)
assert _SPEC and _SPEC.loader
math_reward = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(math_reward)


def test_standard_boxed_and_spaced_boxed():
    assert math_reward.compute_score("Thus the answer is \\boxed{42}.", "42") == 1.0
    assert math_reward.compute_score("Therefore $\\boxed 42$.", "42") == 1.0
    assert math_reward.compute_score("Answer: \\boxed{\\frac{1}{2}}", "0.5") == 1.0


def test_mixed_spaced_and_braced_boxed_prefers_last():
    solution = "Earlier attempt \\boxed 3 was wrong. Final answer is \\boxed{42}."
    assert math_reward.last_boxed_only_string(solution) == "\\boxed{42}"
    assert math_reward.compute_score(solution, "42") == 1.0

    solution_reverse = "First \\boxed{3}, wait actually $\\boxed 42$."
    assert math_reward.last_boxed_only_string(solution_reverse) == "\\boxed 42"
    assert math_reward.compute_score(solution_reverse, "42") == 1.0


def test_fbox_extracted_and_scored():
    assert math_reward.last_boxed_only_string("Result: \\fbox{42}") == "\\fbox{42}"
    assert math_reward.compute_score("Result: \\fbox{42}", "42") == 1.0


def test_multiple_text_units_and_trailing_frac_sqrt_do_not_crash():
    assert math_reward.is_equiv(
        "5 \\text{ apples } + 3 \\text{ oranges }",
        "5\\text{apples}+3\\text{oranges}",
    )
    assert math_reward.compute_score("\\boxed{\\frac}", "\\frac") == 1.0
    assert math_reward.compute_score("\\boxed{\\sqrt}", "\\sqrt") == 1.0
    assert math_reward.fix_fracs("\\frac") == "\\frac"
    assert math_reward.fix_sqrt("\\sqrt") == "\\sqrt"
