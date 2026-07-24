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
"""CPU tests for the OPSD fields on ``DistillationConfig``."""

from verl.workers.config.distillation import DistillationConfig


def test_opsd_defaults_are_off_and_backward_compatible():
    c = DistillationConfig()
    assert c.self_distillation is False
    assert c.privileged_solution_key == "reward_model.ground_truth"
    # markers default to newline-wrapped text so the solution is set off from the prompt
    assert c.privileged_prefix.strip() and c.privileged_suffix.strip()


def test_self_distillation_requires_enabled():
    import pytest

    with pytest.raises(ValueError):
        DistillationConfig(self_distillation=True)  # enabled defaults to False


def test_privileged_mode_defaults_to_append():
    c = DistillationConfig()
    assert c.privileged_mode == "append"
    assert c.privileged_problem_key == "extra_info.problem"
    assert c.privileged_enable_thinking is True


def test_privileged_mode_rejects_unknown_value():
    import pytest

    with pytest.raises(ValueError, match="privileged_mode"):
        DistillationConfig(self_distillation=True, privileged_mode="banana")


def test_chat_turn_requires_problem_key():
    import pytest

    with pytest.raises(ValueError, match="privileged_problem_key"):
        DistillationConfig(self_distillation=True, privileged_mode="chat_turn", privileged_problem_key="")


def test_chat_turn_template_must_have_placeholders():
    import pytest

    with pytest.raises(ValueError, match="placeholders"):
        DistillationConfig(
            self_distillation=True, privileged_mode="chat_turn", privileged_user_template="no placeholders here"
        )


def test_chat_turn_valid_fields_pass_field_checks():
    import pytest

    # With valid chat_turn fields the first failure is the enabled gate, proving
    # the mode/problem-key/template checks all passed.
    with pytest.raises(ValueError, match="enabled"):
        DistillationConfig(self_distillation=True, privileged_mode="chat_turn")
