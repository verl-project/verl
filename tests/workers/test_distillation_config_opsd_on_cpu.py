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
    assert c.privileged_solution_key == "ground_truth"
    # markers default to newline-wrapped text so the solution is set off from the prompt
    assert c.privileged_prefix.strip() and c.privileged_suffix.strip()


def test_opsd_fields_settable_without_enabling_distillation():
    # enabled=False short-circuits teacher-pool validation, so OPSD fields can be
    # exercised standalone.
    c = DistillationConfig(self_distillation=True, privileged_solution_key="solution")
    assert c.self_distillation is True
    assert c.privileged_solution_key == "solution"
