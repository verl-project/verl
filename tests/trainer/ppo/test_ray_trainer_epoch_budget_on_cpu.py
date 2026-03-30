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

import ast
import math
from pathlib import Path


RAY_TRAINER_PATH = Path("/hy-tmp/verl-submit/verl/trainer/ppo/ray_trainer.py")


def _load_helper_functions():
    source = RAY_TRAINER_PATH.read_text()
    module = ast.parse(source, filename=str(RAY_TRAINER_PATH))
    wanted = {
        "_compute_effective_total_epochs",
        "_compute_training_epoch_bounds",
        "_completed_training_steps",
        "_format_early_stop_warning",
    }
    selected = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    helper_module = ast.Module(body=selected, type_ignores=[])
    namespace = {"math": math}
    exec(compile(helper_module, filename=str(RAY_TRAINER_PATH), mode="exec"), namespace)
    return namespace


def test_compute_effective_total_epochs_expands_epoch_budget_for_step_mode():
    helpers = _load_helper_functions()

    assert helpers["_compute_effective_total_epochs"](1, 100, 25) == 4


def test_compute_effective_total_epochs_preserves_larger_configured_epoch_budget():
    helpers = _load_helper_functions()

    assert helpers["_compute_effective_total_epochs"](10, 100, 25) == 10


def test_compute_training_epoch_bounds_for_resume_mid_epoch():
    helpers = _load_helper_functions()

    current_epoch, effective_total_epochs = helpers["_compute_training_epoch_bounds"](
        resumed_global_steps=20,
        configured_total_epochs=1,
        total_training_steps=100,
        steps_per_epoch=25,
    )

    assert current_epoch == 0
    assert effective_total_epochs == 4


def test_compute_training_epoch_bounds_for_resume_after_epoch_boundary():
    helpers = _load_helper_functions()

    current_epoch, effective_total_epochs = helpers["_compute_training_epoch_bounds"](
        resumed_global_steps=25,
        configured_total_epochs=1,
        total_training_steps=100,
        steps_per_epoch=25,
    )

    assert current_epoch == 1
    assert effective_total_epochs == 4


def test_completed_training_steps_uses_next_step_cursor():
    helpers = _load_helper_functions()

    assert helpers["_completed_training_steps"](26) == 25
    assert helpers["_completed_training_steps"](0) == 0


def test_format_early_stop_warning_mentions_root_cause():
    helpers = _load_helper_functions()

    warning = helpers["_format_early_stop_warning"](25, 100)

    assert "step 25" in warning
    assert "total_training_steps=100" in warning
    assert "trainer.total_epochs" in warning
