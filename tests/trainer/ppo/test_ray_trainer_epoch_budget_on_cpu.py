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

import types

import torch

if not hasattr(torch.distributed, "tensor"):
    torch.distributed.tensor = types.SimpleNamespace()
if not hasattr(torch.distributed.tensor, "DTensor"):
    torch.distributed.tensor.DTensor = torch.Tensor

from verl.trainer.ppo.ray_trainer import (
    _completed_training_steps,
    _compute_effective_total_epochs,
    _compute_training_epoch_bounds,
    _format_early_stop_warning,
)



def test_compute_effective_total_epochs_expands_epoch_budget_for_step_mode():
    assert _compute_effective_total_epochs(1, 100, 25) == 4


def test_compute_effective_total_epochs_preserves_larger_configured_epoch_budget():
    assert _compute_effective_total_epochs(10, 100, 25) == 10


def test_compute_training_epoch_bounds_for_resume_mid_epoch():
    current_epoch, effective_total_epochs = _compute_training_epoch_bounds(
        resumed_global_steps=20,
        configured_total_epochs=1,
        total_training_steps=100,
        steps_per_epoch=25,
    )

    assert current_epoch == 0
    assert effective_total_epochs == 4


def test_compute_training_epoch_bounds_for_resume_after_epoch_boundary():
    current_epoch, effective_total_epochs = _compute_training_epoch_bounds(
        resumed_global_steps=25,
        configured_total_epochs=1,
        total_training_steps=100,
        steps_per_epoch=25,
    )

    assert current_epoch == 1
    assert effective_total_epochs == 4


def test_completed_training_steps_uses_next_step_cursor():
    assert _completed_training_steps(26) == 25
    assert _completed_training_steps(0) == 0


def test_format_early_stop_warning_mentions_root_cause():
    warning = _format_early_stop_warning(25, 100)

    assert "step 25" in warning
    assert "total_training_steps=100" in warning
    assert "trainer.total_epochs" in warning
