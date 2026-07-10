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
from pathlib import Path

import pytest
import torch

from verl.trainer.ppo.checkpoint_utils import (
    load_adaptive_kl_controller_state,
    save_adaptive_kl_controller_state,
)
from verl.trainer.ppo.core_algos import AdaptiveKLController


def _advance(controller, observations):
    for current_kl, n_steps in observations:
        controller.update(current_kl=current_kl, n_steps=n_steps)


def test_resume_matches_uninterrupted_controller(tmp_path):
    observations = [(1.7, 128), (0.2, 64), (1.4, 256), (0.5, 32)]
    uninterrupted = AdaptiveKLController(init_kl_coef=0.1, target_kl=1.0, horizon=10_000)
    _advance(uninterrupted, observations)

    checkpointed = AdaptiveKLController(init_kl_coef=0.1, target_kl=1.0, horizon=10_000)
    _advance(checkpointed, observations[:2])
    save_adaptive_kl_controller_state(checkpointed, tmp_path)

    resumed = AdaptiveKLController(init_kl_coef=0.1, target_kl=1.0, horizon=10_000)
    assert load_adaptive_kl_controller_state(resumed, tmp_path)
    _advance(resumed, observations[2:])

    assert resumed.value == pytest.approx(uninterrupted.value)


def test_missing_state_preserves_initial_value(tmp_path, caplog):
    controller = AdaptiveKLController(init_kl_coef=0.1, target_kl=1.0, horizon=10_000)
    assert not load_adaptive_kl_controller_state(controller, tmp_path)
    assert controller.value == 0.1
    assert "No adaptive KL controller state" in caplog.text


def test_malformed_state_fails_closed(tmp_path):
    torch.save({"version": torch.tensor(1), "value": torch.ones(2)}, tmp_path / "kl_ctrl.pt")
    controller = AdaptiveKLController(init_kl_coef=0.1, target_kl=1.0, horizon=10_000)
    with pytest.raises(ValueError, match="scalar"):
        load_adaptive_kl_controller_state(controller, tmp_path)


def test_nonfinite_state_fails_closed(tmp_path):
    torch.save({"version": torch.tensor(1), "value": torch.tensor(float("nan"))}, tmp_path / "kl_ctrl.pt")
    controller = AdaptiveKLController(init_kl_coef=0.1, target_kl=1.0, horizon=10_000)
    with pytest.raises(ValueError, match="finite"):
        load_adaptive_kl_controller_state(controller, tmp_path)


@pytest.mark.parametrize(
    "relative_path",
    [
        "verl/trainer/ppo/ray_trainer.py",
        "verl/trainer/ppo/v1/trainer_base.py",
        "verl/experimental/fully_async_policy/fully_async_trainer.py",
    ],
)
def test_all_trainer_checkpoint_paths_persist_adaptive_kl_state(relative_path):
    """Keep the legacy, V1, and fully-async save/load paths wired to the helper."""
    source_path = Path(__file__).parents[3] / relative_path
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    called_helpers = {
        node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "save_adaptive_kl_controller_state" in called_helpers
    assert "load_adaptive_kl_controller_state" in called_helpers
