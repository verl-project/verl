# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""CPU tests for V1 ESI / save_freq checkpoint gating.

``PPOTrainer._maybe_save_checkpoint`` is the default-trainer counterpart of the
V0 / experimental ``should_save_ckpt_esi`` save condition, and
``PPOTrainer._record_step_duration`` is the ``max_steps_duration`` bookkeeping that feeds
it. These tests bind both methods to a stub so they do not construct a full trainer.

``should_save_ckpt_esi`` only force-saves once ``max_steps_duration > 0``, so the
bookkeeping is load-bearing: #7757 was a silent no-op precisely because V1 never
recorded it. ``test_esi_force_save_needs_a_recorded_step_duration`` covers that
coupling, and ``test_fit_records_the_step_duration`` guards the ``fit()`` call site.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.v1.trainer_base import PPOTrainer

MLP_EXPIRATION_ENV = "MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP"
AWS_EXPIRATION_ENV = "SAGEMAKER_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP"


@pytest.fixture(autouse=True)
def _isolate_esi_env(monkeypatch):
    """Hide any ambient expiration timestamp, and undo whatever a test sets."""
    monkeypatch.delenv(MLP_EXPIRATION_ENV, raising=False)
    monkeypatch.delenv(AWS_EXPIRATION_ENV, raising=False)


def _stub(
    *,
    save_freq: int,
    global_steps: int,
    max_steps_duration: float,
    esi_redundant_time: float = 0,
    with_esi_redundant_time: bool = True,
):
    trainer_config = {"save_freq": save_freq}
    if with_esi_redundant_time:
        trainer_config["esi_redundant_time"] = esi_redundant_time
    stub = SimpleNamespace(
        config=OmegaConf.create({"trainer": trainer_config}),
        global_steps=global_steps,
        max_steps_duration=max_steps_duration,
        timing_raw={},
    )
    stub._save_checkpoint = MagicMock()
    stub._maybe_save_checkpoint = PPOTrainer._maybe_save_checkpoint.__get__(stub)
    stub._record_step_duration = PPOTrainer._record_step_duration.__get__(stub)
    return stub


def test_save_freq_hit_writes_checkpoint():
    stub = _stub(save_freq=10, global_steps=20, max_steps_duration=30)
    assert stub._maybe_save_checkpoint(is_last_step=False) is True
    stub._save_checkpoint.assert_called_once()


def test_save_freq_miss_does_not_write():
    stub = _stub(save_freq=10, global_steps=3, max_steps_duration=30)
    assert stub._maybe_save_checkpoint(is_last_step=False) is False
    stub._save_checkpoint.assert_not_called()


def test_last_step_writes_even_when_not_on_freq():
    stub = _stub(save_freq=10, global_steps=3, max_steps_duration=30)
    assert stub._maybe_save_checkpoint(is_last_step=True) is True
    stub._save_checkpoint.assert_called_once()


def test_save_freq_disabled_skips_esi_force_save(monkeypatch):
    monkeypatch.setenv(MLP_EXPIRATION_ENV, str(time.time() + 90))
    stub = _stub(save_freq=-1, global_steps=3, max_steps_duration=30, esi_redundant_time=30)
    assert stub._maybe_save_checkpoint(is_last_step=False) is False
    stub._save_checkpoint.assert_not_called()


def test_save_freq_disabled_skips_last_step_save():
    """``save_freq <= 0`` disables checkpointing outright, last step included (V0 parity)."""
    stub = _stub(save_freq=-1, global_steps=3, max_steps_duration=30)
    assert stub._maybe_save_checkpoint(is_last_step=True) is False
    stub._save_checkpoint.assert_not_called()


def test_esi_expiry_force_saves_off_freq(monkeypatch):
    monkeypatch.setenv(MLP_EXPIRATION_ENV, str(time.time() + 90))
    stub = _stub(save_freq=100, global_steps=3, max_steps_duration=30, esi_redundant_time=30)
    assert stub._maybe_save_checkpoint(is_last_step=False) is True
    stub._save_checkpoint.assert_called_once()


def test_esi_far_future_does_not_force_save(monkeypatch):
    monkeypatch.setenv(MLP_EXPIRATION_ENV, str(time.time() + 10_000))
    stub = _stub(save_freq=100, global_steps=3, max_steps_duration=30, esi_redundant_time=30)
    assert stub._maybe_save_checkpoint(is_last_step=False) is False
    stub._save_checkpoint.assert_not_called()


def test_zero_max_steps_duration_does_not_force_save(monkeypatch):
    """The first training step is unprotected: nothing has been timed yet (V0 parity)."""
    monkeypatch.setenv(MLP_EXPIRATION_ENV, str(time.time() + 90))
    stub = _stub(save_freq=100, global_steps=3, max_steps_duration=0, esi_redundant_time=30)
    assert stub._maybe_save_checkpoint(is_last_step=False) is False
    stub._save_checkpoint.assert_not_called()


def test_missing_esi_redundant_time_defaults_to_zero(monkeypatch):
    """``trainer.esi_redundant_time`` is read with a default, so an older config still works."""
    monkeypatch.setenv(MLP_EXPIRATION_ENV, str(time.time() + 80))
    stub = _stub(save_freq=100, global_steps=3, max_steps_duration=30, with_esi_redundant_time=False)
    assert "esi_redundant_time" not in stub.config.trainer
    assert stub._maybe_save_checkpoint(is_last_step=False) is True
    stub._save_checkpoint.assert_called_once()


def test_record_step_duration_tracks_the_slowest_step():
    stub = _stub(save_freq=100, global_steps=3, max_steps_duration=0)
    stub.timing_raw = {"step": 12}
    stub._record_step_duration()
    assert stub.max_steps_duration == 12
    stub.timing_raw = {"step": 40}
    stub._record_step_duration()
    assert stub.max_steps_duration == 40
    stub.timing_raw = {"step": 5}
    stub._record_step_duration()
    assert stub.max_steps_duration == 40


def test_record_step_duration_tolerates_missing_step_timing():
    stub = _stub(save_freq=100, global_steps=3, max_steps_duration=0)
    stub.timing_raw = {}
    stub._record_step_duration()
    assert stub.max_steps_duration == 0


def test_esi_force_save_needs_a_recorded_step_duration(monkeypatch):
    """Replay two steps: ESI force-save only arms once a step duration has been recorded."""
    monkeypatch.setenv(MLP_EXPIRATION_ENV, str(time.time() + 90))
    stub = _stub(save_freq=100, global_steps=3, max_steps_duration=0, esi_redundant_time=30)

    # Step 1: no duration recorded yet, so the expiry window is not yet estimable.
    assert stub._maybe_save_checkpoint(is_last_step=False) is False
    stub._save_checkpoint.assert_not_called()

    # End-of-step bookkeeping, exactly as ``fit()`` performs it.
    stub.timing_raw = {"step": 30}
    stub._record_step_duration()
    assert stub.max_steps_duration == 30

    # Step 2: the recorded duration now puts the expiry inside the save window.
    assert stub._maybe_save_checkpoint(is_last_step=False) is True
    stub._save_checkpoint.assert_called_once()
