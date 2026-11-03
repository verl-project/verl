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
"""The trainer waits for a weight update that finishes in the background before it validates or exits.

With a backend that publishes weights (``support_publish_weights=True``), ``CheckpointEngineManager.update_weights``
returns once the trainer has published them and commits them to the rollout on a background loop.
``CheckpointEngineManager.wait_for_weight_update`` is where the trainer waits for that commit: before validation, which
must generate with the new weights, and before the training loop exits.
"""

import concurrent.futures
import threading
import time
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import verl.trainer.ppo.v1.trainer_base as trainer_base
from verl.checkpoint_engine.base import CheckpointEngineManager, WeightUpdateTicket
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync


class _RecordingManager:
    def __init__(self, name, events):
        self.name, self.events = name, events

    def wait_for_weight_update(self):
        self.events.append(f"{self.name}_wait")


def _manager_with_ticket(ticket):
    manager = CheckpointEngineManager.__new__(CheckpointEngineManager)
    manager._update_ticket = ticket
    return manager


def test_wait_for_weight_update_without_an_update_returns_at_once():
    assert _manager_with_ticket(None).wait_for_weight_update() is None


def test_wait_for_weight_update_waits_for_the_rollout_commit():
    # A publish-capable backend: update_weights returned after the trainer published, the commit is still running.
    committed = concurrent.futures.Future()
    manager = _manager_with_ticket(WeightUpdateTicket(version=3, trainer_published=[], rollout_committed=committed))
    threading.Timer(0.2, committed.set_result, args=({"ok": True},)).start()

    start = time.perf_counter()
    manager.wait_for_weight_update()

    assert committed.done()
    assert time.perf_counter() - start >= 0.15


def test_wait_for_weight_update_raises_the_commit_error():
    committed = concurrent.futures.Future()
    committed.set_exception(RuntimeError("rollout install failed"))
    manager = _manager_with_ticket(WeightUpdateTicket(version=3, trainer_published=[], rollout_committed=committed))

    with pytest.raises(RuntimeError, match="rollout install failed"):
        manager.wait_for_weight_update()


def test_separate_async_waits_for_the_standalone_rollout_too():
    events: list[str] = []
    trainer = PPOTrainerSeparateAsync.__new__(PPOTrainerSeparateAsync)
    trainer.checkpoint_manager = _RecordingManager("hybrid", events)
    trainer.standalone_checkpoint_manager = _RecordingManager("standalone", events)

    trainer._wait_for_weight_updates()

    assert events == ["hybrid_wait", "standalone_wait"]


class _StubTrainer(trainer_base.PPOTrainer):
    """Just enough of a trainer for ``fit`` to validate before training and end a loop of zero steps."""

    def __init__(self, events, val_only):
        self.events = events
        self.config = OmegaConf.create(
            {
                "trainer": {
                    "project_name": "p",
                    "experiment_name": "e",
                    "logger": [],
                    "val_before_train": True,
                    "val_only": val_only,
                    "total_epochs": 1,
                },
                "global_profiler": {"steps": None},
            }
        )
        self.global_steps = 0
        self.steps_per_epoch = 1
        self.total_training_steps = 0
        self.checkpoint_manager = _RecordingManager("checkpoint", events)

    def on_validate_begin(self):
        self.events.append("validate_begin")

    def _validate(self):
        self.events.append("validate")
        return {"val/score": 1.0}

    def on_train_begin(self):
        self.events.append("train_begin")

    def on_train_end(self):
        self.events.append("train_end")

    def on_step_end(self):
        raise AssertionError("no training step runs")

    def on_sample_end(self):
        raise AssertionError("no training step runs")

    def _reissue_inflight_prompts(self):
        pass

    def _shutdown_dump_executor(self):
        pass

    def _shutdown_train_dataloader(self):
        pass


def _patch_fit_dependencies(monkeypatch):
    for name in ("Tracking", "ValidationGenerationsLogger", "DapoFilteredRewardTableLogger"):
        monkeypatch.setattr(trainer_base, name, lambda **kwargs: SimpleNamespace(log=lambda *a, **k: None))
    monkeypatch.setattr(trainer_base, "SkipManager", SimpleNamespace(init=lambda config: None, set_step=lambda s: None))


def test_fit_waits_before_validation_and_before_the_loop_exits(monkeypatch):
    _patch_fit_dependencies(monkeypatch)
    events: list[str] = []
    _StubTrainer(events, val_only=False).fit(agent_loop_manager=None)
    assert events == ["validate_begin", "checkpoint_wait", "validate", "train_begin", "checkpoint_wait", "train_end"]


def test_fit_waits_before_validation_only(monkeypatch):
    _patch_fit_dependencies(monkeypatch)
    events: list[str] = []
    _StubTrainer(events, val_only=True).fit(agent_loop_manager=None)
    assert events == ["validate_begin", "checkpoint_wait", "validate"]
