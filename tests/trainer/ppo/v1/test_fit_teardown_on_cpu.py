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
"""How fit() ends a run that reaches its last step.

It calls on_train_end() and stops the train dataloader's worker processes. Before, the last step returned from
inside the loop: on_train_end() never ran, and the train iterator (which StatefulDataLoader also keeps) held its
workers until Ray killed them with the actor, so every successful run ended with
"DataLoader worker (pid ...) is killed by signal: Killed".
"""

import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.trainer.ppo.v1 import trainer_base
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


def _collate(prompts):
    return {"raw_prompt": np.array(prompts, dtype=object)}


class _Trainer(PPOTrainer):
    """Runs fit()'s loop with a step that only draws a batch from the train dataloader."""

    def __init__(self, total_steps: int):
        self.config = OmegaConf.create(
            {
                "trainer": {
                    "project_name": "p",
                    "experiment_name": "e",
                    "logger": ["console"],
                    "val_before_train": False,
                    "total_epochs": 1,
                    "save_freq": -1,
                    "test_freq": -1,
                },
                "global_profiler": {"steps": None},
            }
        )
        self.global_steps = 0
        self.total_training_steps = total_steps
        self.steps_per_epoch = total_steps
        self.train_dataloader = StatefulDataLoader(
            [f"prompt {i}" for i in range(16)], batch_size=2, num_workers=2, collate_fn=_collate
        )
        self.train_dataloader_it = None
        self._dump_executor = ThreadPoolExecutor(max_workers=1)
        self._dump_futures = []
        self.steps = 0
        self.train_ended = 0

    def step(self, metrics, timing_raw):
        self._fetch_one_gen_batch()
        self.steps += 1
        return SimpleNamespace(keys=[], partition_id="train")

    def on_train_end(self):
        self.train_ended += 1

    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass

    def _reissue_inflight_prompts(self, partition_id="train"):
        return 0

    def _start_profiling(self):
        pass

    def _stop_profiling(self):
        pass

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        pass


def _fit(trainer):
    with (
        patch.object(trainer_base, "Tracking"),
        patch.object(trainer_base, "ValidationGenerationsLogger"),
        patch.object(trainer_base, "DapoFilteredRewardTableLogger"),
        patch.object(trainer_base, "SkipManager"),
        patch.object(trainer_base, "tq"),
        patch.object(trainer_base.tu, "get_tensordict", side_effect=lambda batch: batch),
    ):
        trainer.fit(MagicMock())


def test_fit_runs_on_train_end_after_the_last_step():
    trainer = _Trainer(total_steps=3)
    _fit(trainer)
    assert trainer.steps == 3
    assert trainer.train_ended == 1


def test_fit_stops_the_train_dataloader_workers():
    before = set(multiprocessing.active_children())
    trainer = _Trainer(total_steps=3)
    _fit(trainer)
    assert trainer.train_dataloader_it is None
    assert set(multiprocessing.active_children()) - before == set()
