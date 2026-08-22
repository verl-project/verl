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

import os
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.checkpoint_callback import CheckpointCallback, build_checkpoint_callback
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class _RecordingCallback(CheckpointCallback):
    def __init__(self, config=None, events=None):
        super().__init__(config=config)
        self.events = events if events is not None else []

    def on_save(self, trainer, global_step, checkpoint_dir, async_save=False, **kwargs):
        self.events.append(("on_save", global_step, checkpoint_dir, async_save))


def _make_config(tmp_path, async_save=False):
    return OmegaConf.create(
        {
            "trainer": {
                "default_local_dir": str(tmp_path),
                "default_hdfs_dir": None,
                "resume_mode": "auto",
                "resume_from_path": None,
                "del_local_ckpt_after_load": False,
                "checkpoint_callback_class": None,
            },
            "actor_rollout_ref": {"actor": {"checkpoint": {"async_save": async_save}}},
        }
    )


def _make_trainer(tmp_path, events, **config_kwargs):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = _make_config(tmp_path, **config_kwargs)
    trainer.global_steps = 3
    trainer.use_critic = False
    trainer.checkpoint_callback = _RecordingCallback(config=trainer.config, events=events)
    trainer.actor_rollout_wg = MagicMock()
    trainer.actor_rollout_wg.save_checkpoint.side_effect = lambda *a, **k: events.append(("wg_save",))
    dataloader = MagicMock()
    dataloader.state_dict.return_value = {}
    dataloader.__len__ = lambda self: 10
    trainer.train_dataloader = dataloader
    return trainer


def test_build_checkpoint_callback_returns_noop_when_unset(tmp_path):
    config = _make_config(tmp_path)
    callback = build_checkpoint_callback(config)
    assert type(callback) is CheckpointCallback
    assert callback.config is config
    # the hook is a no-op
    callback.on_save(trainer=None, global_step=0, checkpoint_dir="", async_save=True)


def test_build_checkpoint_callback_loads_fqn(tmp_path):
    config = _make_config(tmp_path)
    config.trainer.checkpoint_callback_class = "my_pkg.callbacks.MyCallback"
    with patch(
        "verl.trainer.ppo.checkpoint_callback.load_class_from_fqn", return_value=_RecordingCallback
    ) as load_class:
        callback = build_checkpoint_callback(config)
    load_class.assert_called_once_with("my_pkg.callbacks.MyCallback", "CheckpointCallback")
    assert isinstance(callback, _RecordingCallback)
    assert callback.config is config


def test_save_fires_on_save_after_worker_save(tmp_path):
    events = []
    trainer = _make_trainer(tmp_path, events)

    trainer._save_checkpoint()

    expected_dir = os.path.join(str(tmp_path), "global_step_3")
    assert events == [
        ("wg_save",),
        ("on_save", 3, expected_dir, False),
    ]
    tracker = os.path.join(str(tmp_path), "latest_checkpointed_iteration.txt")
    with open(tracker) as f:
        assert f.read() == "3"


def test_async_save_fires_on_save_with_flag(tmp_path):
    events = []
    trainer = _make_trainer(tmp_path, events, async_save=True)

    trainer._save_checkpoint()

    expected_dir = os.path.join(str(tmp_path), "global_step_3")
    assert events[-1] == ("on_save", 3, expected_dir, True)
    assert not os.path.exists(os.path.join(str(tmp_path), "latest_checkpointed_iteration.txt"))


def test_on_save_not_fired_on_worker_failure(tmp_path):
    events = []
    trainer = _make_trainer(tmp_path, events)
    trainer.actor_rollout_wg.save_checkpoint.side_effect = RuntimeError("save failed")

    with pytest.raises(RuntimeError, match="save failed"):
        trainer._save_checkpoint()

    assert events == []


def test_callback_exception_propagates(tmp_path):
    events = []
    trainer = _make_trainer(tmp_path, events)
    trainer.checkpoint_callback.on_save = MagicMock(side_effect=RuntimeError("callback failed"))

    with pytest.raises(RuntimeError, match="callback failed"):
        trainer._save_checkpoint()
