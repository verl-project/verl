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

"""Exercise the real PPO save entry points with controlled worker completion."""

from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.async_checkpoint import finalize_async_checkpoint
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


class StubPPOTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


@pytest.fixture(params=[RayPPOTrainer, StubPPOTrainer])
def trainer(request, tmp_path):
    obj = request.param.__new__(request.param)
    obj.config = OmegaConf.create(
        {
            "trainer": {"default_local_dir": str(tmp_path), "default_hdfs_dir": None},
            "actor_rollout_ref": {"actor": {"checkpoint": {"async_save": True}}},
            "critic": {"checkpoint": {"async_save": True}},
        }
    )
    obj.global_steps = 3
    obj.use_critic = True
    obj.trainer_mode = "sync"
    obj.checkpoint_callback = MagicMock()
    obj.train_dataloader = MagicMock()
    obj.train_dataloader.state_dict.return_value = {"step": 3}
    obj.actor_rollout_wg = MagicMock()
    obj.critic_wg = MagicMock()
    for worker in (obj.actor_rollout_wg, obj.critic_wg):
        worker.finalize_async_checkpointing.return_value = [True, True]
    return obj


@pytest.mark.parametrize("actor_async,critic_async", [(True, True), (True, False), (False, True)])
def test_save_defers_shared_tracker_until_both_roles_and_data_finish(trainer, tmp_path, actor_async, critic_async):
    trainer.config.actor_rollout_ref.actor.checkpoint.async_save = actor_async
    trainer.config.critic.checkpoint.async_save = critic_async
    trainer._save_checkpoint()
    assert (tmp_path / "global_step_3/data.pt").exists()
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    assert not tracker.exists()
    for worker in (trainer.actor_rollout_wg, trainer.critic_wg):
        assert worker.save_checkpoint.call_args.kwargs["update_tracker"] is False
        assert worker.save_checkpoint.call_args.kwargs["defer_retention"] is True
    finalize_async_checkpoint(trainer, blocking=True)
    assert tracker.read_text() == "3"
    assert trainer.actor_rollout_wg.finalize_async_checkpointing.called == actor_async
    assert trainer.critic_wg.finalize_async_checkpointing.called == critic_async


def test_grpo_without_critic(trainer, tmp_path):
    trainer.use_critic = False
    trainer._save_checkpoint()
    trainer.critic_wg.save_checkpoint.assert_not_called()
    finalize_async_checkpoint(trainer, blocking=True)
    trainer.critic_wg.finalize_async_checkpointing.assert_not_called()
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "3"


def test_sync_save_preserves_existing_behavior(trainer, tmp_path):
    trainer.config.actor_rollout_ref.actor.checkpoint.async_save = False
    trainer.config.critic.checkpoint.async_save = False
    trainer._save_checkpoint()
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "3"
    assert "update_tracker" not in trainer.actor_rollout_wg.save_checkpoint.call_args.kwargs
    finalize_async_checkpoint(trainer, blocking=True)
    trainer.actor_rollout_wg.finalize_async_checkpointing.assert_not_called()
    trainer.critic_wg.finalize_async_checkpointing.assert_not_called()


def test_next_save_drains_old_step_before_scheduling_new_one(trainer, tmp_path):
    trainer._save_checkpoint()
    trainer.global_steps = 4

    def check_previous_step(*args, **kwargs):
        assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "3"

    trainer.actor_rollout_wg.save_checkpoint.side_effect = check_previous_step
    trainer._save_checkpoint()
    assert trainer._async_checkpoint_coordinator.pending_step == 4
    finalize_async_checkpoint(trainer, blocking=True)
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "4"


def test_worker_failure_never_publishes_incomplete_step(trainer, tmp_path):
    trainer.critic_wg.save_checkpoint.side_effect = RuntimeError("save failed")
    with pytest.raises(RuntimeError, match="save failed"):
        trainer._save_checkpoint()
    finalize_async_checkpoint(trainer, blocking=True)
    assert not (tmp_path / "latest_checkpointed_iteration.txt").exists()


@pytest.mark.parametrize("total_steps", [1, 2])
def test_v1_fit_drains_on_last_step_and_epoch_exhaustion(tmp_path, monkeypatch, total_steps):
    import verl.trainer.ppo.v1.trainer_base as trainer_module

    obj = StubPPOTrainer.__new__(StubPPOTrainer)
    obj.config = OmegaConf.create(
        {
            "trainer": {
                "default_local_dir": str(tmp_path),
                "default_hdfs_dir": None,
                "project_name": "test",
                "experiment_name": "test",
                "logger": [],
                "val_before_train": False,
                "total_epochs": 1,
                "save_freq": 1,
                "test_freq": -1,
            },
            "actor_rollout_ref": {"actor": {"checkpoint": {"async_save": True}}},
            "global_profiler": {"steps": None},
        }
    )
    obj.global_steps = 0
    obj.steps_per_epoch = 1
    obj.total_training_steps = total_steps
    obj.use_critic = False
    obj.trainer_mode = "sync"
    obj.checkpoint_callback = MagicMock()
    obj.train_dataloader = MagicMock()
    obj.train_dataloader.state_dict.return_value = {"step": 1}
    obj.actor_rollout_wg = MagicMock()
    obj.actor_rollout_wg.finalize_async_checkpointing.return_value = [True, True]
    obj.step = MagicMock(return_value=MagicMock())
    for name in (
        "_reissue_inflight_prompts",
        "on_train_begin",
        "on_train_end",
        "on_step_begin",
        "_start_profiling",
        "_stop_profiling",
        "_compute_metrics",
        "_shutdown_dump_executor",
    ):
        setattr(obj, name, MagicMock())
    obj._consume_sync_metrics = MagicMock(return_value={})
    for name in ("Tracking", "ValidationGenerationsLogger", "DapoFilteredRewardTableLogger", "SkipManager", "tq"):
        monkeypatch.setattr(trainer_module, name, MagicMock())
    obj.fit(MagicMock())
    obj.actor_rollout_wg.finalize_async_checkpointing.assert_called_with(blocking=True)
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "1"
    assert obj._async_checkpoint_coordinator.pending_step is None


def test_legacy_fit_drains_on_epoch_exhaustion(tmp_path, monkeypatch):
    import verl.utils.tracking as tracking

    obj = RayPPOTrainer.__new__(RayPPOTrainer)
    obj.config = OmegaConf.create(
        {
            "trainer": {
                "default_local_dir": str(tmp_path),
                "default_hdfs_dir": None,
                "project_name": "test",
                "experiment_name": "test",
                "logger": [],
                "val_before_train": False,
                "total_epochs": 0,
            },
            "actor_rollout_ref": {"actor": {"checkpoint": {"async_save": True}}},
            "global_profiler": {"steps": None},
        }
    )
    obj.global_steps = 3
    obj.total_training_steps = 10
    obj.use_critic = False
    obj.train_dataloader = MagicMock()
    obj.train_dataloader.state_dict.return_value = {"step": 3}
    obj.train_dataloader.__len__.return_value = 1
    obj.actor_rollout_wg = MagicMock()
    obj.actor_rollout_wg.finalize_async_checkpointing.return_value = [True, True]
    obj._save_checkpoint()
    obj._dump_executor = MagicMock()
    obj._dump_executor._shutdown = False
    obj._load_checkpoint = MagicMock()
    obj.checkpoint_manager = MagicMock()
    obj._shutdown_dump_executor = MagicMock()
    monkeypatch.setattr(tracking, "Tracking", MagicMock())
    import verl.trainer.ppo.ray_trainer as trainer_module

    monkeypatch.setattr(trainer_module, "SkipManager", MagicMock())
    obj.fit()
    obj.actor_rollout_wg.finalize_async_checkpointing.assert_called_with(blocking=True)
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "3"
