import os
import shutil
import types

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.v1 import trainer_base
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager


@pytest.fixture
def manager():
    manager = BaseCheckpointManager.__new__(BaseCheckpointManager)  # only the retention bookkeeping is exercised
    manager.previous_saved_paths = []
    return manager


@pytest.fixture
def make_step(tmp_path):
    def make(step, actor=True):
        folder = os.path.join(tmp_path, f"global_step_{step}")
        os.makedirs(os.path.join(folder, "actor") if actor else folder)
        open(os.path.join(folder, "data.pt"), "w").close()
        return folder

    return make


def test_resume_registers_the_older_siblings_in_order(tmp_path, manager, make_step):
    for step in (3, 1, 2, 5):
        make_step(step)
    manager.register_existing_checkpoints(os.path.join(tmp_path, "global_step_3", "actor"))
    assert manager.previous_saved_paths == [
        os.path.join(tmp_path, f"global_step_{step}", "actor") for step in (1, 2, 3)
    ]


def test_resume_ignores_paths_outside_the_step_layout(tmp_path, manager):
    manager.register_existing_checkpoints(os.path.join(tmp_path, "somewhere", "actor"))
    assert manager.previous_saved_paths == []


class SavingTrainer(PPOTrainer):
    """PPOTrainer with the step hooks, which a checkpoint save never reaches."""

    def on_sample_end(self):
        return

    def on_step_end(self):
        return


def test_pruned_step_folders_go_before_the_transfer_queue_snapshot(tmp_path, make_step, monkeypatch):
    # saving step 6 with max_actor_ckpt_to_keep=2 prunes step 4's actor: step 4's folder (its data.pt and its snapshot)
    # must be gone when step 6's snapshot is written, so that two old snapshots never sit next to the new one
    make_step(4)
    make_step(5)
    folders_at_snapshot = []

    def save_actor(local_path, remote_path, step, max_ckpt_to_keep):
        os.makedirs(local_path)
        shutil.rmtree(os.path.join(tmp_path, "global_step_4", "actor"))

    monkeypatch.setattr(
        trainer_base.tq,
        "save_checkpoint",
        lambda path, metadata: folders_at_snapshot.append(sorted(os.listdir(tmp_path))),
    )
    trainer = SavingTrainer.__new__(SavingTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {"default_local_dir": str(tmp_path), "default_hdfs_dir": None, "max_actor_ckpt_to_keep": 2},
            "actor_rollout_ref": {"actor": {"checkpoint": {}}},
            "data": {},
        }
    )
    trainer.global_steps = 6
    trainer.actor_rollout_wg = types.SimpleNamespace(save_checkpoint=save_actor)
    trainer.use_critic = False
    trainer.train_dataloader = types.SimpleNamespace(state_dict=lambda: {})
    trainer.trainer_mode = "separate_async"
    trainer.checkpoint_callback = types.SimpleNamespace(on_save=lambda **kwargs: None)

    trainer._save_checkpoint()

    assert folders_at_snapshot == [["global_step_5", "global_step_6"]]


def test_pruned_step_folders_are_removed_but_live_ones_kept(tmp_path, make_step):
    make_step(3, actor=False)
    make_step(4, actor=False)
    make_step(5)
    make_step(6)
    PPOTrainer._remove_pruned_step_folders(str(tmp_path), 6)
    assert sorted(os.listdir(tmp_path)) == ["global_step_5", "global_step_6"]
