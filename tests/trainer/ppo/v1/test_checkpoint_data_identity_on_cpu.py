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

"""Exercise real trainer checkpoint ordering with only worker RPCs stubbed."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.utils.dataset import resume_identity


class _Trainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


def _trainer(tmp_path, policy="error", async_save=False):
    trainer = _Trainer.__new__(_Trainer)
    trainer.trainer_mode = "sync"
    trainer.global_steps = 3
    trainer.use_critic = True
    trainer.config = OmegaConf.create(
        {
            "data": {"resume_check": policy},
            "trainer": {
                "default_local_dir": str(tmp_path),
                "default_hdfs_dir": None,
                "resume_mode": "resume_path",
                "resume_from_path": str(tmp_path / "global_step_3"),
                "del_local_ckpt_after_load": False,
            },
            "actor_rollout_ref": {"actor": {"checkpoint": {"async_save": async_save}}},
        }
    )
    trainer.actor_rollout_wg = MagicMock()
    trainer.critic_wg = MagicMock()
    trainer.checkpoint_callback = MagicMock()
    trainer.train_dataloader = MagicMock()
    trainer.train_dataloader.state_dict.return_value = {"step": 2}
    trainer.train_dataloader.dataset.get_resume_signature.return_value = {"rows": "original"}
    return trainer


@pytest.mark.parametrize("async_save", [False, True])
def test_manifest_exists_before_callback_and_tracker_publication(tmp_path, async_save):
    trainer = _trainer(tmp_path, async_save=async_save)

    def on_save(**kwargs):
        saved = json.loads((tmp_path / "global_step_3" / resume_identity.MANIFEST_NAME).read_text())
        assert saved["dataset"] == {"rows": "original"}
        assert (tmp_path / "global_step_3" / "data.pt").is_file()

    trainer.checkpoint_callback.on_save.side_effect = on_save
    trainer._save_checkpoint()
    trainer.checkpoint_callback.on_save.assert_called_once()
    assert (tmp_path / "latest_checkpointed_iteration.txt").exists() is not async_save
    trainer._load_checkpoint()
    trainer.actor_rollout_wg.load_checkpoint.assert_called_once()
    trainer.critic_wg.load_checkpoint.assert_called_once()
    trainer.train_dataloader.load_state_dict.assert_called_once_with({"step": 2})


def test_changed_data_fails_before_any_worker_or_loader_restore(tmp_path):
    trainer = _trainer(tmp_path)
    trainer._save_checkpoint()
    trainer.train_dataloader.dataset.get_resume_signature.return_value = {"rows": "changed"}
    with pytest.raises(ValueError, match="Data identity changed in: rows"):
        trainer._load_checkpoint()
    trainer.actor_rollout_wg.load_checkpoint.assert_not_called()
    trainer.critic_wg.load_checkpoint.assert_not_called()
    trainer.train_dataloader.load_state_dict.assert_not_called()
    assert trainer.global_steps == 0


def test_unavailable_signature_fails_before_model_save(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.train_dataloader.dataset.get_resume_signature.side_effect = NotImplementedError("custom data")
    with pytest.raises(ValueError, match="custom data"):
        trainer._save_checkpoint()
    trainer.actor_rollout_wg.save_checkpoint.assert_not_called()
    trainer.critic_wg.save_checkpoint.assert_not_called()


def test_failed_manifest_is_not_published_as_latest_checkpoint(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path)
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    tracker.write_text("2")

    def fail_replace(*args):
        raise OSError("disk failure")

    monkeypatch.setattr(resume_identity.os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk failure"):
        trainer._save_checkpoint()
    assert tracker.read_text() == "2"
    trainer.checkpoint_callback.on_save.assert_not_called()


@pytest.mark.parametrize("async_save", [False, True])
def test_async_trainer_checks_identity_before_transfer_queue_restore(tmp_path, monkeypatch, async_save):
    from verl.trainer.ppo.v1 import trainer_base

    trainer = _trainer(tmp_path, async_save=async_save)
    trainer.trainer_mode = "async"
    transfer_queue = MagicMock()
    monkeypatch.setattr(trainer_base, "tq", transfer_queue)
    trainer._save_checkpoint()
    transfer_queue.save_checkpoint.assert_called_once()
    checkpoint = tmp_path / "global_step_3"
    (checkpoint / "transfer_queue").mkdir()
    trainer.train_dataloader.dataset.get_resume_signature.return_value = {"rows": "changed"}
    with pytest.raises(ValueError, match="Data identity changed"):
        trainer._load_checkpoint()
    transfer_queue.load_checkpoint.assert_not_called()
    trainer.actor_rollout_wg.load_checkpoint.assert_not_called()


@pytest.mark.parametrize("async_save", [False, True])
def test_manifest_is_published_after_transfer_queue_save(tmp_path, monkeypatch, async_save):
    from verl.trainer.ppo.v1 import trainer_base

    trainer = _trainer(tmp_path, async_save=async_save)
    trainer.trainer_mode = "async"
    checkpoint = tmp_path / "global_step_3"
    transfer_queue = MagicMock()
    monkeypatch.setattr(trainer_base, "tq", transfer_queue)

    def save_queue(*args, **kwargs):
        assert (checkpoint / "data.pt").is_file()
        assert not (checkpoint / resume_identity.MANIFEST_NAME).exists()

    transfer_queue.save_checkpoint.side_effect = save_queue
    trainer._save_checkpoint()
    transfer_queue.save_checkpoint.assert_called_once()
    assert json.loads((checkpoint / resume_identity.MANIFEST_NAME).read_text())["dataset"] == {"rows": "original"}
    trainer.checkpoint_callback.on_save.assert_called_once()


@pytest.mark.parametrize("async_save", [False, True])
def test_failed_queue_overwrite_cannot_certify_old_prompts(tmp_path, monkeypatch, async_save):
    from verl.trainer.ppo.v1 import trainer_base

    trainer = _trainer(tmp_path, async_save=async_save)
    trainer.trainer_mode = "async"
    transfer_queue = MagicMock()
    monkeypatch.setattr(trainer_base, "tq", transfer_queue)
    checkpoint = tmp_path / "global_step_3"
    queue_path = checkpoint / "transfer_queue"

    def save_original_queue(*args, **kwargs):
        queue_path.mkdir(parents=True)
        (queue_path / "prompts.json").write_text('["original prompt"]')

    transfer_queue.save_checkpoint.side_effect = save_original_queue
    trainer._save_checkpoint()
    trainer.checkpoint_callback.on_save.reset_mock()
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    tracker.write_text("2")
    trainer.train_dataloader.dataset.get_resume_signature.return_value = {"rows": "replacement"}
    trainer.train_dataloader.state_dict.return_value = {"step": 9}
    transfer_queue.save_checkpoint.side_effect = OSError("queue write failed")

    with pytest.raises(OSError, match="queue write failed"):
        trainer._save_checkpoint()
    assert (queue_path / "prompts.json").read_text() == '["original prompt"]'
    assert not (checkpoint / resume_identity.MANIFEST_NAME).exists()
    assert tracker.read_text() == "2"
    trainer.checkpoint_callback.on_save.assert_not_called()

    # The replacement dataset must not validate a checkpoint with the old queue.
    with pytest.raises(ValueError, match="Cannot verify checkpoint data identity"):
        trainer._load_checkpoint()
    transfer_queue.load_checkpoint.assert_not_called()
    trainer.actor_rollout_wg.load_checkpoint.assert_not_called()
    trainer.critic_wg.load_checkpoint.assert_not_called()
    trainer.train_dataloader.load_state_dict.assert_not_called()


@pytest.mark.parametrize("policy", ["off", "warn", "error"])
def test_old_checkpoint_policy_applies_before_any_restore(tmp_path, caplog, policy):
    trainer = _trainer(tmp_path, policy=policy)
    checkpoint = tmp_path / "global_step_3"
    checkpoint.mkdir()
    if policy == "error":
        with pytest.raises(ValueError, match="Cannot verify checkpoint data identity"):
            trainer._load_checkpoint()
        trainer.actor_rollout_wg.load_checkpoint.assert_not_called()
        trainer.critic_wg.load_checkpoint.assert_not_called()
        trainer.train_dataloader.load_state_dict.assert_not_called()
    else:
        trainer._load_checkpoint()
        trainer.actor_rollout_wg.load_checkpoint.assert_called_once()
        assert ("Cannot verify checkpoint data identity" in caplog.text) is (policy == "warn")


@pytest.mark.parametrize("policy", ["off", "warn", "error"])
def test_custom_dataset_without_signature_follows_save_policy(tmp_path, caplog, policy):
    trainer = _trainer(tmp_path, policy=policy)
    trainer.train_dataloader.dataset = SimpleNamespace()
    if policy == "error":
        with pytest.raises(ValueError, match="Data resume identity is unavailable"):
            trainer._save_checkpoint()
        trainer.actor_rollout_wg.save_checkpoint.assert_not_called()
        trainer.critic_wg.save_checkpoint.assert_not_called()
    else:
        trainer._save_checkpoint()
        trainer.actor_rollout_wg.save_checkpoint.assert_called_once()
        sidecar = tmp_path / "global_step_3" / resume_identity.MANIFEST_NAME
        if policy == "warn":
            assert json.loads(sidecar.read_text())["dataset"] is None
            assert "Data resume identity is unavailable" in caplog.text
        else:
            assert not sidecar.exists()


@pytest.mark.parametrize("async_save", [False, True])
def test_worker_save_failure_does_not_publish_data_manifest(tmp_path, async_save):
    trainer = _trainer(tmp_path, async_save=async_save)
    trainer.critic_wg.save_checkpoint.side_effect = RuntimeError("critic write failed")
    with pytest.raises(RuntimeError, match="critic write failed"):
        trainer._save_checkpoint()
    assert not (tmp_path / "global_step_3" / resume_identity.MANIFEST_NAME).exists()
    assert not (tmp_path / "latest_checkpointed_iteration.txt").exists()
    trainer.checkpoint_callback.on_save.assert_not_called()


@pytest.mark.parametrize("async_save", [False, True])
def test_off_overwrite_cannot_reuse_an_older_identity(tmp_path, async_save):
    trainer = _trainer(tmp_path, async_save=async_save)
    trainer._save_checkpoint()
    sidecar = tmp_path / "global_step_3" / resume_identity.MANIFEST_NAME
    assert sidecar.is_file()

    trainer.config.data.resume_check = "off"
    trainer.train_dataloader.dataset.get_resume_signature.return_value = {"rows": "replacement"}
    trainer.train_dataloader.state_dict.return_value = {"step": 9}
    trainer._save_checkpoint()
    assert not sidecar.exists()

    # The old data would match the stale sidecar, but not the replacement checkpoint.
    trainer.config.data.resume_check = "error"
    trainer.train_dataloader.dataset.get_resume_signature.return_value = {"rows": "original"}
    with pytest.raises(ValueError, match="Cannot verify checkpoint data identity"):
        trainer._load_checkpoint()
    trainer.actor_rollout_wg.load_checkpoint.assert_not_called()
    trainer.critic_wg.load_checkpoint.assert_not_called()
    trainer.train_dataloader.load_state_dict.assert_not_called()


@pytest.mark.parametrize("async_save", [False, True])
@pytest.mark.parametrize("failure", ["critic", "manifest"])
def test_failed_overwrite_invalidates_the_previous_identity(tmp_path, monkeypatch, async_save, failure):
    trainer = _trainer(tmp_path, async_save=async_save)
    trainer._save_checkpoint()
    trainer.train_dataloader.dataset.get_resume_signature.return_value = {"rows": "replacement"}

    def fail(*args, **kwargs):
        raise OSError("checkpoint write failed")

    if failure == "critic":
        trainer.critic_wg.save_checkpoint.side_effect = fail
    else:
        monkeypatch.setattr(resume_identity.os, "replace", fail)
    with pytest.raises(OSError, match="checkpoint write failed"):
        trainer._save_checkpoint()
    assert not (tmp_path / "global_step_3" / resume_identity.MANIFEST_NAME).exists()

    trainer.train_dataloader.dataset.get_resume_signature.return_value = {"rows": "original"}
    with pytest.raises(ValueError, match="Cannot verify checkpoint data identity"):
        trainer._load_checkpoint()
    trainer.actor_rollout_wg.load_checkpoint.assert_not_called()
    trainer.critic_wg.load_checkpoint.assert_not_called()
    trainer.train_dataloader.load_state_dict.assert_not_called()
