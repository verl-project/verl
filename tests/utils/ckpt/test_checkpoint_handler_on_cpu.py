# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import pytest
import torch

from verl.utils.checkpoint.checkpoint_handler import CheckpointHandler, OrchestrationMode
from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager


class _Dataloader:
    def __init__(self, fail_on_save=False, events=None):
        self.fail_on_save = fail_on_save
        self.events = events

    def state_dict(self):
        if self.events is not None:
            self.events.append("dataloader")
        if self.fail_on_save:
            raise RuntimeError("dataloader save failed")
        return {"position": 1}

    def load_state_dict(self, state_dict):
        assert state_dict == {"position": 1}


class _Engine:
    def __init__(self, manager, events=None):
        self.manager = manager
        self.events = events

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        if self.events is not None:
            self.events.append("model")
        os.makedirs(local_path, exist_ok=True)
        torch.save({"step": global_step}, os.path.join(local_path, "model.pt"))
        self.manager.register_checkpoint(local_path, max_ckpt_to_keep)

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        assert os.path.isdir(local_path)

    def prepare_checkpoint_retention(self, loaded_path):
        self.manager.record_loaded_checkpoint(loaded_path)

    def finalize_checkpoint_retention(self, new_path, max_ckpt_to_keep=None):
        if self.events is not None:
            self.events.append("finalize")
        self.manager.finalize_loaded_checkpoint_retention(new_path, max_ckpt_to_keep)

    def is_mp_src_rank_with_outputs(self):
        return True

    def get_data_parallel_rank(self):
        return 0


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    return BaseCheckpointManager(model=object(), optimizer=object())


def _write_checkpoint(root, step):
    checkpoint_path = root / f"global_step_{step}"
    checkpoint_path.mkdir()
    torch.save({"step": step}, checkpoint_path / "model.pt")
    torch.save({"position": 1}, checkpoint_path / "data_0.pt")
    return checkpoint_path


def _write_tracker(root, step):
    (root / "latest_checkpointed_iteration.txt").write_text(str(step))


def _handler(root, manager, dataloader=None, **kwargs):
    return CheckpointHandler(
        engine=_Engine(manager),
        train_dataloader=dataloader or _Dataloader(),
        default_local_dir=str(root),
        max_ckpt_to_keep=kwargs.pop("max_ckpt_to_keep", 1),
        mode=kwargs.pop("mode", OrchestrationMode.RAY),
        **kwargs,
    )


def test_restart_loaded_checkpoint_is_removed_after_replacement_commits(tmp_path, manager):
    old_checkpoint = _write_checkpoint(tmp_path, 1)
    _write_tracker(tmp_path, 1)
    handler = _handler(tmp_path, manager)

    assert handler.load_checkpoint() == 1
    handler.save_checkpoint(2)

    assert not old_checkpoint.exists()
    assert (tmp_path / "global_step_2").is_dir()
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "2"


def test_restart_loaded_checkpoint_survives_dataloader_save_failure(tmp_path, manager):
    old_checkpoint = _write_checkpoint(tmp_path, 1)
    _write_tracker(tmp_path, 1)
    handler = _handler(tmp_path, manager, dataloader=_Dataloader(fail_on_save=True))
    assert handler.load_checkpoint() == 1

    with pytest.raises(RuntimeError, match="dataloader save failed"):
        handler.save_checkpoint(2)

    assert old_checkpoint.is_dir()
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "1"


def test_restart_loaded_checkpoint_is_not_rotated_when_retention_exceeds_one(tmp_path, manager):
    old_checkpoint = _write_checkpoint(tmp_path, 1)
    _write_tracker(tmp_path, 1)
    handler = _handler(tmp_path, manager, max_ckpt_to_keep=2)

    assert handler.load_checkpoint() == 1
    handler.save_checkpoint(2)

    assert old_checkpoint.is_dir()
    assert (tmp_path / "global_step_2").is_dir()


def test_external_resume_checkpoint_is_not_deleted(tmp_path, manager):
    checkpoint_root = tmp_path / "run"
    checkpoint_root.mkdir()
    external_root = tmp_path / "external"
    external_root.mkdir()
    old_checkpoint = _write_checkpoint(external_root, 1)
    handler = _handler(
        checkpoint_root,
        manager,
        resume_mode="resume_path",
        resume_from_path=str(old_checkpoint),
    )

    assert handler.load_checkpoint() == 1
    handler.save_checkpoint(2)

    assert old_checkpoint.is_dir()
    assert (checkpoint_root / "global_step_2").is_dir()


def test_restart_loaded_checkpoint_survives_tracker_update_failure(tmp_path, manager, monkeypatch):
    old_checkpoint = _write_checkpoint(tmp_path, 1)
    _write_tracker(tmp_path, 1)
    handler = _handler(tmp_path, manager)
    assert handler.load_checkpoint() == 1

    def fail_rename(src, dst):
        raise OSError("tracker update failed")

    monkeypatch.setattr(os, "rename", fail_rename)
    with pytest.raises(OSError, match="tracker update failed"):
        handler.save_checkpoint(2)

    assert old_checkpoint.is_dir()
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "1"


def test_restart_loaded_checkpoint_survives_failed_hdfs_copy(tmp_path, manager, monkeypatch):
    old_checkpoint = _write_checkpoint(tmp_path, 1)
    _write_tracker(tmp_path, 1)
    handler = _handler(tmp_path, manager, default_hdfs_dir="hdfs://checkpoints")
    assert handler.load_checkpoint() == 1
    monkeypatch.setattr("verl.utils.checkpoint.checkpoint_handler.hdfs_io.makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr("verl.utils.checkpoint.checkpoint_handler.hdfs_io.copy", lambda *args, **kwargs: False)

    handler.save_checkpoint(2)

    assert old_checkpoint.is_dir()
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "2"


def test_spmd_waits_for_dataloader_before_finalizing_retention(tmp_path, manager, monkeypatch):
    events = []
    dataloader = _Dataloader(events=events)
    engine = _Engine(manager, events=events)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: events.append("barrier"))
    real_rename = os.rename

    def record_rename(src, dst):
        events.append("tracker")
        real_rename(src, dst)

    monkeypatch.setattr(os, "rename", record_rename)
    handler = CheckpointHandler(
        engine=engine,
        train_dataloader=dataloader,
        default_local_dir=str(tmp_path),
        max_ckpt_to_keep=1,
        mode=OrchestrationMode.SPMD,
    )

    handler.save_checkpoint(1)

    assert events == ["model", "dataloader", "tracker", "barrier", "finalize"]
