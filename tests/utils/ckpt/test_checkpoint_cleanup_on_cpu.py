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

import os
import shutil
import tempfile

import pytest


class TestCheckpointCleanupLogic:
    """Tests for checkpoint cleanup methods in BaseCheckpointManager."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @pytest.fixture
    def manager(self, monkeypatch):
        """Create a minimal BaseCheckpointManager for testing."""
        import torch.distributed

        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)

        from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager

        class MockModel:
            pass

        class MockOptimizer:
            pass

        return BaseCheckpointManager(
            model=MockModel(),
            optimizer=MockOptimizer(),
            lr_scheduler=None,
            processing_class=None,
            checkpoint_config=None,
        )

    def _create_checkpoint_dir(self, step: int) -> str:
        """Create a mock checkpoint directory."""
        path = os.path.join(self.test_dir, f"global_step_{step}")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "checkpoint.txt"), "w") as f:
            f.write(f"step={step}")
        return path

    def test_max_ckpt_1_preserves_existing_before_save(self, manager):
        """
        Regression test: max_ckpt_to_keep=1 must NOT delete existing checkpoint before save.
        """
        ckpt_100 = self._create_checkpoint_dir(100)
        manager.previous_saved_paths = [ckpt_100]

        manager.ensure_checkpoint_capacity(max_ckpt_to_keep=1)

        assert os.path.exists(ckpt_100), "Bug: checkpoint deleted before save!"
        assert manager.previous_saved_paths == [ckpt_100]

    def test_max_ckpt_1_deletes_old_after_save(self, manager):
        """After save succeeds, old checkpoint should be deleted."""
        ckpt_100 = self._create_checkpoint_dir(100)
        manager.previous_saved_paths = [ckpt_100]

        ckpt_200 = self._create_checkpoint_dir(200)
        manager.register_checkpoint(ckpt_200, max_ckpt_to_keep=1)

        assert not os.path.exists(ckpt_100)
        assert os.path.exists(ckpt_200)
        assert manager.previous_saved_paths == [ckpt_200]

    def test_max_ckpt_2_keeps_one_before_save(self, manager):
        """With max_ckpt_to_keep=2, pre-save cleanup keeps 1 checkpoint."""
        ckpt_100 = self._create_checkpoint_dir(100)
        ckpt_200 = self._create_checkpoint_dir(200)
        manager.previous_saved_paths = [ckpt_100, ckpt_200]

        manager.ensure_checkpoint_capacity(max_ckpt_to_keep=2)

        assert not os.path.exists(ckpt_100)
        assert os.path.exists(ckpt_200)
        assert len(manager.previous_saved_paths) == 1

    def test_max_ckpt_0_keeps_all(self, manager):
        """max_ckpt_to_keep=0 means unlimited - no deletions."""
        ckpt_100 = self._create_checkpoint_dir(100)
        ckpt_200 = self._create_checkpoint_dir(200)
        manager.previous_saved_paths = [ckpt_100, ckpt_200]

        manager.ensure_checkpoint_capacity(max_ckpt_to_keep=0)
        ckpt_300 = self._create_checkpoint_dir(300)
        manager.register_checkpoint(ckpt_300, max_ckpt_to_keep=0)

        assert os.path.exists(ckpt_100)
        assert os.path.exists(ckpt_200)
        assert os.path.exists(ckpt_300)
        assert len(manager.previous_saved_paths) == 3

    def test_full_save_cycle_max_ckpt_1(self, manager):
        """Simulate multiple save cycles with max_ckpt_to_keep=1."""
        # First save
        manager.ensure_checkpoint_capacity(1)
        ckpt_100 = self._create_checkpoint_dir(100)
        manager.register_checkpoint(ckpt_100, 1)
        assert manager.previous_saved_paths == [ckpt_100]

        # Second save - existing checkpoint must survive pre-save
        manager.ensure_checkpoint_capacity(1)
        assert os.path.exists(ckpt_100), "Bug: checkpoint deleted before save!"

        ckpt_200 = self._create_checkpoint_dir(200)
        manager.register_checkpoint(ckpt_200, 1)
        assert not os.path.exists(ckpt_100)
        assert manager.previous_saved_paths == [ckpt_200]

        # Third save
        manager.ensure_checkpoint_capacity(1)
        assert os.path.exists(ckpt_200), "Bug: checkpoint deleted before save!"

        ckpt_300 = self._create_checkpoint_dir(300)
        manager.register_checkpoint(ckpt_300, 1)
        assert not os.path.exists(ckpt_200)
        assert manager.previous_saved_paths == [ckpt_300]

    def test_loaded_checkpoint_restores_retention_after_restart(self, manager):
        ckpt_100 = self._create_checkpoint_dir(100)

        manager.record_loaded_checkpoint(ckpt_100)
        ckpt_200 = self._create_checkpoint_dir(200)
        manager.register_checkpoint(ckpt_200, max_ckpt_to_keep=1)
        manager.finalize_loaded_checkpoint_retention(ckpt_200, max_ckpt_to_keep=1)

        assert not os.path.exists(ckpt_100)
        assert os.path.exists(ckpt_200)
        assert manager.previous_saved_paths == [ckpt_200]

    def test_loaded_checkpoint_accepts_relative_registered_path(self, manager, monkeypatch):
        ckpt_100 = self._create_checkpoint_dir(100)
        ckpt_200 = self._create_checkpoint_dir(200)
        monkeypatch.chdir(self.test_dir)

        manager.record_loaded_checkpoint(ckpt_100)
        manager.register_checkpoint("global_step_200", max_ckpt_to_keep=1)
        manager.finalize_loaded_checkpoint_retention(ckpt_200, max_ckpt_to_keep=1)

        assert not os.path.exists(ckpt_100)
        assert os.path.exists(ckpt_200)

    def test_loaded_checkpoint_retention_does_not_scan_siblings(self, manager):
        ckpt_100 = self._create_checkpoint_dir(100)
        ckpt_200 = self._create_checkpoint_dir(200)
        ckpt_300 = os.path.join(self.test_dir, "global_step_300")

        manager.record_loaded_checkpoint(ckpt_200)
        os.makedirs(ckpt_300)
        manager.register_checkpoint(ckpt_300, max_ckpt_to_keep=1)
        manager.finalize_loaded_checkpoint_retention(ckpt_300, max_ckpt_to_keep=1)

        assert manager.previous_saved_paths == [ckpt_300]
        assert not os.path.exists(ckpt_200)
        assert os.path.exists(ckpt_100)

    @pytest.mark.parametrize("max_ckpt_to_keep", [None, 0, 2])
    def test_loaded_checkpoint_retention_is_limited_to_max_one(self, manager, max_ckpt_to_keep):
        ckpt_100 = self._create_checkpoint_dir(100)
        manager.record_loaded_checkpoint(ckpt_100)
        ckpt_200 = os.path.join(self.test_dir, "global_step_200")
        os.makedirs(ckpt_200)
        manager.register_checkpoint(ckpt_200, max_ckpt_to_keep=max_ckpt_to_keep)

        manager.finalize_loaded_checkpoint_retention(ckpt_200, max_ckpt_to_keep=max_ckpt_to_keep)

        assert os.path.exists(ckpt_100)

    def test_loaded_checkpoint_retention_ignores_different_series(self, manager):
        external_root = tempfile.mkdtemp()
        try:
            loaded_path = os.path.join(external_root, "global_step_100")
            os.makedirs(loaded_path)
            manager.record_loaded_checkpoint(loaded_path)
            ckpt_200 = self._create_checkpoint_dir(200)
            manager.register_checkpoint(ckpt_200, max_ckpt_to_keep=1)

            manager.finalize_loaded_checkpoint_retention(ckpt_200, max_ckpt_to_keep=1)

            assert os.path.exists(loaded_path)
        finally:
            shutil.rmtree(external_root, ignore_errors=True)

    @pytest.mark.parametrize("new_step", [100, 50])
    def test_loaded_checkpoint_retention_ignores_same_or_newer_loaded_step(self, manager, new_step):
        loaded_path = self._create_checkpoint_dir(100)
        manager.record_loaded_checkpoint(loaded_path)
        new_path = os.path.join(self.test_dir, f"global_step_{new_step}")
        os.makedirs(new_path, exist_ok=True)
        manager.register_checkpoint(new_path, max_ckpt_to_keep=1)

        manager.finalize_loaded_checkpoint_retention(new_path, max_ckpt_to_keep=1)

        assert os.path.exists(loaded_path)

    def test_loaded_checkpoint_retention_requires_registered_replacement(self, manager):
        loaded_path = self._create_checkpoint_dir(100)
        new_path = self._create_checkpoint_dir(200)
        manager.record_loaded_checkpoint(loaded_path)

        manager.finalize_loaded_checkpoint_retention(new_path, max_ckpt_to_keep=1)

        assert os.path.exists(loaded_path)
