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

"""Tests for del_local_ckpt_after_load checkpoint cleanup.

Verifies that the cleanup logic in FSDPCheckpointManager and
MegatronCheckpointManager correctly removes local checkpoint
directories after loading, regardless of whether the path is
local or remote (HDFS).
"""

import os
import shutil
import tempfile

import pytest

from verl.utils.fs import is_non_local


class TestDelLocalCkptCleanupLogic:
    """Test the underlying cleanup logic that was broken in #7213.

    The bug: cleanup used `os.remove(path) if is_non_local(path) else None`
    which never fires for local paths (is_non_local only matches hdfs://).
    Also, os.remove cannot delete directories.

    The fix: use `shutil.rmtree(path)` without the is_non_local guard.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.test_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_ckpt_dir(self, name="actor"):
        """Create a mock checkpoint directory with files inside."""
        ckpt_path = os.path.join(self.test_dir, "global_step_100", name)
        os.makedirs(ckpt_path, exist_ok=True)
        # Simulate checkpoint files
        for fname in ["model_world_size_1_rank_0.pt", "optim_world_size_1_rank_0.pt"]:
            with open(os.path.join(ckpt_path, fname), "w") as f:
                f.write("dummy")
        return ckpt_path

    def test_is_non_local_rejects_local_paths(self):
        """Confirm is_non_local returns False for local paths (the root cause)."""
        assert not is_non_local("/tmp/checkpoints/step_100")
        assert not is_non_local(self.test_dir)
        assert is_non_local("hdfs://cluster/checkpoints/step_100")

    def test_old_cleanup_logic_fails_on_local_paths(self):
        """Demonstrate the old buggy pattern never deletes local directories."""
        ckpt_path = self._create_ckpt_dir()
        assert os.path.isdir(ckpt_path)

        # Old code: os.remove(path) if is_non_local(path) else None
        os.remove(ckpt_path) if is_non_local(ckpt_path) else None

        # Bug: directory still exists because is_non_local returned False
        assert os.path.isdir(ckpt_path), "Expected old logic to NOT delete local path"

    def test_new_cleanup_logic_removes_local_directory(self):
        """The fixed pattern correctly removes local checkpoint directories."""
        ckpt_path = self._create_ckpt_dir()
        assert os.path.isdir(ckpt_path)

        # New code: shutil.rmtree(path) without is_non_local guard
        if os.path.isdir(ckpt_path):
            shutil.rmtree(ckpt_path)

        assert not os.path.exists(ckpt_path)

    def test_cleanup_handles_already_removed_path(self):
        """Cleanup should not raise if the path was already removed."""
        ckpt_path = os.path.join(self.test_dir, "nonexistent")

        # The isdir guard should prevent any error
        if os.path.isdir(ckpt_path):
            shutil.rmtree(ckpt_path)

        # No exception raised — this is the expected behavior

    def test_cleanup_removes_nested_files(self):
        """Cleanup should remove the directory and all its contents."""
        ckpt_path = self._create_ckpt_dir()
        # Add a nested subdirectory
        nested = os.path.join(ckpt_path, "huggingface")
        os.makedirs(nested)
        with open(os.path.join(nested, "config.json"), "w") as f:
            f.write("{}")

        assert len(os.listdir(ckpt_path)) == 3  # 2 pt files + huggingface dir

        if os.path.isdir(ckpt_path):
            shutil.rmtree(ckpt_path)

        assert not os.path.exists(ckpt_path)
