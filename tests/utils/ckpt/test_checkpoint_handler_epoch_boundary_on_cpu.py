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
import shutil
import tempfile

import pytest
import torch


class TestAtEpochBoundary:
    """Tests for CheckpointHandler._at_epoch_boundary."""

    def _make_handler(self, monkeypatch, steps_per_epoch, resume_global_step):
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

        class MockEngine:
            def is_mp_src_rank_with_outputs(self):
                return True

            def get_data_parallel_rank(self):
                return 0

        from verl.utils.checkpoint.checkpoint_handler import CheckpointHandler

        handler = CheckpointHandler(
            engine=MockEngine(),
            train_dataloader=None,
            default_local_dir="/tmp",
            steps_per_epoch=steps_per_epoch,
        )
        handler.resume_global_step = resume_global_step
        return handler

    def test_disabled_when_steps_per_epoch_unknown(self, monkeypatch):
        handler = self._make_handler(monkeypatch, steps_per_epoch=None, resume_global_step=3183)
        assert handler._at_epoch_boundary() is False

    def test_resume_from_scratch_is_not_boundary(self, monkeypatch):
        handler = self._make_handler(monkeypatch, steps_per_epoch=1061, resume_global_step=0)
        assert handler._at_epoch_boundary() is False

    def test_mid_epoch_step_is_not_boundary(self, monkeypatch):
        handler = self._make_handler(monkeypatch, steps_per_epoch=1061, resume_global_step=3683)
        assert handler._at_epoch_boundary() is False

    def test_epoch_boundary_step_detected(self, monkeypatch):
        handler = self._make_handler(monkeypatch, steps_per_epoch=1061, resume_global_step=3183)
        assert handler._at_epoch_boundary() is True

    def test_first_epoch_boundary_detected(self, monkeypatch):
        handler = self._make_handler(monkeypatch, steps_per_epoch=1061, resume_global_step=1061)
        assert handler._at_epoch_boundary() is True


class TestLoadDataloaderState:
    """End-to-end tests for CheckpointHandler._load_dataloader_state."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.test_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _make_handler(self, monkeypatch, steps_per_epoch, resume_global_step):
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

        class MockEngine:
            def is_mp_src_rank_with_outputs(self):
                return True

            def get_data_parallel_rank(self):
                return 0

        class MockDataloader:
            def __init__(self):
                self.loaded_state = None

            def load_state_dict(self, state_dict):
                self.loaded_state = state_dict

        from verl.utils.checkpoint.checkpoint_handler import CheckpointHandler

        handler = CheckpointHandler(
            engine=MockEngine(),
            train_dataloader=MockDataloader(),
            default_local_dir=self.test_dir,
            steps_per_epoch=steps_per_epoch,
        )
        handler.resume_global_step = resume_global_step
        return handler

    def _write_state(self, name="data_0.pt"):
        state = {"_sampler_iter_yielded": 1061, "_num_yielded": 1061}
        torch.save(state, os.path.join(self.test_dir, name))
        return state

    def test_epoch_boundary_state_is_dropped(self, monkeypatch):
        handler = self._make_handler(monkeypatch, steps_per_epoch=1061, resume_global_step=3183)
        self._write_state()

        handler._load_dataloader_state(self.test_dir)

        assert handler.train_dataloader.loaded_state is None

    def test_mid_epoch_state_is_restored(self, monkeypatch):
        handler = self._make_handler(monkeypatch, steps_per_epoch=1061, resume_global_step=3683)
        state = self._write_state()

        handler._load_dataloader_state(self.test_dir)

        assert handler.train_dataloader.loaded_state == state

    def test_state_restored_when_detection_disabled(self, monkeypatch):
        handler = self._make_handler(monkeypatch, steps_per_epoch=None, resume_global_step=3183)
        state = self._write_state()

        handler._load_dataloader_state(self.test_dir)

        assert handler.train_dataloader.loaded_state == state

    def test_missing_state_file_keeps_scratch_start(self, monkeypatch):
        handler = self._make_handler(monkeypatch, steps_per_epoch=1061, resume_global_step=3183)

        handler._load_dataloader_state(self.test_dir)

        assert handler.train_dataloader.loaded_state is None
