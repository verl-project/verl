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

"""Checkpoint publication timing for SFT's engine based trainer."""

from unittest.mock import MagicMock, call, patch

from verl.utils.checkpoint.checkpoint_handler import CheckpointHandler, OrchestrationMode


def _make_handler(tmp_path, *, async_save):
    engine = MagicMock()
    dataloader = MagicMock()
    dataloader.state_dict.return_value = {"step": 3}
    handler = CheckpointHandler(
        engine=engine,
        train_dataloader=dataloader,
        default_local_dir=str(tmp_path),
        default_hdfs_dir="hdfs://test/checkpoints",
        mode=OrchestrationMode.RAY,
        async_save=async_save,
    )
    return handler, engine


def test_async_sft_saves_auxiliary_state_before_engine_and_defers_publication(tmp_path):
    handler, engine = _make_handler(tmp_path, async_save=True)
    step_dir = tmp_path / "global_step_3"
    tracker = tmp_path / "latest_checkpointed_iteration.txt"

    def check_auxiliary_state(**kwargs):
        assert (step_dir / "data_0.pt").exists()
        assert kwargs["hdfs_path"] == "hdfs://test/checkpoints"

    engine.save_checkpoint.side_effect = check_auxiliary_state
    with patch("verl.utils.checkpoint.checkpoint_handler.hdfs_io.copy") as copy:
        handler.save_checkpoint(step=3)

    assert not tracker.exists()
    copy.assert_not_called()
    handler.finalize_async_checkpointing(blocking=True)
    assert engine.finalize_async_checkpointing.call_args_list == [call(blocking=False), call(blocking=True)]


def test_sync_sft_publishes_after_engine_save(tmp_path):
    handler, engine = _make_handler(tmp_path, async_save=False)
    step_dir = tmp_path / "global_step_3"

    def check_auxiliary_state(**kwargs):
        assert not (step_dir / "data_0.pt").exists()
        assert kwargs["hdfs_path"] is None

    engine.save_checkpoint.side_effect = check_auxiliary_state
    with (
        patch("verl.utils.checkpoint.checkpoint_handler.hdfs_io.makedirs"),
        patch("verl.utils.checkpoint.checkpoint_handler.hdfs_io.copy") as copy,
    ):
        handler.save_checkpoint(step=3)

    assert (step_dir / "data_0.pt").exists()
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "3"
    copy.assert_called_once_with(src=str(step_dir), dst="hdfs://test/checkpoints", dirs_exist_ok=True)
    engine.finalize_async_checkpointing.assert_not_called()
