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

"""Tests for checkpoint publication around asynchronous engine saves."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from verl.utils.checkpoint.checkpoint_handler import CheckpointHandler, OrchestrationMode
from verl.utils.checkpoint.checkpoint_manager import get_checkpoint_tracker_filename
from verl.workers.engine.base import BaseEngine


def _make_handler(tmp_path: Path, *, async_finalize: bool, default_hdfs_dir: str | None = None):
    engine = MagicMock()
    engine.is_mp_src_rank_with_outputs.return_value = False
    engine.get_data_parallel_rank.return_value = 0
    dataloader = MagicMock()
    # ``save_checkpoint`` persists this via ``torch.save``, which cannot
    # pickle a bare MagicMock; return a plain picklable object instead.
    dataloader.state_dict.return_value = {"dummy": 0}

    handler = CheckpointHandler(
        engine=engine,
        train_dataloader=dataloader,
        default_local_dir=str(tmp_path),
        default_hdfs_dir=default_hdfs_dir,
        mode=OrchestrationMode.RAY,
        async_save=async_finalize,
    )
    return handler, engine


def test_async_engine_finalizes_before_save_and_does_not_publish_tracker(tmp_path):
    handler, engine = _make_handler(tmp_path, async_finalize=True)

    handler.save_checkpoint(step=3)

    engine.finalize_async_checkpointing.assert_called_once_with(blocking=False)
    engine.save_checkpoint.assert_called_once_with(
        local_path=str(tmp_path / "global_step_3"),
        hdfs_path=None,
        global_step=3,
        max_ckpt_to_keep=None,
    )
    assert not Path(get_checkpoint_tracker_filename(str(tmp_path))).exists()


def test_synchronous_engine_publishes_tracker_after_save(tmp_path):
    handler, engine = _make_handler(tmp_path, async_finalize=False)

    handler.save_checkpoint(step=5)

    engine.finalize_async_checkpointing.assert_not_called()
    engine.save_checkpoint.assert_called_once_with(
        local_path=str(tmp_path / "global_step_5"),
        hdfs_path=None,
        global_step=5,
        max_ckpt_to_keep=None,
    )
    assert Path(get_checkpoint_tracker_filename(str(tmp_path))).read_text() == "5"


def test_async_engine_defers_hdfs_publication_to_finalize_callback(tmp_path):
    hdfs_dir = "hdfs://checkpoint-output"
    handler, engine = _make_handler(tmp_path, async_finalize=True, default_hdfs_dir=hdfs_dir)

    with patch("verl.utils.checkpoint.checkpoint_handler.hdfs_io.copy") as copy:
        handler.save_checkpoint(step=3)

    engine.save_checkpoint.assert_called_once_with(
        local_path=str(tmp_path / "global_step_3"),
        hdfs_path=hdfs_dir,
        global_step=3,
        max_ckpt_to_keep=None,
    )
    copy.assert_not_called()


def test_synchronous_engine_uploads_checkpoint_immediately(tmp_path):
    hdfs_dir = "hdfs://checkpoint-output"
    handler, _ = _make_handler(tmp_path, async_finalize=False, default_hdfs_dir=hdfs_dir)

    with (
        patch("verl.utils.checkpoint.checkpoint_handler.hdfs_io.makedirs") as makedirs,
        patch("verl.utils.checkpoint.checkpoint_handler.hdfs_io.copy") as copy,
    ):
        handler.save_checkpoint(step=5)

    makedirs.assert_called_once_with(hdfs_dir, exist_ok=True)
    copy.assert_called_once_with(src=str(tmp_path / "global_step_5"), dst=hdfs_dir, dirs_exist_ok=True)


def test_base_engine_rejects_async_checkpointing():
    with pytest.raises(NotImplementedError, match="checkpoint.async_save=false or use the Megatron engine"):
        BaseEngine().finalize_async_checkpointing()
