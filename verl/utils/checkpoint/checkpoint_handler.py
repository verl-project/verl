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


# TODO: add unit tests

import logging
import os
import re
from enum import Enum

import torch

import verl.utils.hdfs_io as hdfs_io
from verl.single_controller import WorkerGroup
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, get_checkpoint_tracker_filename
from verl.utils.logger import log_with_rank
from verl.workers.engine import BaseEngine


def extract_step(path):
    match = re.search(r"global_step_(\d+)", path)
    if match:
        return int(match.group(1))
    return None


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


class OrchestrationMode(Enum):
    SPMD = 0
    RAY = 1


class CheckpointHandler:
    """
    Checkpoint handler handles the path, global_step of a checkpoint folder.
    Currently, it only works with a single model.
    We can expand it to support multiple models. It is expected to be used with SPMD style (e.g., torchrun)
    """

    def __init__(
        self,
        engine: BaseEngine | WorkerGroup,
        train_dataloader,
        *,
        default_local_dir,
        max_ckpt_to_keep=None,
        default_hdfs_dir=None,
        resume_mode="auto",
        resume_from_path=None,
        mode=OrchestrationMode.SPMD,
    ):
        self.default_local_dir = default_local_dir
        self.max_ckpt_to_keep = max_ckpt_to_keep
        self.default_hdfs_dir = default_hdfs_dir
        self.resume_mode = resume_mode
        self.resume_from_path = resume_from_path
        self.engine = engine
        self.train_dataloader = train_dataloader
        self.mode = mode

        if self.mode == OrchestrationMode.SPMD:
            self.rank = torch.distributed.get_rank()
            self.is_mp_src_rank_with_outputs = self.engine.is_mp_src_rank_with_outputs()
            self.dp_rank = self.engine.get_data_parallel_rank()
        elif self.mode == OrchestrationMode.RAY:
            self.rank = 0
            self.is_mp_src_rank_with_outputs = True
            self.dp_rank = 0
        else:
            raise ValueError(f"Unknown {self.mode=}")

    def save_checkpoint(self, step):
        """Save checkpoint using FSDPCheckpointManager with improved tracking"""
        from verl.utils.fs import local_mkdir_safe

        # Determine checkpoint path
        local_global_step_folder = os.path.join(self.default_local_dir, f"global_step_{step}")
        if self.rank == 0:
            print(f"Saving checkpoint to: {local_global_step_folder}")

        # Get max checkpoints to keep
        max_ckpt_to_keep = self.max_ckpt_to_keep

        # Use checkpoint manager to save
        self.engine.save_checkpoint(
            local_path=local_global_step_folder, global_step=step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        # Save dataloader state. Note that we only save the iterator in the train_dataloader.
        # So it's identical in each dp rank.
        if self.is_mp_src_rank_with_outputs:
            dp_rank = self.dp_rank
            local_mkdir_safe(local_global_step_folder)
            dataloader_local_path = os.path.join(local_global_step_folder, f"data_{dp_rank}.pt")

            # Use StatefulDataLoader's built-in state dict functionality
            dataloader_state_dict = self.train_dataloader.state_dict()
            torch.save(dataloader_state_dict, dataloader_local_path)
            print(f"Saved dataloader state to: {dataloader_local_path}")

        if self.rank == 0:
            # Update latest checkpoint tracker (atomic write)
            tracker_file = get_checkpoint_tracker_filename(self.default_local_dir)
            temp_tracker_file = tracker_file + ".tmp"
            with open(temp_tracker_file, "w") as f:
                f.write(str(step))
            os.rename(temp_tracker_file, tracker_file)
            print(f"Updated checkpoint tracker: {tracker_file}")

        # Copy to HDFS if configured
        if self.rank == 0 and self.default_hdfs_dir:
            hdfs_io.makedirs(self.default_hdfs_dir, exist_ok=True)
            hdfs_io.copy(src=local_global_step_folder, dst=self.default_hdfs_dir, dirs_exist_ok=True)

        if self.mode == OrchestrationMode.SPMD:
            torch.distributed.barrier()

    def load_checkpoint(self):
        # Determine resume path based on configuration
        checkpoint_path = self._determine_resume_path()

        if checkpoint_path is None:
            return 0

        # extract resume step from checkpoint path
        resume_step = extract_step(checkpoint_path)
        if resume_step is None:
            log_with_rank(
                f"Warning: Could not extract step number from {checkpoint_path}, starting from step 0",
                logger=logger,
                rank=self.rank,
                level=logging.WARNING,
                log_only_rank_0=True,
            )
            return 0
        self.resume_global_step = resume_step

        # Use checkpoint manager to load model state
        self.engine.load_checkpoint(checkpoint_path)
        # Always load dataloader state for StatefulDataLoader
        self._load_dataloader_state(checkpoint_path)

        return resume_step

    def _load_dataloader_state(self, checkpoint_path: str):
        """Load dataloader state from checkpoint"""
        dp_rank = self.dp_rank
        dataloader_path = os.path.join(checkpoint_path, f"data_{dp_rank}.pt")

        if os.path.exists(dataloader_path):
            # Use StatefulDataLoader's built-in state dict functionality
            dataloader_state_dict = torch.load(dataloader_path, map_location="cpu", weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)

            log_with_rank(
                f"Successfully loaded dataloader state from {dataloader_path}",
                logger=logger,
                rank=self.rank,
                log_only_rank_0=True,
            )

        else:
            log_with_rank(
                f"Warning: No dataloader state found at {dataloader_path}, will start from scratch",
                logger=logger,
                rank=self.rank,
                level=logging.WARNING,
                log_only_rank_0=True,
            )

    def _determine_resume_path(self):
        """Determine the path to resume from based on resume_mode configuration"""
        resume_mode = self.resume_mode
        resume_from_path = self.resume_from_path

        if resume_mode == "disable":
            return None
        elif resume_mode == "auto":
            if resume_from_path is not None:
                assert os.path.exists(resume_from_path), (
                    "resume_from_path must be null or an existing path when resume_mode is 'auto'"
                )
                assert "global_step_" in resume_from_path, "resume_from_path must specify the global_steps"
                return resume_from_path
            # Try to find the latest checkpoint in the default directory
            return self._find_latest_checkpoint()
        elif resume_mode == "resume_path":
            assert os.path.exists(resume_from_path), (
                "resume_from_path must be an existing path when resume_mode is 'resume_path'"
            )
            assert "global_step_" in resume_from_path, "resume_from_path must specify the global_steps"
            return resume_from_path
        else:
            raise ValueError(f"Invalid resume_mode: {resume_mode}. Must be 'auto', 'disable', or 'resume_path'")

    def _find_latest_checkpoint(self):
        """Find the latest checkpoint in the default local directory"""
        checkpoint_dir = self.default_local_dir

        if not os.path.exists(checkpoint_dir):
            return None

        latest_checkpoint = find_latest_ckpt_path(checkpoint_dir)

        if latest_checkpoint and self.rank == 0:
            step_num = extract_step(latest_checkpoint)
            print(f"Found latest checkpoint: {latest_checkpoint} (step {step_num})")

        return latest_checkpoint
