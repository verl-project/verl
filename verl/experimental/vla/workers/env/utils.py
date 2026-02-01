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
"""
Utility classes and functions for Isaac Server mode.

- TaskBalancedSampler: Ensures each batch has at most max_per_task samples per task/stage
- create_task_balanced_sampler: Factory function with server config integration

The sampler enforces per-stage task balancing. Each stage has its own server group,
so we ensure each stage's portion respects the max_per_task constraint independently.

Batch is assigned to stages via round-robin:
    - Stage 0: batch[0], batch[2], batch[4], ...
    - Stage 1: batch[1], batch[3], batch[5], ...

When stage_num=1 (default), samples_per_stage = batch_size, and the behavior
is equivalent to simple per-batch balancing.

Note:
    num_server_groups must match stage_num (pipeline_stage_num).
"""

import logging
import random
from collections import defaultdict
from typing import Iterator, Optional

from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class TaskBalancedSampler(Sampler):
    """
    A sampler that pre-arranges indices to ensure balanced task distribution per stage.

    Each stage gets samples_per_stage = batch_size // stage_num samples.
    Within each stage, no task exceeds max_per_task samples.
    Samples are interleaved across stages: [s0[0], s1[0], s0[1], s1[1], ...]

    When stage_num=1 (default), this reduces to simple per-batch balancing.

    This is critical for Isaac Server mode where each task has limited env capacity:
        max_envs_per_task = server_group_size / num_envs_per_trajectory

    Args:
        dataset: The dataset to sample from (must have 'task_ids' field)
        batch_size: Total number of samples per batch
        max_per_task: Maximum samples from any single task per stage
        drop_last: Whether to drop the last incomplete batch
        shuffle: Whether to shuffle within and across tasks
        seed: Random seed for reproducibility
        stage_num: Number of pipeline stages (default 1)
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        max_per_task: int,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: Optional[int] = None,
        stage_num: int = 1,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_per_task = max_per_task
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        self.stage_num = stage_num

        # Ensure batch_size is evenly divisible by stage_num
        assert batch_size % stage_num == 0, f"batch_size ({batch_size}) must be divisible by stage_num ({stage_num})"

        # When stage_num > 1, each stage gets batch_size/stage_num samples
        # and each stage has its own max_per_task constraint
        self.samples_per_stage = batch_size // stage_num

        # Build task -> sample indices mapping
        self._build_task_indices()

        logger.info(
            f"TaskBalancedSampler initialized: "
            f"batch_size={batch_size}, max_per_task={max_per_task}, "
            f"stage_num={stage_num}, samples_per_stage={self.samples_per_stage}, "
            f"num_tasks={len(self.task_to_indices)}, total_samples={len(dataset)}"
        )

        # Validate that we can fill batches
        self._validate_config()

    def _build_task_indices(self):
        """Build mapping from task_id to sample indices."""
        self.task_to_indices = defaultdict(list)

        # Try different ways to access task_ids from the dataset
        if hasattr(self.dataset, "__getitem__"):
            for idx in range(len(self.dataset)):
                sample = self.dataset[idx]
                if isinstance(sample, dict):
                    task_id = sample.get("task_ids", sample.get("task_id", 0))
                else:
                    task_id = 0
                self.task_to_indices[task_id].append(idx)
        else:
            # Fallback: assume all samples are from task 0
            self.task_to_indices[0] = list(range(len(self.dataset)))

        # Log task distribution
        task_counts = {t: len(indices) for t, indices in self.task_to_indices.items()}
        logger.info(f"Task distribution: {task_counts}")

    def _validate_config(self):
        """Validate that configuration is viable."""
        num_tasks = len(self.task_to_indices)
        # Each stage must be fillable independently
        # When stage_num=1, samples_per_stage = batch_size, so this works for both cases
        max_possible_per_stage = num_tasks * self.max_per_task
        if max_possible_per_stage < self.samples_per_stage:
            logger.warning(
                f"Cannot fill samples_per_stage={self.samples_per_stage} "
                f"with max_per_task={self.max_per_task} and only {num_tasks} tasks. "
                f"Max possible per stage: {max_possible_per_stage}. "
                f"Consider reducing batch_size, increasing max_per_task, or using more tasks."
            )

    def set_epoch(self, epoch: int):
        """Set epoch for shuffling reproducibility."""
        self._epoch = epoch

    def _generate_balanced_indices(self) -> list[int]:
        """
        Generate indices with per-stage task balancing.

        Works for both single-stage (standard) and multi-stage modes:
        - stage_num=1: Standard mode, samples_per_stage = batch_size
        - stage_num>1: Multi-stage mode, samples are interleaved across stages

        When stage_num > 1, batch is split into stages via round-robin:
            - Stage 0: batch[0], batch[2], batch[4], ...
            - Stage 1: batch[1], batch[3], batch[5], ...

        IMPORTANT: This interleaving is tightly coupled with EnvWorkerServer.reset_envs_to_state_ids()
        in env_worker_server.py, which uses the same round-robin logic (traj_idx % stage_num).
        If you change the interleaving here, you MUST update reset_envs_to_state_ids() too.

        Each stage has its own server group, so we need to ensure each stage's
        portion respects the max_per_task constraint independently.

        Strategy:
            1. Generate samples for each stage separately (each with max_per_task limit)
            2. Interleave stage samples to create the batch (no-op when stage_num=1)
        """
        rng = random.Random(self.seed + self._epoch if self.seed else None)

        # Create separate copies for each stage
        # When stage_num=1, this is just one copy (same as standard mode)
        stage_remaining = []
        for _ in range(self.stage_num):
            remaining = {task_id: list(indices) for task_id, indices in self.task_to_indices.items()}
            if self.shuffle:
                for indices in remaining.values():
                    rng.shuffle(indices)
            stage_remaining.append(remaining)

        all_indices = []

        while True:
            # Generate samples for each stage
            stage_batches = []
            all_stages_done = True

            for stage_id in range(self.stage_num):
                remaining = stage_remaining[stage_id]
                active_tasks = {t for t, indices in remaining.items() if indices}

                if not active_tasks:
                    stage_batches.append([])
                    continue

                all_stages_done = False
                stage_batch = []
                task_counts = defaultdict(int)

                task_list = list(active_tasks)
                if self.shuffle:
                    rng.shuffle(task_list)

                # Fill this stage's portion with max_per_task constraint
                made_progress = True
                while len(stage_batch) < self.samples_per_stage and made_progress:
                    made_progress = False
                    for task_id in task_list:
                        if task_id not in active_tasks:
                            continue
                        if task_counts[task_id] >= self.max_per_task:
                            continue
                        if not remaining[task_id]:
                            active_tasks.discard(task_id)
                            continue

                        idx = remaining[task_id].pop()
                        stage_batch.append(idx)
                        task_counts[task_id] += 1
                        made_progress = True

                        if len(stage_batch) >= self.samples_per_stage:
                            break

                stage_batches.append(stage_batch)

            if all_stages_done:
                break

            # Check if all stages have enough samples
            stage_sizes = [len(sb) for sb in stage_batches]
            if all(s == self.samples_per_stage for s in stage_sizes):
                # Interleave: [s0[0], s1[0], s0[1], s1[1], ...]
                # When stage_num=1, this just extends with stage_batches[0]
                batch = []
                for i in range(self.samples_per_stage):
                    for stage_id in range(self.stage_num):
                        batch.append(stage_batches[stage_id][i])
                all_indices.extend(batch)
            elif not self.drop_last and any(s > 0 for s in stage_sizes):
                # Partial batch: interleave what we have
                max_len = max(stage_sizes)
                batch = []
                for i in range(max_len):
                    for stage_id in range(self.stage_num):
                        if i < len(stage_batches[stage_id]):
                            batch.append(stage_batches[stage_id][i])
                all_indices.extend(batch)
            else:
                break

        return all_indices

    def __iter__(self) -> Iterator[int]:
        """Iterate over pre-arranged indices."""
        indices = self._generate_balanced_indices()
        return iter(indices)

    def __len__(self) -> int:
        """Return total number of samples (after dropping incomplete batches if configured)."""
        total_samples = sum(len(indices) for indices in self.task_to_indices.values())
        if self.drop_last:
            return (total_samples // self.batch_size) * self.batch_size
        else:
            return total_samples


def create_task_balanced_sampler(data_config, dataset):
    """
    Create a task-balanced sampler for Isaac Server mode.

    Args:
        data_config: Configuration with server_group_size, num_envs, batch_size, stage_num, etc.
        dataset: The dataset to sample from

    Returns:
        TaskBalancedSampler instance
    """
    # Get config values
    batch_size = data_config.get("gen_batch_size", data_config.get("train_batch_size", 128))
    server_group_size = data_config.get("server_group_size", 64)
    num_envs = data_config.get("num_envs", 8)
    stage_num = data_config.get("stage_num", 1)

    # Calculate max samples per task
    # In Isaac Server, each task has server_group_size envs
    # Each env produces one trajectory, so max_per_task = server_group_size / num_envs
    max_per_task = server_group_size // num_envs
    samples_per_stage = batch_size // stage_num

    logger.info(
        f"Creating TaskBalancedSampler: "
        f"server_group_size={server_group_size}, num_envs={num_envs}, "
        f"max_per_task={max_per_task}, stage_num={stage_num}, samples_per_stage={samples_per_stage}"
    )

    # Validate: each stage's capacity must fit
    # When stage_num=1, samples_per_stage = batch_size, so this works for both cases
    # Get num_tasks from config or dataset
    num_tasks = data_config.get("num_tasks", None)
    if num_tasks is None and hasattr(dataset, "num_tasks"):
        num_tasks = dataset.num_tasks

    if num_tasks is not None:
        max_possible_per_stage = num_tasks * max_per_task
        if max_possible_per_stage < samples_per_stage:
            raise ValueError(
                f"Configuration error: samples_per_stage={samples_per_stage} exceeds capacity. "
                f"Max possible per stage = {num_tasks} tasks × {max_per_task} max_per_task = {max_possible_per_stage}. "
                f"Solutions: increase server_group_size, decrease batch_size, or increase stage_num."
            )
    # Note: If num_tasks is unknown here, TaskBalancedSampler._validate_config() will check after building task indices

    return TaskBalancedSampler(
        dataset=dataset,
        batch_size=batch_size,
        max_per_task=max_per_task,
        drop_last=True,
        shuffle=data_config.get("shuffle", True),
        seed=data_config.get("seed", None),
        stage_num=stage_num,
    )
