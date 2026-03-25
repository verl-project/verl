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

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
from tensordict import TensorDict

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class _DualPoolState:
    positive_pool: Optional[TensorDict] = None
    negative_pool: Optional[TensorDict] = None
    positive_size: int = 0
    negative_size: int = 0
    positive_position: int = 0
    negative_position: int = 0


class SACReplayPool:
    """Task-aware SAC Replay Pool.

    For each task_id we maintain two independent pools:
    - positive pool
    - negative pool

    `single_pool_capacity` is the size of each single pool.
    """

    def __init__(
        self,
        single_pool_capacity: int,
        pool_device: str = "cpu",
        sample_device: str = "cpu",
    ):
        self.single_pool_capacity = int(single_pool_capacity)
        self.pool_device = pool_device
        self.sample_device = sample_device

        self.task_pools: dict[str, _DualPoolState] = {}

        self.size = 0
        self.positive_size = 0
        self.negative_size = 0

        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    def add_batch(self, batch: TensorDict, task_ids: Sequence[Any]):
        """Add a batch of samples into task-specific positive/negative pools."""

        if batch.batch_size[0] == 0:
            return

        if len(task_ids) != batch.batch_size[0]:
            raise ValueError(f"task_ids length ({len(task_ids)}) must match batch size ({batch.batch_size[0]}).")

        valid_mask = batch["valids"].to(torch.bool)
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
        if valid_indices.numel() == 0:
            return

        batch = self._index_select_batch(batch, valid_indices.to(batch.device))
        selected = valid_indices.cpu().tolist()
        task_ids = [task_ids[i] for i in selected]

        positive_mask = self._extract_positive_mask(batch)

        grouped_indices: dict[str, dict[str, list[int]]] = {}
        for idx in range(batch.batch_size[0]):
            task_key = self._normalize_task_id(task_ids[idx])
            if task_key not in grouped_indices:
                grouped_indices[task_key] = {"positive": [], "negative": []}
            if bool(positive_mask[idx].item()):
                grouped_indices[task_key]["positive"].append(idx)
            else:
                grouped_indices[task_key]["negative"].append(idx)

        for task_key, groups in grouped_indices.items():
            pool_state = self._get_or_create_task_pool(task_key, batch)

            if groups["positive"]:
                positive_idx = torch.tensor(groups["positive"], device=batch.device, dtype=torch.long)
                positive_batch = self._index_select_batch(batch, positive_idx)
                self._insert_block_to_pool(pool_state, positive_batch, is_positive_pool=True)

            if groups["negative"]:
                negative_idx = torch.tensor(groups["negative"], device=batch.device, dtype=torch.long)
                negative_batch = self._index_select_batch(batch, negative_idx)
                self._insert_block_to_pool(pool_state, negative_batch, is_positive_pool=False)

        self._refresh_global_stats()

    def sample_batch(
        self,
        batch_size: int,
        positive_sample_ratio: float = 0.5,
        return_sample_info: bool = False,
    ) -> TensorDict | tuple[TensorDict, dict]:
        """Sample a batch from all task-specific pools."""

        if self.size == 0:
            raise ValueError("Replay pool is empty, unable to sample.")

        positive_sample_ratio = max(0.0, min(1.0, float(positive_sample_ratio)))
        target_positive = int(round(batch_size * positive_sample_ratio))
        target_negative = batch_size - target_positive

        sampled_positive = min(target_positive, self.positive_size)
        sampled_negative = min(target_negative, self.negative_size)

        deficit = batch_size - sampled_positive - sampled_negative
        if deficit > 0:
            remaining_positive = self.positive_size - sampled_positive
            remaining_negative = self.negative_size - sampled_negative

            if remaining_positive >= remaining_negative:
                extra_positive = min(deficit, remaining_positive)
                sampled_positive += extra_positive
                deficit -= extra_positive

                extra_negative = min(deficit, remaining_negative)
                sampled_negative += extra_negative
                deficit -= extra_negative
            else:
                extra_negative = min(deficit, remaining_negative)
                sampled_negative += extra_negative
                deficit -= extra_negative

                extra_positive = min(deficit, remaining_positive)
                sampled_positive += extra_positive
                deficit -= extra_positive

        sampled_parts = []
        if sampled_positive > 0:
            sampled_parts.append(self._sample_from_task_pools(sampled_positive, is_positive_pool=True))
        if sampled_negative > 0:
            sampled_parts.append(self._sample_from_task_pools(sampled_negative, is_positive_pool=False))

        sampled_count = sampled_positive + sampled_negative
        if len(sampled_parts) == 1:
            sampled_batch = sampled_parts[0]
        else:
            sampled_batch = TensorDict(
                {key: torch.cat([part[key] for part in sampled_parts], dim=0) for key in sampled_parts[0].keys()},
                batch_size=[sampled_count],
                device=self.sample_device,
            )

        if sampled_count < batch_size:
            sampled_batch = self._pad_sampled_batch(sampled_batch, target_batch_size=batch_size)
        else:
            sampled_batch = TensorDict(
                {key: value for key, value in sampled_batch.items()},
                batch_size=[batch_size],
                device=self.sample_device,
            )

        shuffle_idx = torch.randperm(batch_size, device=self.sample_device)
        sampled_batch = TensorDict(
            {key: value.index_select(0, shuffle_idx) for key, value in sampled_batch.items()},
            batch_size=[batch_size],
            device=self.sample_device,
        )

        if not return_sample_info:
            return sampled_batch

        sample_info = {
            "actual_positive_sample_ratio": sampled_positive / max(batch_size, 1),
            "positive_size": self.positive_size,
            "negative_size": self.negative_size,
            "task_count": len(self.task_pools),
        }
        return sampled_batch, sample_info

    def insert_and_resample(
        self,
        source: TensorDict,
        task_ids: Sequence[Any],
    ) -> TensorDict:
        """Insert source into replay pool and sample a batch with the same size."""

        self.add_batch(source, task_ids=task_ids)
        return self.sample_batch(source.batch_size[0])

    def save(self, directory: str):
        """Save the replay pool to a directory."""

        os.makedirs(directory, exist_ok=True)
        filepath = f"{directory}/sac_replay_pool_rank_{self.rank}.pt"

        tasks_payload: dict[str, dict[str, Any]] = {}
        for task_id, pool_state in self.task_pools.items():
            assert pool_state.positive_pool is not None
            assert pool_state.negative_pool is not None
            tasks_payload[task_id] = {
                "positive_pool": pool_state.positive_pool.cpu(),
                "negative_pool": pool_state.negative_pool.cpu(),
                "positive_size": pool_state.positive_size,
                "negative_size": pool_state.negative_size,
                "positive_position": pool_state.positive_position,
                "negative_position": pool_state.negative_position,
            }

        payload = {
            "meta_info": {
                "version": 3,
                "single_pool_capacity": self.single_pool_capacity,
                "pool_device": self.pool_device,
                "sample_device": self.sample_device,
                "size": self.size,
                "positive_size": self.positive_size,
                "negative_size": self.negative_size,
                "task_count": len(self.task_pools),
            },
            "tasks": tasks_payload,
        }

        torch.save(payload, filepath)
        logger.info(
            f"[Rank {self.rank}] Task replay pool saved to {filepath} with \
               size={self.size}, tasks={len(self.task_pools)}"
        )

    def load(self, directory: str):
        """Load the replay pool from a directory."""

        filepath = f"{directory}/sac_replay_pool_rank_{self.rank}.pt"
        if not os.path.exists(filepath):
            return False

        payload = torch.load(filepath, weights_only=False)
        meta_info = payload["meta_info"]
        tasks_payload = payload["tasks"]

        self.single_pool_capacity = int(meta_info["single_pool_capacity"])
        self.task_pools = {}

        for task_id, task_payload in tasks_payload.items():
            pool_state = _DualPoolState(
                positive_pool=task_payload["positive_pool"].to(self.pool_device),
                negative_pool=task_payload["negative_pool"].to(self.pool_device),
                positive_size=int(task_payload["positive_size"]),
                negative_size=int(task_payload["negative_size"]),
                positive_position=int(task_payload["positive_position"]),
                negative_position=int(task_payload["negative_position"]),
            )
            self.task_pools[task_id] = pool_state

        self._refresh_global_stats()
        logger.info(
            f"[Rank {self.rank}] Task replay pool loaded from {filepath} with \
              size={self.size}, tasks={len(self.task_pools)}"
        )
        return True

    @classmethod
    def from_path(
        cls,
        directory: str,
    ) -> "SACReplayPool":
        """Load a replay pool from a file."""

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        filepath = f"{directory}/sac_replay_pool_rank_{rank}.pt"
        payload = torch.load(filepath, weights_only=False)
        meta_info = payload["meta_info"]

        replay_pool = cls(
            single_pool_capacity=int(meta_info["single_pool_capacity"]),
            pool_device=meta_info["pool_device"],
            sample_device=meta_info["sample_device"],
        )
        replay_pool.rank = rank

        loaded = replay_pool.load(directory)
        if not loaded:
            raise RuntimeError(f"Failed to load replay pool from {filepath}.")

        return replay_pool

    def _insert_block_to_pool(
        self,
        pool_state: _DualPoolState,
        source: TensorDict,
        is_positive_pool: bool,
    ):
        """Insert a block of data from source into one task pool."""

        source_size = source.batch_size[0]
        if source_size == 0:
            return

        length = min(source_size, self.single_pool_capacity)
        idx = torch.arange(length, device=self.pool_device)

        if is_positive_pool:
            assert pool_state.positive_pool is not None
            idx = (pool_state.positive_position + idx) % self.single_pool_capacity
            for key in source.keys():
                pool_state.positive_pool[key].index_copy_(0, idx, source[key][:length].to(self.pool_device))

            pool_state.positive_position = (pool_state.positive_position + length) % self.single_pool_capacity
            pool_state.positive_size = min(pool_state.positive_size + length, self.single_pool_capacity)
        else:
            assert pool_state.negative_pool is not None
            idx = (pool_state.negative_position + idx) % self.single_pool_capacity
            for key in source.keys():
                pool_state.negative_pool[key].index_copy_(0, idx, source[key][:length].to(self.pool_device))

            pool_state.negative_position = (pool_state.negative_position + length) % self.single_pool_capacity
            pool_state.negative_size = min(pool_state.negative_size + length, self.single_pool_capacity)

    def _get_or_create_task_pool(self, task_id: str, sample: TensorDict) -> _DualPoolState:
        if task_id in self.task_pools:
            return self.task_pools[task_id]

        logger.info(
            f"Initializing replay pools for task_id={task_id} with single_pool_capacity={self.single_pool_capacity}"
        )
        pool_template = TensorDict(
            {
                key: torch.zeros(
                    (self.single_pool_capacity, *value.shape[1:]),
                    dtype=value.dtype,
                    device=self.pool_device,
                )
                for key, value in sample.items()
            },
            batch_size=[self.single_pool_capacity],
            device=self.pool_device,
        )
        pool_state = _DualPoolState(
            positive_pool=pool_template.clone(),
            negative_pool=pool_template.clone(),
            positive_size=0,
            negative_size=0,
            positive_position=0,
            negative_position=0,
        )
        self.task_pools[task_id] = pool_state
        return pool_state

    def _extract_positive_mask(self, batch: TensorDict) -> torch.Tensor:
        positive_mask = batch["positive_sample_mask"].to(torch.bool)
        if positive_mask.ndim == 1:
            return positive_mask
        return positive_mask.reshape(positive_mask.shape[0], -1).any(dim=1)

    def _pad_sampled_batch(self, sampled_batch: TensorDict, target_batch_size: int) -> TensorDict:
        current_size = sampled_batch.batch_size[0]
        if current_size >= target_batch_size:
            return sampled_batch

        pad_size = target_batch_size - current_size
        pad_idx = torch.zeros(pad_size, dtype=torch.long, device=self.sample_device)
        padded_batch = TensorDict(
            {key: torch.cat([value, value.index_select(0, pad_idx)], dim=0) for key, value in sampled_batch.items()},
            batch_size=[target_batch_size],
            device=self.sample_device,
        )

        valid_tensor = padded_batch["valids"].clone()
        if valid_tensor.dtype == torch.bool:
            valid_tensor[current_size:] = False
        else:
            valid_tensor[current_size:] = 0
        padded_batch["valids"] = valid_tensor

        return padded_batch

    def _index_select_batch(self, batch: TensorDict, idx: torch.Tensor) -> TensorDict:
        length = int(idx.numel())
        return TensorDict(
            {key: value.index_select(0, idx) for key, value in batch.items()},
            batch_size=[length],
            device=batch.device,
        )

    def _sample_from_task_pools(self, batch_size: int, is_positive_pool: bool) -> TensorDict:
        task_sizes = {
            task_id: (pool_state.positive_size if is_positive_pool else pool_state.negative_size)
            for task_id, pool_state in self.task_pools.items()
            if (pool_state.positive_size if is_positive_pool else pool_state.negative_size) > 0
        }

        allocation = self._allocate_counts_across_tasks(task_sizes, batch_size)

        sampled_parts = []
        for task_id, count in allocation.items():
            if count == 0:
                continue
            sampled_parts.append(self._sample_from_single_task_pool(self.task_pools[task_id], count, is_positive_pool))

        if len(sampled_parts) == 1:
            return sampled_parts[0]

        return TensorDict(
            {key: torch.cat([part[key] for part in sampled_parts], dim=0) for key in sampled_parts[0].keys()},
            batch_size=[batch_size],
            device=self.sample_device,
        )

    def _sample_from_single_task_pool(
        self,
        pool_state: _DualPoolState,
        batch_size: int,
        is_positive_pool: bool,
    ) -> TensorDict:
        pool = pool_state.positive_pool if is_positive_pool else pool_state.negative_pool
        size = pool_state.positive_size if is_positive_pool else pool_state.negative_size
        assert pool is not None

        idx = torch.randperm(size, device=self.pool_device)[:batch_size]
        return TensorDict(
            {key: value.index_select(0, idx).to(self.sample_device) for key, value in pool.items()},
            batch_size=[batch_size],
            device=self.sample_device,
        )

    def _allocate_counts_across_tasks(self, task_sizes: dict[str, int], total_count: int) -> dict[str, int]:
        total_available = sum(task_sizes.values())
        if total_count > total_available:
            raise ValueError(f"Requested {total_count} samples but only {total_available} available across task pools.")

        allocation: dict[str, int] = {task_id: 0 for task_id in task_sizes}
        task_order = list(task_sizes.keys())

        remaining = total_count
        while remaining > 0:
            progressed = False
            for task_id in task_order:
                if allocation[task_id] < task_sizes[task_id]:
                    allocation[task_id] += 1
                    remaining -= 1
                    progressed = True
                    if remaining == 0:
                        break

            if not progressed:
                raise RuntimeError("No eligible task pool left while allocation is still remaining.")

        return allocation

    def _refresh_global_stats(self):
        self.positive_size = sum(state.positive_size for state in self.task_pools.values())
        self.negative_size = sum(state.negative_size for state in self.task_pools.values())
        self.size = self.positive_size + self.negative_size

    def _normalize_task_id(self, task_id: Any) -> str:
        if isinstance(task_id, torch.Tensor):
            task_id = task_id.item()
        return str(task_id)

    def __repr__(self):
        return (
            f"SACReplayPool(single_pool_capacity={self.single_pool_capacity}, size={self.size}, "
            f"positive_size={self.positive_size}, negative_size={self.negative_size}, "
            f"task_count={len(self.task_pools)}, pool_device={self.pool_device}, sample_device={self.sample_device})"
        )

    def __len__(self):
        return self.size
