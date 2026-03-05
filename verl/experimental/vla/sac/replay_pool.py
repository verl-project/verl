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
from typing import Optional

import torch
from tensordict import TensorDict

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SACReplayPool:
    """SAC Replay Pool for storing samples."""

    def __init__(
        self,
        capacity: int,
        pool_device: str = "cpu",
        sample_device: str = "cpu",
    ):
        self.positive_pool: Optional[TensorDict] = None
        self.negative_pool: Optional[TensorDict] = None
        self.capacity = capacity

        self.size = 0
        self.positive_size = 0
        self.negative_size = 0
        self.positive_position = 0
        self.negative_position = 0

        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        self.pool_device = pool_device
        self.sample_device = sample_device

    def add_batch(self, batch: TensorDict):
        """Add a batch of samples to the replay pool.

        Args:
            batch (TensorDict): A batch of samples to add. The batch should be a TensorDict
                containing the necessary keys for SAC training, each with shape [batch_size, ...].
        """

        if self.positive_pool is None or self.negative_pool is None:
            self._lazy_init_pool(batch)

        positive_mask = self._extract_positive_mask(batch)
        positive_idx = torch.nonzero(positive_mask, as_tuple=False).squeeze(-1)
        negative_idx = torch.nonzero(~positive_mask, as_tuple=False).squeeze(-1)

        if positive_idx.numel() > 0:
            positive_batch = self._index_select_batch(batch, positive_idx)
            self._insert_block_to_pool(positive_batch, is_positive_pool=True)

        if negative_idx.numel() > 0:
            negative_batch = self._index_select_batch(batch, negative_idx)
            self._insert_block_to_pool(negative_batch, is_positive_pool=False)

        self.size = self.positive_size + self.negative_size

    def sample_batch(
        self,
        batch_size: int,
        positive_sample_ratio: float = 0.5,
        return_sample_info: bool = False,
    ) -> TensorDict | tuple[TensorDict, dict]:
        """Sample a batch of experiences from the replay pool.

        Args:
            batch_size (int): The number of samples to draw.

        Returns:
            TensorDict: A batch of sampled experiences.
        """

        assert self.size >= batch_size, "Not enough samples in the replay pool to sample the requested batch size."

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

        assert deficit == 0, "Unable to sample enough data from replay pool."

        sampled_parts = []
        if sampled_positive > 0:
            sampled_parts.append(self._sample_from_single_pool(sampled_positive, is_positive_pool=True))
        if sampled_negative > 0:
            sampled_parts.append(self._sample_from_single_pool(sampled_negative, is_positive_pool=False))

        if len(sampled_parts) == 1:
            sampled_batch = sampled_parts[0]
        else:
            sampled_batch = TensorDict(
                {
                    key: torch.cat([part[key] for part in sampled_parts], dim=0)
                    for key in sampled_parts[0].keys()
                },
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
            "requested_positive_sample_ratio": positive_sample_ratio,
            "actual_positive_sample_ratio": sampled_positive / max(batch_size, 1),
            "sampled_positive_count": sampled_positive,
            "sampled_negative_count": sampled_negative,
        }
        return sampled_batch, sample_info

    def insert_and_resample(
        self,
        source: TensorDict,
    ) -> TensorDict:
        """Insert a block of data from source to the replay pool and sample a batch with the same size."""

        self.add_batch(source)
        return self.sample_batch(source.batch_size[0])

    def save(self, directory: str):
        """Save the replay pool to a directory."""

        os.makedirs(directory, exist_ok=True)

        filepath = f"{directory}/sac_replay_pool_rank_{self.rank}.pt"
        if self.positive_pool is not None or self.negative_pool is not None:
            meta_info = {
                "version": 2,
                "size": self.size,
                "capacity": self.capacity,
                "positive_size": self.positive_size,
                "negative_size": self.negative_size,
                "positive_position": self.positive_position,
                "negative_position": self.negative_position,
                "pool_device": self.pool_device,
                "sample_device": self.sample_device,
            }
            payload = {
                "positive_pool": self.positive_pool.cpu() if self.positive_pool is not None else None,
                "negative_pool": self.negative_pool.cpu() if self.negative_pool is not None else None,
            }
            torch.save((payload, meta_info), filepath)
            logger.info(f"[Rank {self.rank}] Replay pool saved to {filepath} with size: {self.size}")
        else:
            logger.info("Replay pool is empty. Nothing to save.")

    def load(self, directory: str):
        """Load the replay pool from a directory."""

        filepath = f"{directory}/sac_replay_pool_rank_{self.rank}.pt"
        if not os.path.exists(filepath):
            return False

        try:
            payload, meta_info = torch.load(filepath, weights_only=False)
        except (RuntimeError, EOFError, ValueError) as exc:
            logger.warning(
                f"[Rank {self.rank}] Failed to load replay pool from {filepath}: {exc}. "
                "Starting with an empty replay pool."
            )
            return False

        loaded_capacity = meta_info.get("capacity", self.capacity)
        positive_capacity = meta_info.get("positive_capacity", loaded_capacity)
        negative_capacity = meta_info.get("negative_capacity", loaded_capacity)

        loaded_positive_pool = payload.get("positive_pool", None)
        loaded_negative_pool = payload.get("negative_pool", None)

        if loaded_positive_pool is not None:
            loaded_positive_pool = loaded_positive_pool.to(self.pool_device)
        if loaded_negative_pool is not None:
            loaded_negative_pool = loaded_negative_pool.to(self.pool_device)

        self.positive_pool = self._resize_loaded_pool(loaded_positive_pool, positive_capacity)
        self.negative_pool = self._resize_loaded_pool(loaded_negative_pool, negative_capacity)

        if self.positive_pool is None and self.negative_pool is None:
            logger.info(f"[Rank {self.rank}] Replay pool file exists but contains no data.")
            return True

        if self.positive_pool is None:
            self.positive_pool = self._create_empty_pool_like(self.negative_pool)
        if self.negative_pool is None:
            self.negative_pool = self._create_empty_pool_like(self.positive_pool)

        self.positive_size = min(meta_info.get("positive_size", 0), self.capacity)
        self.negative_size = min(meta_info.get("negative_size", 0), self.capacity)
        self.positive_position = meta_info.get("positive_position", self.positive_size) % self.capacity
        self.negative_position = meta_info.get("negative_position", self.negative_size) % self.capacity
        self.size = self.positive_size + self.negative_size

        logger.info(
            f"[Rank {self.rank}] Replay pool loaded from {filepath} with size: {self.size} "
            f"(pos={self.positive_size}, neg={self.negative_size})"
        )

        return True

    @classmethod
    def from_path(
        cls,
        directory: str,
    ) -> "SACReplayPool":
        """Load a replay pool from a file.

        Args:
            directory (str): The directory containing the saved replay pool.
        Returns:
            SACReplayPool: An instance of SACReplayPool with the loaded data.
        """
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        filepath = f"{directory}/sac_replay_pool_rank_{rank}.pt"
        payload, meta_info = torch.load(filepath, weights_only=False)

        loaded_capacity = meta_info.get("capacity", None)
        if loaded_capacity is None:
            if isinstance(payload, TensorDict):
                loaded_capacity = payload.batch_size[0]
            elif isinstance(payload, dict):
                for key in ["positive_pool", "negative_pool"]:
                    value = payload.get(key, None)
                    if value is not None:
                        loaded_capacity = value.batch_size[0]
                        break
        if loaded_capacity is None:
            raise ValueError(f"Cannot determine replay pool capacity from {filepath}.")

        replay_pool = cls(
            capacity=loaded_capacity,
            pool_device=meta_info["pool_device"],
            sample_device=meta_info["sample_device"],
        )
        replay_pool.rank = rank

        loaded = replay_pool.load(directory)
        if not loaded:
            raise RuntimeError(f"Failed to load replay pool from {filepath}.")

        logger.info(
            f"[Rank {rank}] Replay pool loaded from {filepath} with size: {replay_pool.size} "
            f"(pos={replay_pool.positive_size}, neg={replay_pool.negative_size})"
        )
        return replay_pool

    def _insert_block_to_pool(
        self,
        source: TensorDict,
        is_positive_pool: bool,
    ):
        """insert a block of data from source to the replay pool."""

        source_size = source.batch_size[0]
        if source_size == 0:
            return

        length = min(source_size, self.capacity)
        idx = torch.arange(length, device=self.pool_device)

        if is_positive_pool:
            assert self.positive_pool is not None
            idx = (self.positive_position + idx) % self.capacity
            for key in source.keys():
                self.positive_pool[key].index_copy_(0, idx, source[key][:length].to(self.pool_device))

            self.positive_position = (self.positive_position + length) % self.capacity
            self.positive_size = min(self.positive_size + length, self.capacity)
        else:
            assert self.negative_pool is not None
            idx = (self.negative_position + idx) % self.capacity
            for key in source.keys():
                self.negative_pool[key].index_copy_(0, idx, source[key][:length].to(self.pool_device))

            self.negative_position = (self.negative_position + length) % self.capacity
            self.negative_size = min(self.negative_size + length, self.capacity)

    def _lazy_init_pool(self, sample: TensorDict):
        """Lazily initialize the replay pool based on the sample structure."""

        logger.info(f"Initializing dual replay pools with capacity: {self.capacity} per pool")

        pool_template = TensorDict(
            {
                key: torch.zeros((self.capacity, *value.shape[1:]), dtype=value.dtype, device=self.pool_device)
                for key, value in sample.items()
            },
            batch_size=[self.capacity],
            device=self.pool_device,
        )
        self.positive_pool = pool_template.clone()
        self.negative_pool = pool_template.clone()

        self.size = 0
        self.positive_size = 0
        self.negative_size = 0
        self.positive_position = 0
        self.negative_position = 0

    def _extract_positive_mask(self, batch: TensorDict) -> torch.Tensor:
        if "positive_sample_mask" not in batch.keys():
            raise KeyError("`positive_sample_mask` is required in batch for dual replay pool insertion.")

        positive_mask = batch["positive_sample_mask"].to(torch.bool)
        if positive_mask.ndim == 1:
            return positive_mask
        return positive_mask.reshape(positive_mask.shape[0], -1).any(dim=1)

    def _index_select_batch(self, batch: TensorDict, idx: torch.Tensor) -> TensorDict:
        length = int(idx.numel())
        return TensorDict(
            {key: value.index_select(0, idx) for key, value in batch.items()},
            batch_size=[length],
            device=batch.device,
        )

    def _sample_from_single_pool(self, batch_size: int, is_positive_pool: bool) -> TensorDict:
        pool = self.positive_pool if is_positive_pool else self.negative_pool
        size = self.positive_size if is_positive_pool else self.negative_size
        assert pool is not None

        idx = torch.randperm(size, device=self.pool_device)[:batch_size]
        return TensorDict(
            {key: value.index_select(0, idx).to(self.sample_device) for key, value in pool.items()},
            batch_size=[batch_size],
            device=self.sample_device,
        )

    def _resize_loaded_pool(self, pool: Optional[TensorDict], loaded_capacity: int) -> Optional[TensorDict]:
        if pool is None:
            return None

        loaded_pool = pool.to(self.pool_device)
        if loaded_capacity == self.capacity:
            return loaded_pool

        if loaded_capacity > self.capacity:
            logger.warning(
                f"Loaded replay pool capacity {loaded_capacity} is greater than "
                f"the current capacity {self.capacity}. Truncating loaded pool."
            )
            return TensorDict(
                {key: value[: self.capacity] for key, value in loaded_pool.items()},
                batch_size=[self.capacity],
                device=self.pool_device,
            )

        logger.warning(
            f"Loaded replay pool capacity {loaded_capacity} is less than "
            f"the current capacity {self.capacity}. Padding loaded pool."
        )
        return TensorDict(
            {
                key: torch.cat(
                    [
                        value,
                        torch.zeros(
                            (self.capacity - loaded_capacity, *value.shape[1:]),
                            dtype=value.dtype,
                            device=self.pool_device,
                        ),
                    ],
                    dim=0,
                )
                for key, value in loaded_pool.items()
            },
            batch_size=[self.capacity],
            device=self.pool_device,
        )

    def _create_empty_pool_like(self, reference_pool: TensorDict) -> TensorDict:
        return TensorDict(
            {
                key: torch.zeros((self.capacity, *value.shape[1:]), dtype=value.dtype, device=self.pool_device)
                for key, value in reference_pool.items()
            },
            batch_size=[self.capacity],
            device=self.pool_device,
        )

    def __repr__(self):
        return (
            f"SACReplayPool(capacity={self.capacity}, "
            f"size={self.size}, positive_size={self.positive_size}, negative_size={self.negative_size}, "
            f"pool_device={self.pool_device}, sample_device={self.sample_device})"
        )

    def __len__(self):
        return self.size
