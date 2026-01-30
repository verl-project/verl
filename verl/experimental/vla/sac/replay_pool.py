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
        self.pool = None
        self.capacity = capacity
        self.size = 0
        self.position = 0
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        self.pool_device = pool_device
        self.sample_device = sample_device

    def add_batch(self, batch: TensorDict):
        """Add a batch of samples to the replay pool.

        Args:
            batch (TensorDict): A batch of samples to add. The batch should be a TensorDict
                containing the necessary keys for SAC training, each with shape [batch_size, ...].
        """

        if self.pool is None:
            self._lazy_init_pool(batch)

        self._insert_block_to_pool(batch)

    def sample_batch(self, batch_size: int) -> TensorDict:
        """Sample a batch of experiences from the replay pool.

        Args:
            batch_size (int): The number of samples to draw.

        Returns:
            TensorDict: A batch of sampled experiences.
        """

        assert self.size >= batch_size, "Not enough samples in the replay pool to sample the requested batch size."

        idx = torch.randperm(self.size)[:batch_size]
        sampled_batch = TensorDict(
            {key: value.index_select(0, idx).to(self.sample_device) for key, value in self.pool.items()},
            batch_size=[batch_size],
            device=self.sample_device,
        )
        return sampled_batch

    def insert_and_resample(
        self,
        source: TensorDict,
    ) -> TensorDict:
        """Insert a block of data from source to the replay pool and sample a batch with the same size."""

        self.add_batch(source)
        return self.sample_batch(source.size(0))

    def save(self, directory: str):
        """Save the replay pool to a directory."""

        os.makedirs(directory, exist_ok=True)

        filepath = f"{directory}/sac_replay_pool_rank_{self.rank}.pt"
        if self.pool is not None:
            meta_info = {
                "size": self.size,
                "capacity": self.capacity,
                "position": self.position,
                "pool_device": self.pool_device,
                "sample_device": self.sample_device,
            }
            torch.save((self.pool.cpu(), meta_info), filepath)
            logger.info(f"[Rank {self.rank}] Replay pool saved to {filepath} with size: {self.size}")
        else:
            logger.info("Replay pool is empty. Nothing to save.")

    def load(self, directory: str):
        """Load the replay pool from a directory."""

        filepath = f"{directory}/sac_replay_pool_rank_{self.rank}.pt"
        if not os.path.exists(filepath):
            return False

        try:
            pool, meta_info = torch.load(filepath, weights_only=False)
        except (RuntimeError, EOFError, ValueError) as exc:
            logger.warning(
                f"[Rank {self.rank}] Failed to load replay pool from {filepath}: {exc}. "
                "Starting with an empty replay pool."
            )
            return False
        self.pool = pool.to(self.pool_device)

        if meta_info["capacity"] != self.capacity:
            if meta_info["capacity"] > self.capacity:
                logger.warning(
                    f"Loaded replay pool capacity {meta_info['capacity']} is greater than "
                    f"the current capacity {self.capacity}. Truncating the loaded pool."
                )
                self.pool = TensorDict(
                    {key: value[: self.capacity] for key, value in pool.items()},
                    batch_size=[self.capacity],
                    device=self.pool_device,
                )
                self.size = min(self.size, self.capacity)
                self.position = self.position % self.capacity
            else:
                logger.warning(
                    f"Loaded replay pool capacity {meta_info['capacity']} is less than "
                    f"the current capacity {self.capacity}. Keeping the current capacity."
                )

                self.pool = TensorDict(
                    {
                        key: torch.cat(
                            [
                                value,
                                torch.zeros(
                                    (self.capacity - meta_info["capacity"], *value.shape[1:]),
                                    dtype=value.dtype,
                                    device=self.pool_device,
                                ),
                            ],
                            dim=0,
                        )
                        for key, value in pool.items()
                    },
                    batch_size=[self.capacity],
                    device=self.pool_device,
                )

        self.size = min(meta_info["size"], self.capacity)
        self.position = meta_info["position"] % self.capacity

        logger.info(f"[Rank {self.rank}] Replay pool loaded from {filepath} with size: {self.size}")

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
        pool, meta_info = torch.load(filepath, weights_only=False)

        replay_pool = cls(
            capacity=meta_info["capacity"],
            pool_device=meta_info["pool_device"],
            sample_device=meta_info["sample_device"],
        )
        replay_pool.pool = pool.to(replay_pool.pool_device)
        replay_pool.rank = rank
        replay_pool.size = meta_info["size"]
        replay_pool.position = meta_info["position"]
        logger.info(f"[Rank {rank}] Replay pool loaded from {filepath} with size: {replay_pool.size}")
        return replay_pool

    def _insert_block_to_pool(
        self,
        source: TensorDict,
    ):
        """insert a block of data from source to the replay pool."""

        length = min(source.size(0), self.capacity)
        idx = (self.position + torch.arange(length)) % self.capacity
        for key in source.keys():
            self.pool[key].index_copy_(0, idx, source[key][:length].to(self.pool_device))

        self.position = (self.position + length) % self.capacity
        self.size = min(self.size + length, self.capacity)

    def _lazy_init_pool(self, sample: TensorDict):
        """Lazily initialize the replay pool based on the sample structure."""

        logger.info(f"Initializing replay pool with capacity: {self.capacity}")

        self.pool = TensorDict(
            {
                key: torch.zeros((self.capacity, *value.shape[1:]), dtype=value.dtype, device=self.pool_device)
                for key, value in sample.items()
            },
            batch_size=[self.capacity],
            device=self.pool_device,
        )

    def __repr__(self):
        return (
            f"SACReplayPool(capacity={self.capacity}, "
            f"size={self.size}, pool_device={self.pool_device}, sample_device={self.sample_device})"
        )

    def __len__(self):
        return self.size
