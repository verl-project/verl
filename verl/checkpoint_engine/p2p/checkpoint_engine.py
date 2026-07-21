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

from __future__ import annotations

from typing import Any, AsyncGenerator, Generator

import torch

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry

from .trainer_updater import P2PTrainerWeightUpdater
from .transfer_utils import P2PRolloutTopology


@CheckpointEngineRegistry.register("p2p")
class P2PCheckpointEngine(CheckpointEngine):
    """Miles-style P2P checkpoint engine: trainer CPU replica + Mooncake RDMA writes.

    Rollout metadata is collected on the driver and passed via
    :meth:`connect_rollout_metadata` before :meth:`send_weights`.
    """

    def __init__(
        self,
        bucket_size: int,
        is_master: bool = False,
        **_: Any,
    ) -> None:
        self._bucket_size = bucket_size
        self.is_master = is_master
        self._updater: P2PTrainerWeightUpdater | None = None

    @property
    def is_source(self) -> bool:
        """Whether this trainer rank pushes weights to rollout via P2P RDMA."""
        return self._updater.is_source

    def connect_rollout_metadata(self, rollout_metadata: dict[str, Any]) -> None:
        topology = rollout_metadata.get("rollout_topology")
        model_path = rollout_metadata.get("model_path")
        if topology is None or model_path is None:
            raise RuntimeError("P2PCheckpointEngine requires model_path and rollout_topology in metadata")
        if self._updater is None:
            topology = P2PRolloutTopology(**topology)
            engine_kwargs = rollout_metadata.get("engine_kwargs", {})
            self._updater = P2PTrainerWeightUpdater(
                model_path=model_path,
                rollout_topology=topology,
                num_workers=int(engine_kwargs.get("p2p_transfer_num_workers", 4)),
                transfer_timeout=float(engine_kwargs.get("p2p_transfer_timeout", 30.0)),
                is_master=self.is_master,
                bucket_size_bytes=self._bucket_size,
            )

        self._updater.connect_rollout_metadata(rollout_metadata)

    def prepare(self) -> dict[str, Any]:
        return {"backend": "p2p", "is_master": self.is_master}

    @classmethod
    def build_topology(
        cls,
        trainer_world_size: int,
        rollout_world_size: int,
        metadata: list[dict[str, Any]],
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        del metadata
        trainer_kwargs = {
            "rank": [-1] * trainer_world_size,
            "world_size": [0] * trainer_world_size,
            "metadata": [{}] * trainer_world_size,
        }
        rollout_kwargs = {
            "rank": [-1] * rollout_world_size,
            "world_size": [0] * rollout_world_size,
            "metadata": [{}] * rollout_world_size,
        }
        return trainer_kwargs, rollout_kwargs

    def init_process_group(self, rank: int = -1, world_size: int = 0, metadata: dict[str, Any] | None = None):
        del rank, world_size, metadata

    def finalize(self) -> None:
        return None

    def release_for_checkpoint(self) -> None:
        """Drop trainer-side P2P CPU buffers so Megatron save does not OOM/segfault."""
        if self._updater is not None:
            self._updater.release_for_checkpoint()

    def restore_after_checkpoint(self) -> None:
        """Rebuild trainer-side P2P CPU replica after checkpoint save."""
        if self._updater is not None:
            self._updater.restore_after_checkpoint()

    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ) -> None:
        if self._updater is None:
            raise RuntimeError("P2P backend is not connected; manager must call connect_rollout_metadata first")

        self._updater.send_weights(weights)

    async def receive_weights(self, global_steps: int | None = None) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        if False:
            yield ("", torch.empty(0))
