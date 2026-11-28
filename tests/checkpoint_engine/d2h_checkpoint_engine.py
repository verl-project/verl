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
"""Test-only asynchronous D2H checkpoint engine.

The trainer publishes an immutable CPU snapshot and can resume training once
the D2H copies complete. Rollout fetches that snapshot independently through a
Ray actor. The actor stands in for trainer-owned, RDMA-registered host memory;
this backend is intentionally a correctness test double, not a data path for
production use.
"""

import asyncio
from collections import defaultdict
from collections.abc import AsyncGenerator, Generator
from typing import Any

import ray
import torch

from verl.checkpoint_engine import CheckpointEngine, CheckpointEngineRegistry


@ray.remote
class _SnapshotStore:
    def __init__(self):
        self._snapshots: dict[int | None, list[tuple[str, torch.Tensor]]] = {}
        self._published: defaultdict[int | None, asyncio.Event] = defaultdict(asyncio.Event)

    async def publish(self, version: int | None, snapshot: list[tuple[str, torch.Tensor]]) -> None:
        if version in self._snapshots:
            raise ValueError(f"Weight version {version!r} has already been published")
        self._snapshots[version] = snapshot
        self._published[version].set()

    async def get(self, version: int | None) -> list[tuple[str, torch.Tensor]]:
        await self._published[version].wait()
        return self._snapshots[version]


@CheckpointEngineRegistry.register("test_d2h")
class TestD2HCheckpointEngine(CheckpointEngine):
    """Publish trainer weights to CPU before rollout starts consuming them."""

    __test__ = False
    support_publish_weights = True

    def __init__(
        self,
        bucket_size: int,
        is_master: bool = False,
        receive_delay: float = 0.0,
    ) -> None:
        del bucket_size  # The test double snapshots tensors individually.
        self.is_master = is_master
        self.receive_delay = receive_delay
        self.store = None
        self.is_trainer = False

    def prepare(self) -> dict[str, Any]:
        store = _SnapshotStore.remote() if self.is_master else None
        return {"store": store}

    @classmethod
    def build_topology(
        cls,
        actor_wg_world_size: int,
        rollout_world_size: int,
        metadata: list[dict[str, Any]],
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        store = metadata[0]["store"]
        if store is None:
            raise RuntimeError("Actor rank 0 must own the D2H snapshot store")

        actor_kwargs = {
            "store": [store] * actor_wg_world_size,
            "is_trainer": [True] * actor_wg_world_size,
        }
        rollout_kwargs = {
            "store": [store] * rollout_world_size,
            "is_trainer": [False] * rollout_world_size,
        }
        return actor_kwargs, rollout_kwargs

    def init_process_group(self, store, is_trainer: bool) -> None:
        self.store = store
        self.is_trainer = is_trainer

    def finalize(self) -> None:
        self.store = None
        self.is_trainer = False

    @torch.no_grad()
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ) -> None:
        if not self.is_trainer:
            raise RuntimeError("Only trainer workers may publish D2H snapshots")

        # Every trainer rank must drain the generator because producing a full
        # parameter can itself involve collectives. Only rank 0 retains a copy.
        snapshot = []
        copied_from_cuda = False
        for name, weight in weights:
            if not self.is_master:
                continue
            if weight.device.type == "cuda":
                cpu_weight = torch.empty_like(weight, device="cpu", pin_memory=True)
                cpu_weight.copy_(weight.detach(), non_blocking=True)
                copied_from_cuda = True
            else:
                cpu_weight = weight.detach().to(device="cpu", copy=True)
            snapshot.append((name, cpu_weight))

        if not self.is_master:
            return
        if copied_from_cuda:
            torch.cuda.current_stream().synchronize()
        await self.store.publish.remote(global_steps, snapshot)

    @torch.no_grad()
    async def receive_weights(
        self,
        global_steps: int | None = None,
    ) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        if self.is_trainer:
            raise RuntimeError("Trainer workers may not consume D2H snapshots")
        if self.receive_delay:
            await asyncio.sleep(self.receive_delay)
        snapshot = await self.store.get.remote(global_steps)
        for name, weight in snapshot:
            yield name, weight
