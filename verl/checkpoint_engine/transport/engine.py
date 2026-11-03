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
"""Checkpoint engine adapter for publish/pull weight sync.

This lets the publisher and puller run inside today's CheckpointEngineManager flow: the trainer's
`send_weights` publishes a version and every rollout worker's `receive_weights` pulls it. There is
no topology and no per-step setup, since endpoints and descriptors travel through the registry.
Rollout-driven pulls without the manager build on the same publisher and puller.
"""

import logging
import os
from typing import AsyncGenerator, Generator

import torch

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry

from .puller import TornReadError, WeightPuller
from .registry import LeaseError, RayWeightRegistry
from .transport import TransportFactory, default_transport_factory

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@CheckpointEngineRegistry.register("transport_pull")
class TransportPullCheckpointEngine(CheckpointEngine):
    """Trainer rank 0 publishes weight versions; rollout workers pull them with one-sided reads.

    Requires torch >= 2.15 for torch.distributed._transport.

    Args:
        bucket_size: Size in bytes of one remote read.
        device: Device of the publisher slots and puller mirror, "cuda" or "cpu".
        backend: torch.distributed._transport backend.
        num_slots: Versions the publisher keeps at once.
        lease_ttl: Reader lease duration in seconds.
        registry_name: Ray actor name of the shared registry.
        wait_timeout: Seconds a rollout worker waits for the requested version to be published.
        is_master: Whether this is trainer rank 0, the only publisher.
        transport_factory: Override of transport construction, for tests.
    """

    def __init__(
        self,
        bucket_size: int,
        device: str = "cuda",
        backend: str = "nixl",
        num_slots: int = 2,
        lease_ttl: float = 60.0,
        registry_name: str = "verl_weight_registry",
        wait_timeout: float = 600.0,
        is_master: bool = False,
        transport_factory: TransportFactory = default_transport_factory,
    ):
        self.bucket_size = bucket_size
        self.device = device
        self.backend = backend
        self.num_slots = num_slots
        self.lease_ttl = lease_ttl
        self.registry_name = registry_name
        self.wait_timeout = wait_timeout
        self.is_master = is_master
        self.transport_factory = transport_factory
        self._registry = None
        self.publisher = None
        self.puller = None
        # Last version published (trainer) or pulled (rollout) by this engine.
        self.last_version = -1

    @property
    def registry(self):
        if self._registry is None:
            self._registry = RayWeightRegistry(name=self.registry_name)
        return self._registry

    def prepare(self) -> dict:
        return {}

    @classmethod
    def build_topology(cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list[dict]):
        return {}, {}

    def init_process_group(self, **kwargs):
        pass

    def finalize(self):
        # Slots and registrations live across steps; that is what makes per-step setup unnecessary.
        pass

    @torch.no_grad()
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ):
        """Publish the weights as version `global_steps` (trainer rank 0 only).

        Without `global_steps` the version is one past the last published one. A version that was
        already published is not published again.
        """
        version = global_steps if global_steps is not None else self.last_version + 1
        if not self.is_master or version <= self.last_version:
            # Every rank still drains the generator: gathering full tensors is collective.
            for _ in weights:
                pass
            return
        if self.publisher is None:
            from .publisher import WeightPublisher

            self.publisher = WeightPublisher(
                self.registry,
                self.bucket_size,
                device=self.device,
                backend=self.backend,
                num_slots=self.num_slots,
                transport_factory=self.transport_factory,
            )
        await self.publisher.publish(weights, version=version)
        self.last_version = version

    @torch.no_grad()
    async def receive_weights(self, global_steps: int | None = None) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """Pull the newest version that is at least `global_steps`, or newer than the last pull."""
        min_version = global_steps if global_steps is not None else self.last_version + 1
        if self.puller is None:
            self.puller = WeightPuller(
                self.registry,
                device=self.device,
                backend=self.backend,
                lease_ttl=self.lease_ttl,
                transport_factory=self.transport_factory,
            )
        while True:
            manifest = await self.puller.wait_for_version(min_version, timeout=self.wait_timeout)
            try:
                pulled = await self.puller.fetch(manifest)
                break
            except (LeaseError, TornReadError) as e:
                # Retired before it could be leased, or the lease lapsed mid-read: a newer version
                # is already READY, so start over from the latest one.
                logger.warning(f"Pull of version {manifest.version} failed, retrying: {e}")
        self.last_version = pulled.version
        for name, tensor in pulled.named_tensors():
            yield name, tensor
