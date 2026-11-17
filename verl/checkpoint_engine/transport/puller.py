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
"""Rollout-side puller of published weight versions."""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Iterator

import torch

from .registry import LeaseError, TensorSpec, WeightManifest, WeightRegistry
from .transport import (
    TransportFactory,
    alloc_slot_buffer,
    bucket_ranges,
    default_transport_factory,
    same_layout,
    tensor_view,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TornReadError(RuntimeError):
    """The lease ended before the read finished, so the publisher may have overwritten the data."""


@dataclass
class PulledWeights:
    version: int
    buffer: torch.Tensor
    tensors: list[TensorSpec]

    def named_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Views into the puller's local buffer, valid until the next fetch."""
        for spec in self.tensors:
            yield spec.name, tensor_view(self.buffer, spec)


class WeightPuller:
    """Fetches a whole published version into a local mirror of the publisher's slot.

    The whole version lands locally before any of it is used, so a read that turns out to be torn
    (lease lost mid-read) is discarded instead of half-applied. The local mirror and its
    registrations are reused while the publisher's endpoint and layout stay the same.

    Args:
        registry: Where versions are discovered and leased.
        device: Device of the local mirror, "cuda" or "cpu" (pinned host memory).
        backend: torch.distributed._transport backend.
        lease_ttl: Lease duration in seconds; renewed every lease_ttl / 3 during a fetch.
        poll_interval: Seconds between registry polls in `wait_for_version`.
        transport_factory: Override of transport construction, for tests.
    """

    def __init__(
        self,
        registry: WeightRegistry,
        device: str = "cuda",
        backend: str = "nixl",
        lease_ttl: float = 60.0,
        poll_interval: float = 0.1,
        transport_factory: TransportFactory = default_transport_factory,
    ):
        self.registry = registry
        self.device = device
        self.backend = backend
        self.lease_ttl = lease_ttl
        self.poll_interval = poll_interval
        self.transport_factory = transport_factory

        self.transport = None
        self.endpoint = None
        self.buffer = None
        self.ranges = None
        self.memories = []

    async def wait_for_version(self, min_version: int = 0, timeout: float | None = None) -> WeightManifest:
        """Return the newest READY manifest once its version is at least `min_version`."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            manifest = await self.registry.latest_ready()
            if manifest is not None and manifest.version >= min_version:
                return manifest
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(f"No version >= {min_version} published within {timeout}s")
            await asyncio.sleep(self.poll_interval)

    def _setup(self, manifest: WeightManifest):
        ranges = [(b.offset, b.size) for b in manifest.buckets]
        if manifest.endpoint == self.endpoint and same_layout(manifest.buckets, self.ranges or []):
            return
        self.close()
        assert ranges == bucket_ranges(manifest.total_bytes, manifest.buckets[0].size), "Buckets must tile the slot"
        self.buffer = alloc_slot_buffer(manifest.total_bytes, self.device)
        self.transport = self.transport_factory(self.backend, self.device)
        self.memories = [self.transport.register_memory(self.buffer[offset : offset + size]) for offset, size in ranges]
        self.transport.connect(manifest.endpoint)
        self.endpoint, self.ranges = manifest.endpoint, ranges

    async def _renew(self, lease_id: str):
        while True:
            await asyncio.sleep(self.lease_ttl / 3)
            await self.registry.renew_lease(lease_id, self.lease_ttl)

    async def fetch(self, manifest: WeightManifest) -> PulledWeights:
        """Read all buckets of `manifest` under a lease.

        Raises:
            LeaseError: The version was retired before the lease was acquired; fetch a newer one.
            TornReadError: The lease was lost during the read; the local copy must not be used.
        """
        start = time.time()
        self._setup(manifest)
        lease = await self.registry.acquire_lease(manifest.version, self.lease_ttl)
        renewer = asyncio.create_task(self._renew(lease.lease_id))
        try:
            remote_cls = type(self.memories[0].to_remote_buffer())
            await asyncio.gather(
                *[
                    self.transport.read_async(memory.to_mutable_view(), remote_cls.deserialize(bucket.remote_buffer))
                    for memory, bucket in zip(self.memories, manifest.buckets, strict=True)
                ]
            )
            valid = await self.registry.is_lease_valid(lease.lease_id)
        finally:
            renewer.cancel()
            try:
                await renewer
            except (asyncio.CancelledError, LeaseError):
                pass
            # Release even when a read failed: the publisher would otherwise wait out the lease before it
            # can reuse the slot, and a failed fetch is never used.
            await self.registry.release_lease(lease.lease_id)
        if not valid:
            raise TornReadError(f"Lease on version {manifest.version} expired during the read")

        elapsed = time.time() - start
        logger.info(
            f"Pulled version {manifest.version}: {manifest.total_bytes / 2**30:.2f} GiB in {elapsed:.2f}s, "
            f"{manifest.total_bytes / elapsed / 2**30:.2f} GiB/s"
        )
        return PulledWeights(version=manifest.version, buffer=self.buffer, tensors=manifest.tensors)

    def close(self):
        for memory in self.memories:
            self.transport.unregister_memory(memory)
        self.memories = []
        if self.transport is not None:
            self.transport.close()
        self.transport, self.endpoint, self.buffer, self.ranges = None, None, None, None
