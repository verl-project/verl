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
"""Weight version registry for publish/pull weight sync.

The registry is a directory, not a store: weights stay in the publisher's registered buffers and
readers fetch them with one-sided reads. The registry only holds, per version, the manifest needed
to issue those reads, plus the leases that keep the publisher from overwriting a version while it
is being read.

Version lifecycle::

    publish(manifest) -> READY -- retire(v) --> RETIRED -- try_reclaim(v), no live lease --> (gone)

A one-sided read is invisible to the publisher, so a reader must hold a lease on a version for the
whole read and check it is still valid afterwards (fencing). Leases expire on the registry's clock,
so a crashed reader cannot pin a version forever.
"""

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import ray
import torch


class VersionState(str, Enum):
    READY = "ready"
    """Readable; new leases may be acquired."""
    RETIRED = "retired"
    """No new leases; buffers are reclaimed once all leases end."""


@dataclass
class TensorSpec:
    name: str
    shape: torch.Size
    dtype: torch.dtype
    offset: int
    """Byte offset of the tensor in the version's contiguous slot buffer."""


@dataclass
class BucketDesc:
    offset: int
    """Byte offset of the bucket in the slot buffer."""
    size: int
    """Size of the bucket in bytes; the unit of one remote read."""
    remote_buffer: bytes
    """Serialized transport descriptor of this bucket on the publisher."""


@dataclass
class WeightManifest:
    version: int
    """Weight version, i.e. the trainer's global step."""
    endpoint: bytes
    """Transport url readers connect to."""
    total_bytes: int
    """Size of the slot buffer; buckets tile [0, total_bytes)."""
    tensors: list[TensorSpec]
    """Every tensor lies contiguously in the slot, possibly across bucket boundaries."""
    buckets: list[BucketDesc]
    extra: dict[str, Any] = field(default_factory=dict)
    """Backend-specific fields, e.g. transport backend name."""


@dataclass
class Lease:
    lease_id: str
    version: int
    expires_at: float
    """Expiry time on the registry's clock."""


class LeaseError(RuntimeError):
    """Raised when a lease cannot be acquired or renewed."""


class WeightRegistry(ABC):
    """Directory of published weight versions and the leases held on them."""

    @abstractmethod
    async def publish(self, manifest: WeightManifest):
        """Make a version READY. Versions must be published in increasing order."""

    @abstractmethod
    async def latest_ready(self) -> WeightManifest | None:
        """Return the manifest of the newest READY version, or None if there is none."""

    @abstractmethod
    async def get_manifest(self, version: int) -> WeightManifest | None:
        """Return the manifest of a READY or RETIRED version, or None if unknown."""

    @abstractmethod
    async def acquire_lease(self, version: int, ttl: float) -> Lease:
        """Lease a READY version for `ttl` seconds. Raises LeaseError otherwise."""

    @abstractmethod
    async def renew_lease(self, lease_id: str, ttl: float) -> Lease:
        """Extend a live lease to `ttl` seconds from now. Raises LeaseError if it already expired."""

    @abstractmethod
    async def release_lease(self, lease_id: str):
        """Release a lease. Releasing an unknown or expired lease is a no-op."""

    @abstractmethod
    async def is_lease_valid(self, lease_id: str) -> bool:
        """Whether the lease is still live; readers check this after a read to fence torn reads."""

    @abstractmethod
    async def retire(self, version: int):
        """Stop handing out leases on a version so its buffers can be reclaimed."""

    @abstractmethod
    async def try_reclaim(self, version: int) -> bool:
        """Forget a RETIRED version if no live lease remains on it.

        Returns:
            True if the publisher may now overwrite the version's buffers.
        """


class InMemoryWeightRegistry(WeightRegistry):
    """Registry held in a single process. Wrap it in a Ray actor with `RayWeightRegistry`.

    Args:
        clock: Monotonic clock used for lease expiry; injectable for tests.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._manifests: dict[int, WeightManifest] = {}
        self._states: dict[int, VersionState] = {}
        self._leases: dict[str, Lease] = {}

    def _expire_leases(self):
        now = self._clock()
        for lease_id in [k for k, lease in self._leases.items() if lease.expires_at <= now]:
            del self._leases[lease_id]

    async def publish(self, manifest: WeightManifest):
        if self._manifests and manifest.version <= max(self._manifests):
            raise ValueError(f"Version {manifest.version} is not newer than {max(self._manifests)}")
        self._manifests[manifest.version] = manifest
        self._states[manifest.version] = VersionState.READY

    async def latest_ready(self) -> WeightManifest | None:
        ready = [v for v, state in self._states.items() if state == VersionState.READY]
        return self._manifests[max(ready)] if ready else None

    async def get_manifest(self, version: int) -> WeightManifest | None:
        return self._manifests.get(version)

    async def acquire_lease(self, version: int, ttl: float) -> Lease:
        if self._states.get(version) != VersionState.READY:
            raise LeaseError(f"Version {version} is not READY: {self._states.get(version)}")
        lease = Lease(lease_id=uuid.uuid4().hex, version=version, expires_at=self._clock() + ttl)
        self._leases[lease.lease_id] = lease
        return lease

    async def renew_lease(self, lease_id: str, ttl: float) -> Lease:
        # Renewal is allowed on a RETIRED version: the reader may be mid-read when the publisher
        # retires it, and the publisher must wait for that read either way.
        self._expire_leases()
        if lease_id not in self._leases:
            raise LeaseError(f"Lease {lease_id} expired or unknown")
        lease = self._leases[lease_id]
        lease.expires_at = self._clock() + ttl
        return lease

    async def release_lease(self, lease_id: str):
        self._leases.pop(lease_id, None)

    async def is_lease_valid(self, lease_id: str) -> bool:
        self._expire_leases()
        return lease_id in self._leases

    async def retire(self, version: int):
        if version not in self._states:
            raise KeyError(f"Version {version} is not published")
        self._states[version] = VersionState.RETIRED

    async def try_reclaim(self, version: int) -> bool:
        if version not in self._states:
            return True
        if self._states[version] != VersionState.RETIRED:
            return False
        self._expire_leases()
        if any(lease.version == version for lease in self._leases.values()):
            return False
        del self._states[version]
        del self._manifests[version]
        return True


class RayWeightRegistry(WeightRegistry):
    """Client of an `InMemoryWeightRegistry` running as a named Ray actor.

    Every process that constructs it with the same name shares one registry, so the trainer and
    rollout replicas only need to agree on the name. The actor lives as long as the Ray job, so a
    restarted job does not see versions whose publisher endpoints are gone.

    Args:
        name: Ray actor name of the registry.
        namespace: Ray namespace of the registry actor; defaults to the current one.
    """

    def __init__(self, name: str = "verl_weight_registry", namespace: str | None = None):
        self._actor = (
            ray.remote(InMemoryWeightRegistry)
            .options(name=name, namespace=namespace, get_if_exists=True, num_cpus=0)
            .remote()
        )

    async def publish(self, manifest: WeightManifest):
        await self._actor.publish.remote(manifest)

    async def latest_ready(self) -> WeightManifest | None:
        return await self._actor.latest_ready.remote()

    async def get_manifest(self, version: int) -> WeightManifest | None:
        return await self._actor.get_manifest.remote(version)

    async def acquire_lease(self, version: int, ttl: float) -> Lease:
        return await self._actor.acquire_lease.remote(version, ttl)

    async def renew_lease(self, lease_id: str, ttl: float) -> Lease:
        return await self._actor.renew_lease.remote(lease_id, ttl)

    async def release_lease(self, lease_id: str):
        await self._actor.release_lease.remote(lease_id)

    async def is_lease_valid(self, lease_id: str) -> bool:
        return await self._actor.is_lease_valid.remote(lease_id)

    async def retire(self, version: int):
        await self._actor.retire.remote(version)

    async def try_reclaim(self, version: int) -> bool:
        return await self._actor.try_reclaim.remote(version)
