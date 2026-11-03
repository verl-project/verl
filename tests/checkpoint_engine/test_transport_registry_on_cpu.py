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

import asyncio

import pytest
import ray

from verl.checkpoint_engine.transport import (
    InMemoryWeightRegistry,
    LeaseError,
    RayWeightRegistry,
    VersionState,
    WeightManifest,
)


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _manifest(version):
    return WeightManifest(version=version, endpoint=f"ep-{version}".encode(), total_bytes=0, tensors=[], buckets=[])


def _run(coro):
    return asyncio.run(coro)


def test_publish_and_latest_ready():
    async def main():
        registry = InMemoryWeightRegistry()
        assert await registry.latest_ready() is None
        await registry.publish(_manifest(1))
        await registry.publish(_manifest(2))
        assert (await registry.latest_ready()).version == 2
        with pytest.raises(ValueError):
            await registry.publish(_manifest(2))

        await registry.retire(2)
        assert (await registry.latest_ready()).version == 1
        assert (await registry.get_manifest(2)).endpoint == b"ep-2"
        assert registry._states[2] == VersionState.RETIRED

    _run(main())


def test_lease_blocks_reclaim_until_released():
    async def main():
        registry = InMemoryWeightRegistry(clock=_FakeClock())
        await registry.publish(_manifest(1))
        lease = await registry.acquire_lease(1, ttl=10)

        # READY versions are never reclaimed; retired ones only once the lease ends.
        assert not await registry.try_reclaim(1)
        await registry.retire(1)
        with pytest.raises(LeaseError):
            await registry.acquire_lease(1, ttl=10)
        assert not await registry.try_reclaim(1)

        await registry.release_lease(lease.lease_id)
        assert await registry.try_reclaim(1)
        assert await registry.get_manifest(1) is None
        assert await registry.try_reclaim(1)

    _run(main())


def test_lease_expiry_reclaims_crashed_reader():
    async def main():
        clock = _FakeClock()
        registry = InMemoryWeightRegistry(clock=clock)
        await registry.publish(_manifest(1))
        lease = await registry.acquire_lease(1, ttl=10)
        await registry.retire(1)

        # Renewal during a long read keeps the version pinned past the original ttl.
        clock.now = 8
        await registry.renew_lease(lease.lease_id, ttl=10)
        clock.now = 15
        assert await registry.is_lease_valid(lease.lease_id)
        assert not await registry.try_reclaim(1)

        # The reader stops renewing, e.g. it crashed: the lease expires on the registry's clock.
        clock.now = 18
        assert not await registry.is_lease_valid(lease.lease_id)
        with pytest.raises(LeaseError):
            await registry.renew_lease(lease.lease_id, ttl=10)
        assert await registry.try_reclaim(1)

    _run(main())


def test_fencing_detects_expired_lease_after_read():
    async def main():
        clock = _FakeClock()
        registry = InMemoryWeightRegistry(clock=clock)
        await registry.publish(_manifest(1))
        lease = await registry.acquire_lease(1, ttl=5)
        await registry.retire(1)

        # A read that outlives its lease may have raced with the publisher overwriting the buffers.
        clock.now = 6
        assert await registry.try_reclaim(1)
        assert not await registry.is_lease_valid(lease.lease_id)

    _run(main())


def test_ray_registry_shared_by_name():
    ray.init(num_cpus=1, namespace="test_transport_registry")
    try:

        async def main():
            publisher = RayWeightRegistry(name="test_registry")
            reader = RayWeightRegistry(name="test_registry")
            await publisher.publish(_manifest(3))
            manifest = await reader.latest_ready()
            assert manifest.version == 3 and manifest.endpoint == b"ep-3"

            lease = await reader.acquire_lease(3, ttl=60)
            await publisher.retire(3)
            assert not await publisher.try_reclaim(3)
            await reader.release_lease(lease.lease_id)
            assert await publisher.try_reclaim(3)

        _run(main())
    finally:
        ray.shutdown()
