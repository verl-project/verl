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
"""Publisher/puller logic against an in-process fake of torch.distributed._transport."""

import asyncio
import pickle
import uuid

import pytest
import torch

from verl.checkpoint_engine.transport import InMemoryWeightRegistry
from verl.checkpoint_engine.transport.engine import TransportPullCheckpointEngine
from verl.checkpoint_engine.transport.publisher import WeightPublisher
from verl.checkpoint_engine.transport.puller import TornReadError, WeightPuller
from verl.checkpoint_engine.transport.transport import TENSOR_ALIGNMENT

_ENDPOINTS: dict[bytes, "_FakeTransport"] = {}


class _FakeRemoteBuffer:
    def __init__(self, url, key, size):
        self.url, self.key, self.size = url, key, size

    def serialize(self) -> bytes:
        return pickle.dumps((self.url, self.key, self.size))

    @classmethod
    def deserialize(cls, data: bytes):
        return cls(*pickle.loads(data))


class _FakeMemory:
    def __init__(self, transport, tensor):
        self.transport, self.tensor, self.key = transport, tensor, uuid.uuid4().hex

    def to_remote_buffer(self):
        return _FakeRemoteBuffer(self.transport.url, self.key, self.tensor.numel())

    def to_mutable_view(self):
        return self.tensor


class _FakeTransport:
    """One-sided reads copy straight out of the peer's registered tensors."""

    def __init__(self, backend, device):
        self.url, self.peer, self.memories = None, None, {}
        self.before_read = None

    def bind(self):
        self.url = uuid.uuid4().hex.encode()
        _ENDPOINTS[self.url] = self
        return self.url

    def connect(self, url):
        self.peer = url

    def register_memory(self, tensor):
        memory = _FakeMemory(self, tensor)
        self.memories[memory.key] = memory
        return memory

    def unregister_memory(self, memory):
        del self.memories[memory.key]

    async def read_async(self, local, remote):
        assert remote.url == self.peer, "read from an endpoint that was not connected"
        source = _ENDPOINTS[remote.url].memories[remote.key].tensor
        assert local.numel() == source.numel() == remote.size
        if self.before_read is not None:
            self.before_read()
        local.copy_(source)

    def close(self):
        _ENDPOINTS.pop(self.url, None)


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _weights(step: int):
    # A 2-D fp32 tensor larger than one bucket, an odd-sized bf16 one to exercise alignment, an int one.
    return [
        ("big.weight", torch.arange(3000, dtype=torch.float32).reshape(30, 100) + step),
        ("odd.bias", torch.full((7,), float(step), dtype=torch.bfloat16)),
        ("ids", torch.arange(5, dtype=torch.int64) * step),
    ]


def _publisher(registry, **kwargs):
    kwargs.setdefault("rollout_dtype", None)
    return WeightPublisher(registry, bucket_size=4096, device="cpu", transport_factory=_FakeTransport, **kwargs)


def _on_wire(tensor, rollout_dtype=torch.bfloat16):
    return tensor.to(rollout_dtype) if tensor.dtype == torch.float32 else tensor


def _puller(registry, **kwargs):
    return WeightPuller(registry, device="cpu", transport_factory=_FakeTransport, poll_interval=0.01, **kwargs)


def _run(coro):
    return asyncio.run(coro)


def test_publish_then_pull_is_bit_exact():
    async def main():
        registry = InMemoryWeightRegistry()
        publisher, puller = _publisher(registry), _puller(registry)

        manifest = await publisher.publish(iter(_weights(1)), version=1)
        assert len(manifest.buckets) > 1, "the big tensor should span several buckets"
        assert all(spec.offset % TENSOR_ALIGNMENT == 0 for spec in manifest.tensors)

        pulled = await puller.fetch(await puller.wait_for_version(1, timeout=1))
        received = dict(pulled.named_tensors())
        for name, expected in _weights(1):
            assert received[name].dtype == expected.dtype and torch.equal(received[name], expected), name

        # A lease is held only during the fetch.
        assert not registry._leases

    _run(main())


def test_fp32_weights_are_cast_to_rollout_dtype():
    async def main():
        registry = InMemoryWeightRegistry()
        publisher, puller = _publisher(registry, rollout_dtype=torch.bfloat16), _puller(registry)
        for step in (1, 2):
            manifest = await publisher.publish(iter(_weights(step)), version=step)
            received = dict((await puller.fetch(manifest)).named_tensors())
            for name, expected in _weights(step):
                assert received[name].dtype == _on_wire(expected).dtype, name
                assert torch.equal(received[name], _on_wire(expected)), name
        # The fp32 tensor takes half the bytes; bf16 and int64 tensors are unchanged.
        assert {spec.name: spec.dtype for spec in manifest.tensors} == {
            "big.weight": torch.bfloat16,
            "odd.bias": torch.bfloat16,
            "ids": torch.int64,
        }

    _run(main())


def test_registrations_are_reused_across_versions():
    async def main():
        registry = InMemoryWeightRegistry()
        publisher, puller = _publisher(registry), _puller(registry)
        await publisher.publish(iter(_weights(1)), version=1)
        await puller.fetch(await registry.latest_ready())
        publisher_memories = [id(m) for slot in publisher.slots for m in slot.memories]
        puller_memories = [id(m) for m in puller.memories]

        for step in (2, 3, 4):
            await publisher.publish(iter(_weights(step)), version=step)
            pulled = await puller.fetch(await registry.latest_ready())
            assert pulled.version == step
            assert torch.equal(dict(pulled.named_tensors())["big.weight"], _weights(step)[0][1])

        assert [id(m) for slot in publisher.slots for m in slot.memories] == publisher_memories
        assert [id(m) for m in puller.memories] == puller_memories
        # Two slots: only the newest version is READY, older ones are gone once unleased.
        assert sorted(registry._states) == [3, 4]
        assert (await registry.latest_ready()).version == 4

    _run(main())


def test_publisher_waits_for_reader_lease_before_overwriting():
    async def main():
        registry = InMemoryWeightRegistry()
        publisher = _publisher(registry, reclaim_timeout=0.2)
        await publisher.publish(iter(_weights(1)), version=1)
        await publisher.publish(iter(_weights(2)), version=2)

        # A slow reader still holds version 1, whose slot is next to be overwritten.
        lease = await registry.acquire_lease(2, ttl=60)
        registry._leases[lease.lease_id].version = 1
        with pytest.raises(TimeoutError):
            await publisher.publish(iter(_weights(3)), version=3)

        await registry.release_lease(lease.lease_id)
        await publisher.publish(iter(_weights(3)), version=3)
        assert (await registry.latest_ready()).version == 3

    _run(main())


def test_torn_read_is_rejected():
    async def main():
        clock = _FakeClock()
        registry = InMemoryWeightRegistry(clock=clock)
        publisher, puller = _publisher(registry), _puller(registry, lease_ttl=60)
        manifest = await publisher.publish(iter(_weights(1)), version=1)

        puller._setup(manifest)

        def stall():
            # The read outlives the lease, as if the reader was descheduled without renewing.
            clock.now += 120

        puller.transport.before_read = stall
        with pytest.raises(TornReadError):
            await puller.fetch(manifest)

    _run(main())


def test_layout_change_is_rejected():
    async def main():
        registry = InMemoryWeightRegistry()
        publisher = _publisher(registry)
        await publisher.publish(iter(_weights(1)), version=1)
        weights = _weights(2)
        weights[1] = ("odd.bias", torch.zeros(8, dtype=torch.bfloat16))
        with pytest.raises(ValueError, match="does not match layout"):
            await publisher.publish(iter(weights), version=2)
        with pytest.raises(ValueError, match="Got 2 weights"):
            await publisher.publish(iter(_weights(2)[:2]), version=2)

    _run(main())


def test_checkpoint_engine_round_trip():
    async def main():
        registry = InMemoryWeightRegistry()

        def engine(is_master):
            e = TransportPullCheckpointEngine(
                bucket_size=4096, device="cpu", is_master=is_master, transport_factory=_FakeTransport
            )
            e._registry = registry
            return e

        trainer, other_rank, rollout = engine(True), engine(False), engine(False)
        assert TransportPullCheckpointEngine.build_topology(2, 3, [{}] * 5) == ({}, {})

        async def receive(step):
            return {n: t.clone() async for n, t in rollout.receive_weights(global_steps=step)}

        for step in (1, 2):
            # As in CheckpointEngineManager, both sides run concurrently; the rollout waits for the version.
            received, _, _ = await asyncio.gather(
                receive(step),
                trainer.send_weights(iter(_weights(step)), global_steps=step),
                other_rank.send_weights(iter(_weights(step)), global_steps=step),
            )
            for name, expected in _weights(step):
                assert torch.equal(received[name], _on_wire(expected)), name
        assert other_rank.publisher is None

        # Re-syncing an already published step drains the weights without publishing again.
        drained = []
        await trainer.send_weights((drained.append(n) or (n, t) for n, t in _weights(9)), global_steps=2)
        assert len(drained) == 3 and (await registry.latest_ready()).version == 2

        # Without global_steps, each sync is one version past the previous one on both sides.
        for step in (3, 4):
            received, _ = await asyncio.gather(receive(None), trainer.send_weights(iter(_weights(step))))
            assert torch.equal(received["ids"], _weights(step)[2][1])
        assert trainer.last_version == rollout.last_version == 4

    _run(main())
