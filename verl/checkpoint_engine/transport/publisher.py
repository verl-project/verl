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
"""Trainer-side publisher of weight versions for one-sided pulls."""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import AsyncIterator, Iterator

import torch

from verl.utils.device import get_torch_device
from verl.workers.rollout.utils import ensure_async_iterator

from .registry import BucketDesc, WeightManifest, WeightRegistry
from .transport import TransportFactory, bucket_ranges, default_transport_factory, plan_layout, tensor_view

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class _Slot:
    buffer: torch.Tensor
    memories: list
    buckets: list[BucketDesc]
    version: int | None = None


class WeightPublisher:
    """Keeps the last `num_slots` weight versions in registered buffers that readers pull from.

    Each slot holds a full copy of the model laid out by `plan_layout`, and each bucket-sized
    range of a slot is registered once, on the first publish. Publishing a version writes into a
    slot whose previous version no reader holds a lease on, then announces it in the registry
    and retires older versions. The trainer never waits for readers beyond that.

    Device memory overhead is `num_slots * model bytes` on `device`, counted in `rollout_dtype`.

    Args:
        registry: Where versions are announced.
        bucket_size: Size in bytes of one remote read.
        device: Device of the slot buffers, "cuda" or "cpu" (pinned host memory).
        backend: torch.distributed._transport backend.
        num_slots: Versions kept at once; 2 lets readers finish v while v+1 is written.
        rollout_dtype: fp32 weights (e.g. FSDP master weights) are cast to this as they are copied
            into a slot, halving slot memory and transfer for bf16. Other dtypes pass through.
            None sends weights as the trainer produces them.
        reclaim_timeout: Seconds to wait for leases on the slot being overwritten to end.
        transport_factory: Override of transport construction, for tests.
    """

    def __init__(
        self,
        registry: WeightRegistry,
        bucket_size: int,
        device: str = "cuda",
        backend: str = "nixl",
        num_slots: int = 2,
        rollout_dtype: torch.dtype | None = torch.bfloat16,
        reclaim_timeout: float = 600.0,
        transport_factory: TransportFactory = default_transport_factory,
    ):
        assert num_slots >= 1
        self.registry = registry
        self.bucket_size = bucket_size
        self.device = device
        self.backend = backend
        self.num_slots = num_slots
        self.rollout_dtype = rollout_dtype
        self.reclaim_timeout = reclaim_timeout
        self.transport_factory = transport_factory

        self.transport = None
        self.endpoint = None
        self.tensors = None
        self.total_bytes = None
        self.slots: list[_Slot] = []

    def _wire_dtype(self, dtype: torch.dtype) -> torch.dtype:
        return self.rollout_dtype if self.rollout_dtype is not None and dtype == torch.float32 else dtype

    def _setup(self, named_tensors: list[tuple[str, torch.Tensor]]):
        self.tensors, self.total_bytes = plan_layout(
            [(n, t.shape, self._wire_dtype(t.dtype)) for n, t in named_tensors]
        )
        self.transport = self.transport_factory(self.backend, self.device)
        ranges = bucket_ranges(self.total_bytes, self.bucket_size)
        for _ in range(self.num_slots):
            if self.device == "cpu":
                buffer = torch.empty(self.total_bytes, dtype=torch.uint8, pin_memory=True)
            else:
                buffer = torch.empty(self.total_bytes, dtype=torch.uint8, device=self.device)
            memories = [self.transport.register_memory(buffer[offset : offset + size]) for offset, size in ranges]
            self.slots.append(_Slot(buffer=buffer, memories=memories, buckets=[]))
        # Bind after registering: some backends (nixl_native) put registrations in the url.
        self.endpoint = self.transport.bind()
        for slot in self.slots:
            slot.buckets = [
                BucketDesc(offset=offset, size=size, remote_buffer=memory.to_remote_buffer().serialize())
                for (offset, size), memory in zip(ranges, slot.memories, strict=True)
            ]
        logger.info(f"Publisher at {self.endpoint!r}: {self.num_slots} slots of {self.total_bytes} bytes")

    async def _acquire_slot(self) -> _Slot:
        """Pick the free slot, or the one holding the oldest version, and wait until no one reads it."""
        slot = min(self.slots, key=lambda s: -1 if s.version is None else s.version)
        if slot.version is None:
            return slot
        await self.registry.retire(slot.version)
        deadline = time.monotonic() + self.reclaim_timeout
        while not await self.registry.try_reclaim(slot.version):
            if time.monotonic() > deadline:
                raise TimeoutError(f"Version {slot.version} still leased after {self.reclaim_timeout}s")
            await asyncio.sleep(0.05)
        slot.version = None
        return slot

    @torch.no_grad()
    async def publish(
        self, weights: Iterator[tuple[str, torch.Tensor]] | AsyncIterator[tuple[str, torch.Tensor]], version: int
    ) -> WeightManifest:
        """Copy `weights` into a free slot and announce them as `version`.

        The tensor names, shapes and dtypes must be the same, in the same order, on every call.
        """
        start = time.time()
        if self.transport is None:
            # The layout is unknown until the weights are seen once; keep a host copy of this
            # first version to size the slots.
            named = [
                (n, t.detach().to("cpu", dtype=self._wire_dtype(t.dtype), copy=True))
                async for n, t in ensure_async_iterator(weights)
            ]
            self._setup(named)
            weights = iter(named)

        slot = await self._acquire_slot()
        index = 0
        async for name, tensor in ensure_async_iterator(weights):
            if index >= len(self.tensors):
                raise ValueError(f"Unexpected weight {name}: layout has {len(self.tensors)} tensors")
            spec = self.tensors[index]
            if (name, tensor.shape, self._wire_dtype(tensor.dtype)) != (spec.name, spec.shape, spec.dtype):
                raise ValueError(f"Weight {name} {tuple(tensor.shape)} {tensor.dtype} does not match layout {spec}")
            # copy_ also casts to the wire dtype.
            tensor_view(slot.buffer, spec).copy_(tensor, non_blocking=True)
            index += 1
        if index != len(self.tensors):
            raise ValueError(f"Got {index} weights, layout has {len(self.tensors)}")
        if self.device != "cpu":
            get_torch_device().synchronize()

        slot.version = version
        manifest = WeightManifest(
            version=version,
            endpoint=self.endpoint,
            total_bytes=self.total_bytes,
            tensors=self.tensors,
            buckets=slot.buckets,
            extra={"backend": self.backend},
        )
        await self.registry.publish(manifest)
        # Only the newest version is handed out; older ones finish their reads and are reclaimed.
        for other in self.slots:
            if other is not slot and other.version is not None:
                await self.registry.retire(other.version)
        logger.info(f"Published version {version} in {time.time() - start:.2f}s")
        return manifest

    def close(self):
        """Unregister and close. Readers must be done with every version first."""
        for slot in self.slots:
            for memory in slot.memories:
                self.transport.unregister_memory(memory)
        self.slots = []
        if self.transport is not None:
            self.transport.close()
            self.transport = None
