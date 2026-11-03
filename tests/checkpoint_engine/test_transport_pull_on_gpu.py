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
"""Publisher and pullers on separate GPUs, reading over a real transport. No model needed."""

import importlib.util

import pytest
import ray
import torch

from verl.checkpoint_engine.transport import RayWeightRegistry
from verl.checkpoint_engine.transport.publisher import WeightPublisher
from verl.checkpoint_engine.transport.puller import WeightPuller
from verl.utils.device import get_torch_device

_NUM_PULLERS = 3


def _backend_available(backend: str) -> bool:
    if backend == "nixl_native":
        return importlib.util.find_spec("nixl") is not None
    return importlib.util.find_spec("torch.distributed._transport") is not None


def _weights(version: int, device: str):
    # Deterministic from the version, so readers can rebuild the expected values themselves.
    gen = torch.Generator().manual_seed(version)
    shapes = [
        ((4096, 4096), torch.bfloat16),
        ((11008, 4096), torch.bfloat16),
        ((4097,), torch.float32),
        ((3,), torch.int64),
    ]
    out = []
    for i, (shape, dtype) in enumerate(shapes):
        values = (
            torch.randint(0, 1000, shape, generator=gen) if dtype == torch.int64 else torch.randn(shape, generator=gen)
        )
        out.append((f"layer{i}", values.to(dtype).to(device)))
    return out


@ray.remote(num_gpus=1)
class _Publisher:
    def __init__(self, backend, device):
        self.publisher = WeightPublisher(
            RayWeightRegistry(name="test_transport_pull"), bucket_size=64 << 20, device=device, backend=backend
        )

    async def publish(self, version):
        await self.publisher.publish(iter(_weights(version, "cuda")), version=version)


@ray.remote(num_gpus=1)
class _Puller:
    def __init__(self, backend, device):
        self.puller = WeightPuller(RayWeightRegistry(name="test_transport_pull"), device=device, backend=backend)

    async def pull(self, min_version):
        manifest = await self.puller.wait_for_version(min_version, timeout=120)
        pulled = await self.puller.fetch(manifest)
        received = dict(pulled.named_tensors())
        expected = _weights(pulled.version, "cpu")
        mismatched = [name for name, tensor in expected if not torch.equal(received[name].cpu(), tensor)]
        get_torch_device().synchronize()
        return pulled.version, mismatched


@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("backend", ["nixl", "nixl_native"])
def test_publish_and_pull_across_gpus(backend, device):
    if not _backend_available(backend):
        pytest.skip(f"{backend} is not installed")
    if get_torch_device().device_count() < 1 + _NUM_PULLERS:
        pytest.skip(f"needs {1 + _NUM_PULLERS} GPUs")

    ray.init(namespace=f"test_transport_pull_{backend}_{device}")
    try:
        publisher = _Publisher.remote(backend, device)
        pullers = [_Puller.remote(backend, device) for _ in range(_NUM_PULLERS)]
        for version in range(1, 4):
            # Pullers wait on the registry while the publisher writes, as rollout replicas would.
            pulls = [p.pull.remote(version) for p in pullers]
            ray.get(publisher.publish.remote(version))
            for got_version, mismatched in ray.get(pulls):
                assert got_version == version
                assert not mismatched, f"version {version}: {mismatched}"
    finally:
        ray.shutdown()
