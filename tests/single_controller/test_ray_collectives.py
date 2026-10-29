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
"""
Test for using ray collective group.
Suppose we Actor and Rollout. Actor contains 4 workers and Rollout contains 2 workers. We established a Worker to
Rollout relationship by using collective groups
Actor: rank 0, 1 - Rollout rank 0
Rollout rank 2, 3 - Rollout rank 1
Then, we initiate 4 p2p comms from actor to rollout
"""

import pytest
import ray
import ray.util.collective as collective
import torch

from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device, is_device_available


@ray.remote
class Actor(Worker):
    @register(Dispatch.ONE_TO_ALL)
    def init(self):
        remote_rank = self.rank // 2
        self.group_name = f"A{self.rank}_R{remote_rank}"
        collective.init_collective_group(world_size=2, rank=0, backend=get_nccl_backend(), group_name=self.group_name)

    @register(Dispatch.ONE_TO_ALL, blocking=False)
    def send_tensors(self):
        tensor = torch.ones(size=(4,), dtype=torch.float32, device=get_device_name()) * self.rank
        collective.send(tensor=tensor, dst_rank=1, group_name=self.group_name)


@ray.remote
class Rollout(Worker):
    @register(Dispatch.ONE_TO_ALL)
    def init(self):
        self.remote_first_rank = self.rank * 2
        self.remote_second_rank = self.remote_first_rank + 1
        self.first_group_name = f"A{self.remote_first_rank}_R{self.rank}"
        self.second_group_name = f"A{self.remote_second_rank}_R{self.rank}"

        collective.init_collective_group(
            world_size=2, rank=1, backend=get_nccl_backend(), group_name=self.first_group_name
        )
        collective.init_collective_group(
            world_size=2, rank=1, backend=get_nccl_backend(), group_name=self.second_group_name
        )

    @register(Dispatch.ONE_TO_ALL, blocking=False)
    def receive_tensors(self):
        self.tensor1 = torch.randn(size=(4,), dtype=torch.float32, device=get_device_name())
        self.tensor2 = torch.randn(size=(4,), dtype=torch.float32, device=get_device_name())

        collective.recv(self.tensor1, src_rank=0, group_name=self.first_group_name)
        collective.recv(self.tensor2, src_rank=0, group_name=self.second_group_name)

    @register(Dispatch.ONE_TO_ALL)
    def get_tensors(self):
        return {f"src_{self.remote_first_rank}": self.tensor1, f"src_{self.remote_second_rank}": self.tensor2}


def _ray_backend_available(backend: str) -> bool:
    # Newer Ray lets platforms register their own collective backend; older Ray only knows NCCL.
    check = getattr(collective, "is_backend_available", None)
    if check is not None:
        return check(backend)
    return backend.lower() == "nccl" and collective.nccl_available()


def test_ray_collective_group():
    # get_device_name() falls back to "cuda" when no accelerator is detected at all (e.g. a CPU-only
    # box), so is_device_available() is needed too or that case would slip past the skips below.
    if not is_device_available():
        pytest.skip("no accelerator available")
    ngpus = get_torch_device().device_count()
    # 2 actors + 1 rollout must be placed at once; with fewer devices the rollout pool is never satisfied.
    if ngpus < 3:
        pytest.skip(f"needs at least 3 devices, found {ngpus}")
    # Init Ray before probing backends: nccl_available() calls ray.get_gpu_ids(), which auto-starts Ray.
    ray.init()
    backend = get_nccl_backend()
    if not _ray_backend_available(backend):
        ray.shutdown()
        pytest.skip(f"ray.util.collective has no {backend!r} backend registered")
    n_rollout = max(1, ngpus // 3)  # keep 2:1 actor:rollout ratio
    n_actor = n_rollout * 2

    actor_resource_pool = RayResourcePool([n_actor])
    rollout_resource_pool = RayResourcePool([n_rollout])

    actor_cls = RayClassWithInitArgs(cls=Actor)
    rollout_cls = RayClassWithInitArgs(cls=Rollout)

    actor_wg = RayWorkerGroup(
        resource_pool=actor_resource_pool, ray_cls_with_init=actor_cls, name_prefix="collective_group_actor"
    )
    rollout_wg = RayWorkerGroup(
        resource_pool=rollout_resource_pool, ray_cls_with_init=rollout_cls, name_prefix="collective_group_rollout"
    )

    actor_wg.init()
    rollout_wg.init()

    out1 = actor_wg.send_tensors()
    out2 = rollout_wg.receive_tensors()

    # block to wait
    ray.get(out1)
    ray.get(out2)

    output = {}
    for d in rollout_wg.get_tensors():
        output |= d

    print(output)

    for i in range(n_actor):
        assert torch.sum(output[f"src_{i}"]).item() == 4 * i

    ray.shutdown()


if __name__ == "__main__":
    test_ray_collective_group()
