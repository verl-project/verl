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

import os

import ray
import torch

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray.base import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
    split_resource_pool,
)
from verl.utils.device import get_device_name, get_nccl_backend


@ray.remote
class Actor(Worker):
    def __init__(self, worker_id) -> None:
        super().__init__()
        self.worker_id = worker_id
        self.temp_tensor = torch.rand(4096, 4096).to(get_device_name())

        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(backend=get_nccl_backend(), world_size=world_size, rank=rank)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def add(self, data: DataProto):
        data.batch["a"] += self.rank + self.worker_id
        return data


def test_split_resource_pool_with_split_size():
    ray.init()
    # assume we have 2 nodes, with 4 GPUs each
    global_resource_pool = RayResourcePool(process_on_nodes=[4, 4])
    global_resource_pool.get_placement_groups(device_name=get_device_name())

    # first 4 gpus for actor_1, last 4 gpus for actor_2
    actor_1_resource_pool, actor_2_resource_pool = split_resource_pool(resource_pool=global_resource_pool, split_size=4)
    actor_cls_1 = RayClassWithInitArgs(cls=Actor, worker_id=0)
    actor_cls_2 = RayClassWithInitArgs(cls=Actor, worker_id=100)
    actor_worker_1 = RayWorkerGroup(
        resource_pool=actor_1_resource_pool, ray_cls_with_init=actor_cls_1, device_name=get_device_name()
    )
    actor_worker_2 = RayWorkerGroup(
        resource_pool=actor_2_resource_pool, ray_cls_with_init=actor_cls_2, device_name=get_device_name()
    )
    assert actor_worker_1.world_size == 4
    assert actor_worker_2.world_size == 4

    data = DataProto.from_dict({"a": torch.zeros(8)})
    actor_output_1 = actor_worker_1.add(data)
    actor_output_2 = actor_worker_2.add(data)
    assert actor_output_1.batch["a"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert actor_output_2.batch["a"].tolist() == [100, 100, 101, 101, 102, 102, 103, 103]

    ray.shutdown()


def test_split_resource_pool_with_split_size_list():
    ray.init()
    # assume we have 4 nodes, with 2 GPUs each
    global_resource_pool = RayResourcePool(process_on_nodes=[2, 2, 2, 2])
    global_resource_pool.get_placement_groups(device_name=get_device_name())

    # first 2 gpus for actor_1, last 6 gpus for actor_2
    actor_1_resource_pool, actor_2_resource_pool = split_resource_pool(
        resource_pool=global_resource_pool,
        split_size=[2, 6],
    )
    actor_cls_1 = RayClassWithInitArgs(cls=Actor, worker_id=0)
    actor_cls_2 = RayClassWithInitArgs(cls=Actor, worker_id=100)
    actor_worker_1 = RayWorkerGroup(
        resource_pool=actor_1_resource_pool, ray_cls_with_init=actor_cls_1, device_name=get_device_name()
    )
    actor_worker_2 = RayWorkerGroup(
        resource_pool=actor_2_resource_pool, ray_cls_with_init=actor_cls_2, device_name=get_device_name()
    )
    assert actor_worker_1.world_size == 2
    assert actor_worker_2.world_size == 6

    data_1 = DataProto.from_dict({"a": torch.zeros(4)})
    data_2 = DataProto.from_dict({"a": torch.zeros(6)})
    actor_output_1 = actor_worker_1.add(data_1)
    actor_output_2 = actor_worker_2.add(data_2)
    print(actor_output_1.batch["a"].tolist())
    print(actor_output_2.batch["a"].tolist())
    assert actor_output_1.batch["a"].tolist() == [0, 0, 1, 1]
    assert actor_output_2.batch["a"].tolist() == [100, 101, 102, 103, 104, 105]

    ray.shutdown()


def test_split_resource_pool_with_split_size_list_cross_nodes():
    ray.init()
    # assume we have 4 nodes, with 2 GPUs each
    global_resource_pool = RayResourcePool(process_on_nodes=[4, 4])
    global_resource_pool.get_placement_groups(device_name=get_device_name())

    # first 2 gpus for actor_1, last 6 gpus for actor_2
    actor_1_resource_pool, actor_2_resource_pool = split_resource_pool(
        resource_pool=global_resource_pool,
        split_size=[2, 6],
    )
    actor_cls_1 = RayClassWithInitArgs(cls=Actor, worker_id=0)
    actor_cls_2 = RayClassWithInitArgs(cls=Actor, worker_id=100)
    actor_worker_1 = RayWorkerGroup(
        resource_pool=actor_1_resource_pool, ray_cls_with_init=actor_cls_1, device_name=get_device_name()
    )
    actor_worker_2 = RayWorkerGroup(
        resource_pool=actor_2_resource_pool, ray_cls_with_init=actor_cls_2, device_name=get_device_name()
    )

    assert actor_worker_1.world_size == 2
    assert actor_worker_2.world_size == 6

    data_1 = DataProto.from_dict({"a": torch.zeros(4)})
    data_2 = DataProto.from_dict({"a": torch.zeros(6)})
    actor_output_1 = actor_worker_1.add(data_1)
    actor_output_2 = actor_worker_2.add(data_2)
    print(actor_output_1.batch["a"].tolist())
    print(actor_output_2.batch["a"].tolist())
    assert actor_output_1.batch["a"].tolist() == [0, 0, 1, 1]
    assert actor_output_2.batch["a"].tolist() == [100, 101, 102, 103, 104, 105]

    ray.shutdown()


def test_split_resource_pool_with_split_twice():
    ray.init()

    # assume we have 4 nodes, with 2 GPUs each
    global_resource_pool = RayResourcePool(process_on_nodes=[2, 2, 2, 2])
    global_resource_pool.get_placement_groups(device_name=get_device_name())

    # actors with [2, 1, 1, 1, 1, 2] (split twice)
    rp_1, rp_2, rp_3 = split_resource_pool(
        resource_pool=global_resource_pool,
        split_size=[2, 4, 2],
    )
    rp_2_1, rp_2_2, rp_2_3, rp_2_4 = split_resource_pool(
        resource_pool=rp_2,
        split_size=1,
    )
    fp_list = [rp_1, rp_2_1, rp_2_2, rp_2_3, rp_2_4, rp_3]
    correct_world_size = [2, 1, 1, 1, 1, 2]
    correct_output = [
        [0.0, 0.0, 1.0, 1.0],  # 2 worker
        [100.0, 100.0, 100.0, 100.0],  # 1 worker
        [200.0, 200.0, 200.0, 200.0],  # 1 worker
        [300.0, 300.0, 300.0, 300.0],  # 1 worker
        [400.0, 400.0, 400.0, 400.0],  # 1 worker
        [500.0, 500.0, 501.0, 501.0],  # 2 worker
    ]
    for idx, rp in enumerate(fp_list):
        actor_cls = RayClassWithInitArgs(cls=Actor, worker_id=idx * 100)
        actor_worker = RayWorkerGroup(resource_pool=rp, ray_cls_with_init=actor_cls, device_name=get_device_name())
        data = DataProto.from_dict({"a": torch.zeros(4)})
        actor_output = actor_worker.add(data)
        assert actor_worker.world_size == correct_world_size[idx]
        assert actor_output.batch["a"].tolist() == correct_output[idx]

    ray.shutdown()
