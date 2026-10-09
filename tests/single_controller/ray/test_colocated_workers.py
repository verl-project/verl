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

import pytest
import ray

from verl import DataProto
from verl.runtime import ClassWithInitArgs, Runtime
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register


class Actor(Worker):
    def __init__(self) -> None:
        super().__init__()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def add(self, data: DataProto):
        data.batch["a"] += self.rank
        return data

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def plain_add(self, value: int) -> int:
        return value + self.rank


class Critic(Worker):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    async def sub(self, data: DataProto):
        data.batch["a"] -= self.config["b"]
        return data

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def plain_sub(self, value: int) -> int:
        return value - self.config["b"]


def test_colocated_workers():
    ray.init()
    runtime = Runtime.from_config({"backend": "ray", "ray": {}})

    import torch

    # create separate workers on the same resource pool
    actor_cls = ClassWithInitArgs(cls=Actor)
    critic_cls = ClassWithInitArgs(cls=Critic, config={"b": 10})
    resource_pool = runtime.create_resource_pool(nnodes=1, processes_per_node=2, device_type="cpu")

    actor_wg = runtime.create_worker_group(actor_cls, on=resource_pool)
    critic_wg = runtime.create_worker_group(critic_cls, on=resource_pool)

    expected_actor_output = actor_wg.add(DataProto.from_dict({"a": torch.zeros(10)}))
    expected_critic_output = critic_wg.sub(DataProto.from_dict({"a": torch.zeros(10)}))

    # create colocated workers
    cls_dict = {"actor": actor_cls, "critic": critic_cls}
    spawn_wg = runtime.create_worker_group(cls_dict, on=resource_pool)

    colocated_actor_wg = spawn_wg["actor"]
    colocated_critic_wg = spawn_wg["critic"]

    actor_output = colocated_actor_wg.add(DataProto.from_dict({"a": torch.zeros(10)}))
    critic_output = colocated_critic_wg.sub(DataProto.from_dict({"a": torch.zeros(10)}))

    torch.testing.assert_close(
        dict(expected_actor_output.batch.items()), dict(actor_output.batch.items()), atol=0, rtol=0
    )
    torch.testing.assert_close(
        dict(expected_critic_output.batch.items()), dict(critic_output.batch.items()), atol=0, rtol=0
    )
    assert colocated_actor_wg.plain_add(3) == [3, 4]
    assert colocated_critic_wg.plain_sub(13) == [3, 3]
    with pytest.raises(AttributeError):
        colocated_actor_wg.plain_sub(13)
    with pytest.raises(AttributeError):
        colocated_critic_wg.plain_add(3)

    runtime.close()
    ray.shutdown()
