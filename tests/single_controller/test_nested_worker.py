# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


import ray

from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.base.worker import Worker
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.device import get_device_name


class TestActor(Worker):
    # TODO: pass *args and **kwargs is bug prone and not very convincing
    def __init__(self, x) -> None:
        super().__init__()
        self.a = x

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get(self):
        return self.a + self.rank


class TestHighLevelActor(Worker):
    def __init__(self, x=None) -> None:
        super().__init__()
        self.test_actor = TestActor(x=x)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get(self):
        return self.test_actor.get()


def test_nested_worker():
    ray.init(num_cpus=100)

    # create 4 workers, each hold a GPU
    resource_pool = RayResourcePool([4], use_gpu=True)
    class_with_args = RayClassWithInitArgs(cls=ray.remote(TestActor), x=2)

    worker_group = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=class_with_args,
        name_prefix="worker_group_basic",
        device_name=get_device_name(),
    )

    output = worker_group.get()

    assert output == [2, 3, 4, 5]

    class_with_args = RayClassWithInitArgs(cls=ray.remote(TestHighLevelActor), x=2)
    high_level_worker_group = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=class_with_args,
        name_prefix="worker_group_basic_2",
        device_name=get_device_name(),
    )

    output_1 = high_level_worker_group.get()

    assert output_1 == [2, 3, 4, 5]

    ray.shutdown()
