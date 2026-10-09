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

from verl.runtime import ClassWithInitArgs, Dispatch, Worker, register
from verl.single_controller.base.fused import FusedWorker, role_manifests


class Actor(Worker):
    def __init__(self) -> None:
        super().__init__()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def add(self, x):
        x += self.rank
        return x


class Critic(Worker):
    def __init__(self, val) -> None:
        super().__init__()
        self.val = val

    @register(dispatch_mode=Dispatch.ALL_TO_ALL)
    def sub(self, x):
        x -= self.val
        return x


cls_dict = {"actor": ClassWithInitArgs(Actor), "critic": ClassWithInitArgs(Critic, val=10)}


class HybridWorker(FusedWorker):
    def __init__(self, *, defer_role_init: bool = False):
        super().__init__(role_manifests(cls_dict), defer_role_init=defer_role_init)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def foo(self, x):
        return self.critic.sub(self.actor.add(x))


def test_fused_workers(ray_only_runtime):
    pool = ray_only_runtime.create_resource_pool(nnodes=1, processes_per_node=2, device_type="cpu")
    roles = ray_only_runtime.create_worker_group(cls_dict, on=pool)
    hybrid = ray_only_runtime.create_worker_group(HybridWorker, on=pool)

    x = roles["actor"].add(0.1)
    y = roles["critic"].sub(x)
    z = hybrid.foo(0.1)
    assert y == z


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
