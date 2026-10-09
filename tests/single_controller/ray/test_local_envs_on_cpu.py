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
e2e test verl.single_controller.ray
"""

import os

import pytest

from verl.runtime import ClassWithInitArgs, Worker
from verl.single_controller.ray.base import RayClassWithInitArgs, RayWorkerGroup


class EnvWorker(Worker):
    def __init__(self) -> None:
        super().__init__()

    def getenv(self, key):
        val = os.getenv(key, f"{key} not set")
        return val


def test_basics(ray_only_runtime):
    # Create four CPU workers under the initialized Runtime.
    resource_pool = ray_only_runtime.create_resource_pool(nnodes=1, processes_per_node=4, device_type="cpu")
    class_with_args = RayClassWithInitArgs.from_class_init(ClassWithInitArgs(EnvWorker))

    worker_group = RayWorkerGroup(
        resource_pool=resource_pool, ray_cls_with_init=class_with_args, name_prefix="worker_group_basic"
    )

    output = worker_group.execute_all_sync("getenv", key="RAY_LOCAL_WORLD_SIZE")
    assert output == ["4", "4", "4", "4"]
    worker_group.close()


def test_customized_env_vars(ray_only_runtime):
    # Create four CPU workers under the initialized Runtime.
    resource_pool = ray_only_runtime.create_resource_pool(nnodes=1, processes_per_node=4, device_type="cpu")
    class_with_args = RayClassWithInitArgs.from_class_init(ClassWithInitArgs(EnvWorker))

    worker_group = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=class_with_args,
        name_prefix="worker_group_customized",
        worker_env={
            "test_key": "test_value",  # new key will be appended
        },
    )

    output = worker_group.execute_all_sync("getenv", key="test_key")
    assert output == ["test_value", "test_value", "test_value", "test_value"]
    worker_group.close()

    try:
        worker_group = RayWorkerGroup(
            resource_pool=resource_pool,
            ray_cls_with_init=class_with_args,
            name_prefix="worker_group_error",
            worker_env={
                "WORLD_SIZE": "100",  # override system env will result in error
            },
        )
    except ValueError as e:
        assert "WORLD_SIZE" in str(e)
    else:
        raise ValueError("test failed")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
