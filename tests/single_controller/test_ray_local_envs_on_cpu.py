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

import errno
import os
import socket
from datetime import timedelta

import pytest
import ray
import torch
import torch.distributed as dist

from verl.single_controller.base.worker import Worker
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup


class TestActor(Worker):
    def __init__(self, init_process_group=False) -> None:
        """Optionally initialize a process group in the constructor."""
        super().__init__()
        if not init_process_group:
            return

        assert not dist.is_initialized()
        self.assert_master_port_bound()

        dist.init_process_group(backend="gloo", timeout=timedelta(seconds=30))
        value = torch.tensor(self.rank + 1)
        dist.all_reduce(value)
        self.result = value.item()
        dist.destroy_process_group()

    def getenv(self, key):
        val = os.getenv(key, f"{key} not set")
        return val

    def assert_master_port_bound(self):
        """Check that rank zero's master port rejects a competing bind."""
        if self.rank == 0:
            with socket.socket() as competitor:
                try:
                    competitor.bind((self.getenv("MASTER_ADDR"), int(self.getenv("MASTER_PORT"))))
                except OSError as error:
                    assert error.errno == errno.EADDRINUSE
                else:
                    raise AssertionError("The master port was free")

    def get_result(self):
        """Check the master port after construction and return the collective result."""
        self.assert_master_port_bound()
        return self.result


class AsyncTestActor(TestActor):
    """Exercise constructor initialization on a Ray actor with asynchronous methods."""

    async def get_result(self):
        """Check the master port and return the result from the actor's event loop."""
        return super().get_result()


TestActor = ray.remote(TestActor)
AsyncTestActor = ray.remote(AsyncTestActor)


def test_basics():
    ray.init(num_cpus=100)
    try:
        # create 4 workers, each hold a GPU
        resource_pool = RayResourcePool([4], use_gpu=False)
        class_with_args = RayClassWithInitArgs(cls=TestActor)

        worker_group = RayWorkerGroup(
            resource_pool=resource_pool, ray_cls_with_init=class_with_args, name_prefix="worker_group_basic"
        )

        output = worker_group.execute_all_sync("getenv", key="RAY_LOCAL_WORLD_SIZE")
        assert output == ["4", "4", "4", "4"]
    finally:
        ray.shutdown()


def test_customized_worker_env():
    ray.init(num_cpus=100)
    try:
        # create 4 workers, each hold a GPU
        resource_pool = RayResourcePool([4], use_gpu=False)
        class_with_args = RayClassWithInitArgs(cls=TestActor)

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
    finally:
        ray.shutdown()


@pytest.mark.parametrize(
    "worker_cls",
    [TestActor, AsyncTestActor],
    ids=["sync", "async"],
)
def test_master_port_bound_before_process_group(worker_cls):
    """Prebind the master port before PG initialization on ordinary and asynchronous Ray actors."""
    ray.init(num_cpus=4)
    try:

        @ray.remote(num_cpus=0)
        def run_collective():
            """Bound the entire group startup so a constructor deadlock fails the test."""
            group = RayWorkerGroup(
                resource_pool=RayResourcePool([2], use_gpu=False, max_colocate_count=1),
                ray_cls_with_init=RayClassWithInitArgs(worker_cls, init_process_group=True),
            )
            return ray.get([worker.get_result.remote() for worker in group.workers])

        assert ray.get(run_collective.remote(), timeout=60) == [3, 3]
    finally:
        ray.shutdown()


def test_rank_zero_bind_failure_does_not_wait():
    """Report an exhausted port range instead of waiting indefinitely for rank zero's endpoint."""
    ray.init(num_cpus=4)
    try:
        with socket.socket() as listener:
            listener.bind(("", 0))
            listener.listen()
            port = listener.getsockname()[1]

            @ray.remote(num_cpus=0)
            def create_group():
                """Attempt group startup while its only allowed master port is occupied."""
                RayWorkerGroup(
                    resource_pool=RayResourcePool([2], use_gpu=False, max_colocate_count=1),
                    ray_cls_with_init=RayClassWithInitArgs(TestActor),
                    master_port_range=[port, port + 1],
                )

            with pytest.raises(ray.exceptions.RayTaskError, match=rf"\b{port}\b"):
                ray.get(create_group.remote(), timeout=60)
    finally:
        ray.shutdown()


def test_rank_zero_skips_occupied_port_in_range():
    """Use the next master port when the first port in the range is already taken."""
    ray.init(num_cpus=4)
    try:
        with socket.socket() as listener:
            listener.bind(("", 0))
            listener.listen()
            port = listener.getsockname()[1]
            with socket.socket() as successor:
                successor.bind(("", port + 1))

            @ray.remote(num_cpus=0)
            def create_group():
                """Start a group whose first allowed master port is already occupied."""
                group = RayWorkerGroup(
                    resource_pool=RayResourcePool([2], use_gpu=False, max_colocate_count=1),
                    ray_cls_with_init=RayClassWithInitArgs(TestActor),
                    master_port_range=[port, port + 2],
                )
                return group.master_port, group.execute_all_sync("getenv", key="MASTER_PORT")

            master_port, worker_ports = ray.get(create_group.remote(), timeout=60)
            assert master_port == str(port + 1)
            assert worker_ports == [master_port, master_port]
    finally:
        ray.shutdown()


if __name__ == "__main__":
    test_basics()
    test_customized_worker_env()
