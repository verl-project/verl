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
from unittest.mock import patch

import pytest
import ray
import torch
import torch.distributed as dist

from verl.single_controller.base.worker import Worker
from verl.single_controller.ray.base import (
    TRAINING_WORKER_OMP_NUM_THREADS_ENV,
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
    split_resource_pool,
)


class TestActor(Worker):
    """Expose worker state and optionally initialize Gloo during construction."""

    def __init__(self, init_process_group=False, expected_use_libuv=True) -> None:
        """Optionally inspect the master port and initialize a process group."""
        super().__init__()
        if not init_process_group:
            return

        assert not dist.is_initialized()
        self.master_port_was_bound = False
        if self.rank == 0:
            assert self.master_tcp_store.libuvBackend == expected_use_libuv
            with socket.socket() as competitor:
                try:
                    competitor.bind((self.getenv("MASTER_ADDR"), int(self.getenv("MASTER_PORT"))))
                except OSError as error:
                    assert error.errno == errno.EADDRINUSE
                    self.master_port_was_bound = True

        dist.init_process_group(
            backend="gloo",
            timeout=timedelta(seconds=30),
            init_method=os.environ.get("DIST_INIT_METHOD", None),
        )
        value = torch.tensor(self.rank + 1)
        dist.all_reduce(value)
        self.result = value.item()
        dist.destroy_process_group()

    def getenv(self, key):
        """Read an environment variable from the worker process."""
        val = os.getenv(key, f"{key} not set")
        return val

    def was_master_port_bound(self):
        """Return the port occupancy observed before process-group initialization."""
        return self.master_port_was_bound

    def get_result(self):
        """Return the collective result from the constructor."""
        return self.result


class AsyncTestActor(TestActor):
    """Exercise constructor initialization on a Ray actor with asynchronous methods."""

    async def get_result(self):
        """Return the collective result from the actor's event loop."""
        return super().get_result()


TestActor = ray.remote(TestActor)
AsyncTestActor = ray.remote(AsyncTestActor)


@pytest.mark.parametrize(
    "worker_cls, worker_env, expected_use_libuv",
    [
        (TestActor, {}, True),
        (AsyncTestActor, {}, True),
        (TestActor, {"USE_LIBUV": "0"}, False),
        (TestActor, {"USE_LIBUV": "1", "DIST_INIT_METHOD": "env://?use_libuv=0"}, False),
        (TestActor, {"USE_LIBUV": "0", "DIST_INIT_METHOD": "env://?use_libuv=1"}, True),
    ],
    ids=["sync-default-libuv", "async-default-libuv", "sync-env-legacy", "sync-query-legacy", "sync-query-libuv"],
)
def test_basics(worker_cls, worker_env, expected_use_libuv, monkeypatch):
    """Propagate worker environments and initialize Gloo using the prebound port."""
    monkeypatch.delenv("USE_LIBUV", raising=False)
    monkeypatch.delenv("DIST_INIT_METHOD", raising=False)
    ray.init(num_cpus=100)
    try:

        @ray.remote(num_cpus=0)
        def run_collective():
            """Bound group startup, including a constructor waiting for other ranks."""
            resource_pool = RayResourcePool([4], use_gpu=False)
            class_with_args = RayClassWithInitArgs(
                cls=worker_cls, init_process_group=True, expected_use_libuv=expected_use_libuv
            )
            worker_group = RayWorkerGroup(
                resource_pool=resource_pool,
                ray_cls_with_init=class_with_args,
                name_prefix="worker_group_basic",
                worker_env=worker_env,
            )

            output = worker_group.execute_all_sync("getenv", key="RAY_LOCAL_WORLD_SIZE")
            assert output == ["4", "4", "4", "4"]
            endpoint = (worker_group.master_address, worker_group.master_port)
            assert int(endpoint[1]) > 0
            for key, value in zip(("MASTER_ADDR", "MASTER_PORT"), endpoint, strict=True):
                assert worker_group.execute_all_sync("getenv", key=key) == [value] * 4
            assert ray.get(worker_group.workers[0].was_master_port_bound.remote())
            return ray.get([worker.get_result.remote() for worker in worker_group.workers])

        assert ray.get(run_collective.remote(), timeout=60) == [10, 10, 10, 10]
    finally:
        ray.shutdown()


def test_customized_worker_env():
    """Propagate custom environment variables and reject protected-key overrides."""
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

        with pytest.raises(ValueError, match="WG_MASTER_ENDPOINT_CONFIG"):
            RayWorkerGroup(
                resource_pool=resource_pool,
                ray_cls_with_init=class_with_args,
                name_prefix="worker_group_endpoint_config_error",
                worker_env={"WG_MASTER_ENDPOINT_CONFIG": "user override"},
            )
    finally:
        ray.shutdown()


@pytest.mark.parametrize(
    "endpoint_keys",
    [("master_addr", "master_port"), ("master_addr",), ("master_port",)],
    ids=["explicit-pair", "addr-only", "port-only"],
)
def test_master_endpoint(endpoint_keys):
    """Preserve explicit endpoints without prebinding and reject incomplete pairs."""
    ray.init(num_cpus=4)
    try:
        with socket.socket() as listener:
            listener.bind(("", 0))
            listener.listen()
            master_addr = "localhost"
            master_port = str(listener.getsockname()[1])
            endpoint = {"master_addr": master_addr, "master_port": master_port}
            endpoint = {key: endpoint[key] for key in endpoint_keys}

            @ray.remote(num_cpus=0)
            def create_group():
                """Create workers without initializing a process-group store."""
                group = RayWorkerGroup(
                    resource_pool=RayResourcePool([2], use_gpu=False, max_colocate_count=1),
                    ray_cls_with_init=RayClassWithInitArgs(TestActor),
                    **endpoint,
                )
                addresses = ray.get([worker.getenv.remote("MASTER_ADDR") for worker in group.workers], timeout=30)
                ports = ray.get([worker.getenv.remote("MASTER_PORT") for worker in group.workers], timeout=30)
                return group.master_address, group.master_port, addresses, ports

            if len(endpoint_keys) == 1:
                with pytest.raises(ValueError, match="master_addr.*master_port"):
                    ray.get(create_group.remote(), timeout=60)
            else:
                address, port, worker_addresses, worker_ports = ray.get(create_group.remote(), timeout=60)
                assert (address, port) == (master_addr, master_port)
                assert worker_addresses == [master_addr, master_addr]
                assert worker_ports == [master_port, master_port]
    finally:
        ray.shutdown()


@pytest.mark.parametrize("port_count", [2, 1], ids=["skip-occupied", "exhausted"])
def test_master_port_range(port_count):
    """Skip occupied ports in order and report exhaustion without waiting forever."""
    ray.init(num_cpus=4)
    try:
        with socket.socket() as listener:
            listener.bind(("", 0))
            listener.listen()
            port = listener.getsockname()[1]
            if port_count == 2:
                with socket.socket() as successor:
                    successor.bind(("", port + 1))

            @ray.remote(num_cpus=0)
            def create_group():
                """Start a group whose first allowed master port is occupied."""
                group = RayWorkerGroup(
                    resource_pool=RayResourcePool([2], use_gpu=False, max_colocate_count=1),
                    ray_cls_with_init=RayClassWithInitArgs(TestActor),
                    master_port_range=[port, port + port_count],
                )
                return group.master_port, group.execute_all_sync("getenv", key="MASTER_PORT")

            if port_count == 1:
                with pytest.raises(ray.exceptions.RayTaskError, match=rf"\b{port}\b"):
                    ray.get(create_group.remote(), timeout=60)
            else:
                master_port, worker_ports = ray.get(create_group.remote(), timeout=60)
                assert master_port == str(port + 1)
                assert worker_ports == [master_port, master_port]
    finally:
        ray.shutdown()


@pytest.mark.parametrize("use_subresource_pool", [False, True], ids=["resource-pool", "subresource-pool"])
@pytest.mark.parametrize("world_size", [1, 2], ids=["single-rank", "waiting-peer"])
def test_constructor_failure_after_endpoint(use_subresource_pool, world_size):
    """Report rank-zero failure after publication, including while a peer is still constructing."""
    ray.init(num_cpus=4)
    try:

        @ray.remote
        class FailingWorker(Worker):
            """Fail rank zero after peers confirm that the driver received its endpoint."""

            def __init__(self):
                """Publish the endpoint before coordinating a deterministic constructor failure."""
                super().__init__()
                if self.rank == 0:
                    if self.world_size > 1:
                        self.master_tcp_store.wait(["peer-started"], timedelta(seconds=30))
                    raise RuntimeError("rank zero failed after publishing its endpoint")

                addr, port = self.get_master_addr_port()
                store = dist.TCPStore(
                    host_name=addr,
                    port=int(port),
                    is_master=False,
                    timeout=timedelta(seconds=120),
                )
                store.set("peer-started", "1")
                # Group creation must report rank zero's error before this wait times out.
                store.wait(["never-published"])

        @ray.remote(num_cpus=0)
        def create_group():
            """Exercise startup failure through either resource-pool initialization path."""
            resource_pool = RayResourcePool(
                [world_size + int(use_subresource_pool)], use_gpu=False, max_colocate_count=1
            )
            if use_subresource_pool:
                resource_pool = split_resource_pool(resource_pool, [1, world_size])[1]
            RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=RayClassWithInitArgs(FailingWorker))

        with pytest.raises(ray.exceptions.RayTaskError, match="rank zero failed after publishing its endpoint"):
            ray.get(create_group.remote(), timeout=60)
    finally:
        ray.shutdown()


def test_training_worker_omp_num_threads():
    """Propagate the configured thread count and allow worker-specific overrides."""
    with patch.dict(os.environ, {TRAINING_WORKER_OMP_NUM_THREADS_ENV: "7"}):
        ray.init(num_cpus=100)
        try:
            resource_pool = RayResourcePool([2], use_gpu=False)
            class_with_args = RayClassWithInitArgs(cls=TestActor)

            worker_group = RayWorkerGroup(
                resource_pool=resource_pool, ray_cls_with_init=class_with_args, name_prefix="worker_group_omp"
            )
            assert worker_group.execute_all_sync("getenv", key="OMP_NUM_THREADS") == ["7", "7"]

            worker_group = RayWorkerGroup(
                resource_pool=resource_pool,
                ray_cls_with_init=class_with_args,
                name_prefix="worker_group_omp_customized",
                worker_env={"OMP_NUM_THREADS": "3"},
            )
            assert worker_group.execute_all_sync("getenv", key="OMP_NUM_THREADS") == ["3", "3"]
        finally:
            ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
