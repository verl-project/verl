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
import time
from datetime import timedelta

import pytest
import ray
import torch
import torch.distributed as dist

from verl.single_controller.base.worker import Worker
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup


class TestActor(Worker):
    """Expose worker state and optionally initialize Gloo during construction."""

    def __init__(self, init_process_group=False) -> None:
        """Optionally inspect the master port and initialize a process group."""
        super().__init__()
        if not init_process_group:
            return

        assert not dist.is_initialized()
        self.master_port_was_bound = False
        if self.rank == 0:
            with socket.socket() as competitor:
                try:
                    competitor.bind((self.getenv("MASTER_ADDR"), int(self.getenv("MASTER_PORT"))))
                except OSError as error:
                    assert error.errno == errno.EADDRINUSE
                    self.master_port_was_bound = True

        dist.init_process_group(backend="gloo", timeout=timedelta(seconds=30))
        value = torch.tensor(self.rank + 1)
        dist.all_reduce(value)
        self.result = value.item()
        dist.destroy_process_group()

    def getenv(self, key):
        """Read an environment variable from the worker process."""
        return os.getenv(key, f"{key} not set")

    def get_pid(self):
        """Identify the worker process before and after a Ray actor restart."""
        return os.getpid()

    def was_master_port_bound(self):
        """Return the port occupancy observed before process-group initialization."""
        return self.master_port_was_bound

    def release_master_port(self):
        """Drop the retained TCPStore before a competing listener claims the port."""
        del self.master_tcp_store

    def assert_master_port_available(self):
        """Bind and listen after PG teardown to rule out a retained listener."""
        with socket.socket() as competitor:
            competitor.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            competitor.bind((self.getenv("MASTER_ADDR"), int(self.getenv("MASTER_PORT"))))
            competitor.listen()

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


@pytest.mark.parametrize("worker_cls", [TestActor, AsyncTestActor], ids=["sync", "async"])
def test_basics(worker_cls):
    """Propagate worker environments and initialize Gloo using the prebound port."""
    ray.init(num_cpus=100)
    try:

        @ray.remote(num_cpus=0)
        def run_collective():
            """Bound group startup, including a constructor waiting for other ranks."""
            resource_pool = RayResourcePool([4], use_gpu=False)
            class_with_args = RayClassWithInitArgs(cls=worker_cls, init_process_group=True)
            worker_group = RayWorkerGroup(
                resource_pool=resource_pool, ray_cls_with_init=class_with_args, name_prefix="worker_group_basic"
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


@pytest.mark.parametrize(
    "decorator_restarts, option_restarts",
    [(1, None), (0, -1)],
    ids=["decorator-one-restart", "options-override-unlimited"],
)
def test_worker_restart_preserves_endpoint(decorator_restarts, option_restarts):
    """Restore the endpoint after creator exit and release metadata with its placement group."""
    ray.init(num_cpus=4)
    try:

        @ray.remote(num_cpus=0)
        class GroupCreator:
            """Create a detached group from a process that the test can terminate."""

            def create_group(self):
                """Return the worker and its placement group before the creator exits."""
                pool = RayResourcePool([1], use_gpu=False, max_colocate_count=1, detached=True)
                restartable_actor = ray.remote(max_restarts=decorator_restarts, max_task_retries=-1)(
                    TestActor.__ray_actor_class__
                )
                factory = RayClassWithInitArgs(restartable_actor, init_process_group=True)
                if option_restarts is not None:
                    factory.update_options({"max_restarts": option_restarts})
                group = RayWorkerGroup(resource_pool=pool, ray_cls_with_init=factory, detached=True)
                return group.workers[0], pool.pgs[0], (group.master_address, group.master_port)

        creator = GroupCreator.remote()
        worker, pg, endpoint = ray.get(creator.create_group.remote(), timeout=60)
        before = ray.get(worker.get_pid.remote(), timeout=30)
        assert ray.get(worker.get_result.remote(), timeout=30) == 1
        metadata_ids = [
            actor_id
            for actor_id, state in ray._private.state.actors().items()
            if state["ActorClassName"] == "MasterEndpointMetadata" and state["State"] == "ALIVE"
        ]
        assert len(metadata_ids) == 1
        metadata_id = metadata_ids[0]

        ray.kill(creator, no_restart=True)
        deadline = time.monotonic() + 15
        while ray._private.state.actors(creator._actor_id.hex())["State"] != "DEAD":
            assert time.monotonic() < deadline, "The detached worker's creator did not exit"
            time.sleep(0.1)

        ray.kill(worker, no_restart=False)
        after = before
        deadline = time.monotonic() + 30
        while after == before and time.monotonic() < deadline:
            time.sleep(0.1)
            after = ray.get(worker.get_pid.remote(), timeout=30)
        assert after != before
        assert ray.get(worker.getenv.remote("MASTER_ADDR"), timeout=30) == endpoint[0]
        assert ray.get(worker.getenv.remote("MASTER_PORT"), timeout=30) == endpoint[1]
        assert ray.get(worker.get_result.remote(), timeout=30) == 1
        ray.get(worker.assert_master_port_available.remote(), timeout=30)

        ray.util.remove_placement_group(pg)
        deadline = time.monotonic() + 15
        while ray._private.state.actors(metadata_id)["State"] != "DEAD":
            assert time.monotonic() < deadline, "Endpoint metadata outlived its placement group"
            time.sleep(0.1)
    finally:
        ray.shutdown()


def test_worker_restart_with_occupied_endpoint():
    """Fail on the published port during restart even when another candidate is free."""
    ray.init(num_cpus=4)
    try:

        @ray.remote(num_cpus=0)
        def restart_on_occupied_port():
            """Occupy the published port before the restarted constructor initializes PyTorch."""
            with socket.socket() as candidate:
                candidate.bind(("", 0))
                port = candidate.getsockname()[1]
                with socket.socket() as successor:
                    successor.bind(("", port + 1))
            pool = RayResourcePool([1], use_gpu=False, max_colocate_count=1)
            restartable_actor = ray.remote(max_restarts=1, max_task_retries=-1)(TestActor.__ray_actor_class__)
            group = RayWorkerGroup(
                resource_pool=pool,
                ray_cls_with_init=RayClassWithInitArgs(restartable_actor, init_process_group=True),
                master_port_range=[port, port + 2],
            )
            worker = group.workers[0]
            master_addr = group.master_address
            assert group.master_port == str(port)
            before = ray.get(worker.get_pid.remote(), timeout=30)
            assert ray.get(worker.get_result.remote(), timeout=30) == 1
            ray.get(worker.release_master_port.remote(), timeout=30)

            with socket.socket() as listener:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind((master_addr, port))
                listener.listen()
                with socket.socket() as successor:
                    successor.bind(("", port + 1))
                ray.kill(worker, no_restart=False)
                with pytest.raises(ray.exceptions.RayActorError, match="init_process_group") as failure:
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        observed = ray.get(worker.get_pid.remote(), timeout=30)
                        assert observed == before, "The worker restarted on a different master endpoint"
                        time.sleep(0.1)
                    raise AssertionError("The worker did not fail on its occupied master port")
                message = str(failure.value)
                assert "EADDRINUSE" in message or f"errno: {errno.EADDRINUSE}" in message
                assert str(port) in message

        ray.get(restart_on_occupied_port.remote(), timeout=90)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
