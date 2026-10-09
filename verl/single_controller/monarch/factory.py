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

"""Private Monarch RuntimeBackend and factory."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from typing import Literal, TypeVar
from uuid import uuid4

from verl.single_controller.base.actor import ActorSpec
from verl.single_controller.base.errors import PlacementUnavailableError
from verl.single_controller.base.resource_pool import ResourcePool
from verl.single_controller.base.topology import Topology
from verl.single_controller.base.worker import Worker
from verl.single_controller.base.worker_group import WorkerGroup

from .cluster import attach_monarch_cluster, close_monarch_cluster, get_monarch_cluster, init_monarch_cluster
from .config import MonarchRuntimeConfig, parse_monarch_runtime_config
from .resource_pool import MonarchResourcePool
from .topology import required_hosts

WorkerT = TypeVar("WorkerT", bound=Worker)
_ROOT_MESH_NAME = "hosts"


class MonarchRuntimeBackend:
    """Private Monarch backend; initializes the process-level MonarchCluster."""

    def __init__(
        self,
        config: MonarchRuntimeConfig,
        *,
        resource_pools: Mapping[str, ResourcePool] | None = None,
        topology: Topology | None = None,
    ) -> None:
        self._config = config
        self._root = resource_pools is None
        self._topology = topology or Topology()
        self._object_store_backend = None
        if not self._root:
            attach_monarch_cluster(config)
        elif config.job_mode == "process":
            init_monarch_cluster(config, local_meshes={_ROOT_MESH_NAME: required_hosts(self._topology)})
        else:
            init_monarch_cluster(config)
        try:
            if resource_pools is not None:
                self._root_pools = dict(resource_pools)
                return
            cluster = get_monarch_cluster()
            host_mesh = cluster.host_mesh(_ROOT_MESH_NAME)
            controller_mesh = cluster.controller_host_mesh()
            self._root_pools: dict[str, ResourcePool] = {
                name: MonarchResourcePool(
                    monarch_host_mesh=mesh,
                    nnodes=int(mesh.size()),
                    processes_per_node=1,
                    device_type="cpu",
                    bound=False,
                )
                for name, mesh in (("cluster", host_mesh), ("controller", controller_mesh))
            }
        except BaseException:
            with suppress(Exception):
                close_monarch_cluster()
            raise

    def root_resource_pools(self) -> Mapping[str, ResourcePool]:
        return dict(self._root_pools)

    def resolve_topology(self, topology: Topology) -> Mapping[str, ResourcePool]:
        from .topology import resolve_topology

        cluster = get_monarch_cluster()
        try:
            meshes = resolve_topology(topology, cluster.host_mesh(_ROOT_MESH_NAME))
        except TimeoutError as exc:
            raise PlacementUnavailableError(
                f"Monarch DevicePools did not become ready within {self._config.worker_ready_timeout_s:g}s"
            ) from exc
        return {
            device_pool.name: MonarchResourcePool(
                monarch_host_mesh=meshes[device_pool.name],
                nnodes=device_pool.nnodes,
                processes_per_node=device_pool.n_gpus_per_node,
                device_type="gpu",
                bound=True,
            )
            for device_pool in topology.device_pools
        }

    def derive_resource_pool(
        self,
        root: ResourcePool,
        *,
        nnodes: int,
        processes_per_node: int,
        device_type: Literal["gpu", "cpu"],
    ) -> ResourcePool:
        if not isinstance(root, MonarchResourcePool):
            raise TypeError(f"Monarch Runtime requires MonarchResourcePool, got {type(root)!r}")
        if nnodes <= 0:
            raise ValueError("nnodes must be positive")
        if nnodes > root.nnodes:
            raise ValueError(f"Monarch HostMesh has {root.nnodes} nodes; requested nnodes={nnodes}")
        selected = root
        if nnodes < root.nnodes:
            # Legacy configs may request fewer nodes than the job allocation. Selecting
            # the first nodes preserves their ordered placement and matches Ray behavior.
            selected = root.slice(slice(0, nnodes))
        return selected._with_processes(
            processes_per_node=processes_per_node,
            device_type=device_type,
        )

    def select_device_range(self, pool: ResourcePool, start: int, end: int) -> ResourcePool:
        if not isinstance(pool, MonarchResourcePool):
            raise TypeError(f"Monarch Runtime requires MonarchResourcePool, got {type(pool)!r}")
        return pool._select_device_range(start, end)

    def create_worker_group(
        self,
        actor: ActorSpec[WorkerT],
        *,
        pool: ResourcePool,
        topology: Topology,
        resource_pools: Mapping[str, ResourcePool],
    ) -> WorkerGroup[WorkerT]:
        from .worker_group import MonarchWorkerGroup

        if not isinstance(pool, MonarchResourcePool):
            raise TypeError(f"Monarch Runtime requires MonarchResourcePool, got {type(pool)!r}")
        return MonarchWorkerGroup.spawn(
            pool=pool,
            actor=actor,
            topology=topology,
            resource_pools=resource_pools,
        )

    def start_object_store(self, config: dict[str, object]) -> None:
        """Start the root backend or install its client in an attached process."""
        if self._root:
            if self._object_store_backend is None:
                from verl.single_controller.monarch.object_store.backend import TorchStoreBackend

                options = self._config.object_store
                pool = self._root_pools["cluster"]
                if options.strategy == "local_rank":
                    if not self._topology.clusters:
                        raise ValueError("local_rank ObjectStore strategy requires declarative topology")
                    pool = pool._with_processes(
                        processes_per_node=max(cluster.n_gpus_per_node for cluster in self._topology.clusters),
                        device_type="cpu",
                    )
                self._object_store_backend = TorchStoreBackend.start(
                    pool=pool,
                    store_name=f"{options.store_name_prefix}_{uuid4().hex}",
                    timeout_s=options.timeout_s,
                    strategy=options.strategy,
                    local_cache_bytes=options.local_cache_bytes,
                )
            config.update(self._object_store_backend.client_config)
        if not config:
            raise RuntimeError("Monarch ObjectStore client configuration is missing")
        from verl.single_controller.monarch.object_store import TorchStoreObjectStore

        TorchStoreObjectStore.start(**config)

    def close(self) -> None:
        get_monarch_cluster().drain_released_meshes()
        if self._object_store_backend is not None:
            self._object_store_backend.close()
            self._object_store_backend = None
        close_monarch_cluster()

    def child_attach_config(self) -> Mapping[str, object]:
        """Copy the selected Monarch section into child RuntimeContext."""
        return {
            "job_mode": self._config.job_mode,
            "worker_ready_timeout_s": self._config.worker_ready_timeout_s,
            "shutdown_timeout_s": self._config.shutdown_timeout_s,
        }


def create_backend(section: Mapping[str, object] | None, topology: Topology) -> MonarchRuntimeBackend:
    return MonarchRuntimeBackend(parse_monarch_runtime_config(section), topology=topology)


def attach_backend(
    section: Mapping[str, object] | None,
    resource_pools: Mapping[str, ResourcePool],
) -> MonarchRuntimeBackend:
    return MonarchRuntimeBackend(
        parse_monarch_runtime_config(section),
        resource_pools=resource_pools,
    )
