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

"""Private built-in backend factory registry for the concrete Runtime.

Built-in Ray and Monarch adapters register lazy loaders here. This is not a
third-party plugin seam; only built-in backends are accepted.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Protocol, TypeVar

from verl.single_controller.base.actor import ActorSpec
from verl.single_controller.base.resource_pool import ResourcePool
from verl.single_controller.base.topology import Topology
from verl.single_controller.base.worker import Worker
from verl.single_controller.base.worker_group import WorkerGroup

WorkerT = TypeVar("WorkerT", bound=Worker)


class RuntimeBackend(Protocol):
    """Private backend handle owned by one concrete Runtime instance."""

    def root_resource_pools(self) -> Mapping[str, ResourcePool]:
        """Return backend-discovered named root resource domains."""
        ...

    def resolve_topology(self, topology: Topology) -> Mapping[str, ResourcePool]:
        """Resolve declared DevicePools into backend ResourcePools."""
        ...

    def select_device_range(self, pool: ResourcePool, start: int, end: int) -> ResourcePool:
        """Compile one per-node model device range into a non-owning pool view."""
        ...

    def derive_resource_pool(
        self,
        root: ResourcePool,
        *,
        nnodes: int,
        processes_per_node: int,
        device_type: Literal["gpu", "cpu"],
    ) -> ResourcePool:
        """Derive an imperative pool from one backend resource domain."""
        ...

    def create_worker_group(
        self,
        actor: ActorSpec[WorkerT],
        *,
        pool: ResourcePool,
        topology: Topology,
        resource_pools: Mapping[str, ResourcePool],
    ) -> WorkerGroup[WorkerT]:
        """Construct one owning WorkerGroup on ``pool``.

        The backend delivers each spawned process the AttachSpec it needs to
        build its Runtime node and start its process-global ObjectStore.
        """
        ...

    def start_object_store(self, config: dict[str, object]) -> None:
        """Start the root backend or an attached client."""
        ...

    def close(self) -> None:
        """Close backend-owned resources when Runtime closes."""
        ...

    def child_attach_config(self) -> Mapping[str, object]:
        """Return the serializable backend section copied into child RuntimeContext."""
        ...


def create_backend(name: str, section: Mapping[str, object] | None, topology: Topology) -> RuntimeBackend:
    if name == "ray":
        from verl.single_controller.ray.factory import create_backend as create_ray_backend

        return create_ray_backend(section)
    if name == "monarch":
        from verl.single_controller.monarch.factory import create_backend as create_monarch_backend

        return create_monarch_backend(section, topology)
    raise ValueError(f"Unknown runtime backend: {name!r}")


def attach_backend(
    name: str,
    section: Mapping[str, object] | None,
    resource_pools: Mapping[str, ResourcePool],
) -> RuntimeBackend:
    if name == "ray":
        from verl.single_controller.ray.factory import attach_backend as attach_ray_backend

        return attach_ray_backend(section, resource_pools)
    if name == "monarch":
        from verl.single_controller.monarch.factory import attach_backend as attach_monarch_backend

        return attach_monarch_backend(section, resource_pools)
    raise ValueError(f"Unknown runtime backend: {name!r}")
