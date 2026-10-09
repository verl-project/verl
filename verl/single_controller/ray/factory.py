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

"""Private Ray Runtime backend construction."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from typing import Any, Literal, TypeVar, cast

from verl.single_controller.base.actor import ActorSpec
from verl.single_controller.base.errors import PlacementUnavailableError
from verl.single_controller.base.lifecycle import close_owned
from verl.single_controller.base.resource_pool import ResourcePool
from verl.single_controller.base.topology import Topology
from verl.single_controller.base.worker import Worker
from verl.single_controller.base.worker_group import WorkerGroup
from verl.single_controller.ray.config import RayRuntimeConfig, parse_ray_section
from verl.single_controller.ray.resource_pool import RayResourcePool
from verl.single_controller.ray.topology import resolve_topology
from verl.single_controller.ray.worker_group import RayWorkerGroup

WorkerT = TypeVar("WorkerT", bound=Worker)


class RayRuntimeBackend:
    """Private Ray backend handle owned by one concrete Runtime.

    Ray cluster lifecycle (``ray.shutdown``) remains with the composition; this
    backend may call ``ray.init`` once when Ray is not yet initialized.
    """

    def __init__(
        self,
        config: RayRuntimeConfig,
        *,
        resource_pools: Mapping[str, ResourcePool] | None = None,
    ) -> None:
        self._config = config
        # Topology is resolved before imperative allocations. One creation
        # order therefore preserves derived-before-topology teardown.
        self._owned_pools: list[RayResourcePool] = []
        if resource_pools is not None:
            self._root_pools = dict(resource_pools)
            return
        cluster = RayResourcePool._from_cluster()
        self._root_pools = {
            "cluster": cluster,
            "controller": RayResourcePool._for_controller(cluster),
        }

    def root_resource_pools(self) -> Mapping[str, ResourcePool]:
        return dict(self._root_pools)

    def resolve_topology(self, topology: Topology) -> Mapping[str, ResourcePool]:
        import ray
        from ray.exceptions import GetTimeoutError

        resolved = {}
        try:
            for name, strategy in resolve_topology(topology, ray.nodes()).items():
                pool = RayResourcePool._from_placement_strategy(strategy)
                resolved[name] = pool
            ray.get(
                [group.ready() for pool in resolved.values() for group in pool._reserved_groups],
                timeout=self._config.placement_ready_timeout_s,
            )
            for pool in resolved.values():
                pool._finalize_placement(sort_groups=False)
        except BaseException as exc:
            for pool in reversed(list(resolved.values())):
                with suppress(Exception):
                    pool.close()
            if isinstance(exc, GetTimeoutError):
                raise PlacementUnavailableError(
                    f"Ray DevicePools did not become ready within {self._config.placement_ready_timeout_s:g}s"
                ) from exc
            raise
        self._owned_pools.extend(resolved.values())
        return resolved

    def derive_resource_pool(
        self,
        root: ResourcePool,
        *,
        nnodes: int,
        processes_per_node: int,
        device_type: Literal["gpu", "cpu"],
    ) -> ResourcePool:
        if not isinstance(root, RayResourcePool):
            raise TypeError(f"Ray Runtime requires RayResourcePool, got {type(root)!r}")
        if root._bound and device_type == "cpu":
            if nnodes > root.nnodes:
                raise ValueError(f"ResourcePool has {root.nnodes} nodes; requested nnodes={nnodes}")
            selected = root
            if nnodes < root.nnodes:
                selected = root.slice(slice(0, nnodes * root.processes_per_node))
            return selected._with_processes(
                processes_per_node=processes_per_node,
                device_type="cpu",
            )
        pool = root._derive(
            nnodes=nnodes,
            processes_per_node=processes_per_node,
            device_type=device_type,
        )
        if device_type == "gpu":
            self._owned_pools.append(pool)
        return pool

    def select_device_range(self, pool: ResourcePool, start: int, end: int) -> ResourcePool:
        if not isinstance(pool, RayResourcePool):
            raise TypeError(f"Ray Runtime requires RayResourcePool, got {type(pool)!r}")
        return pool._select_device_range(start, end)

    def create_worker_group(
        self,
        actor: ActorSpec[WorkerT],
        *,
        pool: ResourcePool,
        topology: Topology,
        resource_pools: Mapping[str, ResourcePool],
    ) -> WorkerGroup[WorkerT]:
        if not isinstance(pool, RayResourcePool):
            raise TypeError(f"Ray Runtime requires RayResourcePool, got {type(pool)!r}")
        owned = cast(
            WorkerGroup[WorkerT],
            RayWorkerGroup.create(
                actor,
                pool=pool,
                worker_env=self._config.env_vars,
                topology=topology,
                resource_pools=resource_pools,
                profile_steps=self._config.profile_steps,
                worker_nsight_options=self._config.worker_nsight_options,
            ),
        )
        return owned

    def start_object_store(self, config: dict[str, object]) -> None:
        if config:
            raise ValueError("Ray ObjectStore does not accept client configuration")
        from verl.single_controller.ray.object_store import RayObjectStore

        RayObjectStore.start()

    def close(self) -> None:
        close_owned(self._owned_pools)
        if self._config.timeline_json_file:
            import ray

            ray.timeline(filename=self._config.timeline_json_file)

    def child_attach_config(self) -> Mapping[str, object]:
        """Copy the selected Ray section into child RuntimeContext."""
        section: dict[str, object] = {"ray_init": dict(self._config.ray_init)}
        if self._config.timeline_json_file is not None:
            section["timeline_json_file"] = self._config.timeline_json_file
        section["placement_ready_timeout_s"] = self._config.placement_ready_timeout_s
        if self._config.profile_steps is not None:
            section["profile_steps"] = list(self._config.profile_steps)
        if self._config.worker_nsight_options is not None:
            section["worker_nsight_options"] = dict(self._config.worker_nsight_options)
        return section


def _ray_init_kwargs_with_env_vars(config: RayRuntimeConfig) -> dict[str, Any]:
    """Build ``ray.init`` kwargs, injecting root ``env_vars`` into ``runtime_env``.

    Root ``env_vars`` (including values appended while building the composition
    config) always become ``ray.init(..., runtime_env={env_vars: ...})`` so
    behavior matches the historical Ray runtime_env shape.
    """
    ray_init = dict(config.ray_init)
    if not config.env_vars:
        return ray_init
    runtime_env = dict(ray_init.get("runtime_env") or {})
    runtime_env["env_vars"] = dict(config.env_vars)
    ray_init["runtime_env"] = runtime_env
    return ray_init


def create_backend(section: Mapping[str, object] | None) -> RayRuntimeBackend:
    config = parse_ray_section(section)
    import ray

    if not ray.is_initialized():
        ray.init(**_ray_init_kwargs_with_env_vars(config))
    return RayRuntimeBackend(config)


def attach_backend(
    section: Mapping[str, object] | None,
    resource_pools: Mapping[str, ResourcePool],
) -> RayRuntimeBackend:
    config = parse_ray_section(section)
    import ray

    if not ray.is_initialized():
        raise RuntimeError("cannot attach Runtime before Ray initializes the worker process")
    return RayRuntimeBackend(config, resource_pools=resource_pools)
