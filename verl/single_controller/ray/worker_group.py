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

"""Ray WorkerGroup implementing the canonical Runtime contract."""

from __future__ import annotations

import re
import socket
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from typing import Any, Generic, TypeVar

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy

from verl.runtime.runtime_context import AttachSpec, runtime_context
from verl.single_controller.base.actor import ActorRoleMap, ActorSpec, ClassWithInitArgs
from verl.single_controller.base.errors import ExceptionGroup
from verl.single_controller.base.fused import create_fused_worker_groups
from verl.single_controller.base.remote_worker_group import registered_methods
from verl.single_controller.base.resource_pool import ResourcePool
from verl.single_controller.base.topology import Topology
from verl.single_controller.base.worker import Worker
from verl.single_controller.base.worker_group import WorkerGroup
from verl.single_controller.ray.actor import RayClassWithInitArgs
from verl.single_controller.ray.remote_worker_group import RayRemoteWorkerGroup, _submit_ray
from verl.single_controller.ray.resource_pool import RayResourcePool

WorkerT = TypeVar("WorkerT")
_SchedulingStrategy = NodeAffinitySchedulingStrategy | PlacementGroupSchedulingStrategy
_RAY_COLOCATED_RESOURCE_REQUEST = 1e-4
_UNSET = object()


def _device_visibility_env(resource_pool: RayResourcePool) -> dict[str, str]:
    """Keep Ray from clearing visibility for CPU sidecars on a GPU pool.

    ``gpu_ids`` identify placement bundles, not host device IDs. The rollout
    owner resolves physical IDs from the GPU Workers and passes them to the
    sidecar implementation.
    """
    if not resource_pool.gpu_ids or resource_pool.device_type == "gpu":
        return {}
    from verl.plugin.platform import get_platform

    platform = get_platform()
    return {name: "1" for name in platform.ray_noset_envvars()}


@ray.remote
def _get_master_addr_port(master_port_range: list[int] | None = None) -> tuple[str, str]:
    address = ray.util.get_node_ip_address().strip("[]")
    if master_port_range is None:
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
    else:
        start, stop = master_port_range
        for port in range(start, stop):
            try:
                with socket.socket() as sock:
                    sock.bind(("", port))
                    break
            except OSError:
                continue
        else:
            raise RuntimeError(f"Could not find a free port in range {master_port_range}")
    return address, str(port)


def _random_prefix(length: int = 6) -> str:
    import random
    import string

    return "".join(random.choice(string.ascii_letters + string.digits) for _ in range(length))


def _node_ip_for_rank(resource_pool: RayResourcePool, rank: int) -> str | None:
    """Return the Ray-assigned node address for one rank, if the pool pinned hosts."""
    if rank < 0 or rank >= len(resource_pool.node_ids):
        return None
    node_id = resource_pool.node_ids[rank]
    if not node_id:
        return None
    for node in ray.nodes():
        if str(node.get("NodeID")) == node_id:
            address = str(node.get("NodeManagerAddress", "")).strip("[]")
            return address or None
    return None


def _complete_nsight_options(
    worker_nsight_options: Mapping[str, Any] | None,
    profile_steps: Sequence[int] | None,
) -> dict[str, Any] | None:
    """Resolve a deferred ``capture-range-end`` against the profiled step count.

    Configs ship ``capture-range-end: null`` so users do not have to count the
    nvtx ranges each profiled step opens.
    """
    if worker_nsight_options is None:
        return None
    options = dict(worker_nsight_options)
    if options.get("capture-range-end") is None and profile_steps:
        options["capture-range-end"] = f"repeat-shutdown:{6 * len(profile_steps)}"
    return options


def _parse_legacy_options(
    bin_pack: object,
    ray_wait_register_center_timeout: object,
    compat_kwargs: dict[str, Any],
) -> tuple[Sequence[int] | None, Mapping[str, Any] | None]:
    """Validate retained constructor options and extract worker profiling config."""
    # DEPRECATED: bin_pack is retained for constructor compatibility and ignored.
    if bin_pack is not _UNSET and not isinstance(bin_pack, bool):
        raise TypeError(f"bin_pack must be a bool, got {type(bin_pack)!r}")
    # DEPRECATED: ray_wait_register_center_timeout is retained for constructor compatibility and ignored.
    if ray_wait_register_center_timeout is not _UNSET:
        if isinstance(ray_wait_register_center_timeout, bool) or not isinstance(ray_wait_register_center_timeout, int):
            raise TypeError(
                f"ray_wait_register_center_timeout must be an int, got {type(ray_wait_register_center_timeout)!r}"
            )

    profile_steps = compat_kwargs.pop("profile_steps", None)
    worker_nsight_options = compat_kwargs.pop("worker_nsight_options", None)
    legacy_use_gpu = compat_kwargs.pop("use_gpu", _UNSET)
    # DEPRECATED: use_gpu is retained for constructor compatibility and ignored.
    if legacy_use_gpu is not _UNSET and not isinstance(legacy_use_gpu, bool):
        raise TypeError(f"use_gpu must be a bool, got {type(legacy_use_gpu)!r}")
    if compat_kwargs:
        unexpected = ", ".join(sorted(compat_kwargs))
        raise TypeError(f"RayWorkerGroup got unexpected keyword argument(s): {unexpected}")

    return profile_steps, worker_nsight_options


class RayWorkerGroup(WorkerGroup[WorkerT], Generic[WorkerT]):
    """A group of Ray workers that can be managed collectively.

    This class extends WorkerGroup to provide Ray-specific functionality for
    creating and managing groups of Ray actors with specific resource requirements
    and scheduling strategies.
    """

    _backend_submit = _submit_ray

    def __init__(
        self,
        resource_pool: RayResourcePool | None = None,
        ray_cls_with_init: RayClassWithInitArgs | None = None,
        bin_pack: bool | object = _UNSET,
        name_prefix: str | None = None,
        detached: bool = False,
        worker_names: list[str] | None = None,
        worker_handles: list[ray.actor.ActorHandle] | None = None,
        ray_wait_register_center_timeout: int | object = _UNSET,
        *,
        worker_env: Mapping[str, str] | None = None,
        topology: Topology | None = None,
        resource_pools: Mapping[str, ResourcePool] | None = None,
        role_name: str | None = None,
        fused_worker_used: bool = False,
        method_metadata: dict[str, dict[str, Any]] | None = None,
        backend_method_names: Mapping[str, str] | None = None,
        actors: list[ray.actor.ActorHandle] | None = None,
        device_name: str = "cuda",
        master_addr: str | None = None,
        master_port: str | None = None,
        master_port_range: list[int] | None = None,
        **compat_kwargs: Any,
    ) -> None:
        profile_steps, worker_nsight_options = _parse_legacy_options(
            bin_pack, ray_wait_register_center_timeout, compat_kwargs
        )

        self._resource_pool = resource_pool
        self._detached = detached
        self.ray_cls_with_init = ray_cls_with_init
        self.name_prefix = _random_prefix() if name_prefix is None else name_prefix
        worker_env = dict(worker_env or {})
        resource_pools = dict(resource_pools or {})
        topology = Topology() if topology is None else topology
        self.profile_steps = profile_steps
        self.worker_nsight_options = _complete_nsight_options(worker_nsight_options, profile_steps)
        self.device_name = device_name
        self._master_addr = master_addr
        self._master_port = master_port
        self.fused_worker_used = fused_worker_used or (
            False if ray_cls_with_init is None else bool(getattr(ray_cls_with_init, "fused_worker_used", False))
        )
        self.wg_dict: dict[str, RayWorkerGroup[WorkerT]] | None = None
        self._backend_method_names = dict(backend_method_names or {})
        if actors is not None:
            self._actors = list(actors)
            self._worker_names = list(worker_names or [])
        elif worker_handles is not None:
            self._actors = list(worker_handles)
            self._worker_names = list(worker_names or [f"detached_{idx}" for idx in range(len(self._actors))])
        elif worker_names:
            self._actors = [ray.get_actor(name=name) for name in worker_names]
            self._worker_names = list(worker_names)
        else:
            self._actors = []
            self._worker_names = list(worker_names or [])
        # Runtime construction path.
        if not self._actors and resource_pool is not None and ray_cls_with_init is not None:
            if not resource_pool.node_ids and not resource_pool._reserved_groups:
                resource_pool.get_placement_groups()
            context = runtime_context()
            attach_spec = context.child_attach_spec(
                node_path=(*context.node_path, f"worker-group-{self.name_prefix}"),
                topology=topology,
                resource_pools=resource_pools,
            )
            self._create_actors_from_pool(
                ray_cls_with_init,
                worker_env=worker_env,
                attach_spec=attach_spec,
                detached=detached,
                master_port_range=master_port_range,
            )

        metadata = dict(method_metadata or {})
        if self.ray_cls_with_init is not None and not metadata:
            user_cls = self.ray_cls_with_init._runtime_user_cls or self._user_cls(self.ray_cls_with_init.cls)
            metadata = registered_methods(user_cls)
        super().__init__(
            method_metadata=metadata,
            ranks=range(len(self._actors)),
            role_name=role_name,
        )
        self._ray_actors = self._actors

    @classmethod
    def from_detached(
        cls,
        name_prefix: str | None = None,
        worker_names: list[str] | None = None,
        worker_handles: list[ray.actor.ActorHandle] | None = None,
        ray_cls_with_init: RayClassWithInitArgs | None = None,
        role_name: str | None = None,
        fused_worker_used: bool = False,
        method_metadata: dict[str, dict[str, Any]] | None = None,
        backend_method_names: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> RayWorkerGroup[Any]:
        """Create a worker group from existing detached workers.

        Args:
            name_prefix: Prefix for worker names
            worker_names: Names of existing workers to attach to
            worker_handles: Existing worker handles to retain strongly
            ray_cls_with_init: Class with initialization arguments for workers
            role_name: Optional fused role selected by this view
            fused_worker_used: Whether this is a fused WorkerGroup
            method_metadata: Registered Worker method metadata
            backend_method_names: Backend method names for legacy colocated workers

        Returns:
            A new non-owning RayWorkerGroup instance
        """
        group = cls(
            resource_pool=None,
            ray_cls_with_init=ray_cls_with_init,
            name_prefix=name_prefix,
            worker_names=worker_names,
            worker_handles=worker_handles,
            role_name=role_name,
            fused_worker_used=fused_worker_used,
            method_metadata=method_metadata,
            backend_method_names=backend_method_names,
            **kwargs,
        )
        group._owned = False
        return group

    @classmethod
    def create(
        cls,
        actor: ActorSpec[WorkerT] | ActorRoleMap[WorkerT],
        *,
        pool: RayResourcePool,
        worker_env: Mapping[str, str] | None = None,
        topology: Topology | None = None,
        resource_pools: Mapping[str, ResourcePool] | None = None,
        profile_steps: Sequence[int] | None = None,
        worker_nsight_options: Mapping[str, Any] | None = None,
    ) -> RayWorkerGroup[WorkerT] | tuple[dict[str, RayWorkerGroup[WorkerT]], RayWorkerGroup[WorkerT]]:
        """Canonical construction used by the Ray RuntimeBackend."""
        if isinstance(actor, Mapping):

            def spawn_root(init: ClassWithInitArgs[Worker]) -> RayWorkerGroup[WorkerT]:
                fused_cia = RayClassWithInitArgs.from_class_init(init)
                fused_cia.fused_worker_used = True
                return cls(
                    resource_pool=pool,
                    ray_cls_with_init=fused_cia,
                    worker_env=worker_env,
                    topology=topology,
                    resource_pools=resource_pools,
                    fused_worker_used=True,
                    profile_steps=profile_steps,
                    worker_nsight_options=worker_nsight_options,
                )

            return create_fused_worker_groups(actor, spawn_root=spawn_root)

        if isinstance(actor, ClassWithInitArgs):
            cia = RayClassWithInitArgs.from_class_init(actor)
        elif isinstance(actor, type):
            cia = RayClassWithInitArgs.from_class_init(ClassWithInitArgs(actor))
        else:
            raise TypeError(f"actor must be a type, ClassWithInitArgs, or role mapping, got {type(actor)!r}")

        return cls(
            resource_pool=pool,
            ray_cls_with_init=cia,
            worker_env=worker_env,
            topology=topology,
            resource_pools=resource_pools,
            profile_steps=profile_steps,
            worker_nsight_options=worker_nsight_options,
        )

    def _role_views(self, roles: Mapping[str, ClassWithInitArgs[Worker]]) -> dict[str, RayWorkerGroup[WorkerT]]:
        views = {
            name: type(self)(
                resource_pool=self.resource_pool,
                ray_cls_with_init=RayClassWithInitArgs.from_class_init(init),
                role_name=name,
                fused_worker_used=True,
                actors=self._actors,
                worker_names=self._worker_names,
            )
            for name, init in roles.items()
        }
        self.fused_worker_used = True
        self.ray_cls_with_init.fused_worker_used = True
        return views

    @staticmethod
    def _user_cls(cls: Any) -> type[Any]:
        if hasattr(cls, "__ray_actor_class__"):
            return cls.__ray_actor_class__
        return cls

    @property
    def resource_pool(self) -> RayResourcePool:
        if self._resource_pool is None:
            raise RuntimeError("detached RayWorkerGroup has no ResourcePool")
        return self._resource_pool

    @property
    def _workers(self) -> list[ray.actor.ActorHandle]:
        # Legacy compatibility alias used by some callers/tests.
        return self._selected_actors()

    @property
    def workers(self) -> list[ray.actor.ActorHandle]:
        return self._selected_actors()

    @property
    def worker_names(self) -> list[str]:
        if self._worker_names:
            return [self._worker_names[rank] for rank in self._ranks]
        return [f"worker_{rank}" for rank in self._ranks]

    def _selected_actors(self) -> list[ray.actor.ActorHandle]:
        return [self._actors[rank] for rank in self._ranks]

    def _execute_remote_single_worker(
        self,
        worker: ray.actor.ActorHandle,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> ray.ObjectRef:
        """Execute a method on a single worker remotely.

        Args:
            worker: The worker actor handle
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Remote object reference to the method execution
        """
        backend_method, backend_args = self._route_invocation(method_name, args)
        return getattr(worker, backend_method).remote(*backend_args, **kwargs)

    def remote(self) -> RayRemoteWorkerGroup:
        return RayRemoteWorkerGroup(
            actor_handles=self._actors,
            method_metadata=self._method_metadata,
            ranks=self._ranks,
            role_name=self._role_name,
        )

    def __del__(self) -> None:
        if not self._owned or self._detached or not self.__dict__.get("_ranks") or not ray.is_initialized():
            return
        for rank in self._ranks:
            try:
                ray.kill(self._actors[rank], no_restart=True)
            except Exception:
                # Explicit close reports errors; GC may run after interpreter teardown.
                pass

    def close(self) -> None:
        if self._close_non_owning_view():
            return
        failures: dict[int, Exception] = {}
        shutdown_refs: list[tuple[int, ray.ObjectRef]] = []

        # Submit every shutdown before observing any one actor so independent
        # actors can make progress concurrently.
        for rank in self._ranks:
            actor = self._actors[rank]
            try:
                shutdown_refs.append((rank, actor._shutdown.remote()))
            except Exception as exc:  # noqa: BLE001
                failures[rank] = exc

        # One common deadline bounds total shutdown observation time. Giving
        # every rank a fresh timeout would make cleanup grow as ranks * 30s.
        deadline = time.monotonic() + 30.0
        for rank, shutdown_ref in shutdown_refs:
            try:
                ray.get(shutdown_ref, timeout=max(0.0, deadline - time.monotonic()))
            except Exception as exc:  # noqa: BLE001
                failures[rank] = exc

        for rank in self._ranks:
            if rank in failures:
                continue
            actor = self._actors[rank]
            try:
                ray.kill(actor, no_restart=True)
            except Exception as exc:  # noqa: BLE001
                failures[rank] = exc
        self._ranks = tuple(rank for rank in self._ranks if rank in failures)
        errors = [failures[rank] for rank in sorted(failures)]
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("runtime failures", errors)

    def execute_rank_zero(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Alias for execute_rank_zero_async."""
        return self.execute_rank_zero_async(method_name, *args, **kwargs)

    def execute_all(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Alias for execute_all_async."""
        return self.execute_all_async(method_name, *args, **kwargs)

    # --- compatibility helpers retained for migration ---

    def spawn(self, prefix_set):
        """Spawn to a dictionary of worker groups, each with a subset of method with prefix.

        Args:
            prefix_set: Set of prefixes to create worker groups for

        Returns:
            Dictionary of worker groups keyed by prefix
        """
        if self.fused_worker_used:
            return self.spawn_fused(prefix_set)

        if self.ray_cls_with_init is None:
            raise RuntimeError("legacy colocated spawn requires its deferred Ray class")
        role_metadata = getattr(self.ray_cls_with_init, "_colocated_role_method_metadata", {})
        role_backend_methods = getattr(self.ray_cls_with_init, "_colocated_role_backend_methods", {})
        result = {}
        for prefix in prefix_set:
            if prefix not in role_metadata or prefix not in role_backend_methods:
                raise KeyError(f"unknown colocated Worker role {prefix!r}")
            new_wg = self.from_detached(
                name_prefix=self.name_prefix,
                worker_names=self._worker_names,
                worker_handles=self._actors,
                ray_cls_with_init=self.ray_cls_with_init,
                method_metadata=role_metadata[prefix],
                backend_method_names=role_backend_methods[prefix],
            )
            new_wg._owned = False
            new_wg._parent_group = self
            result[prefix] = new_wg
        return result

    def spawn_fused(self, prefix_set):
        """Create a dictionary of worker groups for fused workers.

        Args:
            prefix_set: Set of prefixes to create worker groups for

        Returns:
            Dictionary of worker groups keyed by prefix
        """
        wg_dict = {}
        for key in prefix_set:
            new_wg = self.slice(0, self.world_size)
            new_wg._role_name = key
            if self.ray_cls_with_init is not None and hasattr(self.ray_cls_with_init.cls, "raw_cls_dict"):
                new_wg._method_metadata = registered_methods(self.ray_cls_with_init.cls.raw_cls_dict[key])
            new_wg._owned = False
            wg_dict[key] = new_wg
        return wg_dict

    def fuse(self, prefix_set):
        """Fuse multiple worker groups into the current worker group.

        Args:
            prefix_set: Set of prefixes to fuse into the worker group
        """
        if self.wg_dict is None:
            self.wg_dict = self.spawn(prefix_set)
        for role_name, role_wg in self.wg_dict.items():
            setattr(self, role_name, role_wg)
        if self.ray_cls_with_init is not None:
            user_cls = self.ray_cls_with_init._runtime_user_cls or self._user_cls(self.ray_cls_with_init.cls)
            self._method_metadata = registered_methods(user_cls)

    def _route_invocation(self, method_name: str, args: tuple[Any, ...]) -> tuple[str, tuple[Any, ...]]:
        if self._role_name is not None:
            return super()._route_invocation(method_name, args)
        return self._backend_method_names.get(method_name, method_name), args

    def _create_actors_from_pool(
        self,
        ray_cls_with_init: RayClassWithInitArgs,
        *,
        worker_env: Mapping[str, str],
        attach_spec: AttachSpec | None,
        detached: bool,
        master_port_range: list[int] | None,
    ) -> None:
        resource_pool = self.resource_pool
        workers: list[ray.actor.ActorHandle] = []
        names: list[str] = []
        try:
            placements = [
                (
                    rank,
                    rank % resource_pool.processes_per_node,
                    resource_pool._scheduling_strategy(rank),
                )
                for rank in range(resource_pool.world_size)
            ]
            self._ensure_master_addr_port(placements[0][2], master_port_range)
            for placement in placements:
                worker, name = self._create_one_actor(
                    ray_cls_with_init,
                    placement,
                    worker_env=worker_env,
                    attach_spec=attach_spec,
                    detached=detached,
                )
                workers.append(worker)
                names.append(name)
        except BaseException:
            # The handles are the rollback owners until the whole group exists.
            for worker in reversed(workers):
                with suppress(Exception):
                    ray.kill(worker, no_restart=True)
            raise
        self._actors.extend(workers)
        self._worker_names.extend(names)

    def _ensure_master_addr_port(
        self, scheduling_strategy: _SchedulingStrategy, master_port_range: list[int] | None
    ) -> None:
        if self._master_addr is None and self._master_port is None:
            self._master_addr, self._master_port = ray.get(
                _get_master_addr_port.options(
                    scheduling_strategy=scheduling_strategy,
                ).remote(master_port_range=master_port_range)
            )
        elif self._master_addr is None or self._master_port is None:
            raise ValueError("Both master_addr and master_port must be provided together")

    def _create_one_actor(
        self,
        ray_cls_with_init: RayClassWithInitArgs,
        placement: tuple[int, int, _SchedulingStrategy],
        *,
        worker_env: Mapping[str, str],
        attach_spec: AttachSpec | None,
        detached: bool,
    ) -> tuple[ray.actor.ActorHandle, str]:
        resource_pool = self.resource_pool
        rank, local_rank, scheduling_strategy = placement
        world_size = resource_pool.world_size
        local_world_size = resource_pool.processes_per_node
        uses_resource_bundles = bool(resource_pool._reserved_groups)
        num_gpus = 1 / resource_pool.max_colocate_count if resource_pool.device_type == "gpu" else 0
        actor_env = Worker._spmd_environment(
            world_size=world_size,
            rank=rank,
            local_world_size=local_world_size,
            local_rank=local_rank,
            master_addr=str(self._master_addr or ""),
            master_port=str(self._master_port or ""),
            env_vars=worker_env,
        )
        actor_env.update(
            {
                "WG_PREFIX": self.name_prefix,
                "WG_BACKEND": "ray",
                "RAY_LOCAL_WORLD_SIZE": str(local_world_size),
            }
        )
        node_ip = _node_ip_for_rank(resource_pool, rank)
        if node_ip:
            actor_env["VERL_NODE_IP"] = node_ip
        # Ray maps GPU-accounted actors to their single assigned bundle before
        # user code runs. CPU-accounted sidecars retain the parent GPU view.
        actor_env.update(_device_visibility_env(resource_pool))

        if ray_cls_with_init._runtime_user_cls is not None:
            cia_name = ray_cls_with_init._runtime_user_cls.__name__
        else:
            cia_name = type(ray_cls_with_init.cls).__name__
            match = re.search(r"ActorClass\(([^)]+)\)", cia_name)
            cia_name = match.group(1) if match else cia_name
        pg_idx = rank // local_world_size
        name = f"{self.name_prefix}{cia_name}_{pg_idx}:{local_rank}"
        runtime_env: dict[str, Any] = {"env_vars": actor_env}
        if self.profile_steps and self.worker_nsight_options and self.device_name == "cuda":
            runtime_env["nsight"] = self.worker_nsight_options
        ray_cls_with_init.update_options(
            {
                "runtime_env": runtime_env,
                "name": name,
                "num_cpus": (
                    1
                    if resource_pool.device_type == "cpu" or uses_resource_bundles
                    else _RAY_COLOCATED_RESOURCE_REQUEST
                ),
            }
        )
        if detached:
            ray_cls_with_init.update_options({"lifetime": "detached"})

        worker = ray_cls_with_init(
            use_gpu=resource_pool.device_type == "gpu",
            num_gpus=num_gpus,
            device_name=self.device_name,
            scheduling_strategy=scheduling_strategy,
            attach_spec=attach_spec,
        )

        return worker, name
