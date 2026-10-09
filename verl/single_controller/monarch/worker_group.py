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

"""Monarch WorkerGroup over one ActorMesh."""

from __future__ import annotations

from collections.abc import Mapping
from threading import Lock
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast
from uuid import uuid4

from verl.runtime.runtime_context import runtime_context
from verl.single_controller.base.actor import ClassWithInitArgs
from verl.single_controller.base.errors import PlacementUnavailableError
from verl.single_controller.base.remote_worker_group import registered_methods
from verl.single_controller.base.resource_pool import ResourcePool
from verl.single_controller.base.topology import Topology
from verl.single_controller.base.worker import Worker
from verl.single_controller.base.worker_group import WorkerGroup

from .cluster import MonarchCluster, get_monarch_cluster
from .errors import is_address_in_use, map_monarch_exception
from .remote_worker_group import MonarchRemoteWorkerGroup, _submit_monarch
from .resource_pool import MonarchResourcePool

if TYPE_CHECKING:
    from monarch.actor import ProcMesh

    ActorMesh = Any

WorkerT = TypeVar("WorkerT", bound=Worker)


class MonarchWorkerGroup(WorkerGroup[WorkerT], Generic[WorkerT]):
    """Owning Monarch group whose remotely callable handle is an ActorMesh."""

    _backend_submit = _submit_monarch
    _rendezvous_retry_lock = Lock()

    def __init__(
        self,
        *,
        pool: MonarchResourcePool | None,
        actor_mesh: ActorMesh,
        actor_cls: type[Any],
        role_name: str | None = None,
        proc_mesh: ProcMesh | None = None,
    ) -> None:
        self._resource_pool = pool
        self._actor_mesh: ActorMesh | None = actor_mesh
        self._actor_cls = actor_cls
        self._proc_mesh = proc_mesh
        self._worker_shutdown_complete = False
        super().__init__(
            method_metadata=registered_methods(actor_cls),
            ranks=range(int(actor_mesh.flatten("rank").size())),
            role_name=role_name,
        )

    @classmethod
    def spawn(
        cls,
        *,
        pool: MonarchResourcePool,
        actor: type[WorkerT] | ClassWithInitArgs[WorkerT],
        topology: Topology,
        resource_pools: Mapping[str, ResourcePool],
    ) -> MonarchWorkerGroup[WorkerT]:
        """Spawn one ActorMesh on the ResourcePool."""
        cluster = get_monarch_cluster()
        if isinstance(actor, ClassWithInitArgs):
            actor_cls = actor.actor_class
            init_args = actor.args
            init_kwargs = dict(actor.kwargs)
        else:
            if not isinstance(actor, type):
                raise TypeError(f"actor must be a type or ClassWithInitArgs, got {type(actor)!r}")
            ClassWithInitArgs.check_actor_class(actor)
            actor_cls = actor
            init_args = ()
            init_kwargs = {}

        constructor = ClassWithInitArgs(actor_cls, *init_args, **init_kwargs)

        def spawn_once() -> MonarchWorkerGroup[WorkerT]:
            return cls._spawn_once(pool, constructor, topology, resource_pools, cluster)

        try:
            return spawn_once()
        except Exception as exc:
            mapped = map_monarch_exception(exc)
            if not is_address_in_use(mapped):
                if mapped is exc:
                    raise
                raise mapped from exc

        # Monarch selects a free port and closes its probe socket before
        # Worker construction. Concurrent groups can therefore select the same
        # port. Keep the normal path parallel and serialize only the retry after
        # an observed rendezvous collision.
        with cls._rendezvous_retry_lock:
            try:
                return spawn_once()
            except Exception as exc:
                mapped = map_monarch_exception(exc)
                if mapped is exc:
                    raise
                raise mapped from exc

    @classmethod
    def _spawn_once(
        cls,
        pool: MonarchResourcePool,
        actor: ClassWithInitArgs[WorkerT],
        topology: Topology,
        resource_pools: Mapping[str, ResourcePool],
        cluster: MonarchCluster,
    ) -> MonarchWorkerGroup[WorkerT]:
        """Allocate a process and actor mesh, rolling back incomplete startup."""
        group_id = uuid4().hex
        processes_per_node = pool.processes_per_node
        # The child assembles its own RuntimeContext from one AttachSpec.
        # The parent only hands over identity plus the runtime state the
        # root already decided, including the global ObjectStore client config.
        parent_context = runtime_context()
        attach_spec = parent_context.child_attach_spec(
            node_path=(*parent_context.node_path, f"worker-group-{group_id}"),
            topology=topology,
            resource_pools=resource_pools,
        )
        actor_env = dict(cluster.config.env_vars)
        proc_mesh = cluster.spawn_proc_mesh(
            pool._monarch_host_mesh,
            processes_per_node=processes_per_node,
            device_type=pool.device_type,
            device_indices=pool.gpu_ids,
            name=f"wg_{group_id[:12]}",
            env_vars=actor_env,
        )
        actor_mesh: ActorMesh | None = None
        success = False
        try:
            initialized = proc_mesh.initialized
            if initialized is not None:
                initialized.get(timeout=cluster.config.worker_ready_timeout_s)
            from monarch.spmd import setup_torch_elastic_env

            setup_torch_elastic_env(proc_mesh)
            from .actor import MonarchWorkerActor

            actor_mesh = proc_mesh.spawn(
                f"worker_{group_id[:12]}",
                MonarchWorkerActor,
                actor,
                attach_spec,
                processes_per_node,
                actor_env,
            )
            actor_mesh.initialized.get(timeout=cluster.config.worker_ready_timeout_s)
            success = True
        except TimeoutError as exc:
            raise PlacementUnavailableError(
                f"Monarch WorkerGroup did not become ready within {cluster.config.worker_ready_timeout_s:g}s"
            ) from exc
        finally:
            if not success:
                root_mesh = proc_mesh if proc_mesh is not None else actor_mesh
                if root_mesh is not None:
                    try:
                        root_mesh.stop().get(timeout=cluster.config.shutdown_timeout_s)
                    except Exception:
                        pass

        return cls(
            pool=pool,
            actor_mesh=actor_mesh,
            actor_cls=actor.cls,
            proc_mesh=proc_mesh,
        )

    @property
    def resource_pool(self) -> MonarchResourcePool:
        if self._resource_pool is None:
            raise RuntimeError("detached MonarchWorkerGroup has no ResourcePool")
        return self._resource_pool

    def _view(self, start: int, stop: int) -> MonarchWorkerGroup[WorkerT]:
        view = cast(MonarchWorkerGroup[WorkerT], super()._view(start, stop))
        view._proc_mesh = None
        return view

    def _role_views(self, roles: Mapping[str, ClassWithInitArgs[Worker]]) -> dict[str, MonarchWorkerGroup[WorkerT]]:
        return {
            name: type(self)(
                pool=self.resource_pool,
                actor_mesh=self._actor_mesh,
                actor_cls=init.cls,
                role_name=name,
            )
            for name, init in roles.items()
        }

    def remote(self) -> MonarchRemoteWorkerGroup:
        if self._actor_mesh is None:
            raise RuntimeError("Monarch WorkerGroup targets are unavailable")
        return MonarchRemoteWorkerGroup(
            actor_mesh=self._actor_mesh,
            method_metadata=self._method_metadata,
            ranks=self._ranks,
            role_name=self._role_name,
        )

    def close(self) -> None:
        if self._close_non_owning_view():
            return
        if not self._ranks:
            return
        cluster = get_monarch_cluster()
        timeout = cluster.config.shutdown_timeout_s
        actor_mesh = self._actor_mesh
        proc_mesh = self._proc_mesh
        try:
            if actor_mesh is not None and not self._worker_shutdown_complete:
                selected = actor_mesh.flatten("rank").slice(rank=slice(self._ranks[0], self._ranks[-1] + 1))
                selected.__monarch_call__.call("_shutdown", (), {}).get(timeout=timeout)
                self._worker_shutdown_complete = True
            # Attached Runtime.close only releases a non-owning cluster view.
            # This owner must stop its native mesh before the root job exits.
            # Keep the completed Worker phase on a stop failure: native stop
            # may already have terminated actors, so a retry must not RPC them.
            if proc_mesh is not None:
                proc_mesh.stop().get(timeout=timeout)
            elif actor_mesh is not None:
                actor_mesh.stop().get(timeout=timeout)
        except Exception as exc:
            mapped = map_monarch_exception(exc)
            if mapped is exc:
                raise
            raise mapped from exc
        self._ranks = ()
        self._proc_mesh = None
        self._actor_mesh = None

    def __del__(self) -> None:
        if not self._owned or not self.__dict__.get("_ranks"):
            return
        mesh = self._proc_mesh if self._proc_mesh is not None else self._actor_mesh
        if mesh is not None:
            try:
                get_monarch_cluster().release_mesh(mesh)
            except Exception:
                # Explicit close reports errors; GC may run after interpreter teardown.
                pass
