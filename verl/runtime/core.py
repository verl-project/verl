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

"""Concrete Runtime construction and process-level lifecycle authority."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import uuid
import weakref
from collections.abc import Mapping
from contextlib import suppress
from types import TracebackType
from typing import Literal, TypeVar, cast

from verl.runtime.backend_registry import RuntimeBackend, attach_backend, create_backend
from verl.runtime.config import RuntimeConfig, materialize_backend_section, select_backend
from verl.runtime.executor import RuntimeThreadPoolExecutor, _drain_future, run_in_thread
from verl.runtime.runtime_context import (
    AttachSpec,
    RuntimeContext,
    clear_runtime_context,
    install_runtime_context,
    runtime_context,
)
from verl.single_controller.base.actor import ActorRoleMap, ActorSpec, ClassWithInitArgs
from verl.single_controller.base.fused import create_fused_worker_groups, normalize_role_actors
from verl.single_controller.base.lifecycle import close_owned
from verl.single_controller.base.resource_pool import ResourcePool
from verl.single_controller.base.topology import Topology, _format_topology
from verl.single_controller.base.worker import Worker
from verl.single_controller.base.worker_group import WorkerGroup

WorkerT = TypeVar("WorkerT", bound=Worker)

_RUNTIME: Runtime | None = None
logger = logging.getLogger(__name__)


class Runtime:
    """Create WorkerGroups and own resources created through this Runtime.

    There is at most one live Runtime per process. ``Runtime.from_config``
    establishes a root; nested callers use :func:`verl.runtime.runtime`.
    """

    def __init__(self, handle: RuntimeBackend, *, root: bool) -> None:
        self._backend = handle
        self._root = root
        self._closed = False
        self._closing = False
        self._worker_group_executor_joined = False
        context = runtime_context()
        self._runtime_id = context.runtime_id
        self._node_path = context.node_path
        self._backend_name = context.backend
        self._groups: weakref.WeakValueDictionary[int, WorkerGroup[Worker]] = weakref.WeakValueDictionary()
        self._cleanup_groups: list[WorkerGroup[Worker]] = []
        self._groups_lock = threading.Lock()
        self._worker_group_executor = RuntimeThreadPoolExecutor(
            thread_name_prefix="verl-worker-group",
        )
        self._root_pools = dict(handle.root_resource_pools())
        for name, pool in self._root_pools.items():
            pool._bind_runtime_scope(self._runtime_id, name)
        self._resource_pools = dict(self._root_pools)
        self._model_resource_pools: dict[str, ResourcePool] = {}
        self._topology = Topology()

    @property
    def topology(self) -> Topology:
        """Return the validated topology configured on this Runtime."""
        return self._topology

    @property
    def backend(self) -> str:
        """Return the selected backend name."""
        return self._backend_name

    @property
    def runtime_id(self) -> str:
        """Return the process tree's Runtime identity."""
        return self._runtime_id

    @property
    def node_path(self) -> tuple[str, ...]:
        """Return this Runtime node's path within the process tree."""
        return self._node_path

    def model_resource_pool(self, name: str) -> ResourcePool:
        """Return the model-level resource selection compiled from topology."""
        self._ensure_open()
        try:
            return self._model_resource_pools[name]
        except KeyError as exc:
            raise KeyError(
                f"unknown topology model {name!r}; available: {sorted(self._model_resource_pools)!r}"
            ) from exc

    def current_host_resource_pool(self, name: str = "parent") -> ResourcePool:
        """Return the one-host ResourcePool view containing this Worker rank."""
        self._ensure_open()
        try:
            pool = self._root_pools[name]
        except KeyError as exc:
            raise KeyError(
                f"unknown current Runtime ResourcePool {name!r}; available: {sorted(self._root_pools)!r}"
            ) from exc
        try:
            world_size = int(os.environ["WORLD_SIZE"])
            rank = int(os.environ["RANK"])
            local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
            local_rank = int(os.environ["LOCAL_RANK"])
        except (KeyError, ValueError) as exc:
            raise RuntimeError("current_host_resource_pool requires Worker SPMD rank environment") from exc
        if world_size != pool.world_size:
            raise RuntimeError(
                f"Worker WORLD_SIZE does not match its parent ResourcePool: {world_size} != {pool.world_size}"
            )
        if local_world_size != pool.processes_per_node:
            raise RuntimeError(
                "Worker LOCAL_WORLD_SIZE does not match its parent ResourcePool: "
                f"{local_world_size} != {pool.processes_per_node}"
            )
        if rank < 0 or rank >= world_size:
            raise RuntimeError(f"Worker RANK={rank} is outside WORLD_SIZE={world_size}")
        if local_rank < 0 or local_rank >= local_world_size:
            raise RuntimeError(f"Worker LOCAL_RANK={local_rank} is outside LOCAL_WORLD_SIZE={local_world_size}")
        expected_local_rank = rank % local_world_size
        if local_rank != expected_local_rank:
            raise RuntimeError(
                f"Worker LOCAL_RANK does not match RANK modulo LOCAL_WORLD_SIZE: {local_rank} != {expected_local_rank}"
            )
        node_index = rank // local_world_size
        start = node_index * pool.processes_per_node
        return pool.slice(slice(start, start + pool.processes_per_node))

    @classmethod
    def from_config(cls, config: RuntimeConfig) -> Runtime:
        """Create the root Runtime and publish it for process-local callers.

        Args:
            config: Mapping containing the backend selector and backend sections.

        Returns:
            The open process-level Runtime.

        Raises:
            RuntimeError: If a live Runtime already exists in this process.
        """

        global _RUNTIME
        if _RUNTIME is not None and not _RUNTIME._closed:
            raise RuntimeError("Runtime is already initialized in this process")
        backend = select_backend(config)
        section = materialize_backend_section(config, backend)
        env_vars = cast(dict[str, str], section["env_vars"])
        raw_topology = cast(Mapping[str, object], config).get("topology")
        parsed_topology = Topology.from_mapping(raw_topology)
        # ``models`` is the opt-in switch.  Clusters and DevicePools without a
        # model declaration must not reserve resources or change legacy placement.
        topology = parsed_topology if parsed_topology.models else Topology()
        if topology.models:
            logger.info("Runtime topology plan:\n%s", _format_topology(topology))
        runtime_id = uuid.uuid4().hex
        handle: RuntimeBackend | None = None
        context: RuntimeContext | None = None
        try:
            handle = create_backend(backend, section, topology)
            child_section = dict(handle.child_attach_config())
            context_config: dict[str, object] = {
                "backend": backend,
                "env_vars": dict(env_vars),
                backend: child_section,
            }
            object_store_config: dict[str, object] = {}
            context = RuntimeContext(
                runtime_id=runtime_id,
                node_path=("runtime",),
                config=context_config,
                object_store_config=object_store_config,
            )
            install_runtime_context(context)
            runtime = cls(handle=handle, root=True)
            runtime._resolve_topology(topology)
            handle.start_object_store(object_store_config)
        except BaseException:
            _abort_initialization(context, handle)
            raise
        _RUNTIME = runtime
        return runtime

    @classmethod
    def _attach(cls, spec: AttachSpec) -> Runtime:
        """Build this backend-spawned process's Runtime node from an AttachSpec.

        The parent delivers one :class:`AttachSpec` bundling identity, the static
        config needed to rebuild the backend, and the runtime state the root
        already decided (topology, resource pools, and ObjectStore client
        configuration). This process assembles its own RuntimeContext here
        rather than receiving one pre-derived by the parent.
        """
        global _RUNTIME
        object_store_config = dict(spec.object_store_config)
        context = RuntimeContext(
            runtime_id=spec.runtime_id,
            node_path=spec.node_path,
            config=spec.config,
            object_store_config=object_store_config,
        )
        install_runtime_context(context)
        if _RUNTIME is not None and not _RUNTIME._closed:
            if _RUNTIME._runtime_id != spec.runtime_id:
                raise RuntimeError(
                    f"process already belongs to Runtime {_RUNTIME._runtime_id!r}; cannot attach {spec.runtime_id!r}"
                )
            return _RUNTIME
        backend = context.backend
        section = materialize_backend_section(spec.config, backend)
        handle: RuntimeBackend | None = None
        try:
            handle = attach_backend(backend, section, spec.resource_pools)
            runtime = cls(handle=handle, root=False)
            runtime._topology = spec.topology
            runtime._resource_pools = dict(spec.resource_pools)
            runtime._compile_model_resource_pools()
            handle.start_object_store(object_store_config)
        except BaseException:
            _abort_initialization(context, handle)
            raise
        _RUNTIME = runtime
        return runtime

    def _resolve_topology(self, topology: Topology) -> None:
        resolved = dict(self._backend.resolve_topology(topology)) if topology.models else {}
        overlap = set(resolved) & set(self._root_pools)
        if overlap:
            raise ValueError(f"DevicePool names conflict with Runtime root pools: {sorted(overlap)!r}")
        self._topology = topology
        for name, pool in resolved.items():
            pool._bind_runtime_scope(self._runtime_id, name)
        self._resource_pools = {**self._root_pools, **resolved}
        self._compile_model_resource_pools()
        if topology.models:
            logger.info(
                "Resolved Runtime topology:\n%s",
                _format_topology(topology, self._model_resource_pools),
            )

    def _compile_model_resource_pools(self) -> None:
        views: dict[tuple[str, tuple[int, int]], ResourcePool] = {}
        device_pools = {pool.name: pool for pool in self._topology.device_pools}
        for name, key in self._topology._model_placements().items():
            pool_name, selected = key
            spec = device_pools[pool_name]
            pool = views.get(key)
            if pool is None:
                source = self._resource_pools[pool_name]
                pool = (
                    source
                    if selected == (0, spec.n_gpus_per_node)
                    else self._backend.select_device_range(source, *selected)
                )
                if pool is not source:
                    source._inherit_runtime_scope(pool)
                views[key] = pool
            self._model_resource_pools[name] = pool

    def create_resource_pool(
        self,
        *,
        nnodes: int | None,
        processes_per_node: int,
        device_type: Literal["gpu", "cpu"] = "gpu",
        on: ResourcePool | str | None = None,
    ) -> ResourcePool:
        """Derive an imperative ResourcePool from a selected host domain.

        Args:
            nnodes: Number of nodes to select. ``None`` selects every node in
                the chosen root domain.
            processes_per_node: Homogeneous process count per selected node.
            device_type: Resource class claimed by each process.
            on: Root pool or root name. Omit only when exactly one non-controller
                root exists, or when the controller root is the only root.
        Returns:
            The ResourcePool created and tracked by this Runtime.
        """
        self._ensure_open()
        root = self._resolve_root_pool(on)
        selected_nnodes = root.nnodes if nnodes is None else nnodes
        derived = self._backend.derive_resource_pool(
            root,
            nnodes=selected_nnodes,
            processes_per_node=processes_per_node,
            device_type=device_type,
        )
        root._inherit_runtime_scope(derived)
        return derived

    def _resolve_root_pool(self, on: ResourcePool | str | None) -> ResourcePool:
        if isinstance(on, ResourcePool):
            if on._runtime_owner() != self._runtime_id:
                raise ValueError("ResourcePool does not belong to this Runtime")
            return on
        if isinstance(on, str):
            try:
                return self._root_pools[on]
            except KeyError as exc:
                raise KeyError(f"unknown root ResourcePool {on!r}; available: {sorted(self._root_pools)!r}") from exc
        if on is not None:
            raise TypeError(f"on must be ResourcePool, str, or None, got {type(on)!r}")
        candidates = [pool for name, pool in self._root_pools.items() if name != "controller"]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates and "controller" in self._root_pools:
            return self._root_pools["controller"]
        raise ValueError(
            "create_resource_pool requires on= when Runtime exposes multiple root pools; "
            f"available: {sorted(self._root_pools)!r}"
        )

    def create_worker_group(
        self,
        actor: ActorSpec[WorkerT] | ActorRoleMap[WorkerT],
        *,
        on: ResourcePool,
    ) -> WorkerGroup[WorkerT] | dict[str, WorkerGroup[WorkerT]]:
        """Synchronously create one WorkerGroup or fused role views."""
        self._ensure_open()
        if self._worker_group_executor.owns_current_thread():
            created, _owned = self._create_worker_group_with_owner(actor, on=on)
            return created
        creation = self._worker_group_executor.submit(
            self._create_worker_group_with_owner,
            actor,
            on=on,
        )
        created, _owned = creation.result()
        return created

    async def create_worker_group_async(
        self,
        actor: ActorSpec[WorkerT] | ActorRoleMap[WorkerT],
        *,
        on: ResourcePool,
    ) -> WorkerGroup[WorkerT] | dict[str, WorkerGroup[WorkerT]]:
        """Create one WorkerGroup on this Runtime's bounded creation executor."""
        creation = run_in_thread(
            self._worker_group_executor,
            lambda: self._create_worker_group_with_owner(actor, on=on),
        )
        try:
            created, _owned = await asyncio.shield(creation)
            return created
        except asyncio.CancelledError as cancellation:
            # Backend creation cannot be cancelled safely once submitted. Wait
            # for its terminal outcome despite repeated cancellation so failures
            # are not silently detached.
            try:
                _created, owned = await _drain_future(creation)
            except Exception as creation_error:
                raise cancellation from creation_error

            cleanup: asyncio.Future[None] | None = None
            with self._groups_lock:
                if not self._closing:
                    # Runtime.close() sets _closing under this same lock before
                    # shutting down the executor. Therefore cleanup is either
                    # submitted before that shutdown fence or owned by Runtime.
                    cleanup = run_in_thread(
                        self._worker_group_executor,
                        lambda: self._abandon_worker_group(owned),
                    )
            if cleanup is not None:
                try:
                    await _drain_future(cleanup)
                except Exception as cleanup_error:
                    raise cancellation from cleanup_error
            raise cancellation

    def _create_worker_group_with_owner(
        self,
        actor: ActorSpec[WorkerT] | ActorRoleMap[WorkerT],
        *,
        on: ResourcePool,
    ) -> tuple[WorkerGroup[WorkerT] | dict[str, WorkerGroup[WorkerT]], WorkerGroup[WorkerT]]:
        """Create one WorkerGroup or fused role views.

        Args:
            actor: Worker class, deferred constructor, or role-to-actor mapping.
            on: ResourcePool whose physical nodes form the scheduling domain.
        Returns:
            Created WorkerGroup or non-owning role-name views, plus the owned group.
        """
        self._ensure_open()
        # Validate actor class identity before any allocation side effects.
        normalized_actor: ActorSpec[WorkerT] | dict[str, ActorSpec[WorkerT]]
        if isinstance(actor, Mapping):
            if not actor:
                raise ValueError("fused role mapping must be non-empty")
            normalized_actor = normalize_role_actors(actor)
        elif isinstance(actor, ClassWithInitArgs | type):
            normalized_actor = actor
        else:
            raise TypeError(
                f"actor must be a Worker class, ClassWithInitArgs, or role-name mapping; got {type(actor)!r}"
            )
        if not isinstance(on, ResourcePool):
            raise TypeError(f"on must be ResourcePool, got {type(on)!r}")
        pool = on
        if pool._runtime_owner() != self._runtime_id:
            raise ValueError("ResourcePool does not belong to this Runtime")
        child_topology, child_resources = self._child_resource_scope(pool)

        def spawn_root(actor: ActorSpec[WorkerT]) -> WorkerGroup[WorkerT]:
            return self._backend.create_worker_group(
                actor, pool=pool, topology=child_topology, resource_pools=child_resources
            )

        if isinstance(normalized_actor, Mapping):
            created, owned = create_fused_worker_groups(normalized_actor, spawn_root=spawn_root)
        else:
            created = owned = spawn_root(normalized_actor)
        with self._groups_lock:
            self._groups[id(owned)] = owned
        return created, owned

    def _abandon_worker_group(self, owned: WorkerGroup[Worker]) -> None:
        """Close a WorkerGroup whose caller cancelled before taking ownership."""
        try:
            owned.close()
        except BaseException:
            with self._groups_lock:
                self._cleanup_groups.append(owned)
            raise
        with self._groups_lock:
            self._groups.pop(id(owned), None)

    def _child_resource_scope(self, pool: ResourcePool) -> tuple[Topology, dict[str, ResourcePool]]:
        """Derive the child Runtime scope from topology and the selected pool."""
        if pool._runtime_root() == "controller":
            resources = {"controller": pool}
            if self._topology.device_pools:
                resources.update((item.name, self._resource_pools[item.name]) for item in self._topology.device_pools)
                return self._topology, resources
            cluster = self._resource_pools.get("cluster")
            if cluster is not None:
                resources["cluster"] = cluster
            return Topology(), resources
        return Topology(), {"parent": pool}

    def close(self) -> None:
        """Close the Runtime, clean up resources, and clear the process global.

        The Runtime lifecycle owner must serialize calls to this method.
        Concurrent ``close()`` calls are unsupported.
        """
        if self._closed:
            return
        with self._groups_lock:
            # Stop admission before waiting for already-submitted creation and
            # abandonment cleanup. A failed owner remains retryable, so
            # _closing stays set until every dependency closes.
            self._closing = True
        if not self._worker_group_executor_joined:
            self._worker_group_executor.shutdown(wait=True, cancel_futures=False)
            self._worker_group_executor_joined = True

        # Keep live owners stable during teardown; failed cleanup remains
        # strongly owned for retry, without preventing normal caller-side GC.
        groups = list(self._groups.values())
        try:
            close_owned(groups)
        finally:
            self._cleanup_groups = groups
            self._groups = weakref.WeakValueDictionary((id(group), group) for group in groups)

        if self._root:
            from verl.runtime.object_store import _close_object_store

            # WorkerGroup shutdown has completed. Drain the controller client
            # once, immediately before the backend tears down global storage.
            _close_object_store()
        self._backend.close()

        context = runtime_context()
        self._closed = True
        global _RUNTIME
        if _RUNTIME is self:
            _RUNTIME = None
        clear_runtime_context(context)

    def __enter__(self) -> Runtime:
        """Return this open Runtime for context-managed use."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the Runtime without replacing a context body's exception."""
        try:
            self.close()
        except Exception:
            if exc is None:
                raise

    def _ensure_open(self) -> None:
        if self._closing or self._closed:
            raise RuntimeError("Runtime is closed")


def _abort_initialization(context: RuntimeContext | None, handle: RuntimeBackend | None) -> None:
    """Roll back a failed root or attach while preserving its original error."""
    with suppress(Exception):
        from verl.runtime.object_store import _close_object_store

        _close_object_store()
    with suppress(Exception):
        if handle is not None:
            handle.close()
    if context is not None:
        clear_runtime_context(context)


def current_runtime() -> Runtime:
    """Return the Runtime belonging to the process-global RuntimeContext."""
    if _RUNTIME is None or _RUNTIME._closed:
        raise RuntimeError("Runtime is not initialized in this process")
    return _RUNTIME


def close_attached_runtime() -> None:
    """Close the backend-attached Runtime node in the current worker process."""
    if _RUNTIME is not None and not _RUNTIME._closed and not _RUNTIME._root:
        _RUNTIME.close()
