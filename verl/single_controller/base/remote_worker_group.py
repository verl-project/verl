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

"""Backend-neutral WorkerGroup invocation over backend handles."""

from __future__ import annotations

import inspect
import warnings
from abc import ABC
from collections.abc import Callable, Mapping, Sequence
from copy import copy
from functools import partial
from typing import Any, Generic, TypeVar

from verl.single_controller.base.decorator import (
    MAGIC_ATTR,
    Dispatch,
    Execute,
    _materialize_futures,
    collect_lazy_compute_data_proto,
    get_predefined_dispatch_fn,
    get_predefined_execute_fn,
)
from verl.single_controller.base.dependency import (
    dependency_remote_call,
    resolve_submit_dependencies,
    walk_remote_call_dependencies,
)
from verl.single_controller.base.remote_call import RemoteCall, create_deadline
from verl.single_controller.base.worker import Worker

WorkerT = TypeVar("WorkerT", covariant=True)


def registered_methods(worker_cls: type[Any]) -> dict[str, dict[str, Any]]:
    """Build the Actor method manifest, preserving explicit distributed metadata."""
    registered: dict[str, dict[str, Any]] = {}
    for method_name in dir(worker_cls):
        try:
            method = getattr(worker_cls, method_name)
        except Exception:  # noqa: BLE001 - class properties may reject access
            continue
        if not callable(method):
            continue
        metadata = getattr(method, MAGIC_ATTR, None)
        if metadata is None:
            for base in worker_cls.__mro__[1:]:
                base_method = base.__dict__.get(method_name)
                if base_method is not None and hasattr(base_method, MAGIC_ATTR):
                    metadata = getattr(base_method, MAGIC_ATTR)
                    break
        if metadata is None:
            continue
        if not isinstance(metadata, dict):
            raise TypeError(f"register metadata on {method_name} must be a dict, got {type(metadata)!r}")
        if "dispatch_mode" not in metadata:
            raise ValueError(f"register metadata on {method_name} missing dispatch_mode")
        registered[method_name] = dict(metadata)

    for base in worker_cls.__mro__:
        if base is Worker:
            break
        for method_name, descriptor in base.__dict__.items():
            if method_name in registered or method_name.startswith("_") or method_name == "close":
                continue
            if isinstance(descriptor, staticmethod | classmethod):
                descriptor = descriptor.__func__
            if not callable(descriptor):
                continue
            method = getattr(worker_cls, method_name)
            registered[method_name] = {
                "dispatch_mode": Dispatch.ONE_TO_ALL,
                "execute_mode": Execute.ALL,
                "blocking": not inspect.iscoroutinefunction(method),
                "collect_single": True,
            }
    return registered


def _dispatch_and_collect(metadata: Mapping[str, Any]) -> tuple[Callable[..., Any], Callable[..., Any]]:
    dispatch_mode = metadata["dispatch_mode"]
    if isinstance(dispatch_mode, Dispatch):
        functions = get_predefined_dispatch_fn(dispatch_mode=dispatch_mode)
        dispatch, collect = functions["dispatch_fn"], functions["collect_fn"]
        if metadata.get("collect_single"):
            base_collect = collect

            def collect(worker_group, values):
                result = base_collect(worker_group, values)
                if worker_group.world_size != 1:
                    return result
                if not isinstance(result, list) or len(result) != 1:
                    raise RuntimeError("single-Actor collection must contain exactly one result")
                return result[0]

        return dispatch, collect
    if not isinstance(dispatch_mode, dict) or "dispatch_fn" not in dispatch_mode or "collect_fn" not in dispatch_mode:
        raise TypeError("dispatch_mode must be a Dispatch or a dict containing dispatch_fn and collect_fn")
    return dispatch_mode["dispatch_fn"], dispatch_mode["collect_fn"]


def _returns_data_proto_future(metadata: Mapping[str, Any]) -> bool:
    dispatch_mode = metadata["dispatch_mode"]
    if dispatch_mode in (Dispatch.DP_COMPUTE_PROTO, Dispatch.DP_COMPUTE_PROTO_WITH_FUNC):
        return True
    if not isinstance(dispatch_mode, dict):
        return False
    collect = dispatch_mode["collect_fn"]
    while isinstance(collect, partial):
        collect = collect.func
    return collect is collect_lazy_compute_data_proto


def _rank_inputs(
    dispatched_args: Any,
    dispatched_kwargs: Any,
    *,
    rank: int,
    world_size: int,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    args = dispatched_args if isinstance(dispatched_args, tuple | list) else (dispatched_args,)
    kwargs = dict(dispatched_kwargs) if isinstance(dispatched_kwargs, Mapping) else {}
    if not all(isinstance(value, list | tuple) and len(value) == world_size for value in args):
        return tuple(args), kwargs
    if not all(isinstance(value, list | tuple) and len(value) == world_size for value in kwargs.values()):
        return tuple(args), kwargs
    return tuple(value[rank] for value in args), {key: value[rank] for key, value in kwargs.items()}


def _remove_padding(output: Any, count: int) -> Any:
    if count > 0:
        if hasattr(output, "select_idxs"):
            return output.select_idxs(list(range(len(output)))[:-count])
        if isinstance(output, list):
            return output[:-count]
    return output


def _collect_data_future(output: Any, padding_count: int) -> Any:
    from verl.protocol import DataProtoFuture

    if not isinstance(output, DataProtoFuture):
        raise TypeError("DataProto dispatch collect must return DataProtoFuture")
    if padding_count > 0:
        previous_dispatch = output.dispatch_fn

        def remove_padding(value):
            if previous_dispatch is not None:
                value = previous_dispatch(value)
            return value.select_idxs(list(range(len(value)))[:-padding_count])

        output.dispatch_fn = remove_padding
    return output


class RemoteWorkerGroup(ABC, Generic[WorkerT]):
    """A non-owning WorkerGroup facade backed by remote handles.

    Dispatch, collection, rank selection, and the execute family are shared by
    every backend. A concrete backend only turns rank calls into its RPC
    handle. Resource placement and lifecycle ownership belong to
    ``WorkerGroup`` and are intentionally absent here.
    """

    fused_worker_execute_fn_name = "_fuw_execute"
    _backend_submit: Callable[..., RemoteCall[Any]]

    def __init__(
        self,
        *,
        method_metadata: Mapping[str, Mapping[str, Any]],
        ranks: Sequence[int],
        role_name: str | None = None,
    ) -> None:
        self._method_metadata = {name: dict(metadata) for name, metadata in method_metadata.items()}
        self._ranks = tuple(ranks)
        self._role_name = role_name
        self._dispatch_info: dict[str, Any] = {}
        self._collect_info: dict[str, Any] = {}

    def __getattribute__(self, method_name: str) -> Any:
        if not method_name.startswith("__"):
            state = object.__getattribute__(self, "__dict__")
            metadata = state.get("_method_metadata", {})
            if method_name in metadata:
                descriptor = inspect.getattr_static(type(self), method_name, None)
                if descriptor is None or callable(descriptor):
                    return RemoteWorkerGroup._bind_worker_method(self, method_name)
        return object.__getattribute__(self, method_name)

    def __getattr__(self, method_name: str) -> Any:
        metadata = self.__dict__.get("_method_metadata", {})
        if method_name not in metadata:
            raise AttributeError(f"{type(self).__name__!s} has no attribute {method_name!r}")
        return RemoteWorkerGroup._bind_worker_method(self, method_name)

    def _bind_worker_method(self, method_name: str) -> Callable[..., Any]:
        def bound(*args: Any, timeout: float | None = None, **kwargs: Any) -> Any:
            return RemoteWorkerGroup.submit(self, method_name, args=args, kwargs=kwargs, timeout=timeout)

        bound.__name__ = method_name
        bound.__qualname__ = method_name
        return bound

    @property
    def world_size(self) -> int:
        return len(self._ranks)

    def rank(self, index: int) -> RemoteWorkerGroup[WorkerT]:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(f"index must be a non-bool int, got {type(index)!r}")
        if index < 0 or index >= self.world_size:
            raise IndexError(f"rank index {index} out of range for world_size={self.world_size}")
        return self._view(index, index + 1)

    def slice(self, start: int, size: int) -> RemoteWorkerGroup[WorkerT]:
        if isinstance(start, bool) or not isinstance(start, int):
            raise TypeError(f"start must be a non-bool int, got {type(start)!r}")
        if isinstance(size, bool) or not isinstance(size, int):
            raise TypeError(f"size must be a non-bool int, got {type(size)!r}")
        if start < 0:
            raise IndexError(f"start must be >= 0, got {start}")
        if size <= 0:
            raise ValueError(f"size must be a positive int, got {size}")
        stop = start + size
        if stop > self.world_size:
            raise IndexError(f"slice [{start}, {stop}) out of range for world_size={self.world_size}")
        return self._view(start, stop)

    def _view(self, start: int, stop: int) -> RemoteWorkerGroup[WorkerT]:
        view = copy(self)
        view._ranks = self._ranks[start:stop]
        view._dispatch_info = {}
        view._collect_info = {}
        return view

    def submit(
        self,
        method_name: str,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
        blocking: bool | None = None,
    ) -> Any:
        """Dispatch and collect one Worker method.

        ``blocking`` can synthesize unregistered method metadata or
        override a registered method with an explicit warning.
        """
        if not self._ranks:
            raise RuntimeError("WorkerGroup view has no target ranks")
        if blocking is not None and not isinstance(blocking, bool):
            raise TypeError(f"blocking must be bool or None, got {type(blocking)!r}")
        lookup_name = method_name.split(".", 1)[-1]
        metadata = self._method_metadata.get(lookup_name)
        if metadata is None:
            synthesized_blocking = True if blocking is None else blocking
            warnings.warn(
                f"submitting unregistered method {lookup_name!r}; "
                "synthesizing Dispatch.ALL_TO_ALL / Execute.ALL / "
                f"blocking={synthesized_blocking}",
                stacklevel=2,
            )
            metadata = {
                "dispatch_mode": Dispatch.ALL_TO_ALL,
                "execute_mode": Execute.ALL,
                "blocking": synthesized_blocking,
            }
        elif blocking is not None and blocking != metadata["blocking"]:
            warnings.warn(
                f"overriding registered blocking={metadata['blocking']} with blocking={blocking} "
                f"for method {lookup_name!r}",
                stacklevel=2,
            )

        should_block = metadata["blocking"] if blocking is None else blocking
        deadline = create_deadline(timeout)

        if not should_block and next(walk_remote_call_dependencies(args, kwargs), None) is not None:
            call = dependency_remote_call(
                args,
                kwargs,
                deadline=deadline,
                submit_resolved=lambda resolved_args, resolved_kwargs: RemoteWorkerGroup._dispatch_call(
                    self, lookup_name, metadata, (resolved_args, resolved_kwargs), deadline, preserve_data_future=False
                ),
            )
            if _returns_data_proto_future(metadata):
                from verl.protocol import DataProtoFuture

                return DataProtoFuture.concat([call])
            return call
        resolved_args, resolved_kwargs = resolve_submit_dependencies(
            args,
            kwargs,
            downstream_deadline=deadline,
        )
        call = RemoteWorkerGroup._dispatch_call(
            self,
            lookup_name,
            metadata,
            (resolved_args, resolved_kwargs),
            deadline,
            preserve_data_future=not should_block,
        )
        if not should_block:
            return call
        return call.result()

    def _dispatch_call(
        self,
        method_name: str,
        metadata: Mapping[str, Any],
        resolved: tuple[tuple[Any, ...], dict[str, Any]],
        deadline: float | None,
        *,
        preserve_data_future: bool,
    ) -> Any:
        """Lower resolved inputs to rank calls, then apply the registered collector."""
        from verl.protocol import _padding_size_key

        dispatch, collect = _dispatch_and_collect(metadata)
        args, kwargs = resolved
        dispatched_args, dispatched_kwargs = dispatch(self, *args, **kwargs)
        padding_count = int(dispatched_kwargs.pop(_padding_size_key, 0)) if isinstance(dispatched_kwargs, dict) else 0
        execute_name = get_predefined_execute_fn(execute_mode=metadata["execute_mode"])["execute_fn_name"]
        rank_zero = execute_name == "execute_rank_zero"
        ranks = self._ranks[:1] if rank_zero else self._ranks
        calls = []
        for relative_rank, backend_rank in enumerate(ranks):
            rank_args, rank_kwargs = _rank_inputs(
                dispatched_args, dispatched_kwargs, rank=relative_rank, world_size=self.world_size
            )
            rank_args, rank_kwargs = _materialize_futures(*rank_args, **rank_kwargs)
            calls.append((backend_rank, rank_args, rank_kwargs))

        # Materialize all rank inputs before routing, as custom dispatchers may
        # return lazy values with side effects.
        backend_calls = []
        backend_method = method_name
        for rank, rank_args, rank_kwargs in calls:
            backend_method, backend_args = self._route_invocation(method_name, rank_args)
            backend_calls.append((rank, backend_args, rank_kwargs))

        def collect_result(values: list[Any]) -> Any:
            output = collect(self, values[0] if rank_zero else values)
            return _remove_padding(output, padding_count)

        if preserve_data_future and _returns_data_proto_future(metadata):
            rank_calls = [
                self._backend_submit(
                    method_name=backend_method,
                    calls=[backend_call],
                    collect=lambda values: values[0],
                    deadline=deadline,
                )
                for backend_call in backend_calls
            ]
            return _collect_data_future(collect(self, rank_calls), padding_count)
        return self._backend_submit(
            method_name=backend_method, calls=backend_calls, collect=collect_result, deadline=deadline
        )

    def execute_rank_zero_sync(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        return self.execute_rank_zero_async(method_name, *args, **kwargs).result()

    def execute_rank_zero_async(self, method_name: str, *args: Any, **kwargs: Any) -> RemoteCall[Any]:
        if not self._ranks:
            raise RuntimeError("WorkerGroup view has no target ranks")
        backend_method, backend_args = self._route_invocation(method_name.split(".", 1)[-1], args)
        return self._backend_submit(
            method_name=backend_method,
            calls=[(self._ranks[0], backend_args, dict(kwargs))],
            collect=lambda values: values[0],
            deadline=None,
        )

    def execute_all_sync(self, method_name: str, *args: Any, **kwargs: Any) -> list[Any]:
        return RemoteCall.gather(self.execute_all_async(method_name, *args, **kwargs)).result()

    def execute_all_async(self, method_name: str, *args: Any, **kwargs: Any) -> list[RemoteCall[Any]]:
        if not self._ranks:
            raise RuntimeError("WorkerGroup view has no target ranks")
        execute_name, backend_args = self._route_invocation(method_name.split(".", 1)[-1], args)
        return [
            self._backend_submit(
                method_name=execute_name,
                calls=[(rank, backend_args, dict(kwargs))],
                collect=lambda values: values[0],
                deadline=None,
            )
            for rank in self._ranks
        ]

    def _route_invocation(self, method_name: str, args: tuple[Any, ...]) -> tuple[str, tuple[Any, ...]]:
        if self._role_name is None:
            return method_name, args
        return self.fused_worker_execute_fn_name, (self._role_name, method_name, *args)
