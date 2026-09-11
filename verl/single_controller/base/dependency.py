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

"""Resolve RemoteCall arguments for implicit submit-to-submit dependencies."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

from verl.single_controller.base.errors import ExceptionGroup
from verl.single_controller.base.remote_call import RemoteCall, _ActiveLoopWouldBlockError, remaining_deadline


def walk_remote_call_dependencies(args: Sequence[Any], kwargs: Mapping[str, Any] | None = None) -> Iterator[Any]:
    """Yield distinct RemoteCall values in stable depth-first order."""
    seen_containers: set[int] = set()
    emitted_calls: set[int] = set()

    def walk(value: Any) -> Iterator[Any]:
        if isinstance(value, RemoteCall):
            identity = id(value)
            if identity not in emitted_calls:
                emitted_calls.add(identity)
                yield value
            return
        if type(value) not in (list, tuple, dict):
            return
        container_id = id(value)
        if container_id in seen_containers:
            raise ValueError("cyclic container detected while resolving RemoteCall dependencies")
        seen_containers.add(container_id)
        try:
            items = value.values() if type(value) is dict else value
            for item in items:
                yield from walk(item)
        finally:
            seen_containers.remove(container_id)

    for arg in args:
        yield from walk(arg)
    if kwargs:
        for value in kwargs.values():
            yield from walk(value)


def replace_remote_call_dependencies(
    args: Sequence[Any],
    kwargs: Mapping[str, Any] | None,
    resolved: Mapping[int, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Replace RemoteCall values with their resolved outcomes."""

    def rewrite(value: Any) -> Any:
        if isinstance(value, RemoteCall):
            return resolved[id(value)]
        if type(value) is tuple:
            return tuple(rewrite(item) for item in value)
        if type(value) is list:
            return [rewrite(item) for item in value]
        if type(value) is dict:
            return {key: rewrite(item) for key, item in value.items()}
        return value

    return (
        tuple(rewrite(arg) for arg in args),
        {key: rewrite(value) for key, value in (kwargs or {}).items()},
    )


def earliest_deadline(*deadlines: float | None) -> float | None:
    finite = [deadline for deadline in deadlines if deadline is not None]
    return min(finite) if finite else None


def resolve_submit_dependencies(
    args: Sequence[Any],
    kwargs: Mapping[str, Any] | None,
    *,
    downstream_deadline: float | None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Resolve upstream submit results before dispatching the downstream call."""
    dependencies = list(walk_remote_call_dependencies(args, kwargs))
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        in_event_loop = False
    else:
        in_event_loop = True

    resolved: dict[int, Any] = {}
    failures: list[Exception] = []
    for dependency in dependencies:
        identity = id(dependency)
        observe_deadline = earliest_deadline(dependency._deadline, downstream_deadline)
        remaining = remaining_deadline(observe_deadline)
        try:
            dependency._observe(0.0 if in_event_loop else remaining)
            if in_event_loop and not dependency.done():
                if remaining is not None and remaining <= 0:
                    dependency._result_nowait()
                raise _ActiveLoopWouldBlockError(
                    "RemoteCall.result() cannot wait for submit dependencies in an active event loop; await it"
                )
            resolved[identity] = dependency._result_nowait()
        except _ActiveLoopWouldBlockError:
            raise
        except Exception as exc:
            failures.append(exc)

    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise ExceptionGroup("runtime failures", failures)
    return replace_remote_call_dependencies(args, kwargs, resolved)


async def resolve_submit_dependencies_async(
    args: Sequence[Any],
    kwargs: Mapping[str, Any] | None,
    *,
    downstream_deadline: float | None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Resolve upstream submit results without blocking the caller event loop."""
    dependencies = list(walk_remote_call_dependencies(args, kwargs))
    resolved: dict[int, Any] = {}
    failures: list[Exception] = []
    for dependency in dependencies:
        identity = id(dependency)
        observe_deadline = earliest_deadline(dependency._deadline, downstream_deadline)
        remaining = remaining_deadline(observe_deadline)
        try:
            if remaining is None:
                resolved[identity] = await dependency
            else:
                try:
                    resolved[identity] = await asyncio.wait_for(dependency, timeout=remaining)
                except TimeoutError:
                    dependency._observe(0.0)
                    resolved[identity] = dependency._result_nowait()
        except Exception as exc:
            failures.append(exc)

    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise ExceptionGroup("runtime failures", failures)
    return replace_remote_call_dependencies(args, kwargs, resolved)


class _DependencyRemoteCall(RemoteCall[Any]):
    def __init__(
        self,
        args: Sequence[Any],
        kwargs: Mapping[str, Any] | None,
        *,
        deadline: float | None,
        submit_resolved: Callable[[tuple[Any, ...], dict[str, Any]], RemoteCall[Any]],
    ) -> None:
        super().__init__(deadline=deadline)
        self._args = tuple(args)
        self._kwargs = dict(kwargs or {})
        self._submit_resolved = submit_resolved
        self._target: RemoteCall[Any] | None = None

    def _store_target_outcome(self) -> None:
        if self._target is None or not self._target.done() or self.done():
            return
        try:
            result = self._target._result_nowait()
        except Exception as exc:
            self._set_exception(exc)
        else:
            self._set_result(result)

    def _observe(self, remaining: float | None) -> None:
        if self.done():
            return
        if self._target is None:
            try:
                resolved = resolve_submit_dependencies(
                    self._args,
                    self._kwargs,
                    downstream_deadline=self._deadline,
                )
                self._target = self._submit_resolved(*resolved)
            except _ActiveLoopWouldBlockError:
                raise
            except Exception as exc:
                self._set_exception(exc)
                return
        self._target._observe(remaining)
        self._store_target_outcome()

    async def _wait_async(self) -> None:
        if self.done():
            return
        if self._target is None:
            try:
                resolved = await resolve_submit_dependencies_async(
                    self._args,
                    self._kwargs,
                    downstream_deadline=self._deadline,
                )
                if self._target is None and not self.done():
                    self._target = self._submit_resolved(*resolved)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._set_exception(exc)
                return
        if self._target is not None:
            await RemoteCall._observe_async(self._target)
            self._store_target_outcome()


def dependency_remote_call(
    args: Sequence[Any],
    kwargs: Mapping[str, Any] | None,
    *,
    deadline: float | None,
    submit_resolved: Callable[[tuple[Any, ...], dict[str, Any]], RemoteCall[Any]],
) -> RemoteCall[Any]:
    """Return one RemoteCall that submits only after its argument calls resolve."""
    return _DependencyRemoteCall(
        args,
        kwargs,
        deadline=deadline,
        submit_resolved=submit_resolved,
    )
