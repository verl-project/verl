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

"""Collected remote completion with Future-shaped observation and submit-time deadlines."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Coroutine, Generator, Sequence
from typing import Any, Generic, TypeVar, cast

from verl.single_controller.base.errors import ExceptionGroup, RPCTimeoutError

ResultT = TypeVar("ResultT")


class _ObservationTimeout(Exception):
    pass


class _ActiveLoopWouldBlockError(RuntimeError):
    pass


async def _observe_with_timeout(operation: Coroutine[Any, Any, ResultT], remaining: float | None) -> ResultT:
    """Bound one observation without mistaking a task's TimeoutError for ours."""
    if remaining is None:
        return await operation
    task = asyncio.create_task(operation)
    done, _pending = await asyncio.wait((task,), timeout=remaining)
    if done:
        return task.result()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    raise _ObservationTimeout


def create_deadline(timeout: float | None) -> float | None:
    """Create an absolute monotonic deadline from a submit-time timeout.

    Args:
        timeout: ``None`` for no deadline, or a finite non-negative duration.

    Returns:
        Absolute ``time.monotonic()`` deadline, or None.
    """
    if timeout is None:
        return None
    if isinstance(timeout, bool) or not isinstance(timeout, int | float):
        raise TypeError(f"timeout must be None or a finite non-negative number, got {type(timeout)!r}")
    if timeout < 0 or timeout != timeout or timeout == float("inf"):
        raise ValueError(f"timeout must be a finite non-negative number, got {timeout!r}")
    return time.monotonic() + float(timeout)


def remaining_deadline(deadline: float | None, *, now: float | None = None) -> float | None:
    """Return remaining seconds until deadline, or None when unbounded."""
    if deadline is None:
        return None
    current = time.monotonic() if now is None else now
    return max(0.0, deadline - current)


class RemoteCall(Awaitable[ResultT], ABC, Generic[ResultT]):
    """Represent one collected remote completion with a submit-time deadline.

    A call is observed through ``result()``, ``done()``, or ``await``.
    ``timeout`` exists only on submit; observation uses the stored deadline.
    """

    def __init__(self, *, deadline: float | None) -> None:
        self._deadline = deadline
        self._result: ResultT | None = None
        self._exception: BaseException | None = None
        self._done = False

    def __await__(self) -> Generator[Any, None, ResultT]:
        """Await the same value or exception as result()."""
        return self._await_result().__await__()

    async def _await_result(self) -> ResultT:
        remaining = remaining_deadline(self._deadline)
        try:
            if remaining is None:
                await self._wait_async()
            else:
                await asyncio.wait_for(self._wait_async(), timeout=remaining)
        except TimeoutError:
            self._observe(0.0)
        return self._result_nowait()

    def result(self) -> ResultT:
        """Return the result, using the absolute deadline created by submit()."""
        self._observe(remaining_deadline(self._deadline))
        return self._result_nowait()

    @staticmethod
    def wait(
        calls: Sequence[RemoteCall[Any]],
        *,
        count: int = 1,
    ) -> RemoteCall[tuple[list[RemoteCall[Any]], list[RemoteCall[Any]]]]:
        """Return one awaitable call that completes after ``count`` inputs."""
        if not isinstance(calls, Sequence) or isinstance(calls, str | bytes):
            raise TypeError(f"calls must be a sequence of RemoteCall, got {type(calls)!r}")
        if not calls:
            raise ValueError("calls must be non-empty")
        identities: set[int] = set()
        for call in calls:
            if not isinstance(call, RemoteCall):
                raise TypeError(f"calls must contain RemoteCall instances, got {type(call)!r}")
            call_id = id(call)
            if call_id in identities:
                raise ValueError("calls must have unique ids")
            identities.add(call_id)
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"count must be a non-bool int, got {type(count)!r}")
        if count < 1 or count > len(calls):
            raise ValueError(f"count must satisfy 1 <= count <= len(calls); got count={count}, len={len(calls)}")
        return cast(RemoteCall[_WaitResult], _AggregateRemoteCall(calls, count=count))

    @staticmethod
    def gather(calls: Sequence[RemoteCall[ResultT]]) -> RemoteCall[list[ResultT]]:
        """Return one awaitable call that collects all inputs in order."""
        if not isinstance(calls, Sequence) or isinstance(calls, str | bytes):
            raise TypeError(f"calls must be a sequence of RemoteCall, got {type(calls)!r}")
        for call in calls:
            if not isinstance(call, RemoteCall):
                raise TypeError(f"calls must contain RemoteCall instances, got {type(call)!r}")
        return cast(RemoteCall[list[ResultT]], _AggregateRemoteCall(calls))

    @staticmethod
    def _has_running_loop() -> bool:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return False
        return True

    @staticmethod
    async def _observe_async(call: RemoteCall[Any]) -> RemoteCall[Any]:
        try:
            await call
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        return call

    @staticmethod
    async def _wait_async_many(
        calls: Sequence[RemoteCall[Any]],
        count: int,
    ) -> tuple[list[RemoteCall[Any]], list[RemoteCall[Any]]]:
        tasks = {asyncio.create_task(RemoteCall._observe_async(call)) for call in calls}
        done_ids: set[int] = set()
        try:
            while tasks and len(done_ids) < count:
                completed, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in completed:
                    call = task.result()
                    if call.done():
                        done_ids.add(id(call))
        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        if len(done_ids) < count:
            raise RPCTimeoutError(f"only {len(done_ids)} of {count} requested remote calls completed")
        done = [call for call in calls if id(call) in done_ids][:count]
        selected = {id(call) for call in done}
        not_done = [call for call in calls if id(call) not in selected]
        return done, not_done

    @staticmethod
    async def _gather_async_many(calls: Sequence[RemoteCall[ResultT]]) -> list[ResultT]:
        await asyncio.gather(*(RemoteCall._observe_async(call) for call in calls))
        values: list[ResultT] = []
        errors: list[Exception] = []
        for call in calls:
            try:
                values.append(call._result_nowait())
            except Exception as exc:  # noqa: BLE001 - aggregate completed failures
                errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("runtime failures", errors)
        return values

    def done(self) -> bool:
        """Return whether this call has a terminal local outcome."""
        return self._done

    def _result_nowait(self) -> ResultT:
        """Return the cached result or raise its cached exception without observing."""
        if not self.done():
            raise RPCTimeoutError("remote call deadline expired while outcome remained unknown")
        if self._exception is not None:
            raise self._exception
        return cast(ResultT, self._result)

    def _set_result(self, result: ResultT) -> None:
        """Publish the first successful terminal outcome."""
        if self._done:
            return
        self._result = result
        self._done = True

    def _set_exception(self, exception: BaseException) -> None:
        """Publish the first failed terminal outcome."""
        if self._done:
            return
        self._exception = exception
        self._done = True

    @abstractmethod
    def _observe(self, remaining: float | None) -> None:
        """Poll backend completion using remaining deadline budget.

        ``remaining`` of ``0`` is a zero-time poll. ``None`` waits without a
        common-layer deadline budget.
        """

    @abstractmethod
    async def _wait_async(self) -> None:
        """Observe backend completion without blocking the caller event loop."""


_WaitResult = tuple[list[RemoteCall[Any]], list[RemoteCall[Any]]]


class _AggregateRemoteCall(RemoteCall[list[ResultT] | _WaitResult], Generic[ResultT]):
    """Shared aggregation state; count selects wait versus gather collection."""

    def __init__(self, calls: Sequence[RemoteCall[ResultT]], *, count: int | None = None) -> None:
        super().__init__(deadline=None)
        self._calls = tuple(calls)
        self._count = count
        if count is None and not calls:
            self._set_result([])

    def _observe(self, remaining: float | None) -> None:
        if self.done():
            return
        if self._has_running_loop():
            if self._count is not None:
                raise RuntimeError("RemoteCall.wait(...).result() cannot run in an active event loop; await it")
            self._observe_gather_in_loop(remaining)
            return
        try:
            result = asyncio.run(_observe_with_timeout(self._collect(), remaining))
        except _ObservationTimeout:
            pass
        except Exception as exc:
            self._set_exception(exc)
        else:
            self._set_result(result)

    async def _collect(self) -> list[ResultT] | _WaitResult:
        if self._count is None:
            return await self._gather_async_many(self._calls)
        return await self._wait_async_many(self._calls, self._count)

    async def _wait_async(self) -> None:
        if self.done():
            return
        try:
            result = await self._collect()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._set_exception(exc)
        else:
            self._set_result(result)

    def _observe_gather_in_loop(self, remaining: float | None) -> None:
        common_deadline = None if remaining is None else time.monotonic() + remaining
        values: list[ResultT] = []
        errors: list[Exception] = []
        for call in self._calls:
            deadlines = [deadline for deadline in (call._deadline, common_deadline) if deadline is not None]
            try:
                call._observe(remaining_deadline(min(deadlines) if deadlines else None))
                values.append(call._result_nowait())
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        if errors and all(isinstance(error, RPCTimeoutError | _ActiveLoopWouldBlockError) for error in errors):
            would_block = next(
                (error for error in errors if isinstance(error, _ActiveLoopWouldBlockError)),
                None,
            )
            if would_block is not None:
                raise would_block
            return
        if errors:
            material = [error for error in errors if not isinstance(error, _ActiveLoopWouldBlockError)]
            if len(material) == 1:
                self._set_exception(material[0])
            else:
                self._set_exception(ExceptionGroup("runtime failures", material))
        else:
            self._set_result(values)
