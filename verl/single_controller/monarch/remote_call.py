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

"""Process-local Monarch RemoteCall over Future observation."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from verl.single_controller.base.errors import ExceptionGroup, PlacementUnavailableError, RPCError
from verl.single_controller.base.remote_call import RemoteCall, _ActiveLoopWouldBlockError

from .errors import MonarchAddressInUseError, map_monarch_exception

if TYPE_CHECKING:
    from monarch.actor import Future

ResultT = TypeVar("ResultT")


def _same_application_failure(left: Exception, right: Exception) -> bool:
    infrastructure_error = RPCError | PlacementUnavailableError | MonarchAddressInUseError
    if (
        isinstance(left, infrastructure_error)
        or isinstance(right, infrastructure_error)
        or type(left) is not type(right)
    ):
        return False
    try:
        return bool(left.args == right.args)
    except Exception:
        return False


class MonarchRemoteCall(RemoteCall[ResultT]):
    """Observe one collected Monarch completion in the originating process.

    Futures stay process-local. ``Future.get(timeout)`` is non-consuming
    on timeout, so repeated observations need no auxiliary loop or thread.
    """

    def __init__(
        self,
        *,
        deadline: float | None,
        futures: Sequence[Future[Any]],
        collect: Callable[[list[Any]], ResultT],
    ) -> None:
        super().__init__(deadline=deadline)
        if not futures:
            raise ValueError("MonarchRemoteCall requires at least one Future")
        self._collect = collect
        self._futures = list(futures)

    def _observe(self, remaining: float | None) -> None:
        if self.done():
            return
        values, pending_indices, errors = self._zero_poll()
        if not pending_indices:
            if errors:
                self._store_errors(errors)
            else:
                self._store_values(values)
            return
        if remaining is not None and remaining <= 0:
            if errors:
                self._store_errors(errors)
            return
        if self._has_running_loop():
            if errors:
                self._store_errors(errors)
                return
            raise _ActiveLoopWouldBlockError("MonarchRemoteCall.result() cannot wait in an active event loop; await it")
        deadline = None if remaining is None else time.monotonic() + remaining
        for offset, index in enumerate(pending_indices):
            future = self._futures[index]
            try:
                timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
                values[index] = future.get(timeout=timeout)
            except TimeoutError:
                errors.extend(self._zero_poll_errors(pending_indices[offset:]))
                if not errors:
                    return
                self._store_errors(errors)
                return
            except Exception as exc:
                errors.append(map_monarch_exception(exc))
        if errors:
            self._store_errors(errors)
        else:
            self._store_values(values)

    async def _wait_async(self) -> None:
        if self.done():
            return
        import asyncio

        outcomes = await asyncio.gather(
            *(future.as_asyncio() for future in self._futures),
            return_exceptions=True,
        )
        if self.done():
            return
        errors = [map_monarch_exception(outcome) for outcome in outcomes if isinstance(outcome, Exception)]
        if errors:
            self._store_errors(errors)
            return
        self._store_values(outcomes)

    def _zero_poll(self) -> tuple[list[Any], list[int], list[BaseException]]:
        values: list[Any] = [None] * len(self._futures)
        pending_indices: list[int] = []
        errors: list[BaseException] = []
        for index, future in enumerate(self._futures):
            try:
                values[index] = future.get(timeout=0.0)
            except TimeoutError:
                pending_indices.append(index)
            except Exception as exc:
                errors.append(map_monarch_exception(exc))
        return values, pending_indices, errors

    def _zero_poll_errors(self, indices: Sequence[int]) -> list[BaseException]:
        errors: list[BaseException] = []
        for index in indices:
            try:
                self._futures[index].get(timeout=0.0)
            except TimeoutError:
                continue
            except Exception as exc:
                errors.append(map_monarch_exception(exc))
        return errors

    def _store_errors(self, errors: list[BaseException]) -> None:
        for error in errors:
            if not isinstance(error, Exception):
                self._set_exception(error)
                return

        distinct: list[Exception] = []
        for error in (item for item in errors if isinstance(item, Exception)):
            if not any(_same_application_failure(error, previous) for previous in distinct):
                distinct.append(error)
        if len(distinct) == 1:
            self._set_exception(distinct[0])
            return
        self._set_exception(ExceptionGroup("runtime failures", distinct))

    def _store_values(self, values: list[Any]) -> None:
        try:
            result = self._collect(values)
        except Exception as exc:
            self._set_exception(map_monarch_exception(exc))
        else:
            self._set_result(result)
