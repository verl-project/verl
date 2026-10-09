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

"""Context-inheriting executors owned by Runtime lifecycle objects."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import threading
from collections.abc import Callable
from typing import TypeVar

ResultT = TypeVar("ResultT")
_EXECUTOR_THREAD = threading.local()


def _install_context(
    context_items: tuple[tuple[contextvars.ContextVar[object], object], ...],
    owner_token: object,
) -> None:
    for variable, value in context_items:
        variable.set(value)
    _EXECUTOR_THREAD.owner_token = owner_token


class RuntimeThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
    """Give every owned worker thread the Runtime's construction context."""

    def __init__(
        self,
        max_workers: int | None = None,
        thread_name_prefix: str = "",
    ) -> None:
        # Runtime and backend routing contexts are stable for this executor's
        # lifetime. Install them once per worker instead of copying them for
        # every submitted call.
        context_items = tuple(contextvars.copy_context().items())
        self._owner_token = object()
        super().__init__(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
            initializer=_install_context,
            initargs=(context_items, self._owner_token),
        )

    def owns_current_thread(self) -> bool:
        """Return whether the caller already runs on this executor."""
        return getattr(_EXECUTOR_THREAD, "owner_token", None) is self._owner_token


def run_in_thread(
    executor: RuntimeThreadPoolExecutor,
    function: Callable[[], ResultT],
) -> asyncio.Future[ResultT]:
    """Submit one blocking call to a Runtime-owned worker thread."""
    return asyncio.get_running_loop().run_in_executor(executor, function)


async def _drain_future(future: asyncio.Future[ResultT]) -> ResultT:
    """Observe submitted work despite repeated cancellation of its caller."""
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError:
            continue
    return future.result()


__all__ = ["RuntimeThreadPoolExecutor", "run_in_thread"]
