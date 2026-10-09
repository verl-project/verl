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

"""Owning WorkerGroup capabilities shared by every RPC backend."""

from __future__ import annotations

import logging
import signal
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, Generic, TypeVar, cast

from verl.single_controller.base.actor import ClassWithInitArgs
from verl.single_controller.base.remote_worker_group import RemoteWorkerGroup
from verl.single_controller.base.resource_pool import ResourcePool

WorkerT = TypeVar("WorkerT", covariant=True)
GroupT = TypeVar("GroupT", bound="WorkerGroup[Any]")


class WorkerGroup(RemoteWorkerGroup[WorkerT], ABC, Generic[WorkerT]):
    """Worker invocation facade that may own actors or select a non-owning view."""

    fused_worker_execute_fn_name = "_fuw_execute"
    _owned: bool = True
    _parent_group: WorkerGroup[Any] | None = None

    @property
    @abstractmethod
    def resource_pool(self) -> ResourcePool:
        """Return the ResourcePool corresponding to this selected view."""

    @abstractmethod
    def _role_views(self: GroupT, roles: Mapping[str, ClassWithInitArgs[Any]]) -> dict[str, GroupT]:
        """Build role facades for shared fused composition to mark non-owning."""

    def _view(self, start: int, stop: int) -> WorkerGroup[WorkerT]:
        view = cast(WorkerGroup[WorkerT], super()._view(start, stop))
        view._resource_pool = self.resource_pool.slice(slice(start, stop))
        view._owned = False
        view._parent_group = self._parent_group if self._parent_group is not None else self
        return view

    def _close_non_owning_view(self) -> bool:
        """Invalidate a view and report that no backend cleanup is owned."""
        if self._owned:
            return False
        self._ranks = ()
        self._parent_group = None
        return True

    def _query_dispatch_info(self, mesh_name: str) -> list[int]:
        return self.execute_all_sync("_query_dispatch_info", mesh_name)

    def _query_collect_info(self, mesh_name: str) -> list[bool]:
        return self.execute_all_sync("_query_collect_info", mesh_name)

    @abstractmethod
    def remote(self) -> RemoteWorkerGroup[WorkerT]:
        """Return a serializable non-owning projection of this selected view."""

    @abstractmethod
    def close(self) -> None:
        """Close the selected backend actors and their platform resources."""


def check_workers_alive(workers: list, is_alive: Callable, gap_time: float = 1) -> None:
    """Continuously monitors worker processes and raises SIGABRT if any worker dies.

    Args:
        workers (List):
            List of worker objects to monitor
        is_alive (Callable):
            Function to check if a worker is alive
        gap_time (float):
            Time interval between checks
    """
    while True:
        for worker in workers:
            if not is_alive(worker):
                logging.warning(f"worker {worker} is not alive sending signal to main thread")
                signal.raise_signal(signal.SIGABRT)
        time.sleep(gap_time)


__all__ = ["ClassWithInitArgs", "ResourcePool", "WorkerGroup", "check_workers_alive"]
