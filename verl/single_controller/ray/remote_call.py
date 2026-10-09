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

"""Ray RemoteCall wrapping rank-ordered ObjectRefs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Generic, TypeVar

import ray
from ray.exceptions import GetTimeoutError

from verl.single_controller.base.errors import RPCTimeoutError
from verl.single_controller.base.remote_call import RemoteCall

from .errors import map_ray_exception

ResultT = TypeVar("ResultT")


class RayRemoteCall(RemoteCall[ResultT], Generic[ResultT]):
    """Observe one collected Ray completion with a submit-time deadline."""

    def __init__(
        self,
        refs: Sequence[ray.ObjectRef],
        *,
        collect_fn: Callable[[Sequence[Any]], ResultT] | None = None,
        deadline: float | None,
    ) -> None:
        super().__init__(deadline=deadline)
        if not refs:
            raise ValueError("RayRemoteCall requires at least one ObjectRef")
        self._refs = list(refs)
        self._collect_fn = collect_fn

    def _observe(self, remaining: float | None) -> None:
        if self.done():
            return
        try:
            rank_values = (
                [ray.get(self._refs[0], timeout=remaining)]
                if len(self._refs) == 1
                else self._ray_get_many(self._refs, timeout=remaining)
            )
            collected: Any = rank_values[0] if len(rank_values) == 1 else rank_values
            if self._collect_fn is not None:
                collected = self._collect_fn(rank_values)
            self._set_result(collected)
        except (RPCTimeoutError, GetTimeoutError):
            return
        except Exception as exc:
            self._set_exception(map_ray_exception(exc))

    async def _wait_async(self) -> None:
        if self.done():
            return
        import asyncio

        try:
            outcomes = await asyncio.gather(*self._refs)
            collected: Any = outcomes[0] if len(outcomes) == 1 else outcomes
            if self._collect_fn is not None:
                collected = self._collect_fn(outcomes)
            self._set_result(collected)
        except Exception as exc:
            self._set_exception(map_ray_exception(exc))

    def _ray_get_many(self, refs: list[ray.ObjectRef], *, timeout: float | None) -> list[Any]:
        try:
            _done, not_done = ray.wait(refs, num_returns=len(refs), timeout=timeout)
            if not_done:
                raise RPCTimeoutError("ray.wait deadline expired while ObjectRefs remained pending")
            # Native batch get preserves input-rank result/error order.
            return ray.get(refs)
        except GetTimeoutError as exc:
            raise RPCTimeoutError("ray.get deadline expired while ObjectRefs remained pending") from exc
