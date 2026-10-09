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

"""Serializable Monarch WorkerGroup projection over an ActorMesh."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from verl.single_controller.base.errors import RPCUnavailableError
from verl.single_controller.base.remote_call import RemoteCall
from verl.single_controller.base.remote_worker_group import RemoteWorkerGroup

from .remote_call import MonarchRemoteCall

if TYPE_CHECKING:
    ActorMesh = Any


def _same_inputs(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    other_args: tuple[Any, ...],
    other_kwargs: dict[str, Any],
) -> bool:
    return (
        len(args) == len(other_args)
        and kwargs.keys() == other_kwargs.keys()
        and all(value is other for value, other in zip(args, other_args, strict=True))
        and all(kwargs[key] is other_kwargs[key] for key in kwargs)
    )


def _fanout_runs(
    calls: Sequence[tuple[int, tuple[Any, ...], dict[str, Any]]],
) -> list[tuple[int, int, tuple[Any, ...], dict[str, Any]]]:
    """Group consecutive ranks that share one payload into inclusive rank runs.

    A dispatch fans one payload across a whole tensor-parallel group, so those
    ranks can take a single mesh cast instead of one message each.
    """
    runs: list[tuple[int, int, tuple[Any, ...], dict[str, Any]]] = []
    for rank, args, kwargs in calls:
        if runs:
            start, stop, run_args, run_kwargs = runs[-1]
            if rank == stop + 1 and _same_inputs(args, kwargs, run_args, run_kwargs):
                runs[-1] = (start, rank, run_args, run_kwargs)
                continue
        runs.append((rank, rank, args, kwargs))
    return runs


def _submit_monarch(
    worker_group: Any,
    *,
    method_name: str,
    calls: Sequence[tuple[int, tuple[Any, ...], dict[str, Any]]],
    collect: Callable[[list[Any]], Any],
    deadline: float | None,
) -> RemoteCall[Any]:
    actor_mesh = worker_group._actor_mesh
    if actor_mesh is None:
        raise RPCUnavailableError("Monarch WorkerGroup targets are unavailable")
    flat_mesh = actor_mesh.flatten("rank")

    monarch_futures = []
    cast_runs = []
    for start, stop, args, kwargs in _fanout_runs(calls):
        if start == stop:
            target = flat_mesh.slice(rank=start).__monarch_call__
            monarch_futures.append(target.call_one(method_name, args, kwargs))
        else:
            target = flat_mesh.slice(rank=slice(start, stop + 1)).__monarch_call__
            monarch_futures.append(target.call(method_name, args, kwargs))
        cast_runs.append(start != stop)

    collect_result = collect
    if any(cast_runs):

        def collect_runs(values: list[Any]) -> Any:
            ranked: list[Any] = []
            for is_cast, value in zip(cast_runs, values, strict=True):
                if is_cast:
                    ranked.extend(value.values())
                else:
                    ranked.append(value)
            return collect(ranked)

        collect_result = collect_runs

    return MonarchRemoteCall(
        deadline=deadline,
        futures=monarch_futures,
        collect=collect_result,
    )


class MonarchRemoteWorkerGroup(RemoteWorkerGroup[Any]):
    """Non-owning Monarch invocation facade whose identity is an ActorMesh."""

    _backend_submit = _submit_monarch

    def __init__(
        self,
        *,
        actor_mesh: ActorMesh,
        method_metadata: dict[str, dict[str, Any]],
        ranks: Sequence[int],
        role_name: str | None = None,
    ) -> None:
        self._actor_mesh = actor_mesh
        super().__init__(method_metadata=method_metadata, ranks=ranks, role_name=role_name)
