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

"""Serializable Ray WorkerGroup projection over actor handles."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import ray

from verl.single_controller.base.remote_call import RemoteCall
from verl.single_controller.base.remote_worker_group import RemoteWorkerGroup
from verl.single_controller.ray.remote_call import RayRemoteCall

_INLINE_SCALARS = (type(None), bool, int, float, complex, str, bytes)


def _shared_payload_key(value: Any) -> int | None:
    """Key a value worth publishing once, or None to leave Ray to inline it.

    Scalars and empty containers cost less to inline than to round-trip
    through the object store, and an ObjectRef is already shared.
    """
    if isinstance(value, _INLINE_SCALARS) or isinstance(value, ray.ObjectRef):
        return None
    if isinstance(value, tuple | list | set | frozenset | dict) and not value:
        return None
    return id(value)


def _publish_shared_payloads(
    calls: Sequence[tuple[int, tuple[Any, ...], dict[str, Any]]],
) -> Sequence[tuple[int, tuple[Any, ...], dict[str, Any]]]:
    """Publish a payload several ranks receive to the object store once.

    Ray serializes task arguments per task, so a payload fanned out across a
    tensor-parallel group would otherwise be copied once per rank.
    """
    if len(calls) < 2:
        return calls

    counts: dict[int, int] = {}
    payloads: dict[int, Any] = {}
    for _rank, args, kwargs in calls:
        for value in (*args, *kwargs.values()):
            key = _shared_payload_key(value)
            if key is None:
                continue
            counts[key] = counts.get(key, 0) + 1
            payloads[key] = value

    shared = [key for key, count in counts.items() if count > 1]
    if not shared:
        return calls

    from verl.utils.ray_utils import parallel_put

    max_workers = max(1, min(len(shared), os.cpu_count() or 1))
    refs = parallel_put([payloads[key] for key in shared], max_workers=max_workers)
    published = dict(zip(shared, refs, strict=True))

    def substitute(value: Any) -> Any:
        return published.get(_shared_payload_key(value), value)

    return [
        (
            rank,
            tuple(substitute(value) for value in args),
            {name: substitute(value) for name, value in kwargs.items()},
        )
        for rank, args, kwargs in calls
    ]


def _submit_ray(
    worker_group: Any,
    *,
    method_name: str,
    calls: Sequence[tuple[int, tuple[Any, ...], dict[str, Any]]],
    collect: Callable[[list[Any]], Any],
    deadline: float | None,
) -> RemoteCall[Any]:
    refs = [
        getattr(worker_group._ray_actors[rank], method_name).remote(*args, **kwargs)
        for rank, args, kwargs in _publish_shared_payloads(calls)
    ]
    return RayRemoteCall(refs, collect_fn=collect, deadline=deadline)


class RayRemoteWorkerGroup(RemoteWorkerGroup[Any]):
    """Non-owning Ray invocation facade whose identity is its actor handles."""

    _backend_submit = _submit_ray

    def __init__(
        self,
        *,
        actor_handles: Sequence[ray.actor.ActorHandle],
        method_metadata: Mapping[str, Mapping[str, Any]],
        ranks: Sequence[int] | None = None,
        role_name: str | None = None,
    ) -> None:
        self._ray_actors = list(actor_handles)
        super().__init__(
            method_metadata=method_metadata,
            ranks=range(len(self._ray_actors)) if ranks is None else ranks,
            role_name=role_name,
        )
