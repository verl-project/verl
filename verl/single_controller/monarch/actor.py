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

"""Static Monarch Actor that owns one verl Worker instance."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from monarch.actor import Actor, concurrent_endpoint

from verl.runtime.runtime_context import AttachSpec
from verl.single_controller.base.actor import ClassWithInitArgs, WorkerContainer
from verl.single_controller.base.worker import Worker


def _install_spmd_environment(*, local_world_size: int, env_vars: Mapping[str, str]) -> None:
    """Install Stage 1 Worker SPMD keys on this actor process."""
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank % local_world_size)))
    configured_local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(local_world_size)))
    os.environ.update(
        Worker._spmd_environment(
            world_size=world_size,
            rank=rank,
            local_world_size=configured_local_world_size,
            local_rank=local_rank,
            master_addr=os.environ["MASTER_ADDR"],
            master_port=os.environ["MASTER_PORT"],
            env_vars=dict(env_vars),
        )
    )


class MonarchWorkerActor(Actor):
    """Hold one Worker and forward method calls from an ActorMesh."""

    def __init__(
        self,
        actor: ClassWithInitArgs[Worker],
        attach_spec: AttachSpec,
        local_world_size: int,
        env_vars: Mapping[str, str],
    ) -> None:
        from verl.runtime.core import Runtime

        _install_spmd_environment(local_world_size=local_world_size, env_vars=env_vars)
        Runtime._attach(attach_spec)
        self._inner = WorkerContainer(
            actor,
            thread_name_prefix="verl-monarch-worker",
        )

    # Queue-dispatch actors serialize plain endpoints even when their bodies
    # are async. Async Worker calls must overlap so rollout servers can batch;
    # synchronous methods remain serialized by the one-thread executor.
    @concurrent_endpoint
    async def __monarch_call__(
        self,
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Call one method on the owned Worker."""
        return await self._inner.dispatch(method_name, args, kwargs)
