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
"""RemoteWorkerGroup serialization, membership, and passive invalidation."""

from __future__ import annotations

import pickle

from tests.runtime.workers import EchoWorker
from verl.runtime import RemoteCall, RemoteWorkerGroup


class PlainActorMethodsWorker(EchoWorker):
    async def raw_add_async(self, y: int) -> int:
        return self.raw_add(y)

    def preserve_list_argument(self, values: list[str]) -> list[str]:
        return values


class FacadeCollisionWorker(EchoWorker):
    def rank(self) -> str:
        return "worker-rank"


def test_serialized_remote_group_matches_worker_group_invocation_facade(runtime, cpu_pool):
    wg = runtime.create_worker_group(EchoWorker, on=cpu_pool)
    remote = pickle.loads(pickle.dumps(wg.remote()))

    assert remote.world_size == wg.world_size
    assert remote.submit("ping", args=(1,)) == [1, 2]
    assert remote.ping(2) == wg.ping(2)

    selected = remote.slice(1, 1)
    assert selected.world_size == 1
    assert selected.ping(3) == wg.slice(1, 1).ping(3)

    assert remote.execute_rank_zero_sync("raw_add", 4) == wg.execute_rank_zero_sync("raw_add", 4)
    rank_zero_call = remote.execute_rank_zero_async("raw_add", 5)
    assert isinstance(rank_zero_call, RemoteCall)
    assert rank_zero_call.result() == wg.execute_rank_zero_sync("raw_add", 5)

    assert remote.execute_all_sync("raw_add", 6) == wg.execute_all_sync("raw_add", 6)
    assert RemoteCall.gather(remote.execute_all_async("raw_add", 7)).result() == wg.execute_all_sync("raw_add", 7)


def test_plain_actor_methods_are_exposed_without_register(runtime, cpu_pool):
    wg = runtime.create_worker_group(PlainActorMethodsWorker, on=cpu_pool)

    assert wg.remote().raw_add(3) == [3, 4]
    call = wg.remote().raw_add_async(3)

    assert isinstance(call, RemoteCall)
    assert call.result() == [3, 4]

    single = wg.rank(0).remote()
    assert single.raw_add(3) == 3
    assert single.preserve_list_argument(["s1"]) == ["s1"]


def test_worker_method_can_shadow_facade_name_without_losing_explicit_rank_view(runtime, cpu_pool):
    wg = runtime.create_worker_group(FacadeCollisionWorker, on=cpu_pool)
    remote = wg.remote()

    assert wg.rank() == ["worker-rank", "worker-rank"]
    assert remote.rank() == ["worker-rank", "worker-rank"]

    selected = RemoteWorkerGroup.rank(remote, 1)
    assert selected.world_size == 1
    assert selected.rank() == "worker-rank"
