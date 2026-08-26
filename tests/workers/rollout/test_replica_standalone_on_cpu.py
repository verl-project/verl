# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import asyncio

import pytest

import verl.workers.rollout.replica as rollout_replica
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, _get_standalone_master_port_range


def test_init_standalone_reuses_existing_resource_pool(monkeypatch):
    resource_pool = object()
    worker_group_kwargs = {}

    class _WorkerGroup:
        def __init__(self, *, resource_pool, **kwargs):  # noqa: ANN003
            self.resource_pool = resource_pool
            self.workers = ["worker"]
            worker_group_kwargs.update(kwargs)

    class _Replica(RolloutReplica):
        def get_ray_class_with_init_args(self):
            return object()

        async def launch_servers(self):
            self.launch_count += 1

    replica = object.__new__(_Replica)
    replica.replica_rank = 0
    replica.name_suffix = ""
    replica.is_reward_model = False
    replica.is_teacher_model = False
    replica.launch_count = 0
    monkeypatch.setattr(rollout_replica, "RayWorkerGroup", _WorkerGroup)
    monkeypatch.setattr(
        rollout_replica,
        "ResourcePoolManager",
        lambda **kwargs: pytest.fail("external resource pool must be reused"),
    )

    asyncio.run(replica.init_standalone(resource_pool=resource_pool))

    assert replica.rollout_mode is RolloutMode.STANDALONE
    assert replica.resource_pool is resource_pool
    assert replica.workers == ["worker"]
    assert replica.launch_count == 1
    assert worker_group_kwargs["master_port_range"] == [20000, 20032]


def test_standalone_master_port_ranges_are_partitioned_by_replica_and_role():
    assert _get_standalone_master_port_range(replica_rank=0, role="rollout") == [20000, 20032]
    assert _get_standalone_master_port_range(replica_rank=1, role="rollout") == [20032, 20064]
    assert _get_standalone_master_port_range(replica_rank=0, role="reward") == [36384, 36416]


@pytest.mark.parametrize("rank_role", [(-1, "rollout"), (0, "unknown"), (1423, "rollout")])
def test_standalone_master_port_range_rejects_invalid_or_overflowing_partitions(rank_role):
    with pytest.raises(ValueError):
        _get_standalone_master_port_range(*rank_role)
