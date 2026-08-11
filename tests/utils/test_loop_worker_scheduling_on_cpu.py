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
"""CPU unit tests for agent-loop / reward-loop worker node selection."""

import pytest

from verl.utils import ray_utils
from verl.utils.ray_utils import (
    LOOP_WORKER_NODE_RESOURCE_ENV,
    get_loop_worker_node_resource,
    loop_worker_node_affinity_resources,
    schedulable_loop_worker_node_ids,
)

# A shared/heterogeneous cluster: two managed nodes advertising a group resource,
# one foreign node (other worker group, no group resource), and a dead node.
_GROUP = "group:h200-verl-managed"
_FAKE_NODES = [
    {"NodeID": "managed-a", "Alive": True, "Resources": {"CPU": 16.0, "GPU": 1.0, _GROUP: 1.0}},
    {"NodeID": "managed-b", "Alive": True, "Resources": {"CPU": 16.0, "GPU": 1.0, _GROUP: 1.0}},
    {"NodeID": "foreign", "Alive": True, "Resources": {"CPU": 32.0, "GPU": 1.0}},
    {"NodeID": "dead", "Alive": False, "Resources": {"CPU": 16.0, _GROUP: 1.0}},
    {"NodeID": "cpuless", "Alive": True, "Resources": {"GPU": 1.0, _GROUP: 1.0}},
]


@pytest.fixture(autouse=True)
def _fake_ray_nodes(monkeypatch):
    monkeypatch.setattr(ray_utils.ray, "nodes", lambda: [dict(n) for n in _FAKE_NODES])
    monkeypatch.delenv(LOOP_WORKER_NODE_RESOURCE_ENV, raising=False)


def test_env_parsing(monkeypatch):
    assert get_loop_worker_node_resource() is None
    monkeypatch.setenv(LOOP_WORKER_NODE_RESOURCE_ENV, "   ")
    assert get_loop_worker_node_resource() is None
    monkeypatch.setenv(LOOP_WORKER_NODE_RESOURCE_ENV, f"  {_GROUP}  ")
    assert get_loop_worker_node_resource() == _GROUP


def test_unfiltered_keeps_all_alive_cpu_nodes():
    # Historical behavior: every alive node with CPU, incl. the foreign one.
    assert schedulable_loop_worker_node_ids(None) == ["managed-a", "managed-b", "foreign"]


def test_filter_excludes_foreign_and_dead_and_cpuless():
    assert schedulable_loop_worker_node_ids(_GROUP) == ["managed-a", "managed-b"]


def test_filter_no_match_raises():
    with pytest.raises(RuntimeError, match="does-not-exist"):
        schedulable_loop_worker_node_ids("does-not-exist")


def test_round_robin_stays_within_group():
    node_ids = schedulable_loop_worker_node_ids(_GROUP)
    # num_workers > group size must keep wrapping inside the group, never leaking.
    placed = [node_ids[i % len(node_ids)] for i in range(8)]
    assert set(placed) == {"managed-a", "managed-b"}


def test_single_node_group_does_not_degrade():
    node_ids = schedulable_loop_worker_node_ids(_GROUP)[:1]
    placed = [node_ids[i % len(node_ids)] for i in range(4)]
    assert placed == ["managed-a"] * 4


def test_affinity_resources():
    assert loop_worker_node_affinity_resources(None) is None
    assert loop_worker_node_affinity_resources(_GROUP) == {_GROUP: pytest.approx(1e-4)}
