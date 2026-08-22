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

import dataclasses

import pytest

from verl.utils import ray_utils
from verl.utils.ray_utils import (
    LOOP_WORKER_NODE_RESOURCE_ENV,
    assign_loop_worker_nodes,
    get_loop_worker_node_resource,
    loop_worker_node_affinity_resources,
    schedulable_loop_worker_node_ids,
)
from verl.workers.config.reward import RewardConfig
from verl.workers.config.rollout import AgentLoopConfig

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


def test_assign_falls_back_to_round_robin_without_cpu_data():
    node_ids = ["a", "b", "c"]
    # No available-CPU info -> historical round-robin over the candidate nodes.
    assert assign_loop_worker_nodes(node_ids, 5, available_cpu=None, cpus_per_worker=0.25) == [
        "a",
        "b",
        "c",
        "a",
        "b",
    ]
    assert assign_loop_worker_nodes(node_ids, 5, available_cpu={}, cpus_per_worker=0.25) == [
        "a",
        "b",
        "c",
        "a",
        "b",
    ]


def test_assign_prefers_most_available_and_spreads_as_cpu_drains():
    node_ids = ["a", "b"]
    # "b" starts with more free CPU; each placement debits cpus_per_worker so the
    # two nodes alternate as their available CPU converges.
    placed = assign_loop_worker_nodes(node_ids, 4, available_cpu={"a": 5.0, "b": 6.0}, cpus_per_worker=2.0)
    assert placed == ["b", "a", "b", "a"]


def test_assign_zero_cost_pins_to_single_emptiest():
    node_ids = ["a", "b", "c"]
    # With no per-worker cost nothing drains, so the single emptiest node wins all.
    placed = assign_loop_worker_nodes(node_ids, 3, available_cpu={"a": 1.0, "b": 3.0, "c": 2.0}, cpus_per_worker=0.0)
    assert placed == ["b", "b", "b"]


def test_assign_missing_node_treated_as_zero_cpu():
    node_ids = ["a", "b"]
    # "b" absent from the map -> 0 available, so "a" is always preferred.
    placed = assign_loop_worker_nodes(node_ids, 2, available_cpu={"a": 4.0}, cpus_per_worker=0.0)
    assert placed == ["a", "a"]


def test_assign_empty_nodes_raises():
    with pytest.raises(ValueError):
        assign_loop_worker_nodes([], 3, available_cpu={"a": 1.0})


def test_available_cpu_per_node_returns_empty_when_unreachable():
    # Outside a live Ray driver the internal state view is unavailable; callers
    # must then fall back to round-robin.
    assert ray_utils.available_cpu_per_node() == {}


def _declared_default(config_cls, field_name):
    return next(f for f in dataclasses.fields(config_cls) if f.name == field_name).default


def test_loop_worker_cpu_knob_defaults_are_fractional():
    assert AgentLoopConfig().num_cpus_per_worker == 0.25
    assert _declared_default(AgentLoopConfig, "num_cpus_per_worker") == 0.25
    assert _declared_default(RewardConfig, "num_cpus_per_worker") == 0.25
