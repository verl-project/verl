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

import asyncio
from types import SimpleNamespace

import pytest
import ray
from ray.cluster_utils import Cluster
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from verl.experimental.agent_loop import agent_loop as agent_loop_module
from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.experimental.reward_loop.reward_loop import RewardLoopManager
from verl.utils.ray_utils import LOOP_WORKER_NODE_RESOURCE_ENV, schedulable_loop_worker_node_ids

_RESOURCE = "loop_worker_test_group"


@ray.remote(num_cpus=0)
class _NodeProbe:
    def __init__(self, *_):
        self.node_id = ray.get_runtime_context().get_node_id()

    def get_node_id(self):
        return self.node_id


@ray.remote
class _CpuHog:
    def ready(self):
        return True


@pytest.fixture
def multinode_ray_cluster():
    cluster = Cluster()
    try:
        cluster.add_node(num_cpus=1, include_dashboard=False)
        cluster.add_node(num_cpus=2, resources={_RESOURCE: 1}, include_dashboard=False)
        cluster.add_node(num_cpus=2, resources={_RESOURCE: 1}, include_dashboard=False)
        ray.init(address=cluster.address)
        yield
    finally:
        ray.shutdown()
        cluster.shutdown()


def _agent_config(num_workers):
    return SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(
                agent=SimpleNamespace(num_workers=num_workers, num_cpus_per_worker=0.25),
            ),
            model=None,
        )
    )


def _actor_node_ids(actors):
    return ray.get([actor.get_node_id.remote() for actor in actors])


def test_loop_workers_stay_on_configured_nodes(multinode_ray_cluster, monkeypatch):
    nodes = ray.nodes()
    alive_node_ids = {node["NodeID"] for node in nodes if node["Alive"]}
    eligible_node_ids = {node["NodeID"] for node in nodes if node["Resources"].get(_RESOURCE, 0) > 0}
    assert len(nodes) == 3
    assert len(eligible_node_ids) == 2
    print(
        "ray_nodes=",
        [(node["NodeID"], _RESOURCE in node["Resources"]) for node in nodes],
    )

    monkeypatch.setenv(LOOP_WORKER_NODE_RESOURCE_ENV, _RESOURCE)
    agent_manager = AgentLoopManager(_agent_config(num_workers=4), llm_client=None)
    agent_manager.agent_loop_workers_class = _NodeProbe
    asyncio.run(agent_manager._init_agent_loop_workers())

    reward_manager = object.__new__(RewardLoopManager)
    reward_manager.config = SimpleNamespace(reward=SimpleNamespace(num_workers=3, num_cpus_per_worker=0.25))
    reward_manager.reward_loop_workers_class = _NodeProbe
    reward_manager.reward_router_address = None
    reward_manager._init_reward_loop_workers()

    agent_node_ids = _actor_node_ids(agent_manager.agent_loop_workers)
    reward_node_ids = _actor_node_ids(reward_manager.reward_loop_workers)
    print(f"restricted_agent_nodes={agent_node_ids}")
    print(f"restricted_reward_nodes={reward_node_ids}")
    assert set(agent_node_ids) <= eligible_node_ids
    assert set(reward_node_ids) <= eligible_node_ids

    monkeypatch.delenv(LOOP_WORKER_NODE_RESOURCE_ENV)
    default_manager = AgentLoopManager(_agent_config(num_workers=3), llm_client=None)
    default_manager.agent_loop_workers_class = _NodeProbe
    asyncio.run(default_manager._init_agent_loop_workers())
    default_node_ids = _actor_node_ids(default_manager.agent_loop_workers)
    print(f"default_agent_nodes={default_node_ids}")
    assert set(default_node_ids) <= alive_node_ids

    with pytest.raises(RuntimeError, match="missing-loop-worker-resource"):
        schedulable_loop_worker_node_ids("missing-loop-worker-resource")


def test_loop_workers_spill_off_a_full_preferred_node(multinode_ray_cluster, monkeypatch):
    eligible = [node["NodeID"] for node in ray.nodes() if node["Resources"].get(_RESOURCE, 0) > 0]
    full_node, free_node = eligible
    pin_to_full = NodeAffinitySchedulingStrategy(full_node, soft=False)
    hog = _CpuHog.options(num_cpus=2, scheduling_strategy=pin_to_full).remote()
    ray.get(hog.ready.remote())

    monkeypatch.setenv(LOOP_WORKER_NODE_RESOURCE_ENV, _RESOURCE)
    # A stale availability view still ranks the full node first.
    monkeypatch.setattr(agent_loop_module, "available_cpu_per_node", lambda: {full_node: 2.0, free_node: 0.0})
    manager = AgentLoopManager(_agent_config(num_workers=2), llm_client=None)
    manager.agent_loop_workers_class = _NodeProbe
    asyncio.run(manager._init_agent_loop_workers())

    node_ids = ray.get([actor.get_node_id.remote() for actor in manager.agent_loop_workers], timeout=60)
    assert set(node_ids) == {free_node}
