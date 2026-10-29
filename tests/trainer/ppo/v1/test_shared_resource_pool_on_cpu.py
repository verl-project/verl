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
"""Tests for V1 separate-async shared physical resource pools."""

import asyncio
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.single_controller.ray.base import RayResourcePool, split_resource_pool
from verl.trainer.ppo.v1 import trainer_separate_async as trainer_module
from verl.trainer.ppo.v1.trainer_separate_async import (
    PPOTrainerSeparateAsync,
    SharedResourcePoolManager,
    validate_shared_resource_pool,
)
from verl.utils.config import get_trainer_gpu_count
from verl.workers.rollout.llm_server import LLMServerManager


def make_config(*, nodes=2, per_node=8, rollout_gpus=12, replica_size=2):
    return OmegaConf.create(
        {
            "trainer": {
                "use_v1": True,
                "nnodes": nodes,
                "n_gpus_per_node": per_node,
                "v1": {
                    "trainer_mode": "separate_async",
                    "separate_async": {
                        "shared_rollout_gpus": rollout_gpus,
                        "hybrid_rollout": {"enable_switch": False},
                    },
                },
            },
            "actor_rollout_ref": {
                "rollout": {
                    "nnodes": 1,
                    "n_gpus_per_node": per_node,
                    "tensor_model_parallel_size": replica_size,
                    "data_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "disaggregation": {"enabled": False},
                    "name": "vllm",
                    "prometheus": {"enable": False},
                    "disable_log_stats": True,
                },
                "model": {},
            },
        }
    )


@pytest.mark.parametrize(
    ("nodes", "rollout", "trainer"),
    [(1, 6, 2), (1, 5, 3), (2, 12, 4), (2, 10, 6), (3, 18, 6), (4, 28, 4), (4, 4, 28)],
)
def test_split_counts_use_physical_cluster_size(nodes, rollout, trainer):
    config = make_config(nodes=nodes, rollout_gpus=rollout, replica_size=1)
    assert get_trainer_gpu_count(config) == trainer
    validate_shared_resource_pool(config)


def test_legacy_config_keeps_original_trainer_size():
    config = make_config(rollout_gpus=None)
    assert get_trainer_gpu_count(config) == 16
    assert validate_shared_resource_pool(config) is None


@pytest.mark.parametrize("rollout_gpus", [0, -1, 16, 17, 1.5, True])
def test_invalid_rollout_count_fails_before_allocating(rollout_gpus):
    with pytest.raises(ValueError, match="shared_rollout_gpus"):
        get_trainer_gpu_count(make_config(rollout_gpus=rollout_gpus))


def test_split_requires_separate_async_v1():
    config = make_config()
    config.trainer.v1.trainer_mode = "sync"
    with pytest.raises(ValueError, match="separate_async"):
        get_trainer_gpu_count(config)


@pytest.mark.parametrize(("rollout_gpus", "replica_size"), [(10, 4), (12, 3), (12, 16)])
def test_rejects_replicas_that_cannot_fit_within_node(rollout_gpus, replica_size):
    with pytest.raises(ValueError, match="replica"):
        validate_shared_resource_pool(make_config(rollout_gpus=rollout_gpus, replica_size=replica_size))


def test_rejects_hybrid_switch_and_pd_disaggregation():
    config = make_config()
    config.trainer.v1.separate_async.hybrid_rollout.enable_switch = True
    with pytest.raises(ValueError, match="hybrid"):
        validate_shared_resource_pool(config)
    config.trainer.v1.separate_async.hybrid_rollout.enable_switch = False
    config.actor_rollout_ref.rollout.disaggregation.enabled = True
    with pytest.raises(ValueError, match="disaggregation"):
        validate_shared_resource_pool(config)


@pytest.mark.parametrize(("nodes", "rollout_gpus"), [(2, 12), (2, 10), (3, 18), (4, 28), (4, 4)])
def test_manager_slices_one_parent_without_overlapping_bundles(monkeypatch, nodes, rollout_gpus):
    groups = [f"node{i}" for i in range(nodes)]
    monkeypatch.setattr(RayResourcePool, "get_placement_groups", lambda self, **kwargs: groups)
    monkeypatch.setattr(SharedResourcePoolManager, "_check_resource_available", lambda self: None)
    manager = SharedResourcePoolManager(
        resource_pool_spec={"global_pool": [8] * nodes}, mapping={}, rollout_gpus=rollout_gpus
    )
    manager.create_resource_pool()
    rollout = manager.rollout_resource_pool
    trainer = manager.resource_pool_dict["global_pool"]
    assert (rollout.start_bundle_index, rollout.world_size) == (0, rollout_gpus)
    assert (trainer.start_bundle_index, trainer.world_size) == (rollout_gpus, nodes * 8 - rollout_gpus)
    assert rollout.pgs == trainer.pgs == groups
    assert manager.get_n_gpus() == nodes * 8
    replicas = split_resource_pool(rollout, 2)
    assert [pool.start_bundle_index for pool in replicas] == list(range(0, rollout_gpus, 2))
    assert all(pool.world_size == 2 and pool.start_bundle_index % 8 <= 6 for pool in replicas)


def test_throughput_gpu_count_does_not_double_count_shared_rollout():
    trainer = object.__new__(PPOTrainerSeparateAsync)
    trainer.resource_pool_manager = SharedResourcePoolManager(
        resource_pool_spec={"global_pool": [8, 8]}, mapping={}, rollout_gpus=12
    )
    assert trainer._get_n_gpus_for_throughput() == 16
    trainer.resource_pool_manager = SimpleNamespace(get_n_gpus=lambda: 4)
    trainer.config = make_config()
    trainer.config.actor_rollout_ref.rollout.nnodes = 2
    trainer.config.actor_rollout_ref.rollout.n_gpus_per_node = 6
    assert trainer._get_n_gpus_for_throughput() == 16


def test_trainer_setup_passes_the_split_pool_to_standalone_rollout(monkeypatch):
    trainer = object.__new__(PPOTrainerSeparateAsync)
    trainer._enable_hybrid_replicas = False
    trainer.config = make_config()
    trainer.config.actor_rollout_ref.rollout.checkpoint_engine = {
        "_target_": "verl.workers.config.CheckpointEngineConfig",
        "backend": "nccl",
    }
    trainer.resource_pool_manager = SharedResourcePoolManager(
        resource_pool_spec={"global_pool": [8, 8]}, mapping={}, rollout_gpus=12
    )
    rollout_pool = object()
    trainer.resource_pool_manager.rollout_resource_pool = rollout_pool
    captured = {}

    def base_setup(self):
        self.actor_rollout_wg = object()
        self.llm_server_manager = SimpleNamespace(rollout_replicas=[])

    def create_standalone(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(get_replicas=lambda: [])

    monkeypatch.setattr(trainer_module.PPOTrainer, "_setup", base_setup)
    monkeypatch.setattr(trainer_module.LLMServerManager, "create", create_standalone)
    monkeypatch.setattr(trainer_module, "CheckpointEngineManager", lambda **kwargs: object())
    trainer._setup()
    assert captured["rollout_resource_pool"] is rollout_pool


def test_standalone_manager_consumes_supplied_replica_slices(monkeypatch):
    monkeypatch.setattr(RayResourcePool, "get_placement_groups", lambda self, **kwargs: ["node0", "node1"])
    parent = RayResourcePool(process_on_nodes=[8, 8])
    rollout, _trainer = split_resource_pool(parent, [12, 4])
    seen = []

    class Replica:
        def __init__(self, replica_rank, config, model_config, gpus_per_node):
            self.replica_rank = replica_rank
            self.gpus_per_node = gpus_per_node
            self._server_address = str(replica_rank)
            self._server_handle = object()

        async def init_standalone(self, resource_pool=None):
            seen.append((self.replica_rank, self.gpus_per_node, resource_pool))

    config = make_config()
    manager = LLMServerManager.__new__(LLMServerManager)
    manager.config = config
    manager.rollout_config = config.actor_rollout_ref.rollout
    manager.model_config = config.actor_rollout_ref.model
    manager.worker_group = None
    manager.rollout_resource_pool = rollout
    manager.start_rank = 0
    manager.rollout_replica_class = Replica
    asyncio.run(manager._initialize_llm_servers())
    seen.sort(key=lambda item: item[0])
    assert len(seen) == 6
    assert [pool.start_bundle_index for _, _, pool in seen] == [0, 2, 4, 6, 8, 10]
    assert all(per_node == 8 and pool.world_size == 2 for _, per_node, pool in seen)


def test_standalone_manager_rejects_replica_straddling_node_boundary(monkeypatch):
    monkeypatch.setattr(RayResourcePool, "get_placement_groups", lambda self, **kwargs: ["node0", "node1"])
    parent = RayResourcePool(process_on_nodes=[8, 8])
    _first, rollout, _last = split_resource_pool(parent, [7, 4, 5])
    config = make_config()
    manager = LLMServerManager.__new__(LLMServerManager)
    manager.config = config
    manager.rollout_config = config.actor_rollout_ref.rollout
    manager.model_config = config.actor_rollout_ref.model
    manager.worker_group = None
    manager.rollout_resource_pool = rollout
    manager.start_rank = 0
    with pytest.raises(ValueError, match="node boundaries"):
        asyncio.run(manager._initialize_llm_servers())
