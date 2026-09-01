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
import importlib.util
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.experimental.separation import utils as separation_utils
from verl.single_controller.ray import base as ray_base
from verl.single_controller.ray.base import RayResourcePool, ResourcePoolManager, split_resource_pool
from verl.workers.config.rollout import RolloutConfig


def _stub_optional_transfer_queue():
    """Allow wiring tests to import trainer modules in minimal CPU environments."""
    if "transfer_queue" in sys.modules or importlib.util.find_spec("transfer_queue") is not None:
        return

    transfer_queue = ModuleType("transfer_queue")
    transfer_queue.__spec__ = ModuleSpec("transfer_queue", loader=None)
    transfer_queue.BatchMeta = type("BatchMeta", (), {})
    transfer_queue.KVBatchMeta = type("KVBatchMeta", (), {})
    sys.modules["transfer_queue"] = transfer_queue


def test_resource_pool_manager_preserves_positional_compatibility():
    manager = ResourcePoolManager({"global_pool": [1]}, {}, 7)

    assert manager.max_colocate_count == 7
    assert manager.pool_accelerator_resource_key == {}


@pytest.mark.parametrize("accelerator_type", ["", "   ", "None", "null", 123])
def test_resource_pool_manager_rejects_invalid_accelerator_type(accelerator_type):
    with pytest.raises(ValueError, match="Invalid accelerator resource key"):
        ResourcePoolManager(
            {"global_pool": [1]},
            {},
            pool_accelerator_resource_key={"global_pool": accelerator_type},
        )


def test_resource_pool_manager_rejects_unknown_pool_name():
    with pytest.raises(ValueError, match="unknown resource pools"):
        ResourcePoolManager(
            {"global_pool": [1]},
            {},
            pool_accelerator_resource_key={"typo_pool": "accelerator_type:H20"},
        )


def test_resource_pool_manager_rejects_nonpositive_timeout():
    with pytest.raises(ValueError, match="must be positive"):
        ResourcePoolManager(
            {"global_pool": [1]},
            {},
            accelerator_placement_timeout_s=0,
        )


def test_resource_pool_report_exposes_resolved_automatic_topology():
    manager = ResourcePoolManager(
        {"global_pool": [8, 8]},
        {"ActorRollout": "global_pool"},
        pool_accelerator_resource_key={"global_pool": "accelerator_type:L20X"},
    )

    report = manager.describe_resource_pools()

    assert "global_pool" in report
    assert "ActorRollout" in report
    assert "nodes=2" in report
    assert "total_gpus=16" in report
    assert "accelerator_type:L20X" in report


def test_resource_pool_manager_passes_selector_to_ray_pool(monkeypatch):
    captured = {}

    class FakeResourcePool:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(ray_base, "RayResourcePool", FakeResourcePool)
    monkeypatch.setattr(ResourcePoolManager, "_check_resource_available", lambda self: None)

    manager = ResourcePoolManager(
        {"rollout_pool": [2]},
        {},
        pool_accelerator_resource_key={"rollout_pool": "accelerator_type:H20"},
    )
    manager.create_resource_pool()

    assert captured["accelerator_type"] == "accelerator_type:H20"
    assert captured["placement_group_timeout_s"] == pytest.approx(300.0)


def test_resource_pool_manager_keeps_legacy_waiting_without_selector(monkeypatch):
    captured = {}

    class FakeResourcePool:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(ray_base, "RayResourcePool", FakeResourcePool)
    monkeypatch.setattr(ResourcePoolManager, "_check_resource_available", lambda self: None)

    manager = ResourcePoolManager({"global_pool": [1]}, {})
    manager.create_resource_pool()

    assert captured["accelerator_type"] is None
    assert captured["placement_group_timeout_s"] is None


def test_ray_resource_pool_adds_selector_to_every_gpu_bundle(monkeypatch):
    captured_bundles = []

    class FakePlacementGroup:
        def ready(self):
            return object()

    def fake_placement_group(*, bundles, **kwargs):
        captured_bundles.append(bundles)
        return FakePlacementGroup()

    platform = SimpleNamespace(device_name="cuda", ray_resource_name=lambda: "GPU")
    monkeypatch.setattr(ray_base, "get_platform", lambda: platform)
    monkeypatch.setattr(ray_base, "placement_group", fake_placement_group)
    monkeypatch.setattr(ray_base.ray, "get", lambda refs: refs)
    monkeypatch.setattr(ray_base, "sort_placement_group_by_node_ip", lambda pgs: pgs)

    pool = RayResourcePool(
        process_on_nodes=[2],
        accelerator_type="accelerator_type:H20",
    )
    pool.get_placement_groups()

    assert len(captured_bundles) == 1
    assert all(bundle["accelerator_type:H20"] == pytest.approx(1e-4) for bundle in captured_bundles[0])


def test_accelerator_pool_timeout_cleans_all_created_placement_groups(monkeypatch):
    created = []
    removed = []

    class FakePlacementGroup:
        def ready(self):
            return object()

    def fake_placement_group(**kwargs):
        pg = FakePlacementGroup()
        created.append(pg)
        return pg

    def fake_get(*args, **kwargs):
        raise ray_base.ray.exceptions.GetTimeoutError("timed out")

    platform = SimpleNamespace(device_name="cuda", ray_resource_name=lambda: "GPU")
    monkeypatch.setattr(ray_base, "get_platform", lambda: platform)
    monkeypatch.setattr(ray_base, "placement_group", fake_placement_group)
    monkeypatch.setattr(ray_base, "remove_placement_group", removed.append)
    monkeypatch.setattr(ray_base.ray, "get", fake_get)
    monkeypatch.setattr(
        ray_base.ray._private.state,
        "available_resources_per_node",
        lambda: {
            "l20x-node": {"GPU": 8, "accelerator_type:L20X": 1},
            "h20-node": {"GPU": 4, "accelerator_type:H20": 1},
        },
    )

    pool = RayResourcePool(
        process_on_nodes=[8, 8],
        accelerator_type="accelerator_type:H20",
        placement_group_timeout_s=0.01,
    )

    with pytest.raises(TimeoutError, match="accelerator_type:H20") as exc_info:
        pool.get_placement_groups()

    assert "process_on_nodes=[8, 8]" in str(exc_info.value)
    assert "h20-node" in str(exc_info.value)
    assert removed == created
    assert pool.pgs is None


def test_split_resource_pool_preserves_selector_metadata():
    pool = RayResourcePool(
        process_on_nodes=[2],
        accelerator_type="accelerator_type:H20",
    )
    pool.pgs = [object()]

    subpools = split_resource_pool(pool, split_size=1)

    assert len(subpools) == 2
    assert all(subpool.accelerator_type == "accelerator_type:H20" for subpool in subpools)
    assert all(subpool.placement_group_timeout_s is None for subpool in subpools)
    assert all(subpool.pgs is pool.pgs for subpool in subpools)


def test_rollout_config_defaults_to_no_selector():
    assert RolloutConfig.__dataclass_fields__["accelerator_resource_key"].default is None


@pytest.mark.parametrize(
    "config_name",
    ["fully_async_ppo_trainer.yaml", "fully_async_ppo_megatron_trainer.yaml"],
)
def test_fully_async_does_not_define_a_second_accelerator_selector(config_name):
    config_path = Path(__file__).parents[2] / "verl" / "experimental" / "fully_async_policy" / "config" / config_name
    config = OmegaConf.load(config_path)

    assert "accelerator_resource_key" not in config.rollout


def test_experimental_trainer_pool_uses_trainer_selector():
    config = OmegaConf.create(
        {
            "trainer": {
                "nnodes": 2,
                "n_gpus_per_node": 8,
                "accelerator_resource_key": "accelerator_type:H100",
            },
            "rollout": {"nnodes": 1, "n_gpus_per_node": 8},
            "reward": {"reward_model": {"nnodes": 0, "n_gpus_per_node": 8}},
        }
    )

    manager = separation_utils.create_resource_pool_manager(config, [separation_utils.Role.Actor])

    assert manager.pool_accelerator_resource_key == {"trainer_pool": "accelerator_type:H100"}


def test_v1_global_pool_uses_trainer_selector(monkeypatch):
    _stub_optional_transfer_queue()
    from verl.trainer.ppo.v1 import trainer_base

    class TestPPOTrainer(trainer_base.PPOTrainer):
        def on_step_end(self):
            return None

        def on_sample_end(self):
            return None

    config = OmegaConf.create(
        {
            "actor_rollout_ref": {"model": {"lora": {"rank": 0}, "lora_adapter_path": None}},
            "trainer": {
                "nnodes": 2,
                "n_gpus_per_node": 8,
                "accelerator_resource_key": "accelerator_type:H100",
            },
            "reward": {
                "reward_model": {
                    "enable_resource_pool": False,
                    "nnodes": 0,
                    "n_gpus_per_node": 0,
                }
            },
            "distillation": None,
        }
    )
    monkeypatch.setattr(trainer_base, "need_reference_policy", lambda config: False)
    monkeypatch.setattr(trainer_base, "need_critic", lambda config: False)
    monkeypatch.setattr(trainer_base, "is_distillation_enabled", lambda config: False)
    monkeypatch.setattr(trainer_base.ray, "remote", lambda cls: cls)

    trainer = object.__new__(TestPPOTrainer)
    trainer.config = config
    trainer._init_resource_pool_mgr()

    assert trainer.resource_pool_manager.pool_accelerator_resource_key == {"global_pool": "accelerator_type:H100"}


def test_v0_global_pool_uses_trainer_selector():
    _stub_optional_transfer_queue()
    from verl.trainer import main_ppo_v0

    config = OmegaConf.create(
        {
            "trainer": {
                "nnodes": 2,
                "n_gpus_per_node": 8,
                "accelerator_resource_key": "accelerator_type:H100",
            },
            "reward": {
                "reward_model": {
                    "enable": False,
                    "enable_resource_pool": False,
                    "nnodes": 0,
                    "n_gpus_per_node": 0,
                }
            },
            "distillation": None,
        }
    )
    runner = object.__new__(main_ppo_v0.BaseTaskRunner)
    runner.mapping = {}

    manager = runner.init_resource_pool_mgr(config)

    assert manager.pool_accelerator_resource_key == {"global_pool": "accelerator_type:H100"}


def test_standalone_rollout_pool_uses_rollout_selector(monkeypatch):
    from verl.workers.rollout import replica as replica_module

    captured = {}

    class FakeResourcePoolManager:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.resource_pool_dict = {next(iter(kwargs["resource_pool_spec"])): object()}

        def create_resource_pool(self):
            return None

    class FakeWorkerGroup:
        def __init__(self, **kwargs):
            self.workers = []

    class TestReplica(replica_module.RolloutReplica):
        async def launch_servers(self):
            return None

        def get_ray_class_with_init_args(self):
            return None

    monkeypatch.setattr(replica_module, "ResourcePoolManager", FakeResourcePoolManager)
    monkeypatch.setattr(replica_module, "RayWorkerGroup", FakeWorkerGroup)

    rollout_replica = object.__new__(TestReplica)
    rollout_replica.replica_rank = 0
    rollout_replica.config = SimpleNamespace(accelerator_resource_key="accelerator_type:H20")
    rollout_replica.gpus_per_replica_node = 8
    rollout_replica.nnodes = 1
    rollout_replica.is_reward_model = False
    rollout_replica.is_teacher_model = False
    rollout_replica.name_suffix = ""
    rollout_replica.workers = []

    asyncio.run(rollout_replica.init_standalone())

    assert captured["pool_accelerator_resource_key"] == {"rollout_pool_0": "accelerator_type:H20"}
