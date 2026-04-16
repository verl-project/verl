# Copyright 2025 Meituan Ltd. and/or its affiliates
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

import pytest
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.hybrid_res_pool import (
    build_hybrid_res_pool_layout,
    is_hybrid_res_pool_enabled,
    normalize_hybrid_res_pool_config,
)
from verl.experimental.separation.utils import create_resource_pool_manager
from verl.trainer.ppo.utils import Role


def make_config(*, hybrid_res_pool: bool, partition: str | None = None):
    config = {
        "async_training": {
            "hybrid_res_pool": hybrid_res_pool,
        },
        "trainer": {
            "nnodes": 2,
            "n_gpus_per_node": 8,
        },
        "rollout": {
            "nnodes": 1,
            "n_gpus_per_node": 8,
        },
        "actor_rollout_ref": {
            "hybrid_engine": False,
            "rollout": {
                "mode": "async",
            },
        },
    }
    if partition is not None:
        config["actor_rollout_ref"]["partition"] = partition
    return OmegaConf.create(config)


def test_hybrid_res_pool_disabled_is_false_and_normalization_is_noop():
    config = make_config(hybrid_res_pool=False, partition="8_4-4")
    original = OmegaConf.to_container(config, resolve=True)

    assert is_hybrid_res_pool_enabled(config) is False
    assert normalize_hybrid_res_pool_config(config) is None
    assert OmegaConf.to_container(config, resolve=True) == original


def test_build_hybrid_res_pool_layout_normalizes_8_4_4():
    config = make_config(hybrid_res_pool=True, partition="8_4-4")

    layout = build_hybrid_res_pool_layout(config)

    assert layout.trainer_pool == [4, 4, 4]
    assert layout.rollout_pool == [4]
    assert layout.logical_gpus_per_node == 4


def test_build_hybrid_res_pool_layout_normalizes_4_4_8():
    config = make_config(hybrid_res_pool=True, partition="4-4_8")

    layout = build_hybrid_res_pool_layout(config)

    assert layout.trainer_pool == [4]
    assert layout.rollout_pool == [4, 4, 4]
    assert layout.logical_gpus_per_node == 4


def test_build_hybrid_res_pool_layout_requires_partition():
    config = make_config(hybrid_res_pool=True)

    with pytest.raises(ValueError, match="partition"):
        build_hybrid_res_pool_layout(config)


def test_normalize_hybrid_res_pool_config_returns_layout_and_rewrites_config():
    config = make_config(hybrid_res_pool=True, partition="8_4-4")

    layout = normalize_hybrid_res_pool_config(config)

    assert layout.trainer_pool == [4, 4, 4]
    assert layout.rollout_pool == [4]
    assert layout.logical_gpus_per_node == 4
    assert config.trainer.nnodes == 3
    assert config.trainer.n_gpus_per_node == 4
    assert config.rollout.nnodes == 1
    assert config.rollout.n_gpus_per_node == 4


def test_normalize_hybrid_res_pool_config_rewrites_4_4_8_layout():
    config = make_config(hybrid_res_pool=True, partition="4-4_8")

    layout = normalize_hybrid_res_pool_config(config)

    assert layout.trainer_pool == [4]
    assert layout.rollout_pool == [4, 4, 4]
    assert layout.logical_gpus_per_node == 4
    assert config.trainer.nnodes == 1
    assert config.trainer.n_gpus_per_node == 4
    assert config.rollout.nnodes == 3
    assert config.rollout.n_gpus_per_node == 4


def test_create_resource_pool_manager_uses_hybrid_layout_after_normalization():
    config = make_config(hybrid_res_pool=True, partition="8_4-4")

    layout = normalize_hybrid_res_pool_config(config)
    manager = create_resource_pool_manager(config, roles=[Role.Actor, Role.Critic, Role.Rollout])

    assert layout.trainer_pool == [4, 4, 4]
    assert layout.rollout_pool == [4]
    assert manager.resource_pool_spec == {
        "trainer_pool": [4, 4, 4],
        "rollout_pool": [4],
    }
