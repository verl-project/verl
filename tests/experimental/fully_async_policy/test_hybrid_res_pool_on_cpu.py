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


def make_config(
    *,
    hybrid_res_pool: bool,
    partition: str | None = None,
    trainer_nnodes: int = 2,
    trainer_gpus_per_node: int = 8,
    rollout_nnodes: int = 1,
    rollout_gpus_per_node: int = 8,
    checkpoint_backend: str = "nccl",
):
    config = {
        "async_training": {
            "hybrid_res_pool": hybrid_res_pool,
        },
        "trainer": {
            "nnodes": trainer_nnodes,
            "n_gpus_per_node": trainer_gpus_per_node,
        },
        "rollout": {
            "nnodes": rollout_nnodes,
            "n_gpus_per_node": rollout_gpus_per_node,
        },
        "actor_rollout_ref": {
            "hybrid_engine": False,
            "rollout": {
                "mode": "async",
                "checkpoint_engine": {
                    "backend": checkpoint_backend,
                },
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


def test_build_hybrid_res_pool_layout_allows_nccl_checkpoint_backend():
    config = make_config(hybrid_res_pool=True, partition="8_4-4", checkpoint_backend="nccl")

    layout = build_hybrid_res_pool_layout(config)

    assert layout.trainer_pool == [4, 4, 4]
    assert layout.rollout_pool == [4]
    assert layout.logical_gpus_per_node == 4


def test_build_hybrid_res_pool_layout_rejects_naive_checkpoint_backend():
    config = make_config(hybrid_res_pool=True, partition="8_4-4", checkpoint_backend="naive")

    with pytest.raises(ValueError, match="checkpoint_engine.backend == 'nccl'"):
        build_hybrid_res_pool_layout(config)


def test_build_hybrid_res_pool_layout_rejects_unknown_checkpoint_backend_key():
    config = make_config(hybrid_res_pool=True, partition="8_4-4", checkpoint_backend="hccl")

    with pytest.raises(ValueError, match="checkpoint_engine.backend == 'nccl'"):
        build_hybrid_res_pool_layout(config)


def test_build_hybrid_res_pool_layout_requires_matching_node_counts_for_split_mode():
    config = make_config(
        hybrid_res_pool=True,
        partition="8_4-4",
        trainer_nnodes=2,
        trainer_gpus_per_node=6,
        rollout_nnodes=1,
        rollout_gpus_per_node=4,
    )

    with pytest.raises(ValueError, match=r"trainer\.nnodes == rollout\.nnodes"):
        build_hybrid_res_pool_layout(config)


def test_build_hybrid_res_pool_layout_requires_split_mode_side_totals_to_match():
    config = make_config(
        hybrid_res_pool=True,
        partition="8_2-2_4",
        trainer_nnodes=2,
        trainer_gpus_per_node=6,
        rollout_nnodes=2,
        rollout_gpus_per_node=2,
    )

    with pytest.raises(ValueError, match="trainer partition total"):
        build_hybrid_res_pool_layout(config)


def test_build_hybrid_res_pool_layout_rejects_ambiguous_split_shaped_raw_total():
    config = make_config(
        hybrid_res_pool=True,
        partition="6-6",
        trainer_nnodes=2,
        trainer_gpus_per_node=6,
        rollout_nnodes=2,
        rollout_gpus_per_node=2,
    )

    with pytest.raises(ValueError, match="ambiguous"):
        build_hybrid_res_pool_layout(config)


@pytest.mark.parametrize(
    ("partition", "trainer_pool", "rollout_pool", "trainer_gpus_per_node", "rollout_gpus_per_node"),
    [
        ("8_4-4", [4, 4, 4], [4], 6, 2),
        ("4-4_8", [4], [4, 4, 4], 2, 6),
    ],
)
def test_normalize_hybrid_res_pool_config_accepts_existing_split_script_semantics(
    partition,
    trainer_pool,
    rollout_pool,
    trainer_gpus_per_node,
    rollout_gpus_per_node,
):
    config = make_config(
        hybrid_res_pool=True,
        partition=partition,
        trainer_nnodes=2,
        trainer_gpus_per_node=trainer_gpus_per_node,
        rollout_nnodes=2,
        rollout_gpus_per_node=rollout_gpus_per_node,
    )

    layout = normalize_hybrid_res_pool_config(config)

    assert layout.trainer_pool == trainer_pool
    assert layout.rollout_pool == rollout_pool
    assert layout.logical_gpus_per_node == 4
    assert config.trainer.nnodes == len(trainer_pool)
    assert config.trainer.n_gpus_per_node == 4
    assert config.rollout.nnodes == len(rollout_pool)
    assert config.rollout.n_gpus_per_node == 4


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
