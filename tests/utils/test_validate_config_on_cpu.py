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
"""CPU tests for the pre-flight checks in verl.utils.config.validate_config.

Covers _validate_parallel_and_memory_config (unit tests on minimal configs) and
an integration test that composes the real ppo_trainer.yaml via hydra.
"""

import pytest
from omegaconf import OmegaConf

from verl.utils.config import _validate_parallel_and_memory_config


def make_config(overrides=None) -> OmegaConf:
    """Minimal config covering every field read by the pre-flight checks."""
    base = OmegaConf.create(
        {
            "trainer": {"n_gpus_per_node": 1, "nnodes": 1},
            "data": {"train_batch_size": 32, "max_prompt_length": 512, "max_response_length": 512},
            "actor_rollout_ref": {
                "hybrid_engine": True,
                "actor": {
                    "ppo_mini_batch_size": 16,
                    "ulysses_sequence_parallel_size": 1,
                    "fsdp_config": {"ulysses_sequence_parallel_size": 1},
                },
                "rollout": {
                    "tensor_model_parallel_size": 1,
                    "data_parallel_size": 1,
                    "gpu_memory_utilization": 0.4,
                },
            },
        }
    )
    return OmegaConf.merge(base, OmegaConf.create(overrides or {}))


def n_gpus_of(cfg) -> int:
    return cfg.trainer.n_gpus_per_node * cfg.trainer.nnodes


def test_happy_path_single_gpu():
    _validate_parallel_and_memory_config(make_config(), n_gpus_of(make_config()))


def test_happy_path_multi_gpu_tp():
    cfg = make_config(
        {
            "trainer": {"n_gpus_per_node": 4, "nnodes": 2},
            "actor_rollout_ref": {"rollout": {"tensor_model_parallel_size": 2}},
        }
    )
    _validate_parallel_and_memory_config(cfg, n_gpus_of(cfg))


def test_rollout_tp_must_divide_n_gpus():
    cfg = make_config({"actor_rollout_ref": {"rollout": {"tensor_model_parallel_size": 2}}})
    with pytest.raises(ValueError, match="tensor_model_parallel_size"):
        _validate_parallel_and_memory_config(cfg, n_gpus_of(cfg))


def test_rollout_tp_must_not_exceed_gpus_per_node():
    # 2 nodes x 2 GPUs: tp=4 divides n_gpus=4 but cannot span nodes.
    cfg = make_config(
        {
            "trainer": {"n_gpus_per_node": 2, "nnodes": 2},
            "actor_rollout_ref": {"rollout": {"tensor_model_parallel_size": 4}},
        }
    )
    with pytest.raises(ValueError, match="must not exceed"):
        _validate_parallel_and_memory_config(cfg, n_gpus_of(cfg))


def test_rollout_tp_check_skipped_for_separate_deployment():
    # hybrid_engine=False (v1 separate-async): rollout pool has its own topology.
    cfg = make_config(
        {
            "trainer": {"n_gpus_per_node": 2, "nnodes": 1},
            "actor_rollout_ref": {"hybrid_engine": False, "rollout": {"tensor_model_parallel_size": 4}},
        }
    )
    _validate_parallel_and_memory_config(cfg, n_gpus_of(cfg))


def test_ulysses_sp_must_divide_n_gpus():
    cfg = make_config({"actor_rollout_ref": {"actor": {"ulysses_sequence_parallel_size": 3}}})
    with pytest.raises(ValueError, match="ulysses"):
        _validate_parallel_and_memory_config(cfg, n_gpus_of(cfg))


def test_mini_batch_must_divide_train_batch_size():
    cfg = make_config({"actor_rollout_ref": {"actor": {"ppo_mini_batch_size": 12}}})
    with pytest.raises(ValueError, match="ppo_mini_batch_size"):
        _validate_parallel_and_memory_config(cfg, n_gpus_of(cfg))


def test_gpu_memory_utilization_must_be_fraction():
    # Classic typo: 40 (GB-ish) instead of 0.4 (fraction).
    cfg = make_config({"actor_rollout_ref": {"rollout": {"gpu_memory_utilization": 40}}})
    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        _validate_parallel_and_memory_config(cfg, n_gpus_of(cfg))


def test_max_model_len_must_cover_prompt_plus_response():
    cfg = make_config({"actor_rollout_ref": {"rollout": {"max_model_len": 512}}})
    with pytest.raises(ValueError, match="max_model_len"):
        _validate_parallel_and_memory_config(cfg, n_gpus_of(cfg))


def _compose_ppo_trainer(overrides):
    pytest.importorskip("hydra")
    from hydra import compose, initialize

    # config_path is resolved relative to this test file (tests/) and must be
    # relative (hydra rejects absolute paths).
    with initialize(version_base=None, config_path="../../verl/trainer/config"):
        return compose(config_name="ppo_trainer", overrides=overrides)


def test_validate_config_full_yaml_single_gpu_happy_path():
    cfg = _compose_ppo_trainer(
        [
            "trainer.n_gpus_per_node=1",
            "trainer.nnodes=1",
            "data.train_batch_size=256",
            "actor_rollout_ref.actor.ppo_mini_batch_size=64",
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
            "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
            "actor_rollout_ref.rollout.gpu_memory_utilization=0.4",
        ]
    )
    from verl.utils.config import validate_config

    validate_config(cfg, use_reference_policy=False, use_critic=False)


def test_validate_config_full_yaml_rejects_tp_gt_n_gpus():
    cfg = _compose_ppo_trainer(
        [
            "trainer.n_gpus_per_node=1",
            "trainer.nnodes=1",
            "actor_rollout_ref.rollout.tensor_model_parallel_size=2",
        ]
    )
    from verl.utils.config import validate_config

    with pytest.raises(ValueError, match="tensor_model_parallel_size"):
        validate_config(cfg, use_reference_policy=False, use_critic=False)
