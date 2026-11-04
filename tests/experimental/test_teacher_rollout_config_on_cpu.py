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

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl.experimental.teacher_rollout import validate_teacher_rollout_config


@pytest.fixture
def supported_config(request):
    config_dir = Path(__file__).resolve().parents[2] / "verl/trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        overrides = [f"model_engine={request.param}"] if hasattr(request, "param") else []
        config = compose(config_name="ppo_trainer", overrides=overrides)
    for path, value in {
        "distillation.rollout_source_selector.path": "my_policy.py",
        "distillation.rollout_source_selector.name": "choose",
        "distillation.enabled": True,
        "distillation.n_gpus_per_node": 1,
        "distillation.nnodes": 1,
        "distillation.teacher_models.teacher_model.model_path": "Qwen/Qwen3-1.7B",
        "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size": 1,
        "distillation.teacher_models.teacher_model.inference.name": "vllm",
        "distillation.distillation_loss.loss_mode": "forward_kl_topk",
        "distillation.distillation_loss.use_policy_gradient": False,
        "distillation.distillation_loss.use_task_rewards": False,
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.calculate_log_probs": False,
        "algorithm.adv_estimator": "grpo",
    }.items():
        OmegaConf.update(config, path, value, force_add=True)
    return config


@pytest.mark.parametrize(
    "selector",
    [
        {"path": "my_policy.py"},
        {"name": "choose"},
        {"path": 3, "name": "choose"},
        {"path": "my_policy.py", "name": "choose", "kwargs": []},
    ],
)
def test_malformed_selector_fails_before_allocation(selector):
    config = OmegaConf.create({"distillation": {"rollout_source_selector": selector}})
    with pytest.raises(ValueError, match="rollout_source_selector"):
        validate_teacher_rollout_config(config)


def test_supported_config(supported_config):
    validate_teacher_rollout_config(supported_config)


@pytest.mark.parametrize(
    "path,value",
    [
        ("distillation.enabled", False),
        ("trainer.use_v1", False),
        ("trainer.v1.trainer_mode", "colocate_async"),
        ("trainer.v1.trainer_mode", "separate_async"),
        ("actor_rollout_ref.rollout.name", "sglang"),
        ("distillation.teacher_models.teacher_model.inference.name", "sglang"),
        ("distillation.distillation_loss.use_task_rewards", True),
        ("distillation.distillation_loss.use_policy_gradient", True),
        ("distillation.distillation_loss.loss_mode", "k1"),
        ("distillation.distillation_loss.loss_mode", "k3"),
        ("distillation.distillation_loss.topk", 0),
        ("actor_rollout_ref.rollout.calculate_log_probs", True),
        ("actor_rollout_ref.rollout.multi_turn.enable", True),
        ("actor_rollout_ref.rollout.agent.default_agent_loop", "tool_agent"),
        ("actor_rollout_ref.rollout.agent.agent_loop_manager_class", "custom.Manager"),
        ("actor_rollout_ref.rollout.agent.agent_loop_config_path", "/tmp/custom.yaml"),
        ("trainer.v1.sampler.custom_sampler.name", "CustomSampler"),
        ("trainer.v1.sampler.sync_refill_failed_groups", True),
        ("actor_rollout_ref.rollout.temperature", 0.7),
        ("distillation.teacher_models.teacher_model.inference.temperature", 0.7),
    ],
)
def test_opt_in_rejects_unsupported_config(supported_config, path, value):
    OmegaConf.update(supported_config, path, value, force_add=True)
    with pytest.raises(ValueError, match=path.split(".")[-1]):
        validate_teacher_rollout_config(supported_config)


def test_no_selector_preserves_existing_config_freedom(supported_config):
    supported_config.distillation.rollout_source_selector = {}
    supported_config.trainer.v1.trainer_mode = "separate_async"
    supported_config.distillation.distillation_loss.use_policy_gradient = True
    validate_teacher_rollout_config(supported_config)


def test_counts_resolved_teachers(supported_config):
    teachers = supported_config.distillation.teacher_models
    OmegaConf.update(teachers, "other", OmegaConf.to_container(teachers.teacher_model, resolve=True), force_add=True)
    teachers.other.key = "other"
    validate_teacher_rollout_config(supported_config)
    OmegaConf.update(teachers, "third", OmegaConf.to_container(teachers.other, resolve=True), force_add=True)
    teachers.third.key = "third"
    with pytest.raises(ValueError, match="teacher_models"):
        validate_teacher_rollout_config(supported_config)
