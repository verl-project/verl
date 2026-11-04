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

"""Experimental teacher-generated training rollouts for synchronous V1 distillation."""

import math
import random
from collections.abc import Callable
from functools import partial
from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.import_utils import load_extern_object


def is_teacher_rollout_enabled(config: DictConfig) -> bool:
    """Whether training may use a user-selected teacher generation source."""
    return bool(OmegaConf.select(config, "distillation.rollout_source_selector.path"))


def build_rollout_source_selector(config: DictConfig) -> Callable[..., Literal["student", "teacher"]] | None:
    """Load a synchronous user function once; policies own their selection and reproducibility.

    The function receives keyword arguments ``batch``, ``global_step``, and ``seed``,
    plus configured kwargs. It returns one source for the entire submission.
    No selector preserves the existing student generation path.
    """
    if not is_teacher_rollout_enabled(config):
        return None
    selector = config.distillation.rollout_source_selector
    function = load_extern_object(selector.path, selector.name)
    if not callable(function):
        raise TypeError("distillation.rollout_source_selector must refer to a callable.")
    kwargs = OmegaConf.to_container(selector.get("kwargs", OmegaConf.create({})), resolve=True)
    return partial(function, **kwargs)


def select_source(
    *, batch: TensorDict, global_step: int, seed: int, teacher_probability: float = 1.0
) -> Literal["student", "teacher"]:
    """Built-in fixed-probability selector; draws from (seed, global_step) without touching global RNG state."""
    if not math.isfinite(teacher_probability) or not 0.0 <= teacher_probability <= 1.0:
        raise ValueError("teacher_probability must be finite and in [0, 1].")
    if teacher_probability == 0.0:
        return "student"
    if teacher_probability == 1.0:
        return "teacher"
    draw = random.Random(f"teacher-rollout:{seed}:{global_step}").random()
    return "teacher" if draw < teacher_probability else "student"


def assign_rollout_source(
    batch: TensorDict, selector: Callable[..., Literal["student", "teacher"]], *, global_step: int, seed: int
) -> None:
    """Select one source before TransferQueue publication and worker fan-out."""
    source = selector(batch=batch, global_step=global_step, seed=seed)
    if not isinstance(source, str) or source not in ("student", "teacher"):
        raise ValueError("rollout_source_selector must return 'student' or 'teacher' for the entire submission.")
    tu.assign_non_tensor_data(batch, "rollout_source", source)


def get_student_rollout_tags(config: DictConfig, tags: list[dict], metrics: dict) -> list[dict]:
    """Record the real teacher fraction and return student tags for staleness metrics."""
    real_tags = [tag for tag in tags if not tag.get("is_padding", False)]
    if is_teacher_rollout_enabled(config) and real_tags:
        metrics["distillation/teacher_rollout_fraction"] = sum(
            tag.get("rollout_source", "student") == "teacher" for tag in real_tags
        ) / len(real_tags)
    return [tag for tag in real_tags if tag.get("rollout_source", "student") == "student"]


def validate_teacher_rollout_config(config: DictConfig) -> None:
    """Fail before allocation when teacher generation would enter an unsupported training path."""
    selector = OmegaConf.select(config, "distillation.rollout_source_selector")
    if selector is None:
        return
    if not isinstance(selector, dict | DictConfig):
        raise ValueError("distillation.rollout_source_selector must contain path, name, and optional kwargs.")
    path, name = selector.get("path"), selector.get("name")
    kwargs = selector.get("kwargs", {})
    if (
        bool(path) != bool(name)
        or any(value is not None and (not isinstance(value, str) or not value) for value in (path, name))
        or not isinstance(kwargs, dict | DictConfig)
        or any(key in kwargs for key in ("batch", "global_step", "seed"))
        or (not path and kwargs)
    ):
        raise ValueError("distillation.rollout_source_selector requires a nonempty path/name pair and policy kwargs.")
    if not path:
        return

    def require(path: str, expected: Any, default: Any = None) -> None:
        if OmegaConf.select(config, path, default=default) != expected:
            raise ValueError(f"A rollout_source_selector requires {path}={expected!r}.")

    for path, expected, default in (
        ("distillation.enabled", True, False),
        ("trainer.use_v1", True, True),
        ("trainer.v1.trainer_mode", "sync", "sync"),
        ("actor_rollout_ref.rollout.name", "vllm", None),
        ("actor_rollout_ref.rollout.agent.default_agent_loop", "single_turn_agent", "single_turn_agent"),
        ("actor_rollout_ref.rollout.multi_turn.enable", False, False),
        ("distillation.distillation_loss.loss_mode", "forward_kl_topk", None),
        ("distillation.distillation_loss.use_policy_gradient", False, True),
        ("distillation.distillation_loss.use_task_rewards", False, True),
        ("actor_rollout_ref.rollout.calculate_log_probs", False, True),
        ("actor_rollout_ref.rollout.temperature", 1.0, None),
        ("trainer.v1.sampler.sync_refill_failed_groups", False, False),
    ):
        require(path, expected, default)
    for path in (
        "actor_rollout_ref.rollout.agent.agent_loop_manager_class",
        "actor_rollout_ref.rollout.agent.agent_loop_config_path",
        "trainer.v1.sampler.custom_sampler.path",
        "trainer.v1.sampler.custom_sampler.name",
    ):
        if OmegaConf.select(config, path):
            raise ValueError(f"A rollout_source_selector does not support {path}.")

    topk = config.distillation.distillation_loss.topk
    if topk is None or topk <= 0:
        raise ValueError("A rollout_source_selector requires distillation.distillation_loss.topk > 0.")
    teachers = config.distillation.teacher_models
    # Match DistillationConfig._resolve_teacher_models: custom entries replace the default entry.
    names = [name for name in teachers if name != "teacher_model"] if len(teachers) > 1 else list(teachers)
    if len(names) != 1:
        raise ValueError("A rollout_source_selector requires exactly one resolved distillation.teacher_models entry.")
    prefix = f"distillation.teacher_models.{names[0]}"
    require(f"{prefix}.inference.name", "vllm")
    require(f"{prefix}.inference.temperature", 1.0)
    if not OmegaConf.select(config, f"{prefix}.model_path"):
        raise ValueError(f"A rollout_source_selector requires {prefix}.model_path.")
