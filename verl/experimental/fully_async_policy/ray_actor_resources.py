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

from __future__ import annotations

import os
from numbers import Real
from typing import Any, Iterable

import ray

AUTO_CPU_RESERVATION = "auto"

DEFAULT_TRAINER_NUM_CPUS = 10.0
DEFAULT_ROLLOUTER_NUM_CPUS = 10.0
DEFAULT_MESSAGE_QUEUE_NUM_CPUS = 2.0
DEFAULT_TRAINER_WORKER_GROUP_NUM_CPUS_PER_GPU = 3.0
DEFAULT_ROLLOUT_WORKER_GROUP_NUM_CPUS_PER_GPU = 2.0


def get_actor_num_cpus_config(config: Any, key: str, default: str | float = AUTO_CPU_RESERVATION) -> str | float:
    async_training_config = getattr(config, "async_training", None)
    if async_training_config is None:
        return default

    value = async_training_config.get(key, default)
    if isinstance(value, str) and value.lower() == AUTO_CPU_RESERVATION:
        return AUTO_CPU_RESERVATION
    return _validate_num_cpus(value, key)


def get_actor_num_cpus_config_or_default(config: Any, key: str, default_num_cpus: float) -> float:
    value = get_actor_num_cpus_config(config, key)
    if value == AUTO_CPU_RESERVATION:
        return default_num_cpus
    return value


def resolve_actor_num_cpus(
    config: Any,
    key: str,
    default_num_cpus: float,
    remaining_num_cpus: Iterable[float] = (),
    reserved_num_cpus: float = 0.0,
    available_cpus: float | None = None,
) -> float:
    configured_num_cpus = get_actor_num_cpus_config(config, key)
    if configured_num_cpus != AUTO_CPU_RESERVATION:
        return configured_num_cpus

    if available_cpus is None:
        available_cpus = _get_available_cpus()
    if available_cpus is None:
        return float(default_num_cpus)

    return auto_size_num_cpus(
        default_num_cpus=default_num_cpus,
        remaining_num_cpus=remaining_num_cpus,
        reserved_num_cpus=reserved_num_cpus,
        available_cpus=available_cpus,
    )


def auto_size_num_cpus(
    default_num_cpus: float,
    remaining_num_cpus: Iterable[float] = (),
    reserved_num_cpus: float = 0.0,
    available_cpus: float = 0.0,
) -> float:
    available_cpus = max(float(available_cpus), 0.0)
    actor_available_cpus = max(available_cpus - max(float(reserved_num_cpus), 0.0), 0.0)
    remaining_num_cpus = [max(float(num_cpus), 0.0) for num_cpus in remaining_num_cpus]
    desired_num_cpus = max(float(default_num_cpus), 0.0)
    desired_total = desired_num_cpus + sum(remaining_num_cpus)

    if desired_total <= 0:
        return 0.0
    if actor_available_cpus >= desired_total:
        return _round_num_cpus(desired_num_cpus)

    return _round_num_cpus(actor_available_cpus * desired_num_cpus / desired_total)


def estimate_trainer_worker_group_num_cpus(config: Any) -> float:
    return _estimate_worker_group_num_cpus(
        config=config,
        config_key="trainer",
        num_cpus_per_gpu=DEFAULT_TRAINER_WORKER_GROUP_NUM_CPUS_PER_GPU,
    )


def estimate_rollout_worker_group_num_cpus(config: Any) -> float:
    return _estimate_worker_group_num_cpus(
        config=config,
        config_key="rollout",
        num_cpus_per_gpu=DEFAULT_ROLLOUT_WORKER_GROUP_NUM_CPUS_PER_GPU,
    )


def _get_available_cpus() -> float | None:
    if not ray.is_initialized():
        cpu_count = os.cpu_count()
        return float(cpu_count) if cpu_count is not None else None

    available_cpus = ray.available_resources().get("CPU")
    if available_cpus is not None:
        return float(available_cpus)

    cluster_cpus = ray.cluster_resources().get("CPU")
    if cluster_cpus is not None:
        return float(cluster_cpus)
    return None


def _validate_num_cpus(value: Any, key: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"async_training.{key} must be a non-negative number or 'auto', got {value!r}")

    if isinstance(value, Real):
        num_cpus = float(value)
    elif isinstance(value, str):
        try:
            num_cpus = float(value)
        except ValueError as exc:
            raise ValueError(f"async_training.{key} must be a non-negative number or 'auto', got {value!r}") from exc
    else:
        raise ValueError(f"async_training.{key} must be a non-negative number or 'auto', got {value!r}")

    if num_cpus < 0:
        raise ValueError(f"async_training.{key} must be non-negative, got {value!r}")
    return _round_num_cpus(num_cpus)


def _round_num_cpus(num_cpus: float) -> float:
    return round(float(num_cpus), 4)


def _estimate_worker_group_num_cpus(config: Any, config_key: str, num_cpus_per_gpu: float) -> float:
    worker_group_config = getattr(config, config_key, None)
    if worker_group_config is None:
        return 0.0

    nnodes = float(worker_group_config.get("nnodes", 0))
    n_gpus_per_node = float(worker_group_config.get("n_gpus_per_node", 0))
    return _round_num_cpus(max(nnodes, 0.0) * max(n_gpus_per_node, 0.0) * num_cpus_per_gpu)
