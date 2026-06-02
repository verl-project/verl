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

from verl.experimental.fully_async_policy.ray_actor_resources import (
    AUTO_CPU_RESERVATION,
    auto_size_num_cpus,
    estimate_rollout_worker_group_num_cpus,
    estimate_trainer_worker_group_num_cpus,
    get_actor_num_cpus_config,
    get_actor_num_cpus_config_or_default,
    resolve_actor_num_cpus,
)


def test_auto_cpu_reservation_keeps_default_when_enough_cpus_available():
    assert auto_size_num_cpus(default_num_cpus=10, remaining_num_cpus=[10, 2], available_cpus=32) == 10


def test_auto_cpu_reservation_scales_with_remaining_actor_defaults():
    num_cpus = auto_size_num_cpus(default_num_cpus=10, remaining_num_cpus=[10, 2], available_cpus=17)

    assert num_cpus == pytest.approx(7.7273)


def test_auto_cpu_reservation_leaves_reserved_worker_group_cpus():
    num_cpus = auto_size_num_cpus(
        default_num_cpus=10,
        remaining_num_cpus=[10, 2],
        reserved_num_cpus=5,
        available_cpus=17,
    )

    assert num_cpus == pytest.approx(5.4545)


def test_resolve_actor_num_cpus_uses_explicit_config():
    config = OmegaConf.create({"async_training": {"rollouter_num_cpus": 4}})

    num_cpus = resolve_actor_num_cpus(
        config=config,
        key="rollouter_num_cpus",
        default_num_cpus=10,
        remaining_num_cpus=[2],
        available_cpus=1,
    )

    assert num_cpus == 4


def test_resolve_actor_num_cpus_scales_auto_config():
    config = OmegaConf.create({"async_training": {"rollouter_num_cpus": AUTO_CPU_RESERVATION}})

    num_cpus = resolve_actor_num_cpus(
        config=config,
        key="rollouter_num_cpus",
        default_num_cpus=10,
        remaining_num_cpus=[2],
        available_cpus=4,
    )

    assert num_cpus == pytest.approx(3.3333)


def test_auto_config_is_desired_default_for_pending_actor_reservations():
    config = OmegaConf.create({"async_training": {"message_queue_num_cpus": AUTO_CPU_RESERVATION}})

    assert get_actor_num_cpus_config_or_default(config, "message_queue_num_cpus", 2) == 2


def test_worker_group_cpu_reserve_estimates_default_placement_group_bundles():
    config = OmegaConf.create(
        {
            "trainer": {"nnodes": 1, "n_gpus_per_node": 2},
            "rollout": {"nnodes": 2, "n_gpus_per_node": 4},
        }
    )

    assert estimate_trainer_worker_group_num_cpus(config) == 6
    assert estimate_rollout_worker_group_num_cpus(config) == 16


def test_invalid_actor_num_cpus_config_raises():
    config = OmegaConf.create({"async_training": {"trainer_num_cpus": "many"}})

    with pytest.raises(ValueError, match="trainer_num_cpus"):
        get_actor_num_cpus_config(config, "trainer_num_cpus")
