# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import os
from dataclasses import FrozenInstanceError

import pytest
from hydra import compose, initialize_config_dir

from verl.utils import omega_conf_to_dataclass
from verl.utils.profiler.config import ProfilerConfig


def test_finish_hook_field_defaults_and_validation():
    config = ProfilerConfig()
    assert config.relocate_results is False
    assert config.finish_hook_cmd is None
    assert config.finish_hook_all_ranks is False
    assert config.finish_hook_ranks == []

    with pytest.raises(AssertionError):
        ProfilerConfig(finish_hook_cmd=123)
    with pytest.raises(AssertionError):
        ProfilerConfig(finish_hook_ranks="0,1")


def test_config_init():
    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        cfg = compose(config_name="ppo_trainer")
    for config in [
        cfg.actor_rollout_ref.actor.profiler,
        cfg.actor_rollout_ref.rollout.profiler,
        cfg.actor_rollout_ref.ref.profiler,
        cfg.critic.profiler,
        cfg.reward.profiler,
    ]:
        profiler_config = omega_conf_to_dataclass(config)
        assert isinstance(profiler_config, ProfilerConfig)
        assert profiler_config.tool == config.tool
        assert profiler_config.enable == config.enable
        assert profiler_config.all_ranks == config.all_ranks
        assert profiler_config.ranks == config.ranks
        assert profiler_config.save_path == config.save_path
        with pytest.raises(AttributeError):
            _ = profiler_config.non_existing_key
        assert config.get("non_existing_key") == profiler_config.get("non_existing_key")
        assert config.get("non_existing_key", 1) == profiler_config.get("non_existing_key", 1)


def test_role_configs_inherit_global_finish_hook():
    """Global finish-hook settings must reach each role's profiler configuration."""
    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        cfg = compose(
            config_name="ppo_trainer",
            overrides=[
                "global_profiler.finish_hook_cmd=upload.sh",
                "global_profiler.finish_hook_all_ranks=True",
                "global_profiler.finish_hook_ranks=[0,8]",
                "global_profiler.relocate_results=True",
            ],
        )
    for config in [
        cfg.actor_rollout_ref.actor.profiler,
        cfg.actor_rollout_ref.rollout.profiler,
        cfg.actor_rollout_ref.ref.profiler,
        cfg.critic.profiler,
    ]:
        assert config.finish_hook_cmd == "upload.sh"
        assert config.finish_hook_all_ranks is True
        assert list(config.finish_hook_ranks) == [0, 8]
        assert config.relocate_results is True


@pytest.mark.parametrize(
    "combine, expected_flag, expected_ranks",
    [(ProfilerConfig.union, True, {0, 1, 2}), (ProfilerConfig.intersect, False, {1})],
    ids=["union", "intersect"],
)
def test_union_intersect_combine_fields(combine, expected_flag, expected_ranks):
    left = ProfilerConfig(
        tool="nsys",
        enable=True,
        ranks=[0, 1],
        relocate_results=True,
        finish_hook_cmd="cmd_a",
        finish_hook_all_ranks=False,
        finish_hook_ranks=[0, 1],
    )
    right = ProfilerConfig(
        tool="nsys",
        all_ranks=True,
        ranks=[1, 2],
        relocate_results=False,
        finish_hook_cmd=None,
        finish_hook_all_ranks=True,
        finish_hook_ranks=[1, 2],
    )
    result = combine(left, right)
    assert result.enable is expected_flag
    assert result.all_ranks is expected_flag
    assert result.relocate_results is expected_flag
    assert result.finish_hook_all_ranks is expected_flag
    assert set(result.ranks) == expected_ranks
    assert set(result.finish_hook_ranks) == expected_ranks
    assert result.finish_hook_cmd == "cmd_a"


@pytest.mark.parametrize("combine", [ProfilerConfig.union, ProfilerConfig.intersect], ids=["union", "intersect"])
@pytest.mark.parametrize(
    "left_options, right_options, expected",
    [
        (None, None, None),
        (None, {"o": "right"}, {"o": "right"}),
        ({}, {"o": "right"}, {}),
        ({"o": "left"}, {"o": "right"}, {"o": "left"}),
    ],
    ids=["both-unset", "right-fallback", "empty-left", "left-precedence"],
)
def test_combine_ray_actor_nsight_options(combine, left_options, right_options, expected):
    left = ProfilerConfig(tool=None, ray_actor_nsight_options=left_options)
    right = ProfilerConfig(tool=None, ray_actor_nsight_options=right_options)
    assert combine(left, right).ray_actor_nsight_options == expected


def test_frozen_config():
    config = ProfilerConfig(all_ranks=False, ranks=[0])
    with pytest.raises(FrozenInstanceError):
        config.all_ranks = True
    with pytest.raises(FrozenInstanceError):
        config.ranks = [1, 2, 3]
    with pytest.raises(TypeError):
        config["all_ranks"] = True
    with pytest.raises(TypeError):
        config["ranks"] = [1, 2, 3]
