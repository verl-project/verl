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

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from verl.utils.profiler.config import ProfilerConfig
from verl.utils.profiler.nsight import (
    _build_gpu_metrics_nsight_launch_options,
    _build_rollout_ray_actor_nsight_launch_options,
    _config_select,
    _count_capture_ranges,
    _is_gpu_metrics_profile_enabled,
    _profile_rank_selected,
    _ray_actor_profile_enabled,
    _render_nsight_output_name,
    _render_ray_actor_nsight_options,
    _resolve_nsight_capture_options,
    _rollout_replica_ranks,
    build_controller_nsight_launch_options,
    build_ray_actor_nsight_launch_options,
    is_controller_nsys_api_control_enabled,
    resolve_ray_actor_profiler_config,
)


def _as_namespace(values):
    if isinstance(values, dict):
        return SimpleNamespace(**{key: _as_namespace(value) for key, value in values.items()})
    return values


@pytest.mark.parametrize("make_config", [dict, OmegaConf.create, _as_namespace], ids=["dict", "omegaconf", "object"])
@pytest.mark.parametrize("value", [False, 0, None, [1, 2]])
def test_config_select_preserves_existing_values(make_config, value):
    config = make_config({"profiler": {"ranks": value}})
    result = _config_select(config, "profiler.ranks", default="missing")
    assert result == value
    # OmegaConf wraps lists in ListConfig, so compare their contents without requiring the same type.
    if not isinstance(value, list):
        assert type(result) is type(value)


def test_config_select_returns_default_for_none_config():
    sentinel = object()
    assert _config_select(None, "profiler.ranks", default=sentinel) is sentinel


@pytest.mark.parametrize("make_config", [dict, OmegaConf.create, _as_namespace], ids=["dict", "omegaconf", "object"])
@pytest.mark.parametrize(
    "config, expected",
    [
        ({}, "default"),
        ({"profiler": None}, "default"),
        ({"profiler": {}}, "default"),
        ({"profiler": {"ranks": None}}, None),
    ],
    ids=["missing-intermediate-key", "intermediate-none", "missing-leaf-key", "leaf-none"],
)
def test_config_select_handles_missing_and_none_values(make_config, config, expected):
    config = make_config(config)
    assert _config_select(config, "profiler.ranks", default="default") == expected


def test_config_select_traverses_mixed_dict_and_object_layers():
    config = {"profiler": SimpleNamespace(nsys=SimpleNamespace(enable=True))}
    assert _config_select(config, "profiler.nsys.enable") is True


@pytest.mark.parametrize(
    "config, rank, override, expected",
    [
        (None, 0, None, False),
        (ProfilerConfig(enable=True), 0, None, True),
        (ProfilerConfig(enable=True), 1, None, False),
        (ProfilerConfig(enable=False), 0, None, False),
        (ProfilerConfig(enable=True, ranks=[]), 0, None, True),
        (ProfilerConfig(enable=True, ranks=[1, 2]), 1, None, True),
        (ProfilerConfig(enable=True, ranks=[1, 2]), 0, None, False),
        (ProfilerConfig(enable=False, ranks=[1]), 1, None, True),
        (ProfilerConfig(enable=False, all_ranks=True, ranks=[0]), 3, None, True),
        ({"all_ranks": True}, 1, None, True),
        (OmegaConf.create({"all_ranks": True}), 1, None, True),
        (SimpleNamespace(all_ranks=True), 1, None, True),
        (ProfilerConfig(enable=True, ranks=[2]), 1, [1], True),
    ],
)
def test_profile_rank_selection(config, rank, override, expected):
    # Rank selection is separate from the overall enable gate in DistProfiler.
    assert _profile_rank_selected(config, rank, ranks=override) is expected


@pytest.mark.parametrize(
    "ranks, world_size, expected",
    [
        ([], 1, []),
        ([0 * 3 + 0, 1 * 3 + 1, 2 * 3 + 2], 3, [0, 1, 2]),
        ([1, 0], 1, [0, 1]),
        ([1, 1, 1], 1, [1]),
        (["1"], 1, [1]),
    ],
)
def test_rollout_replica_ranks(ranks, world_size, expected):
    original_ranks = list(ranks)
    assert _rollout_replica_ranks(ranks, world_size) == expected
    assert ranks == original_ranks


@pytest.mark.parametrize(
    "steps, continuous, discrete, expected",
    [
        (None, False, False, 0),
        ([], True, False, 0),
        ([1, 2, 4], False, False, 3),
        ([1, 2, 4], True, False, 2),
        ([4, 2, 1, 2], True, False, 2),
        ([1, 1, 2], False, False, 2),
        ([1, 1, 2], False, True, 6 * 2),
        ([1, 1, 2], True, True, 6 * 2),
        ([1, 2, 4], False, True, 6 * 3),
        ([1, 2, 4], True, True, 6 * 3),
    ],
)
def test_count_capture_ranges(steps, continuous, discrete, expected):
    original_steps = copy.deepcopy(steps)
    assert _count_capture_ranges(steps, continuous, discrete=discrete) == expected
    assert steps == original_steps


@pytest.mark.parametrize(
    "options, capture_count, expected",
    [
        (
            {"capture-range": "cudaProfilerApi"},
            7,
            {"capture-range": "cudaProfilerApi", "capture-range-end": "repeat-shutdown:7"},
        ),
        (
            {"capture-range": "cudaProfilerApi", "capture-range-end": None},
            7,
            {"capture-range": "cudaProfilerApi", "capture-range-end": "repeat-shutdown:7"},
        ),
        (
            {"capture-range": "cudaProfilerApi", "capture-range-end": "stop"},
            None,
            {"capture-range": "cudaProfilerApi", "capture-range-end": "stop"},
        ),
        ({"capture-range": "nvtx"}, None, {"capture-range": "nvtx"}),
        ({}, None, {}),
        ({"capture-range": "cudaProfilerApi"}, 0, {"capture-range": "cudaProfilerApi"}),
    ],
    ids=["missing-end", "null-end", "explicit-end", "nvtx", "empty", "zero-captures"],
)
def test_resolve_nsight_capture_options(options, capture_count, expected):
    with patch("verl.utils.profiler.nsight._count_capture_ranges", return_value=capture_count) as count_ranges:
        result = _resolve_nsight_capture_options(options, [1], False)
    # None marks cases that do not need a capture count; do not constrain those calls.
    if capture_count is not None:
        count_ranges.assert_called_once_with([1], False)
    assert result == expected


@pytest.mark.parametrize(
    "options",
    [
        {"capture-range": "cudaProfilerApi", "o": "worker"},
        OmegaConf.create({"capture-range": "cudaProfilerApi", "o": "worker"}),
    ],
    ids=["dict", "omegaconf"],
)
def test_resolve_nsight_capture_options_returns_independent_dict(options):
    original_options = copy.deepcopy(options)
    with patch("verl.utils.profiler.nsight._count_capture_ranges", return_value=7):
        result = _resolve_nsight_capture_options(options, [1], False)
    assert type(result) is dict
    assert result is not options
    assert result["o"] == "worker"
    result["o"] = "changed"
    assert options == original_options


@pytest.mark.parametrize(
    "profiler_config, expected_options",
    [
        (
            ProfilerConfig(tool="nsys", ray_actor_nsight_options={"o": "role"}),
            {"o": "role"},
        ),
        (
            OmegaConf.create({"tool": "nsys", "ray_actor_nsight_options": {"o": "role"}}),
            {"o": "role"},
        ),
        (
            ProfilerConfig(tool="nsys"),
            {"o": "global"},
        ),
        (ProfilerConfig(tool="nsys", ray_actor_nsight_options={}), {}),
    ],
    ids=["role-options-dataclass", "role-options-omegaconf", "global-fallback", "empty-role-options"],
)
def test_resolve_ray_actor_profiler_config(profiler_config, expected_options):
    config = OmegaConf.create(
        {
            "global_profiler": {
                "steps": [1],
                "profile_continuous_steps": False,
                "global_tool_config": {"nsys": {"worker_nsight_options": {"o": "global"}}},
            }
        }
    )
    original_role = copy.deepcopy(profiler_config)
    original_global = copy.deepcopy(config)
    with patch(
        "verl.utils.profiler.nsight._resolve_nsight_capture_options", return_value={"o": "resolved"}
    ) as resolve_options:
        result = resolve_ray_actor_profiler_config(profiler_config, config)
    resolve_options.assert_called_once_with(expected_options, [1], False)
    assert type(result) is type(profiler_config)
    assert result is not profiler_config
    assert result.ray_actor_nsight_options == {"o": "resolved"}
    result.ray_actor_nsight_options["o"] = "changed"
    assert profiler_config == original_role
    assert config == original_global


@pytest.mark.parametrize("profiler_config", [None, ProfilerConfig(tool="torch"), ProfilerConfig(tool="nsys")])
def test_resolve_ray_actor_profiler_config_without_nsight_options_is_unchanged(profiler_config):
    assert resolve_ray_actor_profiler_config(profiler_config, {}) == profiler_config


@pytest.mark.parametrize(
    "settings, expected",
    [
        ({}, False),
        ({"tool": "nsys"}, False),
        ({"tool": "nsys", "global_tool_config": {"nsys": {"controller_nsight_options": None}}}, False),
        (
            {"tool": "nsys", "global_tool_config": {"nsys": {"controller_nsight_options": {"capture-range": "nvtx"}}}},
            False,
        ),
        (
            {
                "tool": "torch",
                "global_tool_config": {"nsys": {"controller_nsight_options": {"capture-range": "cudaProfilerApi"}}},
            },
            False,
        ),
        (
            {
                "tool": "nsys",
                "global_tool_config": {"nsys": {"controller_nsight_options": {"capture-range": "cudaProfilerApi"}}},
            },
            True,
        ),
    ],
)
def test_is_controller_nsys_api_control_enabled(settings, expected):
    config = OmegaConf.create({"global_profiler": settings})
    assert is_controller_nsys_api_control_enabled(config) is expected


def test_build_controller_nsight_launch_options():
    config = OmegaConf.create(
        {
            "global_profiler": {
                "steps": [1],
                "profile_continuous_steps": False,
                "global_tool_config": {"nsys": {"controller_nsight_options": {}}},
            }
        }
    )
    with patch(
        "verl.utils.profiler.nsight._resolve_nsight_capture_options", return_value={"o": "resolved"}
    ) as resolve_options:
        result = build_controller_nsight_launch_options(config)
    resolve_options.assert_called_once_with({}, [1], False)
    assert result == {"o": "resolved"}


@pytest.mark.parametrize(
    "tool, steps, options, expected",
    [
        ("nsys", [1], {"gpu-metrics-devices": "all"}, True),
        ("torch", [1], {"gpu-metrics-devices": "all"}, False),
        ("nsys", [], {"gpu-metrics-devices": "all"}, False),
        ("nsys", None, {"gpu-metrics-devices": "all"}, False),
        ("nsys", [1], {}, False),
        ("nsys", [1], None, False),
    ],
    ids=["enabled", "other-tool", "empty-steps", "null-steps", "empty-options", "null-options"],
)
def test_is_gpu_metrics_profile_enabled(tool, steps, options, expected):
    config = OmegaConf.create(
        {
            "global_profiler": {
                "tool": tool,
                "steps": steps,
                "global_tool_config": {"nsys": {"gpu_metrics_nsight_options": options}},
            }
        }
    )
    assert _is_gpu_metrics_profile_enabled(config) is expected


def test_gpu_metrics_profiling_defaults_to_disabled():
    assert _is_gpu_metrics_profile_enabled(OmegaConf.create({})) is False


@pytest.mark.parametrize(
    "options, message",
    [
        ({}, "must not be empty"),
        (None, "must not be empty"),
        ({"gpu-metrics-devices": "all"}, "cudaProfilerApi"),
        ({"capture-range": "nvtx"}, "cudaProfilerApi"),
    ],
    ids=["empty-options", "null-options", "missing-capture-range", "other-capture-range"],
)
def test_build_gpu_metrics_nsight_launch_options_rejects_invalid_options(options, message):
    config = OmegaConf.create(
        {"global_profiler": {"global_tool_config": {"nsys": {"gpu_metrics_nsight_options": options}}}}
    )
    with pytest.raises(ValueError, match=message):
        _build_gpu_metrics_nsight_launch_options(config)


def test_build_gpu_metrics_nsight_launch_options():
    options = {"capture-range": "cudaProfilerApi", "gpu-metrics-devices": "all"}
    config = OmegaConf.create(
        {
            "global_profiler": {
                "steps": [1],
                "profile_continuous_steps": False,
                "global_tool_config": {"nsys": {"gpu_metrics_nsight_options": options}},
            }
        }
    )
    with patch(
        "verl.utils.profiler.nsight._resolve_nsight_capture_options", return_value={"o": "resolved"}
    ) as resolve_options:
        result = _build_gpu_metrics_nsight_launch_options(config)
    resolve_options.assert_called_once_with(options, [1], False)
    assert result == {"o": "resolved"}


@pytest.mark.parametrize(
    "output, expected",
    [
        (None, "reward_loop_rank_2_%h_pid%p"),
        ("trace_%h_pid%p", "reward_loop_rank_2_trace_%h_pid%p"),
        ("{role}_rank_{rank}_%h_pid%p", "reward_loop_rank_2_%h_pid%p"),
        ("{role}_%p", "reward_loop_%p"),
        ("rank_{rank}_%p", "rank_2_%p"),
        ("rank_%q{RANK}_%p", "reward_loop_rank_2_rank_2_%p"),
        ("{role}_%q{RANK}_%p", "reward_loop_2_%p"),
    ],
    ids=["default", "prefix", "role-and-rank", "role-only", "rank-only", "rank-env", "role-and-rank-env"],
)
def test_render_nsight_output_name(output, expected):
    assert _render_nsight_output_name(output, rank=2, role="reward_loop") == expected


@pytest.mark.parametrize(
    "config, expected",
    [
        (None, False),
        (ProfilerConfig(tool="torch", enable=True), False),
        (ProfilerConfig(tool="nsys", enable=False), False),
        (ProfilerConfig(tool="nsys", enable=True), True),
    ],
)
def test_ray_actor_profile_enabled(config, expected):
    assert _ray_actor_profile_enabled(config) is expected


@pytest.mark.parametrize(
    "config, output, expected",
    [
        (
            ProfilerConfig(ray_actor_nsight_options={"o": "template", "trace": "cuda"}),
            "template",
            {"o": "rendered", "trace": "cuda"},
        ),
        (
            OmegaConf.create({"ray_actor_nsight_options": {"o": "template", "trace": "cuda"}}),
            "template",
            {"o": "rendered", "trace": "cuda"},
        ),
        (ProfilerConfig(ray_actor_nsight_options={}), None, {"o": "rendered"}),
    ],
    ids=["dataclass", "omegaconf", "default-output"],
)
def test_render_ray_actor_nsight_options(config, output, expected):
    original_config = copy.deepcopy(config)
    with patch("verl.utils.profiler.nsight._render_nsight_output_name", return_value="rendered") as render_output:
        result = _render_ray_actor_nsight_options(config, rank=1, role="reward_loop")
    render_output.assert_called_once_with(output, rank=1, role="reward_loop")
    assert type(result) is dict
    assert result == expected
    result["o"] = "changed"
    assert config == original_config


def test_render_ray_actor_nsight_options_requires_options():
    with pytest.raises(ValueError, match="options"):
        _render_ray_actor_nsight_options(ProfilerConfig(), rank=0, role="reward_loop")


@pytest.mark.parametrize(
    "enabled, selected, expected",
    [(False, True, None), (True, False, None), (True, True, {"o": "rendered"})],
    ids=["disabled", "unselected-rank", "selected-rank"],
)
def test_build_ray_actor_nsight_launch_options(enabled, selected, expected):
    config = ProfilerConfig(tool_config={})
    with (
        patch("verl.utils.profiler.nsight._ray_actor_profile_enabled", return_value=enabled) as profile_enabled,
        patch("verl.utils.profiler.nsight._profile_rank_selected", return_value=selected) as select_rank,
        patch("verl.utils.profiler.nsight._render_ray_actor_nsight_options", return_value={"o": "rendered"}) as render,
    ):
        result = build_ray_actor_nsight_launch_options(config, rank=1, role="reward_loop")
    assert result == expected
    if expected is not None:
        profile_enabled.assert_any_call(config)
        select_rank.assert_any_call(config, 1)
        render.assert_called_once_with(config, rank=1, role="reward_loop")


@pytest.mark.parametrize(
    "enabled, selected, expected",
    [(False, True, None), (True, False, None), (True, True, {"o": "rendered"})],
    ids=["disabled", "unselected-replica", "selected-replica"],
)
def test_build_rollout_ray_actor_nsight_launch_options(enabled, selected, expected):
    config = ProfilerConfig(ranks=[0], tool_config={})
    with (
        patch("verl.utils.profiler.nsight._ray_actor_profile_enabled", return_value=enabled) as profile_enabled,
        patch("verl.utils.profiler.nsight._rollout_replica_ranks", return_value=[7]) as map_ranks,
        patch("verl.utils.profiler.nsight._profile_rank_selected", return_value=selected) as select_rank,
        patch("verl.utils.profiler.nsight._render_ray_actor_nsight_options", return_value={"o": "rendered"}) as render,
    ):
        result = _build_rollout_ray_actor_nsight_launch_options(
            config, replica_rank=1, replica_world_size=2, role="rollout"
        )
    assert result == expected
    if expected is not None:
        profile_enabled.assert_any_call(config)
        map_ranks.assert_any_call([0], 2)
        select_rank.assert_any_call(config, 1, ranks=[7])
        render.assert_called_once_with(config, rank=1, role="rollout")


@pytest.mark.parametrize(
    "build_options, rank_kwargs",
    [
        (build_ray_actor_nsight_launch_options, {"rank": 0}),
        (_build_rollout_ray_actor_nsight_launch_options, {"replica_rank": 0, "replica_world_size": 1}),
    ],
    ids=["actor", "rollout"],
)
def test_ray_actor_launch_options_reject_discrete_mode(build_options, rank_kwargs):
    config = ProfilerConfig(tool_config={"nsys": {"discrete": True}})
    with (
        patch("verl.utils.profiler.nsight._ray_actor_profile_enabled", return_value=True),
        patch("verl.utils.profiler.nsight._profile_rank_selected", return_value=True),
        pytest.raises(ValueError, match="discrete"),
    ):
        build_options(config, role="rollout", **rank_kwargs)
