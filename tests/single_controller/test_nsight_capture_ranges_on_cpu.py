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

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from verl.single_controller.ray.base import RayWorkerGroup


@pytest.mark.parametrize(
    "kwargs, continuous, discrete",
    [
        ({}, False, True),
        ({"profiler_discrete": False, "profile_continuous_steps": True}, True, False),
    ],
    ids=["legacy-defaults", "explicit-overrides"],
)
def test_worker_group_uses_capture_count(kwargs, continuous, discrete):
    with patch("verl.single_controller.ray.base._count_capture_ranges", return_value=7) as count_ranges:
        group = RayWorkerGroup(
            worker_names=["worker"],
            worker_handles=[object()],
            profile_steps=[1],
            worker_nsight_options={"capture-range-end": None},
            **kwargs,
        )

    count_ranges.assert_called_once_with([1], continuous, discrete=discrete)
    assert group.worker_nsight_options["capture-range-end"] == "repeat-shutdown:7"


@pytest.mark.parametrize(
    "options, expected",
    [
        (None, None),
        ({"capture-range-end": "repeat-shutdown:9"}, {"capture-range-end": "repeat-shutdown:9"}),
    ],
    ids=["no-options", "explicit-capture-end"],
)
def test_worker_group_preserves_capture_options(options, expected):
    with patch("verl.single_controller.ray.base._count_capture_ranges", return_value=7):
        group = RayWorkerGroup(
            worker_names=["worker"],
            worker_handles=[object()],
            worker_nsight_options=options,
        )

    assert group.worker_nsight_options == expected


@pytest.mark.parametrize(
    "rank, options, expected",
    [
        (0, {"session-new": "run_{role}_{rank}"}, {"session-new": "run_actor_group_0"}),
        (3, {"session-new": "run_{role}_{rank}"}, {"session-new": "run_actor_group_3"}),
        (0, None, None),
        (0, {"trace": "cuda"}, {"trace": "cuda"}),
    ],
    ids=["rank-zero", "other-rank", "no-options", "no-session-template"],
)
def test_worker_session_names_are_rendered_without_mutating_template(rank, options, expected):
    group = RayWorkerGroup(
        worker_names=["worker"],
        worker_handles=[object()],
        name_prefix="actor_group",
        worker_nsight_options=None,
        device_name="cuda",
    )
    group.profile_steps = [1]
    group.worker_nsight_options = options
    group._master_addr = "localhost"
    group._master_port = "12345"
    worker_class = Mock()
    resource_pool = SimpleNamespace(world_size=4, use_gpu=True, store=[4], max_colocate_count=1)

    group._create_worker(rank, 0, object(), rank, resource_pool, worker_class, None, False)

    runtime_env = worker_class.update_options.call_args.args[0]["runtime_env"]
    assert runtime_env["nsight"] == expected
    if options is not None and "session-new" in options:
        assert options["session-new"] == "run_{role}_{rank}"
