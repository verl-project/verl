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

from unittest.mock import patch

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
