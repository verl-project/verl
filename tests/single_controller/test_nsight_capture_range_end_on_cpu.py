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

import pytest

from verl.single_controller.ray.base import RayWorkerGroup


def _make_group(**kwargs):
    return RayWorkerGroup(worker_names=["worker"], worker_handles=[object()], **kwargs)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, "repeat-shutdown:12"),
        ({"profiler_discrete": True}, "repeat-shutdown:12"),
        ({"profiler_discrete": False}, "repeat-shutdown:2"),
    ],
    ids=["default-discrete", "discrete", "continuous"],
)
def test_capture_range_end_matches_ranges_per_step(kwargs, expected):
    group = _make_group(profile_steps=[1, 3], worker_nsight_options={"capture-range-end": None}, **kwargs)

    assert group.worker_nsight_options["capture-range-end"] == expected


def test_explicit_capture_range_end_is_kept():
    group = _make_group(
        profile_steps=[1, 3],
        worker_nsight_options={"capture-range-end": "repeat-shutdown:9"},
        profiler_discrete=False,
    )

    assert group.worker_nsight_options["capture-range-end"] == "repeat-shutdown:9"
