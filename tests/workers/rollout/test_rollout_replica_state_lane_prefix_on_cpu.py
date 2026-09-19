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

import pytest

from verl.workers.config.rollout import RolloutConfig
from verl.workers.rollout.replica import RolloutReplica


class _StubRolloutReplica(RolloutReplica):
    async def launch_servers(self):
        raise NotImplementedError


def _make_replica(**kwargs) -> RolloutReplica:
    return _StubRolloutReplica(
        replica_rank=kwargs.pop("replica_rank", 0),
        config=RolloutConfig(),
        model_config=None,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("kwargs", "expected_prefix"),
    [
        ({}, "rollout"),
        ({"name_suffix": "student"}, "rollout_student"),
        ({"is_reward_model": True}, "reward"),
        ({"is_reward_model": True, "name_suffix": "rm"}, "reward_rm"),
        ({"is_teacher_model": True}, "teacher"),
        ({"is_teacher_model": True, "name_suffix": "deepseek_r1"}, "teacher_deepseek_r1"),
    ],
)
def test_state_lane_prefix_distinguishes_model_roles(kwargs, expected_prefix):
    replica = _make_replica(**kwargs)

    assert replica.state_lane_prefix == expected_prefix


def test_state_lane_prefix_matches_teacher_metric_label_format():
    replica = _make_replica(is_teacher_model=True, name_suffix="deepseek_r1", replica_rank=2)

    assert f"{replica.state_lane_prefix}_{replica.replica_rank}" == "teacher_deepseek_r1_2"
