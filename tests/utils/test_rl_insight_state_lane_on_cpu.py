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

from verl.utils.tracking import RLInsightLogger


def test_rollout_state_lane_ids_include_role_and_replica():
    lane_ids = {
        RLInsightLogger.rollout_state_lane_id(replica_rank, prefix)
        for prefix in ("rollout", "teacher_default")
        for replica_rank in (0, 1)
    }

    assert lane_ids == {
        "rollout_replica_0",
        "rollout_replica_1",
        "teacher_default_replica_0",
        "teacher_default_replica_1",
    }
