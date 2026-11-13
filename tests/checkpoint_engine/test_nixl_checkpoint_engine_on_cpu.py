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

from verl.checkpoint_engine.nixl_checkpoint_engine import NIXLCheckpointEngine


def test_build_topology_with_single_rollout_worker():
    metadata = ["trainer", "rollout"]

    _, rollout_kwargs = NIXLCheckpointEngine.build_topology(
        actor_wg_world_size=1,
        rollout_world_size=1,
        metadata=metadata,
    )

    assert rollout_kwargs["prev_agent_metadata"] == ["trainer"]
    assert rollout_kwargs["next_agent_metadata"] == [None]
