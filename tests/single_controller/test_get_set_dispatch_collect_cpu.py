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
import os

import pytest

from verl.single_controller.base import Worker


def test_get_set_dispatch_collect_cpu():
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "2"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"

    ref = Worker()
    ref._register_dispatch_collect_info(mesh_name="actor", dp_rank=0, is_collect=True)

    actor = Worker()
    actor._register_dispatch_collect_info(mesh_name="actor", dp_rank=1, is_collect=False)

    actor_rollout_ref = Worker()
    actor_rollout_ref.set_dispatch_collect(mesh_name="ref", **ref.get_dispatch_collect())
    actor_rollout_ref.set_dispatch_collect(mesh_name="actor", **actor.get_dispatch_collect())

    assert actor_rollout_ref._query_dispatch_info("ref") == 0
    assert actor_rollout_ref._query_collect_info("ref")
    assert actor_rollout_ref._query_dispatch_info("actor") == 1
    assert not actor_rollout_ref._query_collect_info("actor")

    # test conflict mesh_name
    actor2 = Worker()
    actor2._register_dispatch_collect_info(mesh_name="actor", dp_rank=1, is_collect=False)
    with pytest.raises(AssertionError):
        actor_rollout_ref.set_dispatch_collect(mesh_name="actor", **actor2.get_dispatch_collect())
