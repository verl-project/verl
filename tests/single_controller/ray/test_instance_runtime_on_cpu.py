# Copyright 2025 Individual Contributor: Zehan Zhu
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
"""Two instances share real Ray names and placement groups in one namespace."""

import uuid

import ray
from ray.util.placement_group import placement_group_table, remove_placement_group

from verl.single_controller.ray.base import RayResourcePool
from verl.single_controller.ray.instance_naming import InstanceNameBuilder


@ray.remote(num_cpus=0)
class IdentityWorker:
    def __init__(self, identity):
        self.identity = identity

    def ping(self):
        return self.identity


def test_two_instances_share_runtime_and_release_independently():
    owns_runtime = not ray.is_initialized()
    if owns_runtime:
        ray.init(num_cpus=2, include_dashboard=False)
    actors, groups = [], []
    namespace = "instance_test_" + uuid.uuid4().hex
    try:
        for identity in ("policy_a", "policy_b"):
            names = InstanceNameBuilder(identity)
            pool = RayResourcePool(
                [1], use_gpu=False, name_prefix=namespace, max_colocate_count=1, instance_name_builder=names
            )
            groups.extend(pool.get_placement_groups())
            actor = IdentityWorker.options(name=names.actor("worker"), namespace=namespace).remote(identity)
            actors.append(actor)
        # Ray rejects a duplicate placement-group name, so both creations succeeding
        # already implies the names differ; assert the instance prefix explicitly.
        pg_names = [placement_group_table()[group.id.hex()]["name"] for group in groups]
        assert pg_names[0].startswith("policy_a__")
        assert pg_names[1].startswith("policy_b__")
        assert ray.get([actor.ping.remote() for actor in actors], timeout=30) == ["policy_a", "policy_b"]
        for identity in ("policy_a", "policy_b"):
            handle = ray.get_actor(InstanceNameBuilder(identity).actor("worker"), namespace=namespace)
            assert ray.get(handle.ping.remote(), timeout=30) == identity
        ray.kill(actors.pop(0))
        remove_placement_group(groups.pop(0))
        assert ray.get(actors[0].ping.remote(), timeout=30) == "policy_b"
    finally:
        for actor in actors:
            ray.kill(actor)
        for group in groups:
            remove_placement_group(group)
        if owns_runtime:
            ray.shutdown()
