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
"""Exercise cross-node placement using two local Ray nodes with synthetic GPUs.

Run manually on a host with enough CPU resources. No CUDA kernels are executed.
Use a short TMPDIR so Ray's Unix socket paths fit within the OS limit.
"""

from collections import Counter

import ray
from ray.cluster_utils import Cluster
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from verl.single_controller.ray.base import RayResourcePool, split_resource_pool


@ray.remote(num_cpus=0.1, num_gpus=0.1)
class Probe:
    def location(self):
        ctx = ray.get_runtime_context()
        return ctx.get_node_id(), str(ctx.get_accelerator_ids()["GPU"][0])


def main():
    cluster = Cluster()
    try:
        for _ in range(2):
            cluster.add_node(num_cpus=32, num_gpus=8, object_store_memory=100 * 1024 * 1024)
        ray.init(address=cluster.address, log_to_driver=False)
        parent = RayResourcePool(process_on_nodes=[8, 8], max_colocate_count=3)
        parent.get_placement_groups(device_name="cuda")
        for rollout_size in (12, 10):
            rollout, trainer = split_resource_pool(parent, [rollout_size, 16 - rollout_size])
            actors, memberships = [], []
            for role, pool in (("rollout", rollout), ("trainer", trainer)):
                for index in range(pool.start_bundle_index, pool.start_bundle_index + pool.world_size):
                    actor = Probe.options(
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pool.pgs[index // 8], placement_group_bundle_index=index % 8
                        )
                    ).remote()
                    actors.append(actor)
                    memberships.append(role)
            locations = ray.get([actor.location.remote() for actor in actors])
            rollout_locations = {loc for loc, role in zip(locations, memberships, strict=True) if role == "rollout"}
            trainer_locations = {loc for loc, role in zip(locations, memberships, strict=True) if role == "trainer"}
            assert len(rollout_locations) == rollout_size
            assert len(trainer_locations) == 16 - rollout_size
            assert rollout_locations.isdisjoint(trainer_locations)
            assert sorted(Counter(node for node, _ in rollout_locations).values()) == [rollout_size - 8, 8]
            print(f"rollout={rollout_size}, trainer={16 - rollout_size}: disjoint GPU slots on two Ray nodes")
            for actor in actors:
                ray.kill(actor)
    finally:
        ray.shutdown()
        cluster.shutdown()


if __name__ == "__main__":
    main()
