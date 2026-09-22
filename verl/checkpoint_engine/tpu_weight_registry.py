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

import ray

"""Lightweight CPU-only Ray actor registry for tracking TPU weight checkpoint references."""


@ray.remote(num_cpus=0)
class TPUWeightRegistry:
    """Ray actor for holding references to synchronized TPU model weights across steps."""

    def __init__(self):
        self.weights = {}
        self.peers = []
        self.stats = {}
        self.controller_address = None
        self.global_shapes = {}

    def set_global_shapes(self, shapes: dict):
        self.global_shapes = shapes

    def get_global_shapes(self) -> dict:
        return self.global_shapes

    def set_controller_address(self, address: str):
        self.controller_address = address

    def get_controller_address(self) -> str:
        return self.controller_address

    def set_peers(self, peers):
        self.peers = peers

    def get_peers(self):
        return self.peers

    def set_stats(self, step, stats):
        self.stats[step] = {"master": stats} if "master" not in stats else stats
        steps_to_keep = sorted(self.stats.keys())
        if len(steps_to_keep) > 5:
            for old_step in steps_to_keep[:-5]:
                del self.stats[old_step]

    def set_rank_stats(self, step, rank, stats):
        if rank == 0:
            self.set_stats(step, stats)

    def get_stats(self, step):
        return self.stats.get(step, None)

    def set_weights(self, step, ref):
        self.weights[step] = ref
        # Keep only the entry just written, to bound memory and disk.
        #
        # Evict by write, not by step order. This actor is detached, so a step
        # left by a previous job (say 5) outranks the step 0 a fresh job just
        # published, and an ordering-based policy would delete the new entry.
        for old_step in [s for s in self.weights if s != step]:
            old_ref = self.weights[old_step]
            if isinstance(old_ref, str):
                try:
                    import os

                    if os.path.exists(old_ref):
                        os.remove(old_ref)
                except Exception:
                    pass
            del self.weights[old_step]

    def get_weights(self, step):
        return self.weights.get(step, None)

    def clear(self):
        """Drops every cached entry. Used to reset state left by a previous job."""
        self.weights.clear()
        self.stats.clear()
        self.global_shapes.clear()

def get_tpu_weight_registry():
    """Gets or creates the singleton detached TPUWeightRegistry actor in the 'verl' namespace."""
    try:
        return ray.get_actor("TPUWeightRegistry", namespace="verl")
    except ValueError:
        pass

    try:
        return TPUWeightRegistry.options(
            name="TPUWeightRegistry", namespace="verl", lifetime="detached"
        ).remote()
    except ValueError:
        # Handled race condition: another worker created it concurrently
        return ray.get_actor("TPUWeightRegistry", namespace="verl")

