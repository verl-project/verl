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

    def set_weights(self, step, ref):
        self.weights[step] = ref
        # Keep only the last 1 step to prevent memory leaks and disk spilling.
        steps_to_keep = sorted(self.weights.keys())
        if len(steps_to_keep) > 1:
            for old_step in steps_to_keep[:-1]:
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
