# Copyright 2025 Meituan Ltd. and/or its affiliates
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

"""Built-in "static_fully_async" dynamic scheduling policy."""

from .base import DynamicScheduleContext, DynamicSchedulePolicyBase, register_policy


@register_policy("static_fully_async")
class StaticFullyAsyncPolicy(DynamicSchedulePolicyBase):
    """Static fully-async policy: deactivate whenever hybrid is active, with adaptive ratio.

    Wait threshold: deactivate_ratio × required_samples × trigger_parameter_sync_step.

    Ratio adaptation (skipped when only_hybrid=True):
      - trainer wait > 10 s  → ratio += 0.05  (rollout is bottleneck, deactivate later)
      - sample buffer excess  → ratio -= 0.05  (training is bottleneck, deactivate earlier)

    Args:
        deactivate_ratio: Initial ratio in (0, 1].
        only_hybrid: No standalone replicas; forces ratio=1.0, disables adaptation.
    """

    def __init__(self, deactivate_ratio: float = 0.3, only_hybrid: bool = False):
        self.deactivate_ratio = deactivate_ratio
        self.only_hybrid = only_hybrid
        if only_hybrid:
            print("[StaticFullyAsyncPolicy] only_hybrid=True: forcing deactivate_ratio=1.0")
            self.deactivate_ratio = 1.0

    def should_deactivate(self, global_steps: int, is_hybrid_active: bool, ctx: DynamicScheduleContext) -> bool:
        return is_hybrid_active

    def deactivate_wait_samples(self, ctx: DynamicScheduleContext) -> int:
        return 0

    def update_after_step(self, global_steps: int, ctx: DynamicScheduleContext) -> None:
        return

    def should_activate_after_step(
        self, global_steps: int, is_hybrid_active: bool, ctx: DynamicScheduleContext
    ) -> bool:
        return False
