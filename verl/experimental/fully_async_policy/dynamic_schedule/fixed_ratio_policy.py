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

"""Built-in "fixed_ratio" dynamic scheduling policy (no adaptive ratio update)."""

import ray

from .base import DynamicScheduleContext, DynamicSchedulePolicyBase, register_policy


@register_policy("fixed_ratio")
class FixedRatioDynamicSchedulePolicy(DynamicSchedulePolicyBase):
    """Fixed-ratio policy: same as default but deactivate_ratio is never updated.

    Identical to :class:`DefaultDynamicSchedulePolicy` except that
    ``update_after_step`` is a no-op — the ratio stays at its initial value
    throughout training regardless of trainer wait time or sample-buffer state.

    Wait threshold: deactivate_ratio × step_required_samples
    (= required_samples × trigger_parameter_sync_step).

    Args:
        deactivate_ratio: Fixed ratio in (0, 1].
        only_hybrid: No standalone replicas; forces ratio=1.0.
    """

    def __init__(self, deactivate_ratio: float = 0.3, only_hybrid: bool = False):
        self.deactivate_ratio = deactivate_ratio
        self.only_hybrid = only_hybrid
        if only_hybrid:
            print("[FixedRatioDynamicSchedulePolicy] only_hybrid=True: forcing deactivate_ratio=1.0")
            self.deactivate_ratio = 1.0

    def should_deactivate(self, global_steps: int, is_hybrid_active: bool, ctx: DynamicScheduleContext) -> bool:
        return is_hybrid_active

    def deactivate_wait_samples(self, ctx: DynamicScheduleContext) -> int:
        return int(ctx.step_required_samples * self.deactivate_ratio)

    def update_after_step(self, global_steps: int, ctx: DynamicScheduleContext) -> None:
        """No-op: deactivate_ratio is fixed and never updated."""
        print(
            f"[FixedRatioDynamicSchedulePolicy] step={global_steps} "
            f"deactivate_ratio={self.deactivate_ratio:.3f} (fixed, no update) "
            f"(generated={ctx.total_generated_samples}, expected={ctx.expected_samples}, "
            f"activate_dur={ctx.last_activate_duration_s:.2f}s, deactivate_dur={ctx.last_deactivate_duration_s:.2f}s)"
        )

    def should_activate_after_step(
        self, global_steps: int, is_hybrid_active: bool, ctx: DynamicScheduleContext
    ) -> bool:
        if self.only_hybrid:
            return True

        print(
            f"[should_activate_after_step] ctx.total_generated_samples:{ctx.total_generated_samples}, "
            f"ctx.expected_samples:{ctx.expected_samples}, self.deactivate_ratio:{self.deactivate_ratio},"
            f" ctx.buffer_samples:{ctx.buffer_samples}"
        )
        print(f"DynamicScheduleContext:{ctx}")

        return ctx.total_generated_samples - ctx.expected_samples < self.deactivate_ratio * ctx.step_required_samples

    def request_rebalance(self, global_steps: int, ctx: DynamicScheduleContext) -> None:
        """Redistribute requests across all active replicas after activation."""
        if not hasattr(self, "_rollouter") or self._rollouter is None:
            print("[FixedRatioDynamicSchedulePolicy] request_rebalance skipped: no rollouter reference available")
            return

        try:
            result = ray.get(self._rollouter.rebalance_requests.remote())
            print(
                f"[FixedRatioDynamicSchedulePolicy] request_rebalance done at step {global_steps}: "
                f"cleared {result.get('cleared_entries', 0)} sticky entries, "
                f"server loads: {result.get('server_loads', {})}"
            )
        except Exception as e:
            print(f"[FixedRatioDynamicSchedulePolicy] request_rebalance failed at step {global_steps}: {e}")
