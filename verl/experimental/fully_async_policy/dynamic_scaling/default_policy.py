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

"""Built-in "default" dynamic scaling policy."""

import ray

from .base import DynamicScaleContext, DynamicScalingPolicyBase, register_policy


@register_policy("default")
class DefaultDynamicScalingPolicy(DynamicScalingPolicyBase):
    """Default policy: deactivate whenever hybrid is active, with adaptive ratio.

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
            print("[DefaultDynamicScalingPolicy] only_hybrid=True: forcing deactivate_ratio=1.0")
            self.deactivate_ratio = 1.0

    def should_deactivate(self, global_steps: int, is_hybrid_active: bool, ctx: DynamicScaleContext) -> bool:
        return is_hybrid_active

    def deactivate_wait_samples(self, ctx: DynamicScaleContext) -> int:
        total = ctx.required_samples * ctx.trigger_parameter_sync_step
        return int(total * self.deactivate_ratio)

    def update_after_step(self, global_steps: int, ctx: DynamicScaleContext) -> None:
        if global_steps <= 0 or self.only_hybrid:
            return

        total_wait = sum(ctx.step_wait_times)

        if total_wait > 10:  # trainer wait over 10 seconds in last step
            self.deactivate_ratio = min(1.0, self.deactivate_ratio + 0.05)
        else:
            self.deactivate_ratio -= 0.2

        print(
            f"[DefaultDynamicScalingPolicy] step={global_steps} "
            f"deactivate_ratio={self.deactivate_ratio:.3f} "
            f"(wait={total_wait:.1f}s, generated={ctx.total_generated_samples}, expected={ctx.expected_samples}, "
            f"activate_dur={ctx.last_activate_duration_s:.2f}s, deactivate_dur={ctx.last_deactivate_duration_s:.2f}s)"
        )

    def should_activate_after_step(self, global_steps: int, is_hybrid_active: bool, ctx: DynamicScaleContext) -> bool:
        if self.only_hybrid:
            return True

        print(
            f"[should_activate_after_step] ctx.total_generated_samples:{ctx.total_generated_samples},"
            f" ctx.expected_samples:{ctx.expected_samples},"
            f" self.deactivate_ratio:{self.deactivate_ratio},"
            f" ctx.buffer_samples:{ctx.buffer_samples}"
        )
        print(f"DynamicScaleContext:{ctx}")

        return ctx.total_generated_samples - ctx.expected_samples < (
            self.deactivate_ratio * ctx.required_samples * ctx.trigger_parameter_sync_step
        )

    def request_rebalance(self, global_steps: int, ctx: DynamicScaleContext) -> None:
        """Redistribute requests across all active replicas after activation.

        Performs a full rebalance via the rollouter:

        1. Clears the load-balancer sticky-session cache.
        2. Aborts in-flight requests on all active replicas (standalone + hybrid),
           triggering :class:`FullyAsyncLLMServerClient` retry.
        3. Resumes generation so retried requests are accepted and routed via
           least-loaded selection — naturally balancing load toward the newly
           activated hybrid replicas (which start with 0 in-flight requests).
        """
        if not hasattr(self, "_rollouter") or self._rollouter is None:
            print("[DefaultDynamicScalingPolicy] request_rebalance skipped: no rollouter reference available")
            return

        try:
            result = ray.get(self._rollouter.rebalance_requests.remote())
            print(
                f"[DefaultDynamicScalingPolicy] request_rebalance done at step {global_steps}: "
                f"cleared {result.get('cleared_entries', 0)} sticky entries, "
                f"server loads: {result.get('server_loads', {})}"
            )
        except Exception as e:
            print(f"[DefaultDynamicScalingPolicy] request_rebalance failed at step {global_steps}: {e}")
