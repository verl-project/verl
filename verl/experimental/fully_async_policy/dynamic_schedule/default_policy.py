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

"""Built-in "default" dynamic scheduling policy."""

import ray

from .base import DynamicScheduleContext, DynamicSchedulePolicyBase, register_policy

DEFAULT_SWITCH_THRESHOLD_S = 10.0
SWITCH_OVERHEAD_WINDOW_SIZE = 3
# Bounds for the deactivate_ratio increase step (see update_after_step): the raw
# step is total_wait_samples / ctx.step_required_samples, i.e. the fraction of
# this cycle's samples that actually had to be waited on.
DEACTIVATE_RATIO_INCREASE_MIN = 0.02
DEACTIVATE_RATIO_INCREASE_MAX = 0.1
DEACTIVATE_RATIO_DECREASE_STEP = 0.02
# Deliberately pessimistic initial per-sample-time estimate (seconds), used before
# any cycle has produced a real generation-rate signal. Being large makes the
# estimated standalone-only shortfall look expensive, biasing should_activate_after_step
# toward activating hybrid until real timing data narrows the estimate.
INITIAL_PER_SAMPLE_TIME_S = 1000.0


@register_policy("default")
class DefaultDynamicSchedulePolicy(DynamicSchedulePolicyBase):
    """Default policy: deactivate whenever hybrid is active, with adaptive ratio.

    Wait threshold: deactivate_ratio × step_required_samples
    (= required_samples × trigger_parameter_sync_step).

    Ratio adaptation (skipped when only_hybrid=True), based on whether this
    cycle had to really wait on rollout generation (``step_wait_samples``,
    excluding samples served instantly from queue backlog):
      - any real wait occurred → ratio += clip(total_wait_samples /
        step_required_samples, MIN, MAX)
        (rollout is bottleneck, deactivate later; the larger the shortfall
        fraction, the bigger the step, bounded to
        [DEACTIVATE_RATIO_INCREASE_MIN, DEACTIVATE_RATIO_INCREASE_MAX])
      - no real wait occurred → ratio -= DEACTIVATE_RATIO_DECREASE_STEP
        (training is bottleneck, deactivate earlier)

    Activation is additionally gated by a cost/benefit check (skipped when
    only_hybrid=True): re-activating hybrid replicas costs one activate +
    one (future) deactivate cycle — ``switch_cost`` seconds, the rolling
    average of the last SWITCH_OVERHEAD_WINDOW_SIZE measured activate+
    deactivate cycle durations. This is only worth paying if the
    standalone-only generation shortfall would otherwise cost the trainer
    more wall-clock time than the switch itself. If the estimated benefit
    does not exceed ``switch_cost``, activation is skipped.

    Args:
        deactivate_ratio: Initial ratio in (0, 1].
        only_hybrid: No standalone replicas; forces ratio=1.0, disables adaptation.
    """

    def __init__(self, deactivate_ratio: float = 0.3, only_hybrid: bool = False):
        self.deactivate_ratio = deactivate_ratio
        self.only_hybrid = only_hybrid
        self._recent_switch_overheads: list[float] = []
        # Rolling estimate of standalone-only wall-clock time to produce one sample.
        # Starts pessimistic (see INITIAL_PER_SAMPLE_TIME_S) and is only updated when
        # a cycle actually observed real waiting for newly-generated samples.
        self._last_per_sample_time: float = INITIAL_PER_SAMPLE_TIME_S
        if only_hybrid:
            print("[DefaultDynamicSchedulePolicy] only_hybrid=True: forcing deactivate_ratio=1.0")
            self.deactivate_ratio = 1.0

    def should_deactivate(self, global_steps: int, is_hybrid_active: bool, ctx: DynamicScheduleContext) -> bool:
        return is_hybrid_active

    def deactivate_wait_samples(self, ctx: DynamicScheduleContext) -> int:
        return int(ctx.step_required_samples * self.deactivate_ratio)

    def _switch_cost(self) -> float:
        if self._recent_switch_overheads:
            recent = self._recent_switch_overheads[-SWITCH_OVERHEAD_WINDOW_SIZE:]
            return sum(recent) / len(recent)
        return DEFAULT_SWITCH_THRESHOLD_S

    def update_after_step(self, global_steps: int, ctx: DynamicScheduleContext) -> None:
        if global_steps <= 0 or self.only_hybrid:
            return

        switch_overhead = ctx.last_activate_duration_s + ctx.last_deactivate_duration_s
        if switch_overhead > 0:
            self._recent_switch_overheads.append(switch_overhead)

        total_wait = sum(ctx.step_wait_times)
        total_wait_samples = sum(ctx.step_wait_samples)

        # Use step_wait_samples (real waits for newly-generated samples), not
        # total_wait, as the bottleneck signal: total_wait can look small purely
        # because the queue had backlog, even while standalone generation is
        # actually struggling to keep up.
        if total_wait_samples > 0:
            # Scale the increase by how much of this cycle's samples actually had
            # to be waited on, so a near-total shortfall pushes the ratio up faster
            # than a marginal one. Clipped to keep single-step adaptation bounded.
            cycle_samples = ctx.step_required_samples
            raw_increase = total_wait_samples / cycle_samples if cycle_samples > 0 else DEACTIVATE_RATIO_INCREASE_MAX
            ratio_delta = min(DEACTIVATE_RATIO_INCREASE_MAX, max(DEACTIVATE_RATIO_INCREASE_MIN, raw_increase))
            self.deactivate_ratio = min(1.0, self.deactivate_ratio + ratio_delta)
        else:
            ratio_delta = -DEACTIVATE_RATIO_DECREASE_STEP
            self.deactivate_ratio += ratio_delta

        print(
            f"[DefaultDynamicSchedulePolicy] step={global_steps} "
            f"deactivate_ratio={self.deactivate_ratio:.3f} (delta={ratio_delta:+.3f}) "
            f"(wait={total_wait:.1f}s, wait_samples={total_wait_samples}, "
            f"switch_samples={len(self._recent_switch_overheads)}/{SWITCH_OVERHEAD_WINDOW_SIZE}, "
            f"generated={ctx.total_generated_samples}, expected={ctx.expected_samples}, "
            f"activate_dur={ctx.last_activate_duration_s:.2f}s, deactivate_dur={ctx.last_deactivate_duration_s:.2f}s)"
        )
        print(
            f"[DefaultDynamicSchedulePolicy] step={global_steps} "
            f"switch_overheads_history(tail10)={self._recent_switch_overheads[-10:]}"
        )

    def _deactivate_gap(self, ctx: DynamicScheduleContext) -> float:
        """How many more samples are needed to reach the deactivate-wait threshold.

        Shared by :meth:`should_activate_after_step` and
        :meth:`_has_positive_net_benefit` so both use the same reference point.
        """
        deactivate_threshold = self.deactivate_ratio * ctx.step_required_samples
        surplus = ctx.total_generated_samples - ctx.expected_samples
        return deactivate_threshold - surplus

    def _estimate_per_sample_time(self, ctx: DynamicScheduleContext) -> float:
        """Estimate standalone-only wall-clock time to produce one sample.

        Divides this cycle's total wait time by ``step_wait_samples`` — the
        count of samples that actually had to be waited on for generation,
        excluding samples already sitting in the queue at collection start
        (those are served instantly and would otherwise skew the estimate
        toward 0 when there's a queue backlog).

        If this cycle collected 0 real-wait samples (every collection was
        served entirely from backlog), there's no fresh signal: keep the
        last known estimate rather than treating it as "instant" (0.0) or
        recomputing garbage. Before any cycle has ever produced a signal,
        falls back to a deliberately pessimistic INITIAL_PER_SAMPLE_TIME_S.
        """
        total_wait = sum(ctx.step_wait_times)
        total_wait_samples = sum(ctx.step_wait_samples)
        if total_wait > 0 and total_wait_samples > 0:
            self._last_per_sample_time = total_wait / total_wait_samples
        return self._last_per_sample_time

    def _has_positive_net_benefit(self, global_steps: int, ctx: DynamicScheduleContext) -> bool:
        """Cost/benefit gate: is activating hybrid worth its switch overhead?

        Converts the outstanding sample gap into an estimated wall-clock cost
        and compares it against ``switch_cost`` (the rolling-average activate
        + deactivate overhead). Activation is only justified when the
        estimated benefit exceeds the cost.

        Before any real generation-rate signal has ever been observed,
        ``per_sample_time`` defaults to a large pessimistic constant (see
        ``_estimate_per_sample_time``), which naturally biases this check
        toward activation until real timing data narrows the estimate.
        """
        switch_cost = self._switch_cost()
        per_sample_time = self._estimate_per_sample_time(ctx)

        gap = max(0.0, self._deactivate_gap(ctx))
        expected_benefit_s = gap * per_sample_time
        net_benefit = expected_benefit_s - switch_cost

        print(
            f"[DefaultDynamicSchedulePolicy] step={global_steps} cost/benefit check: "
            f"gap={gap}, per_sample_time={per_sample_time:.2f}s, "
            f"expected_benefit={expected_benefit_s:.2f}s, switch_cost={switch_cost:.2f}s, "
            f"net_benefit={net_benefit:.2f}s -> {'activate' if net_benefit > 0 else 'skip'}"
        )
        return net_benefit > 0

    def should_activate_after_step(
        self, global_steps: int, is_hybrid_active: bool, ctx: DynamicScheduleContext
    ) -> bool:
        if self.only_hybrid:
            return True

        print(
            f"[should_activate_after_step] ctx.total_generated_samples:{ctx.total_generated_samples},"
            f" ctx.expected_samples:{ctx.expected_samples},"
            f" self.deactivate_ratio:{self.deactivate_ratio},"
            f" ctx.buffer_samples:{ctx.buffer_samples}"
        )
        print(f"DynamicScheduleContext:{ctx}")

        has_gap = self._deactivate_gap(ctx) > 0
        if not has_gap:
            return False

        return self._has_positive_net_benefit(global_steps, ctx)

    def request_rebalance(self, global_steps: int, ctx: DynamicScheduleContext) -> None:
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
            print("[DefaultDynamicSchedulePolicy] request_rebalance skipped: no rollouter reference available")
            return

        try:
            result = ray.get(self._rollouter.rebalance_requests.remote())
            print(
                f"[DefaultDynamicSchedulePolicy] request_rebalance done at step {global_steps}: "
                f"cleared {result.get('cleared_entries', 0)} sticky entries, "
                f"server loads: {result.get('server_loads', {})}"
            )
        except Exception as e:
            print(f"[DefaultDynamicSchedulePolicy] request_rebalance failed at step {global_steps}: {e}")
