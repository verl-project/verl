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

"""Policy base class and registry.

Custom policy example::

    from verl.experimental.fully_async_policy.dynamic_schedule import (
        DynamicSchedulePolicyBase, DynamicScheduleContext, register_policy,
    )

    @register_policy("my_policy")
    class MyPolicy(DynamicSchedulePolicyBase):
        def should_deactivate(self, global_steps, is_hybrid_active, ctx): ...
        def deactivate_wait_samples(self, ctx): ...
        def should_activate_after_step(self, global_steps, is_hybrid_active, ctx): ...
        def request_rebalance(self, global_steps, ctx): ...

Then set ``async_training.dynamic_schedule_policy: "my_policy"`` in the training config.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

_policy_registry: dict[str, type["DynamicSchedulePolicyBase"]] = {}


def register_policy(name: str):
    """Decorator: register a DynamicSchedulePolicyBase subclass under *name*."""

    def decorator(cls: type["DynamicSchedulePolicyBase"]) -> type["DynamicSchedulePolicyBase"]:
        _policy_registry[name] = cls
        return cls

    return decorator


def build_policy(name: str, **kwargs) -> "DynamicSchedulePolicyBase":
    """Instantiate a registered policy by name, forwarding **kwargs to its __init__."""
    if name not in _policy_registry:
        raise KeyError(f"Unknown dynamic scheduling policy '{name}'. Registered: {list(_policy_registry.keys())}")
    return _policy_registry[name](**kwargs)


@dataclass
class DynamicScheduleContext:
    """Unified context for all dynamic scheduling policy decisions and state updates.

    Combines static training-run configuration (constant across a run) with
    per-step runtime state and activation/deactivation timing metrics.

    Built by :class:`FullyAsyncTrainer` before each policy call and passed to
    all :class:`DynamicSchedulePolicyBase` methods as the single context entry point.

    Attributes:
        required_samples: Minimum samples per collection (ppo_mini_batch_size × require_batches).
        trigger_parameter_sync_step: How many collections before a weight-sync step.
        step_required_samples: Derived, cached ``required_samples × trigger_parameter_sync_step``
            — the total sample count expected in one weight-sync cycle. Computed once in
            :meth:`__post_init__` since both operands are constant across a training run.
        total_generated_samples: Cumulative rollout samples since training began.
        expected_samples: Theoretical samples needed up to the current sync step.
        buffer_samples: Allowed buffer headroom (expected × staleness_threshold).
        step_wait_times: Per-collection wait times within the latest step (seconds).
        step_wait_samples: Per-collection count of samples that actually had to be
            waited on (i.e. not already sitting in the queue) — ``max(0,
            required_samples - queue_size_at_collection_start)``. Parallel to
            ``step_wait_times``; used together to compute a generation-rate signal
            that isn't skewed by samples served instantly from queue backlog.
        only_hybrid: True when standalone rollout replicas are zero.
        last_activate_duration_s: Duration of the last activate cycle (weight sync + onload), in seconds.
        last_deactivate_duration_s: Duration of the last deactivate cycle (offload), in seconds.
    """

    # Static config (constant across a training run)
    required_samples: int
    trigger_parameter_sync_step: int

    # Per-step runtime state
    total_generated_samples: int
    expected_samples: int
    buffer_samples: int
    step_wait_times: list[float] = field(default_factory=list)
    step_wait_samples: list[int] = field(default_factory=list)
    only_hybrid: bool = False

    # Activation / deactivation timing (seconds)
    last_activate_duration_s: float = 0.0
    last_deactivate_duration_s: float = 0.0

    # Derived field, computed once in __post_init__ (not passed to the constructor).
    step_required_samples: int = field(init=False)

    def __post_init__(self) -> None:
        self.step_required_samples = self.required_samples * self.trigger_parameter_sync_step


class DynamicSchedulePolicyBase(ABC):
    """Abstract base class for dynamic resource scheduling policies.

    Subclasses must implement :meth:`should_deactivate` and
    :meth:`deactivate_wait_samples` and :meth:`should_activate_after_step`,
    and may override :meth:`update_after_step` to adapt internal parameters.

    Call order per training step:
      1. ``should_deactivate()``      – before training; returns whether to deactivate.
      2. ``deactivate_wait_samples()`` – if (1) is True; returns sample threshold.
      3. ``should_activate_after_step()`` – after weight sync; whether to (re-)activate.
      4. ``request_rebalance()``             – after activation; redistribute requests across replicas.
      5. ``update_after_step()``      – after weight sync; update internal state.
    """

    @abstractmethod
    def should_deactivate(self, global_steps: int, is_hybrid_active: bool, ctx: DynamicScheduleContext) -> bool:
        """Return True to trigger deactivation this step, False to keep hybrid active.

        Args:
            global_steps: Current global step (parameter version).
            is_hybrid_active: Whether hybrid replicas are currently active.
            ctx: Unified policy context (static config + per-step state + timing).
        """

    @abstractmethod
    def deactivate_wait_samples(self, ctx: DynamicScheduleContext) -> int:
        """Return the minimum buffered-sample count before deactivation proceeds.

        Args:
            ctx: Unified policy context (static config + per-step state + timing).
        """

    def update_after_step(self, global_steps: int, ctx: DynamicScheduleContext) -> None:  # noqa: B027
        """Update policy state after weight sync. Override to adapt parameters.

        Args:
            global_steps: Current global step (parameter version).
            ctx: Unified policy context (static config + per-step state + timing).
        """

    @abstractmethod
    def should_activate_after_step(
        self, global_steps: int, is_hybrid_active: bool, ctx: DynamicScheduleContext
    ) -> bool:
        """Return True to trigger activation after this step, False to keep hybrid deactivated.

        Args:
            global_steps: Current global step (parameter version).
            is_hybrid_active: Whether hybrid replicas are currently active.
            ctx: Unified policy context (static config + per-step state + timing).
        """

    def request_rebalance(self, global_steps: int, ctx: DynamicScheduleContext) -> None:  # noqa: B027
        """Redistribute requests across inference replicas after activation.

        Called immediately after hybrid replicas are activated and registered
        in the load balancer.  Override to implement custom load distribution
        logic (e.g. weighted routing, request migration).

        Default implementation is a no-op.

        Args:
            global_steps: Current global step (parameter version).
            ctx: Unified policy context (static config + per-step state + timing).
        """
