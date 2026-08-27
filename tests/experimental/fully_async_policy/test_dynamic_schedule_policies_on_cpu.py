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

"""CPU-only unit tests for fully-async dynamic scheduling policies."""

import pytest

from verl.experimental.fully_async_policy.dynamic_schedule import (
    DefaultDynamicSchedulePolicy,
    DynamicScheduleContext,
    FixedRatioDynamicSchedulePolicy,
    StaticFullyAsyncPolicy,
    build_policy,
)


def _context(**overrides) -> DynamicScheduleContext:
    values = {
        "required_samples": 10,
        "trigger_parameter_sync_step": 4,
        "total_generated_samples": 100,
        "expected_samples": 100,
        "buffer_samples": 10,
    }
    values.update(overrides)
    return DynamicScheduleContext(**values)


def test_context_derives_samples_per_parameter_sync_step():
    assert _context().step_required_samples == 40


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        ("default", DefaultDynamicSchedulePolicy),
        ("static_fully_async", StaticFullyAsyncPolicy),
        ("fixed_ratio", FixedRatioDynamicSchedulePolicy),
    ],
)
def test_all_documented_builtin_policies_are_registered(name, expected_type):
    assert isinstance(build_policy(name), expected_type)


def test_build_policy_rejects_unknown_policy():
    with pytest.raises(KeyError, match="Unknown dynamic scheduling policy"):
        build_policy("missing")


def test_default_policy_deactivation_tracks_active_state_and_ratio():
    policy = DefaultDynamicSchedulePolicy(deactivate_ratio=0.5)
    ctx = _context()

    assert policy.should_deactivate(global_steps=1, is_hybrid_active=True, ctx=ctx)
    assert not policy.should_deactivate(global_steps=1, is_hybrid_active=False, ctx=ctx)
    assert policy.deactivate_wait_samples(ctx) == 20


def test_default_policy_adapts_ratio_from_real_waited_samples():
    policy = DefaultDynamicSchedulePolicy(deactivate_ratio=0.5)

    policy.update_after_step(
        global_steps=1,
        ctx=_context(step_wait_times=[2.0], step_wait_samples=[2]),
    )
    assert policy.deactivate_ratio == pytest.approx(0.55)

    policy.update_after_step(
        global_steps=2,
        ctx=_context(step_wait_times=[0.0], step_wait_samples=[0]),
    )
    assert policy.deactivate_ratio == pytest.approx(0.53)


def test_default_policy_activation_requires_benefit_above_switch_cost():
    policy = DefaultDynamicSchedulePolicy(deactivate_ratio=0.5)
    low_benefit = _context(
        total_generated_samples=110,
        step_wait_times=[1.0],
        step_wait_samples=[10],
    )
    high_benefit = _context(
        total_generated_samples=110,
        step_wait_times=[20.0],
        step_wait_samples=[10],
    )

    assert not policy.should_activate_after_step(global_steps=1, is_hybrid_active=False, ctx=low_benefit)
    assert policy.should_activate_after_step(global_steps=2, is_hybrid_active=False, ctx=high_benefit)


def test_only_hybrid_default_policy_forces_full_ratio_and_activation():
    policy = DefaultDynamicSchedulePolicy(deactivate_ratio=0.2, only_hybrid=True)

    assert policy.deactivate_ratio == 1.0
    assert policy.deactivate_wait_samples(_context()) == 40
    assert policy.should_activate_after_step(global_steps=1, is_hybrid_active=False, ctx=_context())


def test_static_policy_matches_baseline_fully_async_behavior():
    policy = StaticFullyAsyncPolicy()
    ctx = _context()

    assert policy.should_deactivate(global_steps=1, is_hybrid_active=True, ctx=ctx)
    assert policy.deactivate_wait_samples(ctx) == 0
    assert not policy.should_activate_after_step(global_steps=1, is_hybrid_active=False, ctx=ctx)


def test_fixed_ratio_policy_never_adapts_and_uses_sample_gap():
    policy = FixedRatioDynamicSchedulePolicy(deactivate_ratio=0.5)
    ctx = _context(step_wait_times=[10.0], step_wait_samples=[40])

    policy.update_after_step(global_steps=1, ctx=ctx)

    assert policy.deactivate_ratio == 0.5
    assert policy.deactivate_wait_samples(ctx) == 20
    assert policy.should_activate_after_step(
        global_steps=1,
        is_hybrid_active=False,
        ctx=_context(total_generated_samples=119),
    )
    assert not policy.should_activate_after_step(
        global_steps=1,
        is_hybrid_active=False,
        ctx=_context(total_generated_samples=120),
    )
