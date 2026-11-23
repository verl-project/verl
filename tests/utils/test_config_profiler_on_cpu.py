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

"""Profiler configuration warnings: the two halves of the profiler switch must agree.

``global_profiler.steps`` arms the trainer's start/stop calls, and each role's own
``profiler.enable`` decides whether a worker acts on them. Arming the first without the second makes
the whole run a silent no-op, which is what these tests pin down.
"""

import warnings

import pytest
from omegaconf import OmegaConf

from verl.utils.config import _validate_profiler_config

_ACTOR = "actor_rollout_ref.actor.profiler.enable"
_REF = "actor_rollout_ref.ref.profiler.enable"
_ROLLOUT = "actor_rollout_ref.rollout.profiler.enable"
_CRITIC = "critic.profiler.enable"
_ALL_ROLES = (_ACTOR, _REF, _ROLLOUT, _CRITIC)


def _config(*, tool="npu", steps=(1,), enabled=()):
    """Build a config where every role is off unless its key is listed in ``enabled``."""
    state = dict.fromkeys(_ALL_ROLES, False)
    state.update(dict.fromkeys(enabled, True))
    return OmegaConf.create(
        {
            "global_profiler": {"tool": tool, "steps": None if steps is None else list(steps)},
            "actor_rollout_ref": {
                "actor": {"profiler": {"enable": state[_ACTOR]}},
                "ref": {"profiler": {"enable": state[_REF]}},
                "rollout": {"profiler": {"enable": state[_ROLLOUT]}},
            },
            "critic": {"profiler": {"enable": state[_CRITIC]}},
        }
    )


def _warnings(config):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _validate_profiler_config(config)
    return [str(item.message) for item in caught]


def test_steps_without_any_enabled_role_names_the_missing_half():
    # The issue's first repro: global_profiler alone. The trainer calls start/stop_profile on every
    # profiled step, every worker drops the call, and the run finishes with an empty save_path.
    messages = _warnings(_config())

    assert len(messages) == 1
    assert "global_profiler.steps=[1] schedules profiling but no role enables it" in messages[0]
    assert f"{_ACTOR}, {_REF}, {_ROLLOUT}, {_CRITIC} are all False" in messages[0]
    assert "actor_rollout_ref.actor.profiler.enable=True" in messages[0]


@pytest.mark.parametrize("steps", [None, ()], ids=["null", "empty-list"])
def test_enabled_role_without_a_trigger_stays_quiet(steps):
    # The documented code-instrumentation setup: the user drives the profiler from their own script
    # and keeps global_profiler.steps null so verl does not open a second, conflicting profiler.
    # examples/ascend_extras/grpo_trainer/run_qwen3_30b_a3b_megatron.sh relies on this.
    assert _warnings(_config(steps=steps, enabled=(_ACTOR, _ROLLOUT))) == []


def test_null_tool_with_a_trigger_is_reported():
    # DistProfiler silently falls back to a no-op impl for an unrecognised tool, so a null tool
    # collects nothing even though the trigger and the enable flag both line up.
    messages = _warnings(_config(tool=None, enabled=(_ACTOR,)))

    assert len(messages) == 1
    assert "global_profiler.tool is null" in messages[0]


@pytest.mark.parametrize("role", _ALL_ROLES)
def test_any_single_role_satisfies_the_trigger(role):
    # Each role block is read by a different worker; enabling one is enough for profiling to happen.
    assert _warnings(_config(enabled=(role,))) == []


def test_fully_configured_profiler_is_silent():
    assert _warnings(_config(enabled=(_ACTOR,))) == []


def test_disabled_profiler_is_silent():
    # Default run: no steps, nothing enabled. This is the common path and must stay quiet.
    assert _warnings(_config(tool=None, steps=None)) == []


def test_schema_without_profiler_blocks_is_read_as_disabled():
    # A schema from an older release may have no role profiler node at all; the check must treat
    # that as "not enabled" rather than raising.
    messages = _warnings(OmegaConf.create({"global_profiler": {"tool": "npu", "steps": [1]}}))

    assert len(messages) == 1
    assert "no role enables it" in messages[0]


def test_unresolvable_interpolation_is_read_as_disabled():
    # Generated role configs interpolate from global_profiler; if that base is missing, resolving the
    # key raises and the check must degrade to "not enabled" instead of crashing the driver.
    config = OmegaConf.create(
        {
            "global_profiler": {"tool": "npu", "steps": [1]},
            "actor_rollout_ref": {"actor": {"profiler": {"enable": "${global_profiler.missing_key}"}}},
        }
    )

    assert len(_warnings(config)) == 1
