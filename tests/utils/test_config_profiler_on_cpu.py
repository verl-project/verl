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

import logging
import warnings

import pytest
from omegaconf import OmegaConf

from verl.utils.config import _reference_shares_actor_group, _validate_profiler_config

_ACTOR = "actor_rollout_ref.actor.profiler.enable"
_REF = "actor_rollout_ref.ref.profiler.enable"
_ROLLOUT = "actor_rollout_ref.rollout.profiler.enable"
_CRITIC = "critic.profiler.enable"
_ALL_ROLES = (_ACTOR, _REF, _ROLLOUT, _CRITIC)


def _config(*, tool="npu", steps=(1,), enabled=(), role_tools=None, role_discrete=None, actor_use_kl_loss=False):
    """Build a config where every role is off unless its key is listed in ``enabled``.

    Each role's `tool` interpolates `global_profiler.tool` as its default, exactly like
    `verl/trainer/config/{actor,ref,rollout,critic}/*.yaml`
    (`tool: ${oc.select:global_profiler.tool,null}`), so a role only overrides it by naming one.
    ``role_discrete`` sets each role's ``tool_config.<tool>.discrete``; the shipped default is False.
    """
    state = dict.fromkeys(_ALL_ROLES, False)
    state.update(dict.fromkeys(enabled, True))
    role_tools = role_tools or {}
    role_discrete = role_discrete or {}

    def block(key):
        profiler = {
            "enable": state[key],
            "tool_config": {name: {"discrete": role_discrete.get(key, False)} for name in ("npu", "torch")},
        }
        if key in role_tools:
            profiler["tool"] = role_tools[key]
        else:
            profiler["tool"] = "${oc.select:global_profiler.tool,null}"
        return profiler

    return OmegaConf.create(
        {
            "global_profiler": {"tool": tool, "steps": None if steps is None else list(steps)},
            "actor_rollout_ref": {
                "actor": {
                    "profiler": block(_ACTOR),
                    "use_kl_loss": actor_use_kl_loss,
                },
                "ref": {"profiler": block(_REF)},
                "rollout": {"profiler": block(_ROLLOUT)},
            },
            "critic": {"profiler": block(_CRITIC)},
        }
    )


def _run(config, *, use_reference_policy=False, use_critic=False, ref_aliased=True):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _validate_profiler_config(
            config,
            use_reference_policy=use_reference_policy,
            use_critic=use_critic,
            ref_aliased=ref_aliased,
        )
    return [str(item.message) for item in caught]


def test_steps_without_any_enabled_role_names_the_missing_half():
    # The issue's first repro: global_profiler alone. The trainer calls start/stop_profile on every
    # profiled step, every worker drops the call, and the run finishes with an empty save_path.
    messages = _run(_config())

    assert len(messages) == 1
    assert "global_profiler.steps=[1] schedules profiling but no active role collects it" in messages[0]
    assert f"{_ACTOR}, {_REF}, {_ROLLOUT}, {_CRITIC} are all False" in messages[0]
    assert "actor_rollout_ref.actor.profiler.enable=True" in messages[0]


@pytest.mark.parametrize("steps", [None, ()], ids=["null", "empty-list"])
def test_enabled_role_without_a_trigger_stays_quiet(steps):
    # The documented code-instrumentation setup: the user drives the profiler from their own script
    # and keeps global_profiler.steps null so verl does not open a second, conflicting profiler.
    # examples/ascend_extras/grpo_trainer/run_qwen3_30b_a3b_megatron.sh relies on this.
    assert _run(_config(steps=steps, enabled=(_ACTOR, _ROLLOUT))) == []


def test_null_tool_with_a_trigger_is_reported():
    # DistProfiler silently falls back to a no-op impl for an unrecognised tool, so a null tool
    # collects nothing even though the trigger and the enable flag both line up.
    messages = _run(_config(tool=None, enabled=(_ACTOR,)))

    assert len(messages) == 1
    assert "effective profiler tool is null" in messages[0]


def test_role_level_tool_satisfies_the_tool_check():
    # A role that names its own tool profiles fine even when the global default is null, so warning
    # about the null default would be a false positive.
    assert _run(_config(tool=None, enabled=(_ACTOR,), role_tools={_ACTOR: "npu"})) == []


def test_explicit_null_role_tool_does_not_inherit_the_global_tool():
    # The role block interpolates the global tool as its *default*, but an explicit null overrides
    # it: the worker passes that null to DistProfiler and gets the no-op backend. The global `npu`
    # must not mask that. The rollout is turned off here because it inherits the actor's `enable`
    # and its own tool defaults to the global one -- with it on, the job really does collect.
    messages = _run(
        _config(tool="npu", enabled=(_ACTOR,), role_tools={_ACTOR: None, _ROLLOUT: "npu"}),
    )

    assert len(messages) == 1
    assert "effective profiler tool is null" in messages[0]


def test_colocated_reference_alone_is_not_a_trigger():
    # In the hybrid engine the reference shares the actor/rollout worker group, and the trainer only
    # drives ref_policy_wg when it is a distinct group (_start_profiling). A colocated
    # `ref.profiler.enable` therefore cannot catch the scheduled steps; the actor block is what runs.
    messages = _run(_config(enabled=(_REF,)), use_reference_policy=True)

    assert len(messages) == 1
    assert "no active role collects it" in messages[0]


def test_standalone_reference_alone_is_a_trigger():
    # v1 separate_async without hybrid replicas gives the reference its own worker group, so its
    # enabled profiler block is a real collector and must not trigger the colocated warning.
    messages = _run(_config(enabled=(_REF,)), use_reference_policy=True, ref_aliased=False)

    assert messages == []


def _aliasing_config(*, trainer_mode="sync", enable_switch=False, hybrid_engine=True, lora_rank=0):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {"lora": {"rank": lora_rank}},
                "hybrid_engine": hybrid_engine,
            },
            "trainer": {
                "v1": {
                    "trainer_mode": trainer_mode,
                    "separate_async": {"hybrid_rollout": {"enable_switch": enable_switch}},
                }
            },
        }
    )


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, True),
        ({"trainer_mode": "separate_async", "enable_switch": False}, False),
        ({"trainer_mode": "separate_async", "enable_switch": True}, True),
        ({"trainer_mode": "separate_async", "enable_switch": True, "hybrid_engine": False}, False),
        ({"trainer_mode": "separate_async", "enable_switch": False, "lora_rank": 8}, True),
    ],
)
def test_reference_group_aliasing_follows_the_trainer_config(kwargs, expected):
    assert _reference_shares_actor_group(_aliasing_config(**kwargs)) is expected


def test_profiler_enable_on_an_inactive_critic_is_not_a_trigger():
    # critic.enable=False means no critic worker exists. Its profiler.enable flag then gates nothing,
    # so treating it as "someone collects" would hide exactly the no-op this check exists to report.
    config = _config(enabled=(_CRITIC,))
    config.critic.enable = False

    messages = _run(config, use_critic=False)

    assert len(messages) == 1
    assert "no active role collects it" in messages[0]


def test_inactive_role_is_named_in_the_driver_log(caplog):
    config = _config(enabled=(_CRITIC,))
    config.critic.enable = False

    with caplog.at_level(logging.WARNING, logger="verl.utils.config"):
        messages = _run(config, use_critic=False)

    assert len(messages) == 1
    assert f"{_CRITIC} is set, but that role does not run in this job" in caplog.text


def test_active_critic_counts_as_a_trigger():
    # The mirror of the previous test: when the critic does run, its profiler block is a real one.
    config = _config(enabled=(_CRITIC,))
    config.critic.enable = True

    assert _run(config, use_critic=True) == []


@pytest.mark.parametrize("role", _ALL_ROLES)
def test_any_single_active_role_satisfies_the_trigger(role):
    # Each role block is read by a different worker; enabling one is enough for profiling to happen.
    # The reference is the exception: colocated, it leaves the scheduled work to the actor block.
    if role == _REF:
        return
    if role == _CRITIC:
        assert _run(_config(enabled=(role,)), use_critic=True) == []
    elif role == _ROLLOUT:
        # The rollout server forwards scheduled start/stop only in discrete mode.
        assert _run(_config(enabled=(role,), role_discrete={role: True})) == []
    else:
        assert _run(_config(enabled=(role,))) == []


def test_rollout_outside_discrete_mode_is_not_a_trigger(caplog):
    # Both vLLM and SGLang gate start_profile on is_discrete_mode(); continuous mode never starts a
    # profiler for the trainer's scheduled steps, so the rollout must not count as a collector.
    with caplog.at_level(logging.WARNING, logger="verl.utils.config"):
        messages = _run(_config(enabled=(_ROLLOUT,)))

    assert len(messages) == 1
    assert "no active role collects it" in messages[0]
    assert (
        "rollout profiler only starts for the trainer's scheduled start_profile calls in discrete mode" in caplog.text
    )


def test_rollout_in_discrete_mode_counts_as_a_trigger():
    assert _run(_config(enabled=(_ROLLOUT,), role_discrete={_ROLLOUT: True})) == []


def test_null_tool_warns_even_when_another_role_can_collect():
    # The actor explicitly nulls its tool while the critic inherits npu. A working critic must not
    # hide the actor's no-op backend.
    config = _config(enabled=(_ACTOR, _CRITIC), role_tools={_ACTOR: None})
    config.critic.enable = True

    messages = _run(config, use_critic=True)

    assert len(messages) == 1
    assert f"effective profiler tool is null for: {_ACTOR}" in messages[0]


def test_fully_configured_profiler_is_silent():
    assert _run(_config(enabled=(_ACTOR,))) == []


def test_disabled_profiler_is_silent():
    # Default run: no steps, nothing enabled. This is the common path and must stay quiet.
    assert _run(_config(tool=None, steps=None)) == []


def test_schema_without_profiler_blocks_is_read_as_disabled():
    # A schema from an older release may have no role profiler node at all; the check must treat
    # that as "not enabled" rather than raising.
    messages = _run(OmegaConf.create({"global_profiler": {"tool": "npu", "steps": [1]}}))

    assert len(messages) == 1
    assert "no active role collects it" in messages[0]


def test_unresolvable_interpolation_is_read_as_disabled():
    # Generated role configs interpolate from global_profiler; if that base is missing, resolving the
    # key raises and the check must degrade to "not enabled" instead of crashing the driver.
    config = OmegaConf.create(
        {
            "global_profiler": {"tool": "npu", "steps": [1]},
            "actor_rollout_ref": {"actor": {"profiler": {"enable": "${global_profiler.missing_key}"}}},
        }
    )

    assert len(_run(config)) == 1
