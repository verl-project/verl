# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Config-level tests for the freeze_vision_tower plumbing.

The flag was a silent no-op after the 0.8 engine refactor: the actor-level key
existed in configs and docs but nothing consumed it. These tests lock the
restored chain at the config layer: the actor-level key resolves, the ref
section mirrors it (pre-0.8 froze ref together with actor), the critic carries
its own key, and engines that do not implement the field ignore it with a
warning instead of failing.
"""

import dataclasses
import logging
import pathlib

from hydra import compose, initialize_config_dir

from verl.workers.config.engine import FSDPEngineConfig, McoreEngineConfig
from verl.workers.engine_workers import plumb_freeze_vision_tower

_TRAINER_CONFIG_DIR = str(pathlib.Path(__file__).resolve().parents[2] / "verl" / "trainer" / "config")


def _compose_ppo_trainer(overrides):
    with initialize_config_dir(config_dir=_TRAINER_CONFIG_DIR, version_base=None):
        return compose(config_name="ppo_trainer", overrides=overrides)


def test_actor_flag_resolves_through_the_trainer_config():
    cfg = _compose_ppo_trainer(["actor_rollout_ref.actor.freeze_vision_tower=true"])

    assert cfg.actor_rollout_ref.actor.freeze_vision_tower is True

    # ref mirrors the actor flag (pre-0.8 behavior: ref froze together with actor)
    assert cfg.actor_rollout_ref.ref.freeze_vision_tower is True

    # the FSDP engine section (fsdp_config) declares the field; the value is
    # forwarded by the worker config __post_init__s, not set in the yaml
    assert cfg.actor_rollout_ref.actor.fsdp_config.freeze_vision_tower is False

    # default is off everywhere
    cfg_off = _compose_ppo_trainer([])
    assert cfg_off.actor_rollout_ref.actor.freeze_vision_tower is False
    assert cfg_off.actor_rollout_ref.ref.freeze_vision_tower is False


def test_critic_flag_resolves_through_the_trainer_config():
    cfg = _compose_ppo_trainer(["critic.freeze_vision_tower=true"])
    assert cfg.critic.freeze_vision_tower is True


def test_plumb_sets_supported_engine_and_warns_on_unsupported(caplog):
    fsdp = FSDPEngineConfig()
    plumb_freeze_vision_tower(fsdp, True, role="actor")
    assert fsdp.freeze_vision_tower is True

    # Forward, never clear: a worker-level False must not silently turn off an
    # engine-level freeze_vision_tower=true set directly in the yaml.
    plumb_freeze_vision_tower(fsdp, False, role="actor")
    assert fsdp.freeze_vision_tower is True

    mcore = McoreEngineConfig()
    with caplog.at_level(logging.WARNING):
        plumb_freeze_vision_tower(mcore, True, role="actor")
    assert any("only implemented for the FSDP engine" in r.message for r in caplog.records)
    # no inert attribute on the unsupported engine
    assert "freeze_vision_tower" not in {f.name for f in dataclasses.fields(mcore)}
    assert not hasattr(mcore, "freeze_vision_tower")


def test_worker_config_post_init_forwards_the_flag(caplog):
    """The choke point: every engine-bearing worker config forwards the
    worker-level flag in __post_init__ (next to the strategy sync), so all
    construction paths — current and future — pick it up. Non-FSDP engines
    warn when the flag is set instead of silently ignoring it."""
    from verl.workers.config.actor import FSDPActorConfig, McoreActorConfig
    from verl.workers.config.critic import FSDPCriticConfig

    actor = FSDPActorConfig(freeze_vision_tower=True, rollout_n=1, ppo_micro_batch_size=8)
    assert actor.engine.freeze_vision_tower is True

    critic = FSDPCriticConfig(freeze_vision_tower=True, ppo_micro_batch_size=1)
    assert critic.engine.freeze_vision_tower is True

    # Forward, never clear: a worker-level False must not turn off an
    # engine-level true set directly in the yaml (the SFT-style knob).
    engine = FSDPEngineConfig(freeze_vision_tower=True)
    actor = FSDPActorConfig(freeze_vision_tower=False, rollout_n=1, ppo_micro_batch_size=8, fsdp_config=engine)
    assert actor.engine.freeze_vision_tower is True

    # Non-FSDP engines: the flag warns and does not grow a field.
    with caplog.at_level(logging.WARNING, logger="verl.workers.config.engine"):
        mcore = McoreActorConfig(freeze_vision_tower=True, rollout_n=1, ppo_micro_batch_size=8)
    assert any("only implemented for the FSDP engine" in r.message for r in caplog.records)
    assert not hasattr(mcore.engine, "freeze_vision_tower")
