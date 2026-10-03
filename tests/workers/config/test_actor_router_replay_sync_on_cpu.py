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

from dataclasses import fields

from verl.workers.config.actor import ActorConfig, McoreActorConfig, VeOmniActorConfig
from verl.workers.config.engine import EngineRouterReplayConfig, McoreEngineConfig, VeOmniEngineConfig
from verl.workers.config.optimizer import OptimizerConfig


def test_actor_config_has_no_top_level_router_replay():
    assert "router_replay" not in {f.name for f in fields(ActorConfig)}


def test_mcore_router_replay_lives_on_engine():
    cfg = McoreActorConfig(
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        megatron=McoreEngineConfig(router_replay=EngineRouterReplayConfig(mode="R3")),
        optim=OptimizerConfig(lr=1e-6),
    )
    assert not hasattr(cfg, "router_replay")
    assert cfg.megatron.router_replay.mode == "R3"


def test_veomni_router_replay_lives_on_engine():
    cfg = VeOmniActorConfig(
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        use_remove_padding=True,
        veomni=VeOmniEngineConfig(router_replay=EngineRouterReplayConfig(mode="R2")),
        optim=OptimizerConfig(lr=1e-6),
    )
    assert not hasattr(cfg, "router_replay")
    assert cfg.veomni.router_replay.mode == "R2"
