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

from omegaconf import OmegaConf

from verl.runtime import Topology
from verl.trainer.ppo.model_config import PPOModelConfigs, PPORoleConfigs


def _policy(rollout_n: int) -> dict:
    return {"model": {"path": "m"}, "actor": {}, "rollout": {"n": rollout_n}, "ref": {}}


def _config():
    return OmegaConf.create(
        {
            "actor_rollout_ref": _policy(rollout_n=1),
            "rollout_policy": _policy(rollout_n=4),
            "critic": {"enable": False},
            "reward": {"reward_model": {"enable": False}},
        }
    )


def test_legacy_config_reads_roles_from_legacy_keys():
    config = _config()

    roles = PPORoleConfigs.resolve(config, None)

    assert roles.actor is roles.rollout is roles.ref is roles.actor_rollout_worker_config()
    assert roles.actor == config.actor_rollout_ref
    assert roles.critic == config.critic
    assert roles.reward_model == config.reward.reward_model


def test_topology_selects_declared_roles_and_falls_back_for_undeclared():
    config = _config()
    topology = Topology.from_mapping(
        {
            "clusters": [{"name": "c", "nnodes": 1, "n_gpus_per_node": 1}],
            "device_pools": [{"name": "p", "cluster": "c", "nnodes": 1, "n_gpus_per_node": 1}],
            "models": [
                {"name": "actor", "worker": "actor", "config_key": "actor_rollout_ref", "resource_pool": "p"},
                {"name": "rollout", "worker": "rollout", "config_key": "rollout_policy", "resource_pool": "p"},
            ],
        }
    )

    roles = PPORoleConfigs.resolve(config, PPOModelConfigs(config, topology))

    assert roles.rollout == config.rollout_policy
    assert roles.ref is roles.actor
    assert roles.critic == config.critic
    assert roles.actor_rollout_worker_config().rollout.n == 4
