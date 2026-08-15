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

"""Every RolloutConfig field must have a key in rollout.yaml.

Hydra runs in struct mode, so a dataclass field with no YAML counterpart cannot be
set with a plain ``a.b.c=value`` override -- the run aborts with
ConfigCompositionException. Fields listed in KNOWN_UNMAPPED have no YAML key yet;
anything else that drifts fails here instead of at a user's first training run.
"""

from dataclasses import fields
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from verl.workers.config import RolloutConfig

CONFIG_DIR = Path(__file__).resolve().parents[2] / "verl" / "trainer" / "config"

# Fields that are deliberately not exposed in rollout.yaml.
KNOWN_UNMAPPED = {
    # Mandatory (???) in the YAML, so OmegaConf reports it as absent.
    "name",
    # Nested ServerConfig block; read by the trtllm rollout only.
    "server",
    # No reader in the tree.
    "custom",
    "layer_name_map",
    "sglang_engine_mode",
}


def _rollout_node(config_name):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name=config_name).actor_rollout_ref.rollout


@pytest.mark.parametrize("config_name", ["ppo_trainer", "ppo_megatron_trainer"])
def test_every_field_is_reachable(config_name):
    node = _rollout_node(config_name)
    missing = sorted(f.name for f in fields(RolloutConfig) if f.name not in node and f.name not in KNOWN_UNMAPPED)
    assert not missing, (
        f"{config_name}: RolloutConfig fields {missing} have no key in rollout.yaml, so "
        f"actor_rollout_ref.rollout.<field>=... is rejected by Hydra. Add each one to "
        f"verl/trainer/config/rollout/rollout.yaml, or to KNOWN_UNMAPPED if it is not "
        f"meant to be set from the command line."
    )


@pytest.mark.parametrize("config_name", ["ppo_trainer", "ppo_megatron_trainer"])
def test_known_unmapped_is_not_stale(config_name):
    node = _rollout_node(config_name)
    mapped = sorted(name for name in KNOWN_UNMAPPED if name in node)
    assert not mapped, f"{config_name}: {mapped} are in rollout.yaml now; drop them from KNOWN_UNMAPPED"
