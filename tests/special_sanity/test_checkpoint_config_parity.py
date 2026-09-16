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

"""Every ``CheckpointConfig`` field must be reachable from the command line.

Hydra composes the trainer configs in struct mode, so a key that is absent from
the YAML cannot be set with a plain ``a.b.c=value`` override -- the user gets
``Could not override ... To append to your config use +a.b.c=value`` even though
the dataclass declares the field and the docs describe it. Adding a field to
``CheckpointConfig`` without adding it to the YAML therefore ships a knob that
looks configurable and is not.

This walks the composed configs, finds every node whose ``_target_`` is
``CheckpointConfig`` or a subclass, and asserts the node exposes every field the
dataclass declares.
"""

import os
from dataclasses import fields
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import get_class
from omegaconf import DictConfig

from verl.trainer.config import CheckpointConfig

CONFIG_DIR = Path(__file__).resolve().parents[2] / "verl" / "trainer" / "config"

# Top-level configs that own at least one checkpoint block.
TRAINER_CONFIGS = ["ppo_trainer", "ppo_megatron_trainer", "sft_trainer_engine"]


def _checkpoint_nodes(node: DictConfig, path: str = ""):
    """Yield ``(path, node, cls)`` for every CheckpointConfig node under ``node``."""
    target = node["_target_"] if "_target_" in node else None
    if target:
        cls = get_class(str(target))
        if isinstance(cls, type) and issubclass(cls, CheckpointConfig):
            yield path, node, cls

    for key in node:
        child = node._get_node(key)
        if isinstance(child, DictConfig):
            yield from _checkpoint_nodes(child, f"{path}.{key}" if path else key)


@pytest.mark.parametrize("config_name", TRAINER_CONFIGS)
def test_checkpoint_config_fields_are_all_in_yaml(config_name):
    with initialize_config_dir(config_dir=os.fspath(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=config_name)

    found = list(_checkpoint_nodes(cfg))
    assert found, f"no CheckpointConfig node found in {config_name}.yaml"

    missing = {
        path: sorted(f.name for f in fields(cls) if f.name not in node)
        for path, node, cls in found
        if any(f.name not in node for f in fields(cls))
    }
    assert not missing, (
        f"{config_name}.yaml: checkpoint fields declared on the dataclass but absent from the YAML, "
        f"so `{config_name}.py ... <path>.<field>=<value>` is rejected by Hydra: {missing}"
    )
