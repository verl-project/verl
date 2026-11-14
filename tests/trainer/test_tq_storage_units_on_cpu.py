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

import os

import pytest
from hydra import compose, initialize_config_dir

from verl.trainer.main_ppo import set_default_tq_storage_units

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "verl", "trainer", "config")


def _compose(overrides):
    with initialize_config_dir(config_dir=os.path.abspath(CONFIG_DIR), version_base=None):
        return compose(config_name="ppo_trainer", overrides=overrides)


@pytest.mark.parametrize("nnodes, expected", [(1, 2), (4, 8)])
def test_storage_units_default_to_two_per_node(nnodes, expected):
    config = _compose([f"trainer.nnodes={nnodes}"])

    set_default_tq_storage_units(config)

    assert config.transfer_queue.backend.SimpleStorage.num_data_storage_units == expected


def test_explicit_storage_units_are_kept():
    config = _compose(["trainer.nnodes=4", "transfer_queue.backend.SimpleStorage.num_data_storage_units=3"])

    set_default_tq_storage_units(config)

    assert config.transfer_queue.backend.SimpleStorage.num_data_storage_units == 3
