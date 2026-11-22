# Copyright contributors to the PEAR project
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

import json

import pytest

from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager


def _manager(world_size):
    manager = FSDPCheckpointManager.__new__(FSDPCheckpointManager)
    manager.world_size = world_size
    return manager


def test_a_checkpoint_of_another_world_size_is_refused_with_both_sizes(tmp_path):
    (tmp_path / "fsdp_config.json").write_text(json.dumps({"FSDP_version": 2, "world_size": 8}))

    with pytest.raises(ValueError, match="saved by 8 trainer processes and this run has 16"):
        _manager(16).load_checkpoint(str(tmp_path))


def test_a_checkpoint_of_this_world_size_or_without_its_config_passes_the_check(tmp_path):
    _manager(8)._require_saved_world_size(str(tmp_path))

    (tmp_path / "fsdp_config.json").write_text(json.dumps({"FSDP_version": 2, "world_size": 8}))
    _manager(8)._require_saved_world_size(str(tmp_path))
