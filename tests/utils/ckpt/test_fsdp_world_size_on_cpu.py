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
