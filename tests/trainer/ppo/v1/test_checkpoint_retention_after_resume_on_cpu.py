import os

import pytest

from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager


@pytest.fixture
def manager():
    manager = BaseCheckpointManager.__new__(BaseCheckpointManager)  # only the retention bookkeeping is exercised
    manager.previous_saved_paths = []
    return manager


@pytest.fixture
def make_step(tmp_path):
    def make(step, actor=True):
        folder = os.path.join(tmp_path, f"global_step_{step}")
        os.makedirs(os.path.join(folder, "actor") if actor else folder)
        open(os.path.join(folder, "data.pt"), "w").close()
        return folder

    return make


def test_resume_registers_the_older_siblings_in_order(tmp_path, manager, make_step):
    for step in (3, 1, 2, 5):
        make_step(step)
    manager.register_existing_checkpoints(os.path.join(tmp_path, "global_step_3", "actor"))
    assert manager.previous_saved_paths == [
        os.path.join(tmp_path, f"global_step_{step}", "actor") for step in (1, 2, 3)
    ]


def test_resume_ignores_paths_outside_the_step_layout(tmp_path, manager):
    manager.register_existing_checkpoints(os.path.join(tmp_path, "somewhere", "actor"))
    assert manager.previous_saved_paths == []
