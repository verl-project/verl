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

"""Real directory retention across checkpoint publication and simulated interruptions."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from verl.trainer.ppo.async_checkpoint import AsyncCheckpointCoordinator
from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager
from verl.workers.engine_workers import TrainingWorker


class CheckpointWorker:
    """Simulate writer completion while using the real registration/deletion methods."""

    def __init__(self, root, role, ready=False, synchronous=False):
        self.root = root
        self.role = role
        self.ready = ready
        self.manager = BaseCheckpointManager.__new__(BaseCheckpointManager)
        old = root / "global_step_1" / role
        old.mkdir(parents=True)
        (old / "weights").write_text("old")
        self.manager.previous_saved_paths = [str(old)]
        self.registered = False
        self.prune_calls = 0
        self.synchronous = synchronous
        engine = SimpleNamespace(
            supports_deferred_checkpoint_retention=True,
            save_checkpoint=self.save_on_engine,
            prune_checkpoints=self.manager.prune_checkpoints,
        )
        self.worker = SimpleNamespace(engine=engine)
        self.step = 2
        self.start_save(2)

    def start_save(self, step):
        self.step = step
        self.ready = False
        self.registered = False
        TrainingWorker.save_checkpoint(
            self.worker,
            str(self.root / f"global_step_{step}" / self.role),
            global_step=step,
            max_ckpt_to_keep=1,
            defer_retention=True,
            update_tracker=False,
        )

    def save_on_engine(self, local_path, hdfs_path, global_step, max_ckpt_to_keep, **kwargs):
        self.limit_during_save = max_ckpt_to_keep
        self.manager.ensure_checkpoint_capacity(max_ckpt_to_keep)
        if self.synchronous:
            self.complete_write()

    def complete_write(self):
        if not self.registered:
            new = self.root / f"global_step_{self.step}" / self.role
            new.mkdir(parents=True, exist_ok=True)
            (new / "weights").write_text("new")
            # TrainingWorker disables retention until the coordinator publishes the tracker.
            self.manager.register_checkpoint(str(new), self.limit_during_save)
            self.registered = True

    def finalize_async_checkpointing(self, blocking=False):
        if self.ready or blocking:
            self.complete_write()
        return [self.registered]

    def prune_checkpoints(self, max_ckpt_to_keep):
        # Every destructive call must observe the new complete checkpoint as published.
        assert (self.root / "latest_checkpointed_iteration.txt").read_text() == str(self.step)
        self.prune_calls += 1
        TrainingWorker.prune_checkpoints(self.worker, max_ckpt_to_keep=max_ckpt_to_keep)


def setup_checkpoint(tmp_path, *, actor_only=False, sync_critic=False):
    actor = CheckpointWorker(tmp_path, "actor")
    roles = [actor]
    if not actor_only:
        roles.append(CheckpointWorker(tmp_path, "critic", synchronous=sync_critic))
    (tmp_path / "global_step_1/data.pt").write_text("old dataloader")
    (tmp_path / "global_step_2").mkdir(exist_ok=True)
    (tmp_path / "global_step_2/data.pt").write_text("new dataloader")
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("1")
    async_roles = [actor] if sync_critic else roles
    coordinator = AsyncCheckpointCoordinator(async_roles, str(tmp_path), [(role, 1) for role in roles])
    coordinator.pending_step = 2
    return coordinator, roles


def assert_recoverable(root, roles, step):
    assert (root / "latest_checkpointed_iteration.txt").read_text() == str(step)
    path = root / f"global_step_{step}"
    assert (path / "data.pt").exists()
    for role in roles:
        assert (path / role.role / "weights").exists()


@pytest.mark.parametrize("first", [0, 1])
def test_keep_one_preserves_old_checkpoint_while_other_role_is_pending(tmp_path, first):
    coordinator, roles = setup_checkpoint(tmp_path)
    roles[first].ready = True
    coordinator.finalize()
    # A process exit at this point still leaves the tracker-selected checkpoint complete.
    assert_recoverable(tmp_path, roles, 1)
    assert all(role.prune_calls == 0 for role in roles)
    roles[1 - first].ready = True
    coordinator.finalize()
    assert_recoverable(tmp_path, roles, 2)
    for role in roles:
        assert not (tmp_path / "global_step_1" / role.role).exists()
        assert len(role.manager.previous_saved_paths) == 1


def test_keep_one_prunes_each_previous_actor_only_after_three_step_publication(tmp_path):
    coordinator, roles = setup_checkpoint(tmp_path, actor_only=True)
    actor = roles[0]

    coordinator.finalize()
    assert_recoverable(tmp_path, roles, 1)
    assert actor.prune_calls == 0
    actor.ready = True
    coordinator.finalize()
    assert_recoverable(tmp_path, roles, 2)
    assert not (tmp_path / "global_step_1/actor").exists()

    (tmp_path / "global_step_3").mkdir()
    (tmp_path / "global_step_3/data.pt").write_text("third dataloader")
    actor.start_save(3)
    coordinator.pending_step = 3
    coordinator.finalize()
    assert_recoverable(tmp_path, roles, 2)
    assert actor.prune_calls == 1

    actor.ready = True
    coordinator.finalize()
    assert_recoverable(tmp_path, roles, 3)
    assert not (tmp_path / "global_step_2/actor").exists()
    assert actor.prune_calls == 2
    assert actor.manager.previous_saved_paths == [str(tmp_path / "global_step_3/actor")]


@pytest.mark.parametrize("actor_only,sync_critic", [(True, False), (False, False), (False, True)])
def test_interruption_before_tracker_replace_keeps_old_files(tmp_path, actor_only, sync_critic):
    coordinator, roles = setup_checkpoint(tmp_path, actor_only=actor_only, sync_critic=sync_critic)
    with patch("verl.trainer.ppo.async_checkpoint.os.replace", side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            coordinator.finalize(blocking=True)
    assert_recoverable(tmp_path, roles, 1)
    assert all(role.prune_calls == 0 for role in roles)
    # Retry publication and retention; completed writers must not register duplicate paths.
    coordinator.finalize(blocking=True)
    assert_recoverable(tmp_path, roles, 2)
    assert all(len(role.manager.previous_saved_paths) == 1 for role in roles)


def test_sync_critic_does_not_prune_while_async_actor_is_pending(tmp_path):
    coordinator, roles = setup_checkpoint(tmp_path, sync_critic=True)
    coordinator.finalize()
    assert_recoverable(tmp_path, roles, 1)
    assert roles[1].registered
    assert all(role.prune_calls == 0 for role in roles)
    coordinator.finalize(blocking=True)
    assert_recoverable(tmp_path, roles, 2)
    assert all(role.prune_calls == 1 for role in roles)


def test_interruption_during_cleanup_leaves_new_checkpoint_recoverable(tmp_path):
    coordinator, roles = setup_checkpoint(tmp_path)
    with patch.object(roles[1], "prune_checkpoints", side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            coordinator.finalize(blocking=True)
    assert_recoverable(tmp_path, roles, 2)
    assert not (tmp_path / "global_step_1/actor").exists()
    assert (tmp_path / "global_step_1/critic").exists()
    assert coordinator.pending_step == 2
    coordinator.finalize(blocking=True)
    assert_recoverable(tmp_path, roles, 2)
    assert coordinator.pending_step is None
    assert all(len(role.manager.previous_saved_paths) == 1 for role in roles)


def test_finalizer_failure_keeps_old_checkpoint(tmp_path):
    coordinator, roles = setup_checkpoint(tmp_path)
    roles[0].ready = True
    with patch.object(roles[1], "finalize_async_checkpointing", side_effect=RuntimeError("writer failed")):
        with pytest.raises(RuntimeError, match="writer failed"):
            coordinator.finalize()
    assert_recoverable(tmp_path, roles, 1)
    assert all(role.prune_calls == 0 for role in roles)


@pytest.mark.parametrize("limit", [None, 0, 1, 2])
def test_post_publication_pruning_does_not_register_again(tmp_path, limit):
    manager = BaseCheckpointManager.__new__(BaseCheckpointManager)
    manager.previous_saved_paths = []
    for step in range(3):
        path = tmp_path / str(step)
        path.mkdir()
        manager.register_checkpoint(str(path), None)
    manager.prune_checkpoints(limit)
    manager.prune_checkpoints(limit)
    expected = 3 if not limit else limit
    assert len(manager.previous_saved_paths) == expected
    assert all(Path(path).exists() for path in manager.previous_saved_paths)
