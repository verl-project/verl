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

"""Driver publication must wait for every async role and every rank."""

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.async_checkpoint import finalize_async_checkpoint, prepare_async_checkpoint


def make_trainer(tmp_path, actor_async=True, critic_async=True, use_critic=True):
    trainer = SimpleNamespace(
        config=OmegaConf.create(
            {
                "trainer": {"default_local_dir": str(tmp_path)},
                "actor_rollout_ref": {"actor": {"checkpoint": {"async_save": actor_async}}},
                "critic": {"checkpoint": {"async_save": critic_async}},
            }
        ),
        actor_rollout_wg=Mock(),
        critic_wg=Mock(),
        use_critic=use_critic,
    )
    for worker in (trainer.actor_rollout_wg, trainer.critic_wg):
        worker.finalize_async_checkpointing.return_value = [True, True]
    return trainer


def test_tracker_waits_for_all_roles_and_ranks(tmp_path):
    trainer = make_trainer(tmp_path)
    coordinator = prepare_async_checkpoint(trainer)
    coordinator.pending_step = 3
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    tracker.write_text("1")
    trainer.actor_rollout_wg.finalize_async_checkpointing.return_value = [True, False]
    finalize_async_checkpoint(trainer)
    assert tracker.read_text() == "1"
    trainer.critic_wg.finalize_async_checkpointing.assert_called_once_with(blocking=False)
    trainer.actor_rollout_wg.finalize_async_checkpointing.return_value = [True, True]
    trainer.critic_wg.finalize_async_checkpointing.return_value = [False, False]
    finalize_async_checkpoint(trainer)
    assert tracker.read_text() == "1"
    trainer.critic_wg.finalize_async_checkpointing.return_value = [True, True]
    finalize_async_checkpoint(trainer, blocking=True)
    assert tracker.read_text() == "3"
    assert coordinator.pending_step is None
    assert not (tmp_path / "latest_checkpointed_iteration.txt.tmp").exists()


@pytest.mark.parametrize(
    "actor_async,critic_async,use_critic",
    [
        (True, False, False),
        (True, False, True),
        (False, True, True),
        (False, False, True),
    ],
)
def test_only_async_roles_receive_collective_rpcs(tmp_path, actor_async, critic_async, use_critic):
    trainer = make_trainer(tmp_path, actor_async, critic_async, use_critic)
    coordinator = prepare_async_checkpoint(trainer)
    if coordinator:
        coordinator.pending_step = 3
    finalize_async_checkpoint(trainer, blocking=True)
    assert trainer.actor_rollout_wg.finalize_async_checkpointing.called == actor_async
    assert trainer.critic_wg.finalize_async_checkpointing.called == (critic_async and use_critic)
    assert (tmp_path / "latest_checkpointed_iteration.txt").exists() == bool(coordinator)


def test_next_save_drains_previous_checkpoint(tmp_path):
    trainer = make_trainer(tmp_path, use_critic=False)
    coordinator = prepare_async_checkpoint(trainer)
    coordinator.pending_step = 3
    assert prepare_async_checkpoint(trainer) is coordinator
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "3"
    assert coordinator.pending_step is None
    assert trainer.actor_rollout_wg.finalize_async_checkpointing.call_args_list == [call(blocking=True)]


def test_failed_finalization_does_not_publish(tmp_path):
    trainer = make_trainer(tmp_path)
    coordinator = prepare_async_checkpoint(trainer)
    coordinator.pending_step = 3
    trainer.critic_wg.finalize_async_checkpointing.side_effect = RuntimeError("writer failed")
    with pytest.raises(RuntimeError, match="writer failed"):
        finalize_async_checkpoint(trainer, blocking=True)
    assert not (tmp_path / "latest_checkpointed_iteration.txt").exists()
    assert coordinator.pending_step == 3


@pytest.mark.parametrize("results", [[], [None], [True, False]])
def test_blocking_drain_requires_completion_acknowledgements(tmp_path, results):
    trainer = make_trainer(tmp_path, use_critic=False)
    prepare_async_checkpoint(trainer).pending_step = 3
    trainer.actor_rollout_wg.finalize_async_checkpointing.return_value = results
    with pytest.raises(RuntimeError, match="pending writes"):
        finalize_async_checkpoint(trainer, blocking=True)
    assert not (tmp_path / "latest_checkpointed_iteration.txt").exists()


@pytest.mark.parametrize("remove_previous", [False, True])
def test_retention_includes_sync_roles_and_preserves_configured_limits(tmp_path, remove_previous):
    trainer = make_trainer(tmp_path, critic_async=False)
    trainer.config.trainer.max_actor_ckpt_to_keep = 2
    trainer.config.trainer.max_critic_ckpt_to_keep = 3
    trainer.config.trainer.remove_previous_ckpt_in_save = remove_previous
    coordinator = prepare_async_checkpoint(trainer)
    coordinator.pending_step = 3
    finalize_async_checkpoint(trainer, blocking=True)
    trainer.actor_rollout_wg.prune_checkpoints.assert_called_once_with(max_ckpt_to_keep=1 if remove_previous else 2)
    trainer.critic_wg.prune_checkpoints.assert_called_once_with(max_ckpt_to_keep=1 if remove_previous else 3)
    trainer.critic_wg.finalize_async_checkpointing.assert_not_called()
