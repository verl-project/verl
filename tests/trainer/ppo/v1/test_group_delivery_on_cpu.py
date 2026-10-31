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
"""Group delivery, strict on-policy batch membership, and optimizer barriers."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.v1 import replay_buffer as replay_module
from verl.trainer.ppo.v1.group_delivery import GroupDeliveryReplayBuffer
from verl.trainer.ppo.v1.trainer_group_delivery import PPOTrainerGroupDelivery, validate_group_delivery_config


@pytest.mark.parametrize(("mode", "weight"), [("token-mean", 3), ("seq-mean-token-mean", 2)])
def test_actor_delivery_excludes_padding_and_tool_observations(monkeypatch, mode, weight):
    from verl.trainer.ppo.v1 import trainer_group_delivery as trainer_module

    trainer = object.__new__(PPOTrainerGroupDelivery)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {"loss_agg_mode": mode, "calculate_entropy": False, "entropy_coeff": 0.0},
                "rollout": {"temperature": 1.0},
            }
        }
    )
    trainer.global_steps = 1
    trainer._overlapped_groups = 0
    trainer.actor_rollout_wg = MagicMock()
    trainer.actor_rollout_wg.backward_group.return_value = {"metrics": {"pg_loss": [1.0]}}
    # Real agent actions occupy 3 positions; observations and synthetic DP lanes are masked.
    masks = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
    monkeypatch.setattr(trainer_module.tq, "kv_batch_get", lambda **kwargs: {"response_mask": masks})
    monkeypatch.setattr(
        trainer_module.tq, "kv_list", lambda *_: {"train": {"b": {"is_prompt": True, "status": "running"}}}
    )
    batch = KVBatchMeta(
        partition_id="train", keys=["a_0_0", "a_1_0", "padding_0_0"], tags=[{}, {}, {"is_padding": True}]
    )
    metrics = {}
    assert trainer._update_actor(batch, metrics) is batch
    assert batch.extra_info["group_delivery_weight"] == weight
    assert batch.extra_info["global_batch_size"] == 2
    assert batch.extra_info["group_delivery_id"] == "a"
    assert trainer._overlapped_groups == 1
    trainer.actor_rollout_wg.backward_group.assert_called_once_with(batch)


@pytest.fixture
def queue(monkeypatch):
    """In-memory TransferQueue metadata; trajectory payloads are not needed by sampling."""
    entries = {}
    monkeypatch.setattr(replay_module.tq, "kv_list", lambda *args: {"train": deepcopy(entries)})

    def clear(*, partition_id, keys):
        for key in keys:
            entries.pop(key)

    monkeypatch.setattr(replay_module.tq, "kv_clear", clear)
    return entries


def add_prompt(queue, uid, status="finished", step=1, sessions=2):
    queue[uid] = {"is_prompt": True, "status": status, "global_steps": step}
    if status == "finished":
        for session in range(sessions):
            queue[f"{uid}_{session}_0"] = {
                "seq_len": 8,
                "global_steps": step,
                "min_global_steps": step - 1,
                "max_global_steps": step - 1,
            }


def make_buffer():
    return GroupDeliveryReplayBuffer(
        trainer_mode="group_delivery",
        trainer_config={},
        max_off_policy_threshold=None,
        max_off_policy_strategy="drop",
        sampler_kwargs={},
        poll_interval=0.001,
    )


def test_ready_group_is_delivered_while_sibling_is_running(queue):
    add_prompt(queue, "a", status="running")
    add_prompt(queue, "b")
    buffer = make_buffer()
    buffer.begin_batch(["a", "b"], 1, 2)
    first, _ = buffer.sample(1, "train", 1)
    assert first.keys == ["b_0_0", "b_1_0"]
    assert queue["a"]["status"] == "running"
    with pytest.raises(RuntimeError, match="Still waiting"):
        buffer.finish_batch()
    add_prompt(queue, "a")
    second, _ = buffer.sample(1, "train", 1)
    assert second.keys == ["a_0_0", "a_1_0"]
    buffer.finish_batch()
    with pytest.raises(ValueError):
        buffer.sample(1, "train", 1)


@pytest.mark.parametrize(
    "bad",
    ["failure", "wrong_version", "foreign", "missing", "partial", "duplicate_session", "trajectory_wrong_version"],
)
def test_invalid_group_fails_batch_instead_of_dropping_or_refilling(queue, bad):
    add_prompt(queue, "a")
    add_prompt(queue, "b")
    buffer = make_buffer()
    buffer.begin_batch(["a", "b"], 1, 2)
    if bad == "failure":
        queue["b"]["status"] = "failure"
    elif bad == "wrong_version":
        queue["b"]["global_steps"] = 0
    elif bad == "foreign":
        add_prompt(queue, "c")
    elif bad == "missing":
        queue.pop("b")
    elif bad == "partial":
        queue.pop("a_1_0")
    elif bad == "duplicate_session":
        queue["a_0_1"] = queue.pop("a_1_0")
    else:
        queue["a_0_0"]["max_global_steps"] = 1
    with pytest.raises(RuntimeError):
        buffer.sample(1, "train", 1)


def test_controller_only_steps_after_all_group_backwards(queue):
    trainer = object.__new__(PPOTrainerGroupDelivery)
    trainer.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"n": 2}}})
    trainer.global_steps = 1
    trainer.replay_buffer = make_buffer()
    trainer._next_train_batch = lambda: {"uid": ["a", "b"]}
    events = []
    trainer.actor_rollout_wg = SimpleNamespace(
        begin_group_batch=lambda ids: events.append(("begin", tuple(ids))),
        finish_group_batch=lambda: events.append("step") or [{"lr": 0.01}, {"grad_norm": 1.0, "lr": 0.01}],
        abort_group_batch=lambda: events.append("abort"),
    )
    trainer.standalone_checkpoint_manager = SimpleNamespace(abort_replicas=lambda: events.append("abort_rollout"))

    def dispatch(prompts):
        events.append("rollout")
        add_prompt(queue, "a", status="running")
        add_prompt(queue, "b")

    def group_step(metrics, timing, sample_batch_size):
        batch, _ = trainer.replay_buffer.sample(1, "train", sample_batch_size)
        uid = batch.keys[0].split("_", 1)[0]
        events.append(("backward", uid))
        if uid == "b":
            assert queue["a"]["status"] == "running"
            assert "step" not in events
            add_prompt(queue, "a")
        batch.extra_info["group_delivery_weight"] = 8
        return batch

    trainer._submit_batch_to_rollout = dispatch
    trainer._step_once = group_step
    metrics = {}
    batch = trainer.step(metrics, {})
    assert isinstance(batch, KVBatchMeta) and len(batch) == 4
    assert events == [("begin", ("a", "b")), "rollout", ("backward", "b"), ("backward", "a"), "step"]
    assert metrics["training/optimizer_steps"] == 1
    assert metrics["actor/grad_norm"] == 1.0


def test_controller_aborts_without_optimizer_step_on_backward_failure():
    trainer = object.__new__(PPOTrainerGroupDelivery)
    trainer.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"n": 2}}})
    trainer.global_steps = 1
    trainer.replay_buffer = make_buffer()
    trainer._next_train_batch = lambda: {"uid": ["a", "b"]}
    trainer.actor_rollout_wg = MagicMock()
    trainer.standalone_checkpoint_manager = MagicMock()
    trainer._submit_batch_to_rollout = MagicMock()
    trainer._step_once = MagicMock(side_effect=RuntimeError("backward failed"))
    with pytest.raises(RuntimeError, match="backward failed"):
        trainer.step({}, {})
    trainer.actor_rollout_wg.finish_group_batch.assert_not_called()
    trainer.actor_rollout_wg.abort_group_batch.assert_called_once()
    trainer.standalone_checkpoint_manager.abort_replicas.assert_called_once()


@pytest.fixture
def config():
    root = Path(__file__).resolve().parents[4] / "verl/trainer/config"
    with initialize_config_dir(config_dir=str(root), version_base=None):
        return compose(
            config_name="ppo_trainer",
            overrides=[
                "model_engine=megatron",
                "trainer.v1.trainer_mode=group_delivery",
                "actor_rollout_ref.hybrid_engine=False",
                "algorithm.adv_estimator=grpo",
                "actor_rollout_ref.rollout.name=vllm",
                "actor_rollout_ref.rollout.n=2",
                "actor_rollout_ref.rollout.nnodes=1",
                "actor_rollout_ref.rollout.checkpoint_engine.backend=nccl",
                "data.train_batch_size=4",
                "actor_rollout_ref.actor.ppo_mini_batch_size=4",
            ],
        )


def test_config_and_registry(config):
    from verl.trainer.ppo.v1 import get_trainer_cls

    validate_group_delivery_config(config)
    assert get_trainer_cls("group_delivery") is PPOTrainerGroupDelivery


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("actor_rollout_ref.actor.ppo_epochs", 2),
        ("actor_rollout_ref.actor.ppo_mini_batch_size", 2),
        ("actor_rollout_ref.hybrid_engine", True),
        ("algorithm.adv_estimator", "gae"),
        ("algorithm.filter_groups.enable", True),
        ("algorithm.rollout_correction.rollout_is", "token"),
        ("actor_rollout_ref.actor.policy_loss.loss_mode", "clip-cov"),
    ],
)
def test_incompatible_algorithms_fail_early(config, key, value):
    OmegaConf.update(config, key, value, force_add=True)
    with pytest.raises(ValueError, match="group_delivery requires"):
        validate_group_delivery_config(config)


@pytest.mark.parametrize(
    ("actor_dp", "ref_dp", "micro", "expected"),
    [
        (2, 1, 1, 2),
        (1, 2, 1, 2),
        (2, 4, 3, 12),
        (4, 2, 2, 8),
    ],
)
def test_padding_aligns_actor_and_reference_parallel_topologies(config, actor_dp, ref_dp, micro, expected):
    trainer = object.__new__(PPOTrainerGroupDelivery)
    trainer.config = config
    trainer.use_reference_policy = True
    trainer.ref_in_actor = False
    config.actor_rollout_ref.actor.use_dynamic_bsz = False
    config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = micro
    config.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz = True
    config.actor_rollout_ref.ref.log_prob_use_dynamic_bsz = False
    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu = micro
    trainer.ref_policy_wg = SimpleNamespace(_dispatch_info={}, _query_dispatch_info=lambda mesh: list(range(ref_dp)))
    assert trainer._get_required_batch_multiple(actor_dp) == expected


def test_cleanup_errors_do_not_hide_original_backward_error():
    trainer = object.__new__(PPOTrainerGroupDelivery)
    trainer.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"n": 2}}})
    trainer.global_steps = 1
    trainer.replay_buffer = make_buffer()
    trainer._next_train_batch = lambda: {"uid": ["a"]}
    trainer.actor_rollout_wg = MagicMock()
    trainer.actor_rollout_wg.abort_group_batch.side_effect = RuntimeError("cleanup failed")
    trainer.standalone_checkpoint_manager = MagicMock()
    trainer._submit_batch_to_rollout = MagicMock()
    trainer._step_once = MagicMock(side_effect=RuntimeError("original failure"))
    with pytest.raises(RuntimeError, match="original failure"):
        trainer.step({}, {})
    trainer.standalone_checkpoint_manager.abort_replicas.assert_called_once()


def test_wait_for_full_batch_control_disables_early_delivery(queue, monkeypatch):
    from verl.trainer.ppo.v1 import group_delivery

    add_prompt(queue, "a", status="running")
    add_prompt(queue, "b")
    buffer = make_buffer()
    buffer.trainer_config["wait_for_full_batch"] = True
    buffer.begin_batch(["a", "b"], 1, 2)
    polls = []

    def wait(_):
        assert queue["b"]["is_prompt"]
        polls.append(True)
        add_prompt(queue, "a")

    monkeypatch.setattr(group_delivery.time, "sleep", wait)
    first, _ = buffer.sample(1, "train", 1)
    assert first.keys == ["a_0_0", "a_1_0"] and polls == [True]
