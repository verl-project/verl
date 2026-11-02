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

"""Exercise legacy training skip through real fit control flow and CPU optimizer state."""

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo import ray_trainer
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils import tracking
from verl.utils.skip import SkipManager


@pytest.fixture(autouse=True)
def reset_manager(monkeypatch):
    monkeypatch.setattr(SkipManager, "config", None)
    monkeypatch.setattr(SkipManager, "step", -1)
    monkeypatch.setattr(SkipManager, "skip_instances", {})


def _snapshot(states):
    return copy.deepcopy(
        {
            name: [model.state_dict(), optimizer.state_dict(), scheduler.state_dict()]
            for name, (model, optimizer, scheduler) in states.items()
        }
    )


def _assert_same(actual, expected):
    if isinstance(actual, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_same(actual[key], expected[key])
    elif isinstance(actual, list | tuple):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_same(left, right)
    else:
        assert actual == expected


def _make_trainer(monkeypatch, *, scope="train_compute", steps=None, enable=True, start_step=0, use_critic=True):
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "skip": {"train": {"enable": enable, "scope": scope, "steps": steps}},
            "trainer": {
                "use_v1": False,
                "v1": {"trainer_mode": "sync"},
                "project_name": "test",
                "experiment_name": "skip_train_v0",
                "logger": ["console"],
                "total_epochs": 1,
                "balance_batch": True,
                "critic_warmup": 0,
                "save_freq": 1,
                "test_freq": 1,
                "val_before_train": True,
                "rollout_data_dir": "unused",
                "esi_redundant_time": 0,
            },
            "actor_rollout_ref": {
                "rollout": {"temperature": 1.0, "n": 2},
                "actor": {"loss_agg_mode": "token-mean", "loss_scale_factor": None, "policy_loss": {}},
            },
            "algorithm": {"adv_estimator": "grpo", "gamma": 1.0, "lam": 1.0, "use_kl_in_reward": False},
            "global_profiler": {"steps": None, "profile_continuous_steps": False},
        }
    )
    trainer.global_steps = start_step
    trainer.use_critic = use_critic
    trainer.use_reference_policy = True
    trainer.use_rm = True
    trainer.total_training_steps = 3
    trainer._dump_executor = SimpleNamespace(_shutdown=False)
    trainer.train_dataset = SimpleNamespace(on_batch_end=MagicMock())
    trainer.resource_pool_manager = SimpleNamespace(get_n_gpus=lambda: 1)
    trainer.train_dataloader = [
        {"prompts": torch.ones(2, 2, dtype=torch.long), "raw_prompt": np.array(["a", "b"], dtype=object)}
        for _ in range(3)
    ]
    events, logged, saved = [], {}, {}
    states = {}
    for name in ["actor", "critic"]:
        model = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.ones_(model.weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, foreach=False)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        states[name] = (model, optimizer, scheduler)
    snapshots = {start_step: _snapshot(states)}

    def record(name, result=None):
        def call(*args, **kwargs):
            events.append((trainer.global_steps, name))
            return result

        return MagicMock(side_effect=call)

    def load():
        trainer.global_steps = start_step

    trainer._load_checkpoint = MagicMock(side_effect=load)
    trainer.checkpoint_manager = SimpleNamespace(
        update_weights=record("sync"),
        sleep_replicas=record("sleep"),
    )
    trainer.actor_rollout_wg = SimpleNamespace(async_calls_finalize_fn_exec=record("finalize"))
    trainer._start_profiling = record("profile_start")
    trainer._stop_profiling = record("profile_end")
    trainer._save_checkpoint = record("save")
    trainer._validate = record("validate", {"validation/reward": 1.0})
    trainer._shutdown_dump_executor = record("shutdown")
    trainer._balance_batch = record("balance")
    monkeypatch.setattr(ray_trainer, "should_save_ckpt_esi", lambda **kwargs: False)

    def generate(batch):
        events.append((trainer.global_steps, "rollout"))
        n = len(batch)
        return DataProto.from_dict(
            tensors={
                "prompts": torch.ones(n, 2, dtype=torch.long),
                "responses": torch.ones(n, 2, dtype=torch.long),
                "attention_mask": torch.ones(n, 4, dtype=torch.long),
            },
            non_tensors={
                "multi_modal_inputs": np.array([{} for _ in range(n)], dtype=object),
                "accuracy": np.arange(n, dtype=float),
            },
            meta_info={"timing": {}, "reward_extra_keys": ["accuracy"]},
        )

    trainer.async_rollout_manager = SimpleNamespace(generate_sequences=MagicMock(side_effect=generate))

    def reward(batch):
        events.append((trainer.global_steps, "reward"))
        return DataProto.from_dict(tensors={"rm_scores": torch.arange(len(batch) * 2).reshape(len(batch), 2).float()})

    trainer._compute_reward_colocate = MagicMock(side_effect=reward)

    def forward(name, fields):
        def call(batch):
            events.append((trainer.global_steps, name))
            output = DataProto.from_dict(
                tensors={key: torch.ones(len(batch), 2) * value for key, value in fields.items()}
            )
            return (output, 0.0) if name == "old_log_prob" else output

        return MagicMock(side_effect=call)

    trainer._compute_old_log_prob = forward("old_log_prob", {"old_log_probs": -1.0, "entropys": 1.0})
    trainer._compute_ref_log_prob = forward("ref", {"ref_log_prob": -0.8})
    trainer._compute_values = forward("values", {"values": 0.0})
    advantage_impl = ray_trainer.compute_advantage

    def advantage(*args, **kwargs):
        events.append((trainer.global_steps, "advantage"))
        return advantage_impl(*args, **kwargs)

    monkeypatch.setattr(ray_trainer, "compute_advantage", advantage)

    def update(name):
        def call(batch):
            events.append((trainer.global_steps, f"update_{name}"))
            model, optimizer, scheduler = states[name]
            optimizer.zero_grad()
            loss = model(torch.ones(1, 1)).square().sum()
            loss.backward()
            optimizer.step()
            scheduler.step()
            return DataProto.from_dict(tensors={}, meta_info={"metrics": {f"{name}/loss": loss.item()}})

        return MagicMock(side_effect=call)

    trainer._update_actor = update("actor")
    trainer._update_critic = update("critic")

    def dump(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir):
        events.append((trainer.global_steps, "dump"))
        saved[trainer.global_steps] = copy.deepcopy(batch)

    trainer._log_rollout_data = MagicMock(side_effect=dump)

    def log(data, step):
        if "training/global_step" in data:
            logged[step] = dict(data)
            snapshots[step] = _snapshot(states)

    monkeypatch.setattr(tracking, "Tracking", lambda **kwargs: SimpleNamespace(log=log))
    return trainer, events, logged, saved, snapshots, states


@pytest.mark.parametrize("scope", ["updates", "train_compute"])
@pytest.mark.parametrize("steps, enable", [(None, True), ([], True), ([2], True), (None, False)])
def test_fit_skip_selection_preserves_rollout_state_and_lifecycle(monkeypatch, scope, steps, enable):
    trainer, events, logs, batches, snapshots, states = _make_trainer(
        monkeypatch, scope=scope, steps=steps, enable=enable
    )
    trainer.fit()
    assert set(logs) == {1, 2, 3}
    for step in [1, 2, 3]:
        selected = enable and (steps is None or step in steps)
        compute_skipped = selected and scope == "train_compute"
        names = [name for s, name in events if s == step]
        assert names.count("rollout") == names.count("reward") == names.count("sleep") == names.count("sync") == 1
        assert names.count("validate") == names.count("save") == names.count("dump") == 1
        for name in ["balance", "old_log_prob", "ref", "values", "advantage"]:
            assert names.count(name) == int(not compute_skipped)
        if enable:
            assert logs[step]["training/skip_train"] == int(selected)
            assert logs[step]["training/skip_train_compute"] == int(compute_skipped)
        else:
            assert "training/skip_train" not in logs[step]
            assert "training/skip_train_compute" not in logs[step]
        assert logs[step]["critic/score/mean"] > 0
        assert logs[step]["response_length/mean"] == 2
        for role in states:
            assert names.count(f"update_{role}") == int(not selected)
            assert (f"{role}/loss" in logs[step]) == (not selected)
            if selected:
                _assert_same(snapshots[step][role], snapshots[step - 1][role])
            else:
                assert not torch.equal(snapshots[step][role][0]["weight"], snapshots[step - 1][role][0]["weight"])
        for key in ["old_log_probs", "ref_log_prob", "values", "advantages", "returns", "token_level_rewards"]:
            assert (key in batches[step].batch) == (not compute_skipped)
        assert "token_level_scores" in batches[step].batch
        assert "accuracy" in batches[step].non_tensor_batch
        for prefix in ["critic/advantages/", "critic/returns/", "critic/rewards/", "critic/values/"]:
            assert any(key.startswith(prefix) for key in logs[step]) == (not compute_skipped)
    assert trainer.global_steps == SkipManager.step == 4
    trainer._shutdown_dump_executor.assert_called_once()


@pytest.mark.parametrize("scope", ["updates", "train_compute"])
def test_fit_resumed_global_step_can_skip_then_restore_optimizer_updates(monkeypatch, scope):
    trainer, events, logs, _, snapshots, states = _make_trainer(monkeypatch, scope=scope, steps=[2], start_step=1)
    trainer.fit()
    assert list(logs) == [2, 3]
    _assert_same(snapshots[2], snapshots[1])
    for role in states:
        assert not torch.equal(snapshots[3][role][0]["weight"], snapshots[2][role][0]["weight"])
        assert [step for step, name in events if name == f"update_{role}"] == [3]


@pytest.mark.parametrize("use_critic", [False, True])
@pytest.mark.parametrize("scope", ["updates", "train_compute"])
def test_critic_warmup_and_checkpoint_cadence_remain_unchanged(monkeypatch, use_critic, scope):
    trainer, events, logs, _, _, _ = _make_trainer(monkeypatch, scope=scope, steps=[2], use_critic=use_critic)
    trainer.config.trainer.critic_warmup = 3
    trainer.fit()
    assert [step for step, name in events if name == "update_actor"] == [3]
    assert [step for step, name in events if name == "update_critic"] == ([1, 3] if use_critic else [])
    assert [s for s, name in events if name == "save"] == [3]
    assert [s for s, name in events if name == "sync"] == [0, 1, 2, 3]


@pytest.mark.parametrize("scope", ["updates", "train_compute"])
def test_inference_errors_are_not_hidden_by_train_skip(monkeypatch, scope):
    trainer, _, _, _, _, _ = _make_trainer(monkeypatch, scope=scope)
    trainer.async_rollout_manager.generate_sequences.side_effect = RuntimeError("inference failure")
    with pytest.raises(RuntimeError, match="inference failure"):
        trainer.fit()
    trainer._compute_reward_colocate.assert_not_called()
    trainer._update_actor.assert_not_called()


def test_update_failure_on_unselected_step_propagates(monkeypatch):
    trainer, _, _, _, _, _ = _make_trainer(monkeypatch, steps=[2])
    trainer._update_actor.side_effect = RuntimeError("optimizer failure")
    with pytest.raises(RuntimeError, match="optimizer failure"):
        trainer.fit()


@pytest.mark.parametrize("scope", ["updates", "train_compute"])
def test_remax_baseline_still_runs_real_rollout_and_reward(monkeypatch, scope):
    trainer, events, logs, batches, _, _ = _make_trainer(monkeypatch, scope=scope)
    trainer.config.algorithm.adv_estimator = "remax"
    trainer.fit()
    assert all(
        call.args[0].non_tensor_batch["__do_sample__"].tolist() == [True] * 4 + [False] * 2
        for call in trainer.async_rollout_manager.generate_sequences.call_args_list
    )
    for step in [1, 2, 3]:
        assert [name for s, name in events if s == step].count("reward") == 2
        assert len(batches[step]) == 4
        assert "reward_baselines" in batches[step].batch
    trainer._update_actor.assert_not_called()


@pytest.mark.parametrize("scope", ["updates", "train_compute"])
def test_kl_controller_and_reward_shaping_follow_selected_scope(monkeypatch, scope):
    trainer, _, logs, batches, _, _ = _make_trainer(monkeypatch, scope=scope, steps=[2])
    trainer.config.algorithm.use_kl_in_reward = True
    trainer.config.algorithm.kl_penalty = "kl"
    trainer.kl_ctrl_in_reward = SimpleNamespace(value=0.1, update=MagicMock())
    trainer.fit()
    assert trainer.kl_ctrl_in_reward.update.call_count == (2 if scope == "train_compute" else 3)
    for step in [1, 2, 3]:
        computed = scope == "updates" or step != 2
        assert ("token_level_rewards" in batches[step].batch) == computed
        assert ("actor/reward_kl_penalty" in logs[step]) == computed
        if computed:
            assert not torch.equal(
                batches[step].batch["token_level_scores"], batches[step].batch["token_level_rewards"]
            )


@pytest.mark.parametrize("scope", ["updates", "train_compute"])
@pytest.mark.parametrize("bypass", [False, True])
def test_rollout_correction_follows_scope_and_preserves_bypass_mode(monkeypatch, scope, bypass):
    from verl.trainer.ppo import rollout_corr_helper

    trainer, events, _, _, _, _ = _make_trainer(monkeypatch, scope=scope, steps=[2])
    trainer.config.algorithm.rollout_correction = {"bypass_mode": bypass}
    generate_impl = trainer.async_rollout_manager.generate_sequences.side_effect

    def generate(batch):
        result = generate_impl(batch)
        result.batch["rollout_log_probs"] = torch.full((len(result), 2), -1.0)
        return result

    trainer.async_rollout_manager.generate_sequences.side_effect = generate
    correction = MagicMock(side_effect=lambda batch, config: (batch, {"rollout_corr/test": 1.0}))
    monkeypatch.setattr(rollout_corr_helper, "compute_rollout_correction_and_add_to_batch", correction)

    def apply_bypass(*, batch, **kwargs):
        batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]

    bypass_call = MagicMock(side_effect=apply_bypass)
    monkeypatch.setattr(rollout_corr_helper, "apply_bypass_mode", bypass_call)
    trainer.fit()
    compute_calls = 2 if scope == "train_compute" else 3
    assert correction.call_count == (0 if bypass else compute_calls)
    assert bypass_call.call_count == (compute_calls if bypass else 0)
    assert trainer._compute_old_log_prob.call_count == (0 if bypass else compute_calls)
    assert [step for step, name in events if name == "update_actor"] == [1, 3]


@pytest.mark.parametrize("scope", ["updates", "train_compute"])
@pytest.mark.parametrize("action", ["cache", "repeat"])
def test_train_skip_composes_with_rollout_replay(monkeypatch, scope, action, tmp_path):
    trainer, events, logs, _, _, _ = _make_trainer(monkeypatch, scope=scope, steps=[2])
    trainer.config.skip.rollout = {"enable": True, "action": action, "steps": [2], "dump_dir": str(tmp_path)}
    generate_impl = trainer.async_rollout_manager.generate_sequences.side_effect
    cached = generate_impl(DataProto.from_dict(tensors={"prompts": torch.ones(4, 2, dtype=torch.long)}))
    events.clear()
    SkipManager.init(trainer.config)
    SkipManager.skip_instances["rollout"].prepare_data(2 if action == "cache" else 1, cached)
    trainer.async_rollout_manager.generate_sequences = MagicMock(
        side_effect=SkipManager.annotate(role="rollout")(generate_impl)
    )
    trainer.fit()
    assert [step for step, name in events if name == "rollout"] == [1, 3]
    assert [step for step, name in events if name == "update_actor"] == [1, 3]


@pytest.mark.parametrize("scope, selected", [("train_compute", False), ("train_compute", True), ("updates", True)])
def test_router_replay_conflict_is_preserved_unless_training_forward_is_skipped(monkeypatch, scope, selected):
    trainer, _, logs, _, snapshots, _ = _make_trainer(monkeypatch, scope=scope, steps=[1] if selected else [])
    trainer.total_training_steps = 1
    generate_impl = trainer.async_rollout_manager.generate_sequences.side_effect
    forward_impl = trainer._compute_old_log_prob.side_effect

    def generate(batch):
        result = generate_impl(batch)
        result.batch["routed_experts"] = torch.ones(len(result), 2, dtype=torch.long)
        return result

    def old_log_prob(batch):
        result, mfu = forward_impl(batch)
        result.batch["routed_experts"] = torch.ones(len(result), 2, dtype=torch.long)
        return result, mfu

    trainer.async_rollout_manager.generate_sequences.side_effect = generate
    trainer._compute_old_log_prob.side_effect = old_log_prob
    if selected and scope == "train_compute":
        trainer.fit()
        trainer._compute_old_log_prob.assert_not_called()
        assert logs[1]["training/skip_train_compute"] == 1
        _assert_same(snapshots[1], snapshots[0])
    else:
        with pytest.raises(ValueError, match="Detected conflicting router replay configuration"):
            trainer.fit()
        trainer._compute_old_log_prob.assert_called_once()
    trainer._update_actor.assert_not_called()
    trainer._update_critic.assert_not_called()
