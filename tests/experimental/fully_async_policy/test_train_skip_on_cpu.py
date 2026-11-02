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

"""Fully async training skip with production queue/step/sync code and CPU optimizers."""

import asyncio
import copy
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import ray
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.fully_async_policy import fully_async_trainer as module
from verl.experimental.fully_async_policy.detach_utils import MetricsAggregator, RolloutSample
from verl.experimental.fully_async_policy.dynamic_schedule import DynamicScheduleContext
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer, TrainingStopException
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.utils.skip import SkipManager

Trainer = FullyAsyncTrainer.__ray_metadata__.modified_class


@pytest.fixture(autouse=True)
def reset_manager(monkeypatch):
    monkeypatch.setattr(SkipManager, "config", None)
    monkeypatch.setattr(SkipManager, "step", -1)
    monkeypatch.setattr(SkipManager, "skip_instances", {})


class _ObjectRef:
    def __init__(self, value):
        self.value = value

    def future(self):
        result = Future()
        result.set_result(self.value)
        return result

    def __await__(self):
        async def get():
            return self.value

        return get().__await__()


def _batch():
    return DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([[1, 2], [1, 3]]),
            "responses": torch.tensor([[4, 5], [6, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
            "response_mask": torch.tensor([[1, 1], [1, 0]]),
            "rm_scores": torch.tensor([[0.0, 1.0], [0.5, 0.0]]),
            "rollout_log_probs": torch.zeros(2, 2),
        },
        non_tensors={
            "uid": np.array(["a", "a"]),
            "min_global_steps": np.array([0, 0]),
            "max_global_steps": np.array([0, 0]),
            "accuracy": np.array([1.0, 0.5]),
        },
        meta_info={
            "reward_extra_keys": ["accuracy"],
            "global_token_num": [4, 3],
            "metrics": [{"generate_sequences": 0.1, "tool_calls": 0.0}] * 2,
        },
    )


def _make_trainer(monkeypatch, *, scope="train_compute", steps=None, enable=True, sync_steps=1, use_critic=True):
    trainer = object.__new__(Trainer)
    trainer.config = OmegaConf.create(
        {
            "skip": {"train": {"enable": enable, "scope": scope, "steps": steps}},
            "async_training": {"trigger_parameter_sync_step": sync_steps, "use_trainer_do_validate": False},
            "trainer": {
                "use_v1": False,
                "v1": {"trainer_mode": "sync"},
                "balance_batch": True,
                "critic_warmup": 0,
                "test_freq": 1,
                "save_freq": 1,
                "esi_redundant_time": 0,
                "rollout_data_dir": "/unused",
            },
            "actor_rollout_ref": {"rollout": {"temperature": 1.0}},
            "global_profiler": {"steps": None},
        }
    )
    trainer.global_steps = 1
    trainer.epoch = 0
    trainer.local_trigger_step = 1
    trainer.current_param_version = 0
    trainer._old_log_prob_param_version = None
    trainer.trigger_parameter_sync_step = sync_steps
    trainer.required_samples = 1
    trainer.stale_trajectory_processed = 0
    trainer.use_critic = use_critic
    trainer.use_reference_policy = True
    trainer.use_rm = False
    trainer.dynamic_schedule_enabled = False
    trainer.dynamic_resource_controller = None
    trainer.dynamic_schedule_ctx = DynamicScheduleContext(
        required_samples=1,
        trigger_parameter_sync_step=sync_steps,
        total_generated_samples=0,
        expected_samples=0,
        buffer_samples=0,
    )
    trainer.staleness_threshold = 1.0
    trainer.only_hybrid = False
    trainer._step_wait_times = []
    trainer._step_wait_samples = []
    trainer.last_ckpt_version = 0
    trainer.max_steps_duration = 0
    trainer.tokenizer = None
    trainer.progress_bar = MagicMock()
    trainer.resource_pool_manager = SimpleNamespace(get_n_gpus=lambda: 1)
    trainer.metrics_aggregator = MetricsAggregator(total_gpus=2)
    trainer._shutdown_dump_executor = MagicMock()
    events = []
    samples = []
    logs = []

    def queue_batch():
        sample = RolloutSample(full_batch=_batch(), sample_id="a", epoch=0, rollout_status={})
        samples.append(ray.cloudpickle.dumps(sample))

    async def get_sample():
        events.append((trainer.global_steps, "queue_get"))
        return (samples.pop(0), len(samples)) if samples else (None, 0)

    trainer.message_queue_client = SimpleNamespace(
        get_sample=AsyncMock(side_effect=get_sample),
        get_queue_size=AsyncMock(side_effect=lambda: len(samples)),
        get_queue_size_sync=MagicMock(side_effect=lambda: len(samples)),
    )
    trainer.logger = SimpleNamespace(log=lambda *, data, step: logs.append((step, dict(data))))

    def record(name, result=None):
        def call(*args, **kwargs):
            events.append((trainer.global_steps, name))
            return result

        return MagicMock(side_effect=call)

    trainer._balance_batch = record("balance")
    trainer._fit_start_profile = record("profile_start")
    trainer._fit_stop_profile = record("profile_stop")
    trainer._log_rollout_data = record("dump")
    trainer._save_checkpoint = record("checkpoint")
    trainer.checkpoint_manager = SimpleNamespace(update_weights=AsyncMock(side_effect=record("weight_sync")))
    trainer.rollouter = SimpleNamespace(
        get_total_produced_samples=SimpleNamespace(remote=MagicMock(return_value=100)),
        get_completed_steps=SimpleNamespace(remote=MagicMock(return_value=0)),
        reset_staleness=SimpleNamespace(remote=record("reset_staleness", _ObjectRef({}))),
        do_validate=SimpleNamespace(remote=record("validation", _ObjectRef({"validation/reward": 1.0}))),
    )
    monkeypatch.setattr(module.ray, "get", lambda value: value)
    monkeypatch.setattr(module, "should_save_ckpt_esi", lambda **kwargs: False)

    def forward(name, fields):
        def call(batch):
            events.append((trainer.global_steps, name))
            for field in fields:
                batch.batch[field] = torch.ones_like(batch.batch["rm_scores"])
            return batch

        return MagicMock(side_effect=call)

    trainer._fit_compute_log_prob = forward("old_log_prob", ["old_log_probs"])
    trainer._fit_compute_ref_log_prob = forward("ref", ["ref_log_prob"])
    trainer._fit_compute_critic = forward("values", ["values"] if use_critic else [])
    trainer._fit_compute_advantage = forward("adv", ["token_level_rewards", "advantages", "returns"])
    states = {}
    for role in ["actor", "critic"]:
        model = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.ones_(model.weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, foreach=False)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        states[role] = model, optimizer, scheduler

    def update(role):
        def call(batch):
            events.append((trainer.global_steps, f"update_{role}"))
            model, optimizer, scheduler = states[role]
            optimizer.zero_grad()
            loss = model(torch.ones(1, 1)).square().sum()
            loss.backward()
            optimizer.step()
            scheduler.step()
            return DataProto.from_dict(
                tensors={}, meta_info={"metrics": {f"{role}/loss": loss.item(), f"{role}/grad_norm": 1.0}}
            )

        return MagicMock(side_effect=call)

    trainer._update_actor = update("actor")
    trainer._update_critic = update("critic")
    SkipManager.init(trainer.config)
    return trainer, events, states, queue_batch, logs


def _snapshot(states):
    return copy.deepcopy({role: [part.state_dict() for part in parts] for role, parts in states.items()})


def _assert_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_equal(actual[key], expected[key])
    elif isinstance(actual, list | tuple):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_equal(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize("scope", ["updates", "train_compute"])
@pytest.mark.parametrize(
    "steps, enable, selected", [(None, True, True), ([], True, False), ([2], True, True), (None, False, False)]
)
def test_selected_step_freezes_training_state_and_keeps_real_queue_and_lifecycle(
    monkeypatch, scope, steps, enable, selected
):
    trainer, events, states, queue_batch, logs = _make_trainer(monkeypatch, scope=scope, steps=steps, enable=enable)
    trainer.global_steps = 2
    before = _snapshot(states)
    queue_batch()
    asyncio.run(trainer.fit_step())
    names = [name for _, name in events]
    for name in ["queue_get", "weight_sync", "reset_staleness", "validation", "checkpoint", "dump"]:
        assert names.count(name) == 1
    for name in ["balance", "old_log_prob", "ref", "values", "adv"]:
        assert names.count(name) == int(not (selected and scope == "train_compute"))
    assert trainer.global_steps == 3
    assert trainer.current_param_version == 1
    if enable:
        assert trainer.metrics["training/skip_train"] == int(selected)
        assert trainer.metrics["training/skip_train_compute"] == int(selected and scope == "train_compute")
    else:
        assert "training/skip_train" not in trainer.metrics
        assert "training/skip_train_compute" not in trainer.metrics
    for role in states:
        assert names.count(f"update_{role}") == int(not selected)
        if selected:
            _assert_equal(_snapshot(states)[role], before[role])
            assert f"{role}/loss" not in trainer.metrics
        else:
            assert not torch.equal(states[role][0].weight, before[role][0]["weight"])
    assert trainer.metrics["critic/score/mean"] == 0.75
    assert ("critic/advantages/mean" not in trainer.metrics) == (selected and scope == "train_compute")
    training_logs = [data for _, data in logs if "training/global_step" in data]
    assert len(training_logs) == 1
    assert training_logs[0]["training/global_step"] == 2
    assert trainer.metrics_aggregator.step_count == 0


@pytest.mark.parametrize("scope", ["updates", "train_compute"])
@pytest.mark.parametrize("sync_steps", [1, 2, 3])
def test_skip_then_resume_in_and_across_sync_cycles(monkeypatch, scope, sync_steps):
    trainer, events, states, queue_batch, logs = _make_trainer(
        monkeypatch, scope=scope, steps=[1, 3], sync_steps=sync_steps
    )
    snapshots = {0: _snapshot(states)}
    for step in range(1, 7):
        queue_batch()
        asyncio.run(trainer.fit_step())
        snapshots[step] = _snapshot(states)
        if step in [1, 3]:
            _assert_equal(snapshots[step], snapshots[step - 1])
        else:
            for role in states:
                assert not torch.equal(snapshots[step][role][0]["weight"], snapshots[step - 1][role][0]["weight"])
    assert [name for _, name in events].count("queue_get") == 6
    assert [name for _, name in events].count("weight_sync") == 6 // sync_steps
    assert trainer.current_param_version == 6 // sync_steps
    assert trainer.global_steps == 7
    training_logs = [data for _, data in logs if "training/global_step" in data]
    for role in states:
        assert [step for step, name in events if name == f"update_{role}"] == [2, 4, 5, 6]
    assert [data["training/skip_train"] for data in training_logs] == [
        sum(s in [1, 3] for s in range(end - sync_steps + 1, end + 1)) / sync_steps
        for end in range(sync_steps, 7, sync_steps)
    ]
    assert [data["training/global_step"] for data in training_logs] == list(range(sync_steps, 7, sync_steps))
    assert all(data["training/partial_sync_cycle"] == 0 for data in training_logs)
    assert all(data["training/sync_cycle_micro_steps"] == sync_steps for data in training_logs)


@pytest.mark.parametrize("use_critic", [False, True])
def test_critic_warmup_is_preserved(monkeypatch, use_critic):
    trainer, _, _, queue_batch, _ = _make_trainer(monkeypatch, steps=[], use_critic=use_critic)
    trainer.config.trainer.critic_warmup = 3
    queue_batch()
    asyncio.run(trainer.fit_step())
    trainer._update_actor.assert_not_called()
    assert trainer._update_critic.call_count == int(use_critic)
    trainer.checkpoint_manager.update_weights.assert_awaited_once()


@pytest.mark.parametrize("scope", ["updates", "train_compute"])
def test_termination_and_queue_failure_are_not_hidden(monkeypatch, scope):
    trainer, _, states, _, _ = _make_trainer(monkeypatch, scope=scope)
    before = _snapshot(states)
    with pytest.raises(TrainingStopException):
        asyncio.run(trainer.fit_step())
    _assert_equal(_snapshot(states), before)
    trainer.checkpoint_manager.update_weights.assert_not_awaited()
    trainer.message_queue_client.get_sample.side_effect = RuntimeError("producer failure")
    with pytest.raises(RuntimeError, match="producer failure"):
        asyncio.run(trainer.fit_step())


def test_unselected_update_failure_propagates(monkeypatch):
    trainer, _, _, queue_batch, _ = _make_trainer(monkeypatch, steps=[2], use_critic=False)
    queue_batch()
    trainer._update_actor.side_effect = RuntimeError("optimizer failure")
    with pytest.raises(RuntimeError, match="optimizer failure"):
        asyncio.run(trainer.fit_step())
    trainer.checkpoint_manager.update_weights.assert_not_awaited()


@pytest.mark.parametrize("test_freq", [0, 1])
def test_fit_consumes_termination_and_closes_resources(monkeypatch, test_freq):
    trainer, _, _, queue_batch, _ = _make_trainer(monkeypatch)
    trainer.global_steps = 0
    trainer.config.trainer.test_freq = test_freq
    queue_batch()
    asyncio.run(trainer.fit())
    trainer.progress_bar.close.assert_called_once()
    assert trainer.global_steps == 2
    trainer._shutdown_dump_executor.assert_called_once()


@pytest.mark.parametrize("scope", ["updates", "train_compute"])
@pytest.mark.parametrize("skip_steps", [[1], [1, 2]])
@pytest.mark.parametrize("start_version, complete_cycles", [(0, 0), (0, 1), (2, 0)])
def test_partial_cycle_eof_flushes_metrics_without_sync_or_version_advance(
    monkeypatch, scope, skip_steps, start_version, complete_cycles
):
    tail_start = (start_version + complete_cycles) * 3
    skip_steps = [tail_start + step for step in skip_steps]
    trainer, events, states, queue_batch, logs = _make_trainer(monkeypatch, scope=scope, steps=skip_steps, sync_steps=3)
    trainer.global_steps = start_version * 3
    trainer.current_param_version = start_version
    trainer.last_ckpt_version = start_version
    before = _snapshot(states)
    for _ in range(complete_cycles * 3 + 2):
        queue_batch()
    asyncio.run(trainer.fit())
    assert trainer.global_steps == tail_start + 3
    assert trainer.current_param_version == start_version + complete_cycles
    assert trainer.local_trigger_step == 3
    assert trainer.checkpoint_manager.update_weights.await_count == complete_cycles
    assert trainer.rollouter.reset_staleness.remote.call_count == complete_cycles
    assert trainer._save_checkpoint.call_count == complete_cycles
    assert trainer.rollouter.do_validate.remote.call_count == complete_cycles
    assert [name for _, name in events].count("queue_get") == complete_cycles * 3 + 3
    training_logs = [(step, data) for step, data in logs if "training/global_step" in data]
    assert [step for step, _ in training_logs] == list(range(start_version + 1, start_version + complete_cycles + 2))
    for _, full_cycle in training_logs[:-1]:
        assert full_cycle["training/skip_train"] == 0
        assert full_cycle["training/partial_sync_cycle"] == 0
        assert full_cycle["training/sync_cycle_micro_steps"] == 3
    _, data = training_logs[-1]
    assert data["training/global_step"] == tail_start + 2
    assert data["training/partial_sync_cycle"] == 1
    assert data["training/sync_cycle_micro_steps"] == 2
    assert not any("param_sync" in key for key in data)
    assert data["training/skip_train"] == len(skip_steps) / 2
    for role in ["actor", "critic"]:
        assert [step for step, name in events if name == f"update_{role}"] == [
            step for step in range(start_version * 3 + 1, tail_start + 3) if step not in skip_steps
        ]
        assert (f"{role}/loss" in data) == (len(skip_steps) == 1)
    if len(skip_steps) == 2 and complete_cycles == 0:
        _assert_equal(_snapshot(states), before)
    assert trainer.metrics_aggregator.step_count == 0
    trainer._shutdown_dump_executor.assert_called_once()


@pytest.mark.parametrize("scope", ["updates", "train_compute"])
def test_dynamic_deactivation_and_activation_remain_outside_skip(monkeypatch, scope):
    trainer, events, _, queue_batch, _ = _make_trainer(monkeypatch, scope=scope)
    trainer.dynamic_schedule_enabled = True
    trainer._dynamic_schedule_enable_rebalance = True
    trainer.dynamic_resource_controller = SimpleNamespace(
        is_hybrid_active=True,
        policy=SimpleNamespace(
            should_deactivate=MagicMock(return_value=True),
            deactivate_wait_samples=MagicMock(return_value=1),
            should_activate_after_step=MagicMock(return_value=True),
            request_rebalance=MagicMock(),
            update_after_step=MagicMock(),
        ),
        deactivate_hybrid_replicas=AsyncMock(),
        sync_hybrid_weights=AsyncMock(),
        activate_hybrid_replicas=AsyncMock(),
    )
    trainer.rollouter.wait_for_enough_samples = SimpleNamespace(remote=MagicMock(return_value=1))
    queue_batch()
    asyncio.run(trainer.fit_step())
    ctrl = trainer.dynamic_resource_controller
    ctrl.deactivate_hybrid_replicas.assert_awaited_once_with(0)
    ctrl.sync_hybrid_weights.assert_awaited_once_with(global_steps=1)
    ctrl.activate_hybrid_replicas.assert_awaited_once_with(1)
    ctrl.policy.request_rebalance.assert_called_once()
    trainer._update_actor.assert_not_called()
    assert "reset_staleness" in [name for _, name in events]


def test_lazy_snapshot_and_restore_across_skipped_cycle_starts(monkeypatch):
    trainer, _, _, _, _ = _make_trainer(monkeypatch, steps=[1, 4], sync_steps=3)
    weights = {"current": 1.0}
    snapshots = {}
    trainer.actor_rollout_wg = SimpleNamespace(
        save_model_to_cpu=lambda version: snapshots.update({version: weights["current"]}),
        restore_model_from_cpu=lambda version: weights.update(current=snapshots[version]),
        clear_cpu_model=lambda version: snapshots.pop(version),
    )
    seen = []

    def forward(self, batch):
        seen.append(weights["current"])
        return batch, 0.1

    monkeypatch.setattr(SeparateRayPPOTrainer, "_compute_old_log_prob", forward)
    batch = _batch()
    trainer.local_trigger_step = 2
    trainer._compute_old_log_prob(batch)
    assert seen == [1.0] and snapshots == {1: 1.0}
    weights["current"] = 2.0
    trainer.local_trigger_step = 3
    trainer._compute_old_log_prob(batch)
    assert seen == [1.0, 1.0] and weights["current"] == 2.0
    assert snapshots == {1: 1.0}
    trainer.current_param_version = 1
    trainer.local_trigger_step = 2
    trainer._compute_old_log_prob(batch)
    assert seen[-1] == 2.0 and snapshots == {1: 2.0}


def test_snapshot_model_restored_after_forward_failure(monkeypatch):
    trainer, _, _, _, _ = _make_trainer(monkeypatch)
    trainer._old_log_prob_param_version = trainer.current_param_version
    trainer.local_trigger_step = 2
    trainer.actor_rollout_wg = MagicMock()
    monkeypatch.setattr(SeparateRayPPOTrainer, "_compute_old_log_prob", MagicMock(side_effect=RuntimeError("forward")))
    with pytest.raises(RuntimeError, match="forward"):
        trainer._compute_old_log_prob(_batch())
    assert trainer.actor_rollout_wg.restore_model_from_cpu.call_args_list[-1].args == (2,)
    trainer.actor_rollout_wg.clear_cpu_model.assert_called_once_with(2)


@pytest.mark.parametrize("skip_steps", [[1], [2], [1, 2, 3, 4]])
def test_snapshot_anchor_tracks_sync_cycle_with_selective_training(monkeypatch, skip_steps):
    trainer, _, _, _, _ = _make_trainer(monkeypatch, steps=skip_steps, sync_steps=3, use_critic=False)
    weights = {"current": 1.0}
    snapshots = {}
    observed = []
    trainer.actor_rollout_wg = SimpleNamespace(
        save_model_to_cpu=lambda version: snapshots.update({version: weights["current"]}),
        restore_model_from_cpu=lambda version: weights.update(current=snapshots[version]),
        clear_cpu_model=lambda version: snapshots.pop(version),
    )

    def forward(self, batch):
        observed.append(weights["current"])
        return batch, 0.0

    def update(batch):
        weights["current"] += 1
        return DataProto.from_dict(tensors={}, meta_info={"metrics": {"actor/loss": 1.0}})

    monkeypatch.setattr(SeparateRayPPOTrainer, "_compute_old_log_prob", forward)
    trainer._fit_compute_log_prob = lambda batch: trainer._compute_old_log_prob(batch)[0]
    trainer._update_actor = update
    expected = []
    cycle_start = weights["current"]
    for step in range(1, 7):
        trainer.global_steps = step
        SkipManager.set_step(step)
        trainer.metrics = {}
        trainer.timing_raw = {}
        if trainer.local_trigger_step == 1:
            cycle_start = weights["current"]
        before = weights["current"]
        trainer._fit_train_batch(_batch())
        if step not in skip_steps:
            expected.append(cycle_start)
            assert weights["current"] == before + 1
        else:
            assert weights["current"] == before
        trainer._fit_update_local_step()
    assert observed == expected
    assert set(snapshots) == {1}


def test_checkpoint_resume_selects_next_actual_micro_step(monkeypatch):
    trainer, _, _, queue_batch, _ = _make_trainer(monkeypatch, steps=[7], sync_steps=3)
    trainer.config.trainer.resume_mode = "resume_path"
    trainer.config.trainer.resume_from_path = "/unused/global_step_2"
    trainer.config.trainer.default_local_dir = "/unused"
    trainer.config.trainer.default_hdfs_dir = None
    trainer.config.trainer.del_local_ckpt_after_load = False
    trainer.actor_rollout_wg = MagicMock()
    trainer.critic_wg = MagicMock()
    monkeypatch.setattr(module, "find_latest_ckpt_path", lambda path: None)
    asyncio.run(trainer.load_checkpoint())
    assert trainer.global_steps == 6
    queue_batch()
    asyncio.run(trainer.fit())
    assert trainer.global_steps == 8
    # The attempted terminal read starts step 8; the only consumed batch was step 7.
    trainer._update_actor.assert_not_called()
    trainer._update_critic.assert_not_called()
