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

"""Exercise V1 train skip with real trainer control flow and CPU optimizers."""

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from transfer_queue import KVBatchMeta

from verl.protocol import DataProto
from verl.trainer.ppo.metric_utils import RolloutMoELoadBalanceMetricsAccumulator, compute_data_metrics
from verl.trainer.ppo.v1 import trainer_base
from verl.trainer.ppo.v1.trainer_colocate_async import PPOTrainerColocateAsync
from verl.trainer.ppo.v1.trainer_separate_async import HybridEngineMode, PPOTrainerSeparateAsync
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils import tensordict_utils as tu
from verl.utils.skip import SkipManager

TRAINERS = {
    "sync": PPOTrainerSync,
    "colocate_async": PPOTrainerColocateAsync,
    "separate_async": PPOTrainerSeparateAsync,
}
FORWARDS = ["balance", "old_log_prob", "ref_log_prob", "values", "advantage"]


@pytest.fixture(autouse=True)
def reset_manager(monkeypatch):
    monkeypatch.setattr(SkipManager, "config", None)
    monkeypatch.setattr(SkipManager, "step", -1)
    monkeypatch.setattr(SkipManager, "skip_instances", {})


def _make_trainer(monkeypatch, *, mode="sync", scope="train_compute", steps=None, enable=True, use_critic=True):
    """Use production step/hooks; replace RPCs with a queue and small CPU models."""
    trainer = object.__new__(TRAINERS[mode])
    sync_steps = 2 if mode == "separate_async" else 1
    trainer.config = OmegaConf.create(
        {
            "skip": {"train": {"enable": enable, "scope": scope, "steps": steps}, "rollout_tq": {"enable": False}},
            "trainer": {
                "use_v1": True,
                "v1": {
                    "trainer_mode": mode,
                    mode: {"parameter_sync_step": sync_steps, "num_warmup_batches": 0},
                },
                "critic_warmup": 0,
                "project_name": "test",
                "experiment_name": "skip_train",
                "logger": ["console"],
                "total_epochs": 1,
                "save_freq": 1,
                "test_freq": 1,
                "val_before_train": True,
            },
            "actor_rollout_ref": {"rollout": {"n": 2, "temperature": 1.0}, "model": {}},
            "data": {"train_batch_size": 4, "gen_batch_size": 4},
            "global_profiler": {"steps": None},
        }
    )
    trainer.trainer_mode = mode
    trainer.parameter_sync_step = sync_steps
    trainer.global_steps = 1
    trainer.local_trigger_step = 0
    trainer.use_critic = use_critic
    trainer.use_reference_policy = True
    trainer.curr_step_profile = trainer.prev_step_profile = False
    trainer.total_training_steps = trainer.steps_per_epoch = 3
    trainer._restored_tq_prompt_count = 0
    trainer.current_mode = HybridEngineMode.TRAINER
    trainer.hybrid_rollout_config = SimpleNamespace(enable_switch=False)
    trainer._enable_hybrid_replicas = False
    trainer._pending_sync_metrics = {}
    events = []
    queue = []

    def record(name, result=None):
        def call(*args, **kwargs):
            events.append((trainer.global_steps, name))
            return result

        return MagicMock(side_effect=call)

    trainer.checkpoint_manager = SimpleNamespace(
        update_weights=record("weight_sync", {}),
        sleep_replicas=record("sleep"),
        abort_replicas=record("abort"),
        resume_generation_replicas=record("resume"),
    )
    trainer.standalone_checkpoint_manager = SimpleNamespace(update_weights=record("weight_sync", {}))

    def next_batch(num_prompts=None):
        events.append((trainer.global_steps, "dataloader"))
        size = num_prompts or trainer.config.data.train_batch_size
        return tu.get_tensordict(
            {"uid": np.array([f"step{trainer.global_steps}prompt{i}" for i in range(size)], dtype=object)}
        )

    trainer._next_train_batch = MagicMock(side_effect=next_batch)

    def generate(batch):
        events.append((trainer.global_steps, "rollout"))
        queue.extend(f"{uid}_{session}_0" for uid in batch["uid"] for session in range(2))

    trainer.agent_loop_manager = SimpleNamespace(generate_sequences=MagicMock(side_effect=generate))

    def sample(global_steps, partition_id, batch_size):
        events.append((trainer.global_steps, "sample"))
        size = batch_size * 2
        assert len(queue) >= size, "train skip must still submit real rollout before sampling"
        keys = queue[:size]
        del queue[:size]
        tags = [{"min_global_steps": global_steps, "max_global_steps": global_steps}] * size
        return KVBatchMeta(partition_id=partition_id, keys=keys, tags=tags), {}

    trainer.replay_buffer = SimpleNamespace(sample=MagicMock(side_effect=sample))
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=None)

    def passthrough(name):
        def call(batch, metrics):
            events.append((trainer.global_steps, name))
            return batch

        return MagicMock(side_effect=call)

    trainer._compute_reward_colocate = passthrough("reward")
    for method, name in zip(
        ["_balance_batch", "_compute_old_log_prob", "_compute_ref_log_prob", "_compute_values", "_compute_advantage"],
        FORWARDS,
        strict=True,
    ):
        setattr(trainer, method, passthrough(name))

    states = {}
    for role in ["actor", "critic"]:
        model = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.ones_(model.weight)
        # Keep this a CPU test even when torch-npu patches automatic foreach selection.
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, foreach=False)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        states[role] = (model, optimizer, scheduler)

    def update(role):
        def call(batch, metrics):
            events.append((trainer.global_steps, f"update_{role}"))
            model, optimizer, scheduler = states[role]
            optimizer.zero_grad()
            loss = model(torch.ones(1, 1)).square().sum()
            loss.backward()
            optimizer.step()
            scheduler.step()
            metrics[f"{role}/loss"] = loss.item()
            return batch

        return MagicMock(side_effect=call)

    trainer._update_actor = update("actor")
    trainer._update_critic = update("critic")
    monkeypatch.setattr(trainer_base.tq, "kv_batch_put", record("queue_put"))
    clear = record("queue_clear")
    monkeypatch.setattr(trainer_base.tq, "kv_clear", clear)
    SkipManager.init(trainer.config)
    return trainer, events, states, clear


def _snapshot(states):
    return copy.deepcopy(
        {
            role: (model.state_dict(), optimizer.state_dict(), scheduler.state_dict())
            for role, (model, optimizer, scheduler) in states.items()
        }
    )


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


@pytest.mark.parametrize("mode", TRAINERS)
@pytest.mark.parametrize("scope", ["updates", "train_compute"])
@pytest.mark.parametrize(
    "steps, enable, selected", [(None, True, True), ([], True, False), ([2], True, True), (None, False, False)]
)
def test_step_selection_preserves_rollout_and_model_state(monkeypatch, mode, scope, steps, enable, selected):
    trainer, events, states, _ = _make_trainer(monkeypatch, mode=mode, scope=scope, steps=steps, enable=enable)
    trainer.global_steps = 2
    trainer.timing_raw = {}
    SkipManager.set_step(2)
    before = _snapshot(states)
    trainer.on_step_begin()
    metrics = {}
    batch = trainer.step(metrics, trainer.timing_raw)
    trainer.on_step_end()
    names = [name for _, name in events]
    assert names.count("rollout") == 1
    assert names.count("sample") == names.count("reward") == trainer.parameter_sync_step
    assert names.count("weight_sync") == 1
    assert len(batch.keys) == 8
    for stage in FORWARDS:
        assert names.count(stage) == (0 if selected and scope == "train_compute" else trainer.parameter_sync_step)
    if mode in ("sync", "colocate_async"):
        assert names.count("sleep") == 1
    if mode == "colocate_async":
        assert names.count("abort") == names.count("resume") == 1
    if enable:
        assert metrics["training/skip_train"] == int(selected)
        assert metrics["training/skip_train_compute"] == int(selected and scope == "train_compute")
    else:
        assert "training/skip_train" not in metrics
        assert "training/skip_train_compute" not in metrics
    for role in states:
        assert names.count(f"update_{role}") == (0 if selected else trainer.parameter_sync_step)
        if selected:
            _assert_equal(_snapshot(states)[role], before[role])
            assert f"{role}/loss" not in metrics
        else:
            assert not torch.equal(states[role][0].weight, before[role][0]["weight"])


@pytest.mark.parametrize("mode", TRAINERS)
@pytest.mark.parametrize("scope", ["updates", "train_compute"])
@pytest.mark.parametrize("start_step", [0, 1])
def test_fit_skips_middle_step_then_resumes_and_finishes_lifecycle(monkeypatch, mode, scope, start_step):
    trainer, events, states, clear = _make_trainer(monkeypatch, mode=mode, scope=scope, steps=[2])
    trainer.global_steps = start_step
    snapshots = {start_step: _snapshot(states)}
    logs = {}

    def log(data, step):
        if "training/skip_train" in data:
            snapshots[step] = _snapshot(states)
            logs[step] = dict(data)

    monkeypatch.setattr(trainer_base, "Tracking", lambda **kwargs: SimpleNamespace(log=log))
    monkeypatch.setattr(trainer_base, "ValidationGenerationsLogger", MagicMock())
    monkeypatch.setattr(trainer_base, "DapoFilteredRewardTableLogger", MagicMock())
    for name in ["_start_profiling", "_stop_profiling", "_reissue_inflight_prompts", "_shutdown_dump_executor"]:
        setattr(trainer, name, MagicMock())
    trainer._compute_metrics = MagicMock()
    trainer._save_checkpoint = MagicMock()
    trainer._validate = MagicMock(return_value={"validation/reward": 1.0})
    trainer.on_validate_begin = MagicMock()
    trainer.on_validate_end = MagicMock()
    trainer.fit(trainer.agent_loop_manager)

    run_steps = list(range(start_step + 1, 4))
    assert set(logs) == set(run_steps)
    _assert_equal(snapshots[2], snapshots[1])
    for role in states:
        assert not torch.equal(snapshots[3][role][0]["weight"], snapshots[2][role][0]["weight"])
        assert [events.count((s, f"update_{role}")) for s in run_steps] == [
            0 if s == 2 else trainer.parameter_sync_step for s in run_steps
        ]
    assert trainer._validate.call_count == len(run_steps) + 1
    assert trainer._save_checkpoint.call_count == len(run_steps)
    assert trainer.on_validate_begin.call_count == trainer.on_validate_end.call_count == len(run_steps) + 1
    assert clear.call_count == len(run_steps)
    assert [call.kwargs["partition_id"] for call in clear.call_args_list] == ["train"] * len(run_steps)
    assert [name for _, name in events].count("rollout") == len(run_steps)
    assert [name for _, name in events].count("weight_sync") == len(run_steps)
    assert trainer.global_steps == SkipManager.step == 4
    trainer._shutdown_dump_executor.assert_called_once()


@pytest.mark.parametrize("use_critic", [False, True])
def test_normal_warmup_and_optional_critic_are_preserved(monkeypatch, use_critic):
    trainer, _, _, _ = _make_trainer(monkeypatch, steps=[], use_critic=use_critic)
    trainer.config.trainer.critic_warmup = 3
    SkipManager.set_step(1)
    metrics = {}
    trainer.step(metrics, {})
    trainer._update_actor.assert_not_called()
    assert trainer._update_critic.call_count == int(use_critic)


@pytest.mark.parametrize("mode", ["colocate_async", "separate_async"])
def test_train_skip_does_not_disable_async_warmup(monkeypatch, mode):
    trainer, _, _, _ = _make_trainer(monkeypatch, mode=mode)
    trainer.config.trainer.v1[mode].num_warmup_batches = 2
    trainer._add_prompts_to_generate = MagicMock()
    trainer.on_train_begin()
    trainer._add_prompts_to_generate.assert_called_once_with(8)


@pytest.mark.parametrize("scope", ["updates", "train_compute"])
def test_rollout_failures_are_not_hidden_by_training_skip(monkeypatch, scope):
    trainer, _, _, _ = _make_trainer(monkeypatch, scope=scope)
    SkipManager.set_step(1)
    trainer.agent_loop_manager.generate_sequences.side_effect = RuntimeError("inference backend failure")
    with pytest.raises(RuntimeError, match="inference backend failure"):
        trainer.step({}, {})
    trainer.replay_buffer.sample.assert_not_called()


def test_update_failures_outside_selected_steps_are_not_hidden(monkeypatch):
    trainer, _, _, _ = _make_trainer(monkeypatch, steps=[2], use_critic=False)
    SkipManager.set_step(1)
    trainer._update_actor.side_effect = RuntimeError("optimizer failure")
    with pytest.raises(RuntimeError, match="optimizer failure"):
        trainer.step({}, {})


def _rollout_data():
    return TensorDict(
        {
            "prompts": torch.nested.nested_tensor([torch.tensor([1, 2]), torch.tensor([1, 2, 3])], layout=torch.jagged),
            "responses": torch.nested.nested_tensor([torch.tensor([4, 5]), torch.tensor([6])], layout=torch.jagged),
            "response_mask": torch.nested.nested_tensor([torch.ones(2), torch.ones(1)], layout=torch.jagged),
            "rm_scores": torch.nested.nested_tensor(
                [torch.tensor([0.0, 1.0]), torch.tensor([0.5])], layout=torch.jagged
            ),
            "num_turns": torch.tensor([1, 2]),
        },
        batch_size=2,
    )


@pytest.mark.parametrize("use_critic", [False, True])
def test_train_compute_metrics_do_not_request_missing_fields(monkeypatch, use_critic):
    trainer, _, _, _ = _make_trainer(monkeypatch, use_critic=use_critic)
    trainer.resource_pool_manager = SimpleNamespace(get_n_gpus=lambda: 1)
    trainer._rollout_moe_lb_metrics_accumulator = RolloutMoELoadBalanceMetricsAccumulator()
    SkipManager.set_step(1)
    data = _rollout_data()

    def get(*, keys, partition_id, select_fields):
        assert set(select_fields) == set(data.keys()), "must not request missing training fields from TQ"
        return data.clone()

    monkeypatch.setattr(trainer_base.tq, "kv_batch_get", get)
    batch = KVBatchMeta(
        partition_id="train",
        keys=["a_0_0", "b_0_0"],
        tags=[{"min_global_steps": 0, "max_global_steps": 0}] * 2,
    )
    metrics = {}
    trainer._compute_metrics(batch, metrics, {"step": 1.0, "gen": 0.5}, global_steps=1, epoch=0)
    assert metrics["critic/score/mean"] == 0.75
    assert metrics["response_length/mean"] == 1.5
    assert metrics["training/num_turns/mean"] == 1.5
    assert "timing_s/gen" in metrics
    assert not any(
        k.startswith(("critic/advantages", "critic/returns", "critic/values", "critic/rewards")) for k in metrics
    )


@pytest.mark.parametrize("aborted", [False, True])
def test_rollout_metrics_allow_missing_training_fields_but_default_remains_strict(aborted):
    data = _rollout_data().to_padded_tensor()
    data["token_level_scores"] = data["rm_scores"]
    data["prompt_length"] = torch.tensor([2.0, 3.0])
    data["response_length"] = torch.zeros(2) if aborted else torch.tensor([2.0, 1.0])
    if aborted:
        data["response_mask"].zero_()
    batch = DataProto(batch=data)
    metrics = compute_data_metrics(batch, include_training=False)
    assert metrics["response/aborted_ratio"] == float(aborted)
    assert "critic/advantages/mean" not in metrics
    with pytest.raises(KeyError):
        compute_data_metrics(batch)
