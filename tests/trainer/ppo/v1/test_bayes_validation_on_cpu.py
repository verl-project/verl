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

import asyncio
import json
from math import sqrt
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.v1 import trainer_base
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ
from verl.trainer.ppo.v1.trainer_base import PPOTrainer

_WorkerTQ = AgentLoopWorkerTQ.__ray_metadata__.modified_class


class _Trainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


class _Tokenizer:
    pad_token_id = 0

    def decode(self, ids, **kwargs):
        return str(int(ids[-1]))


class _Worker:
    _run_prompt = _WorkerTQ._run_prompt
    _compute_score = AgentLoopWorker._compute_score
    _compute_multi_modal_inputs = AgentLoopWorker._compute_multi_modal_inputs
    _compute_position_ids = AgentLoopWorker._compute_position_ids
    _get_mm_processor_kwargs = AgentLoopWorker._get_mm_processor_kwargs
    _compute_teacher_logprobs = AgentLoopWorker._compute_teacher_logprobs
    distillation_enabled = False
    processor = None
    mm_processor_kwargs = {}

    def __init__(self, config, outcomes, reward_manager):
        self.config = config
        self.rollout_config = config.actor_rollout_ref.rollout
        self.outcomes = outcomes
        self.background_tasks = set()
        self.reward_loop_worker_handles = [
            SimpleNamespace(compute_score=SimpleNamespace(remote=reward_manager.run_single))
        ]

    async def _run_agent_loop(self, sampling_params, trajectory, trace, **kwargs):
        value = self.outcomes[int(kwargs["index"])][kwargs["session_id"]]
        if value is None:
            raise RuntimeError("injected generation failure")
        output = AgentLoopOutput(
            prompt_ids=[10 + int(kwargs["index"])],
            response_ids=[int(value)],
            response_mask=[1],
            num_turns=2,
            metrics=AgentLoopMetrics(),
        )
        await _WorkerTQ._agent_loop_postprocess(self, output, validate=True, **kwargs)


def _config():
    config_dir = Path(__file__).resolve().parents[4] / "verl/trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(
            config_name="ppo_trainer",
            overrides=[
                "trainer.val_bayes_at_n=true",
                "actor_rollout_ref.rollout.val_kwargs.n=2",
                "actor_rollout_ref.rollout.val_kwargs.do_sample=true",
                "actor_rollout_ref.rollout.val_kwargs.temperature=1.0",
            ],
        )


def _object_array(values):
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def _validation_trainer(monkeypatch, outcomes, split_batches=False, key_transform=None):
    trainer = _Trainer.__new__(_Trainer)
    trainer.config = _config()
    trainer.trainer_mode = "sync"
    trainer.global_steps = 7
    trainer.tokenizer = _Tokenizer()
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=[object()])
    batches = [[i] for i in range(len(outcomes))] if split_batches else [list(range(len(outcomes)))]
    trainer.val_dataloader = [
        {
            "raw_prompt": _object_array([[{"role": "user", "content": f"q{i}"}] for i in indices]),
            "index": np.array(indices),
            "data_source": _object_array(["source"] * len(indices)),
            "reward_model": _object_array([{"ground_truth": "1"} for _ in indices]),
        }
        for indices in batches
    ]
    storage, submitted, statuses = {}, [], {}

    def kv_batch_put(*, keys, partition_id, tags):
        submitted.extend(keys)

    async def async_kv_put(*, key, partition_id, tag):
        statuses[key] = tag["status"]

    async def async_kv_batch_put(*, keys, fields, tags, partition_id):
        for i, key in enumerate(keys):
            key = key_transform(key) if key_transform else key
            assert key not in storage
            storage[key] = fields[i : i + 1]

    def kv_batch_get(*, keys, partition_id, select_fields):
        if select_fields == ["prompts", "responses"]:
            return {
                name: torch.nested.as_nested_tensor([storage[key][name][0] for key in keys], layout=torch.jagged)
                for name in select_fields
            }
        return torch.cat([storage[key].select(*select_fields) for key in keys], dim=0)

    def kv_clear(*, keys, partition_id):
        for key in keys:
            del storage[key]

    def sample(**kwargs):
        return SimpleNamespace(keys=list(reversed(storage)), partition_id="val"), None

    async def generate(batch):
        def grade(data_source, solution_str, ground_truth, extra_info):
            acc = int(solution_str == ground_truth)
            return {"score": 0.2 + 0.6 * acc, "acc": acc}

        reward_manager = NaiveRewardManager(trainer.config, trainer.tokenizer, grade)
        worker = _Worker(trainer.config, outcomes, reward_manager)
        await _WorkerTQ.generate_sequences(worker, batch)
        await asyncio.gather(*worker.background_tasks)

    trainer.agent_loop_manager = SimpleNamespace(generate_sequences=lambda batch: asyncio.run(generate(batch)))
    trainer.replay_buffer = SimpleNamespace(sample=sample)
    for name, fn in {
        "kv_batch_put": kv_batch_put,
        "async_kv_put": async_kv_put,
        "async_kv_batch_put": async_kv_batch_put,
        "kv_batch_get": kv_batch_get,
        "kv_clear": kv_clear,
    }.items():
        monkeypatch.setattr(trainer_base.tq, name, fn)
    return trainer, submitted, statuses


@pytest.mark.parametrize(
    ("outcomes", "expected_sd"),
    [([[0, 0], [1, 1]], sqrt(3 / 160)), ([[0, 1], [0, 1]], sqrt(1 / 40))],
)
@pytest.mark.parametrize("split_batches", [False, True])
def test_bayes_validation_public_path(monkeypatch, outcomes, expected_sd, split_batches):
    """Run native scheduling, grading, queue serialization, and validation metrics on CPU."""
    trainer, submitted, statuses = _validation_trainer(monkeypatch, outcomes, split_batches)
    metrics = trainer._validate()
    assert len(submitted) == 2 and len(set(submitted)) == 2
    assert set(statuses.values()) == {"finished"}
    assert metrics["val-aux/source/acc/bayes@2/mean"] == 0.5
    assert metrics["val-aux/source/acc/bayes@2/std"] == pytest.approx(expected_sd, abs=1e-14)
    assert metrics["val-core/source/acc/mean@2"] == 0.5
    assert metrics["val-aux/source/reward/mean@2"] == pytest.approx(0.5)
    posterior = {key: value for key, value in metrics.items() if "/bayes@" in key}
    assert json.loads(json.dumps(posterior)) == posterior


@pytest.mark.parametrize("outcomes", [[[0, None], [1, 1]], [[None, None], [1, 1]]])
def test_bayes_validation_rejects_native_failed_sessions(monkeypatch, outcomes):
    """Expected questions come from submission, so failures cannot shrink the cohort."""
    trainer, submitted, statuses = _validation_trainer(monkeypatch, outcomes)
    with pytest.raises(ValueError, match="observed trials"):
        trainer._validate()
    assert len(submitted) == 2
    assert "failure" in statuses.values()


def test_bayes_validation_checks_declared_uid_before_generation(monkeypatch):
    """Distinct scheduled questions must not merge under a duplicate generated UID."""
    trainer, submitted, _ = _validation_trainer(monkeypatch, [[0, 1], [0, 1]])
    monkeypatch.setattr(trainer_base.uuid, "uuid4", lambda: "duplicate")
    with pytest.raises(ValueError, match="duplicate question uid"):
        trainer._validate()
    assert submitted == []


@pytest.mark.parametrize("override", ["agent_name", "__rollout_n__", "data_source"])
def test_bayes_validation_checks_batch_overrides_before_generation(monkeypatch, override):
    """Per-question overrides cannot alter the declared experiment."""
    trainer, submitted, _ = _validation_trainer(monkeypatch, [[0, 1], [0, 1]])
    trainer.val_dataloader[0][override] = {
        "agent_name": _object_array(["tool_agent", "single_turn_agent"]),
        "__rollout_n__": np.array([1, 2]),
        "data_source": _object_array([True, "source"]),
    }[override]
    with pytest.raises(ValueError, match="Bayes@N"):
        trainer._validate()
    assert submitted == []


@pytest.mark.parametrize("change", ["session", "uid", "output"])
def test_bayes_validation_rejects_corrupted_trajectory_identity(monkeypatch, change):
    """Rectangular counts alone cannot validate the native trajectory identity."""

    def corrupt(key):
        uid, session, output = key.rsplit("_", 2)
        if change == "session":
            session = str(int(session) + 1)
        elif change == "uid":
            uid = "unexpected"
        else:
            output = "1"
        return f"{uid}_{session}_{output}"

    trainer, _, _ = _validation_trainer(monkeypatch, [[0, 1]], key_transform=corrupt)
    with pytest.raises(ValueError, match="Bayes@N"):
        trainer._validate()


def test_disabled_bayes_validation_keeps_failure_padding(monkeypatch):
    """Ordinary validation still includes missing sessions as incorrect."""
    trainer, _, _ = _validation_trainer(monkeypatch, [[0, None], [1, 1]])
    trainer.config.trainer.val_bayes_at_n = False
    metrics = trainer._validate()
    assert not any("/bayes@" in key for key in metrics)
    assert metrics["val-core/source/acc/mean@2"] == 0.5


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("actor_rollout_ref.rollout.val_kwargs.do_sample", False),
        ("actor_rollout_ref.rollout.val_kwargs.temperature", 0),
        ("actor_rollout_ref.rollout.val_kwargs.temperature", float("nan")),
        ("actor_rollout_ref.rollout.val_kwargs.temperature", float("inf")),
        ("actor_rollout_ref.rollout.val_kwargs.top_k", 1),
        ("actor_rollout_ref.rollout.val_kwargs.top_p", 0),
        ("actor_rollout_ref.rollout.val_kwargs.top_p", -0.1),
        ("actor_rollout_ref.rollout.val_kwargs.top_p", 1.1),
        ("actor_rollout_ref.rollout.val_kwargs.top_p", float("nan")),
        ("actor_rollout_ref.rollout.val_kwargs.top_p", float("inf")),
        ("actor_rollout_ref.rollout.full_determinism", True),
        ("actor_rollout_ref.rollout.agent.default_agent_loop", "tool_agent"),
        ("actor_rollout_ref.rollout.agent.agent_loop_manager_class", "custom.Manager"),
        ("actor_rollout_ref.rollout.agent.agent_loop_config_path", "/tmp/custom-agent.yaml"),
        ("reward.reward_manager.name", "limited"),
        ("reward.reward_model.enable", True),
        ("actor_rollout_ref.rollout.val_kwargs.n", 2.5),
    ],
)
def test_bayes_validation_rejects_unsupported_config(monkeypatch, path, value):
    """Unsupported experiments fail before any remote generation is submitted."""
    trainer, submitted, _ = _validation_trainer(monkeypatch, [[0, 1]])
    OmegaConf.update(trainer.config, path, value, force_add=True)
    with pytest.raises(ValueError, match="Bayes@N"):
        trainer._validate()
    assert submitted == []


def test_bayes_validation_rejects_async_and_legacy_trainers(monkeypatch):
    """The opt-in is restricted to fixed-checkpoint V1 synchronous validation."""
    trainer, submitted, _ = _validation_trainer(monkeypatch, [[0, 1]])
    trainer.trainer_mode = "separate_async"
    with pytest.raises(ValueError, match="trainer_mode=sync"):
        trainer._validate()
    with pytest.raises(ValueError, match="trainer.use_v1=true"):
        RayPPOTrainer._validate(trainer)
    assert submitted == []


@pytest.mark.parametrize("val_before_train", [False, True])
@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("trainer.v1.trainer_mode", "separate_async"),
        ("actor_rollout_ref.rollout.val_kwargs.do_sample", False),
        ("actor_rollout_ref.rollout.val_kwargs.top_k", 1),
        ("actor_rollout_ref.rollout.agent.agent_loop_config_path", "/tmp/custom-agent.yaml"),
        ("reward.reward_manager.name", "limited"),
        ("actor_rollout_ref.rollout.val_kwargs.n", 0),
    ],
)
def test_bayes_static_config_rejected_during_v1_construction(monkeypatch, path, value, val_before_train):
    """Reject unsupported experiments before callback setup, even without initial validation."""
    config = _config()
    config.trainer.val_before_train = val_before_train
    OmegaConf.update(config, path, value, force_add=True)
    callback_setup = Mock(side_effect=AssertionError("callback setup must not run"))
    monkeypatch.setattr(trainer_base, "build_checkpoint_callback", callback_setup)

    with pytest.raises(ValueError, match="Bayes@N"):
        _Trainer(config)
    callback_setup.assert_not_called()


@pytest.mark.parametrize("val_before_train", [False, True])
def test_bayes_legacy_rejected_during_construction(monkeypatch, val_before_train):
    """Legacy Bayes@N must fail before logger or dataset setup."""
    config = _config()
    config.trainer.val_before_train = val_before_train
    logger_setup = Mock(side_effect=AssertionError("logger setup must not run"))
    monkeypatch.setattr("verl.trainer.ppo.ray_trainer.ValidationGenerationsLogger", logger_setup)

    with pytest.raises(ValueError, match="trainer.use_v1=true"):
        RayPPOTrainer(
            config,
            tokenizer=None,
            role_worker_mapping={trainer_base.Role.ActorRollout: object()},
            resource_pool_manager=None,
        )
    logger_setup.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
def test_bayes_supported_or_disabled_constructor_accepts_config(monkeypatch, enabled):
    """Valid Bayes@N and disabled non-Bayes configurations still construct normally."""
    config = _config()
    config.trainer.val_before_train = False
    config.trainer.val_bayes_at_n = enabled
    if not enabled:
        config.trainer.v1.trainer_mode = "separate_async"
        config.actor_rollout_ref.rollout.val_kwargs.do_sample = False
    callback_setup = Mock(return_value=None)
    replay_setup = Mock(return_value=object())
    monkeypatch.setattr(trainer_base, "build_checkpoint_callback", callback_setup)
    monkeypatch.setattr(_Trainer, "_build_replay_buffer", replay_setup)
    monkeypatch.setattr(trainer_base, "RolloutMoELoadBalanceMetricsAccumulator", Mock(return_value=object()))

    trainer = _Trainer(config)
    assert trainer.trainer_mode == config.trainer.v1.trainer_mode
    callback_setup.assert_called_once_with(config)
    replay_setup.assert_called_once_with()
