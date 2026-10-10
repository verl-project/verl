# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.workers.config.rollout import validate_extra_val_kwargs

TRAIN_SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": -1}


def test_validate_runs_val_kwargs_pass_then_each_extra_profile():
    calls = []

    def run_profile(profile, val_sampling):
        calls.append((profile, val_sampling))
        return {f"val-core/{profile or 'default'}": 1.0}

    trainer = SimpleNamespace(
        config=OmegaConf.create(
            {"actor_rollout_ref": {"rollout": {"extra_val_kwargs": {"train_sampling": TRAIN_SAMPLING}}}}
        ),
        _validate_profile=run_profile,
    )

    metrics = PPOTrainer._validate(trainer)

    assert calls == [(None, None), ("train_sampling", TRAIN_SAMPLING)]
    assert type(calls[1][1]) is dict
    assert metrics == {"val-core/default": 1.0, "val-core/train_sampling": 1.0}


def test_validate_without_extra_profiles_runs_single_pass():
    calls = []
    trainer = SimpleNamespace(
        config=OmegaConf.create({"actor_rollout_ref": {"rollout": {}}}),
        _validate_profile=lambda profile, val_sampling: calls.append(profile) or {},
    )
    PPOTrainer._validate(trainer)
    assert calls == [None]


@pytest.mark.parametrize(
    "extra",
    [
        {"train-sampling": TRAIN_SAMPLING},
        {"train_sampling": {"temperature": 1.0, "n": 16}},
        {"train_sampling": 1.0},
    ],
)
def test_extra_val_kwargs_rejects_bad_profiles(extra):
    with pytest.raises(ValueError):
        validate_extra_val_kwargs(OmegaConf.create(extra))


class _FakeNested:
    def __init__(self, tensor):
        self.tensor = tensor

    def to_padded_tensor(self, padding):
        return self.tensor


class _FakeTQ:
    """Serves one validation group of two sessions from the uid the trainer registered."""

    def __init__(self):
        self.uid = None
        self.cleared = []
        self.data_source = "aime2024"

    def kv_batch_put(self, keys, partition_id, tags):
        assert partition_id == "val"
        self.uid = keys[0]

    def kv_batch_get(self, keys, partition_id, select_fields):
        assert partition_id == "val"
        if select_fields == ["prompts", "responses"]:
            return {
                "prompts": _FakeNested(torch.tensor([[1], [1]])),
                "responses": _FakeNested(torch.tensor([[2], [3]])),
            }
        return {
            "uid": np.array([self.uid, self.uid], dtype=object),
            "rm_scores": torch.tensor([[1.0], [0.0]]),
            "num_turns": np.array([2, 2]),
            "reward_model": np.array([{"ground_truth": "7"}] * 2, dtype=object),
            "data_source": np.array([self.data_source] * 2, dtype=object),
        }

    def kv_clear(self, keys, partition_id):
        assert partition_id == "val"
        self.cleared.extend(keys)


@pytest.mark.parametrize("data_source", ["aime2024", "aime2024_train_sampling"])
def test_validate_profile_namespaces_core_and_aux_metrics(monkeypatch, tmp_path, data_source):
    fake_tq = _FakeTQ()
    fake_tq.data_source = data_source
    monkeypatch.setattr("verl.trainer.ppo.v1.trainer_base.tq", fake_tq)
    generated, dumps, logged = [], [], []

    trainer = SimpleNamespace(
        config=OmegaConf.create(
            {
                "actor_rollout_ref": {"rollout": {"val_kwargs": {"n": 2}}},
                "trainer": {"validation_data_dir": str(tmp_path)},
            }
        ),
        global_steps=40,
        val_dataloader=[
            {"raw_prompt": np.array(["q"], dtype=object), "data_source": np.array(["aime2024"], dtype=object)}
        ],
        agent_loop_manager=SimpleNamespace(generate_sequences=generated.append),
        replay_buffer=SimpleNamespace(
            sample=lambda global_steps, partition_id, batch_size: (
                SimpleNamespace(keys=[f"{fake_tq.uid}_0_0", f"{fake_tq.uid}_1_0"], partition_id=partition_id),
                None,
            )
        ),
        reward_loop_manager=SimpleNamespace(reward_loop_worker_handles=[object()]),
        tokenizer=SimpleNamespace(pad_token_id=0, decode=lambda ids, skip_special_tokens: str(ids.tolist())),
        _maybe_log_val_generations=lambda **kwargs: logged.append(kwargs),
        _dump_generations=lambda **kwargs: dumps.append(kwargs["dump_path"]),
    )
    trainer._val_metrics_update = lambda *args, **kwargs: PPOTrainer._val_metrics_update(trainer, *args, **kwargs)

    metrics = PPOTrainer._validate_profile(trainer, profile="train_sampling", val_sampling=TRAIN_SAMPLING)

    assert generated[0]["val_sampling"] == TRAIN_SAMPLING
    assert metrics[f"val-core/profiles/train_sampling/{data_source}/reward/mean@2"] == pytest.approx(0.5)
    assert not any(key.startswith("val-core/aime2024/") for key in metrics)
    assert metrics["val-aux/profiles/train_sampling/num_turns/mean"] == 2
    assert "val-aux/num_turns/mean" not in metrics
    profile_metrics = metrics.copy()
    assert dumps == [str(tmp_path / "train_sampling")]
    assert logged == []
    assert fake_tq.cleared == [f"{fake_tq.uid}_0_0", f"{fake_tq.uid}_1_0"]

    generated.clear()
    dumps.clear()
    metrics = PPOTrainer._validate_profile(trainer, profile=None, val_sampling=None)

    assert "val_sampling" not in generated[0]
    assert metrics[f"val-core/{data_source}/reward/mean@2"] == pytest.approx(0.5)
    assert dumps == [str(tmp_path)]
    assert len(logged) == 1
    assert metrics["val-aux/num_turns/mean"] == 2
    assert not set(profile_metrics).intersection(metrics)
