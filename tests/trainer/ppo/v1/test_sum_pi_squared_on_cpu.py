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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

import verl.trainer.ppo.v1.trainer_base as trainer_base
from verl.trainer.ppo.v1.trainer_base import PPOTrainer

PROMPT_LENS = [3, 2]
RESPONSE_LENS = [4, 2]


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


class _FakeTQ:
    """In-memory stand-in for the TransferQueue key-value API used by the trainer."""

    def __init__(self, fields: dict):
        self.store = dict(fields)

    def kv_batch_get(self, keys, partition_id, select_fields):
        return TensorDict({k: self.store[k] for k in select_fields if k in self.store}, batch_size=len(keys))

    def kv_batch_put(self, keys, partition_id, fields):
        for k in fields.keys():
            self.store[k] = fields[k]
        return SimpleNamespace(keys=keys, partition_id=partition_id, extra_info={})


class _Batch(SimpleNamespace):
    def __len__(self):
        return len(self.keys)


def _nested(rows):
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def _make_trainer(calculate_sum_pi_squared: bool, adv_estimator: str = "optimal_token_baseline"):
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {
                "adv_estimator": adv_estimator,
                "gamma": 1.0,
                "lam": 1.0,
                "use_kl_in_reward": False,
                "norm_adv_by_std_in_grpo": True,
                "rollout_correction": None,
            },
            "actor_rollout_ref": {
                "actor": {
                    "calculate_sum_pi_squared": calculate_sum_pi_squared,
                    "loss_agg_mode": "token-mean",
                    "loss_scale_factor": None,
                },
                "rollout": {"n": 2, "temperature": 1.0, "calculate_log_probs": False},
            },
        }
    )
    trainer.actor_rollout_wg = MagicMock()
    trainer.actor_rollout_wg.compute_log_prob.side_effect = lambda batch: list(batch.keys)
    return trainer


def _fake_tq_with_actor_outputs():
    torch.manual_seed(0)
    seq_lens = [p + r for p, r in zip(PROMPT_LENS, RESPONSE_LENS, strict=True)]
    # The actor writes per-position outputs over the whole sequence; position t predicts token t + 1.
    full = {name: _nested([torch.rand(n) for n in seq_lens]) for name in ("log_probs", "entropy", "sum_pi_squared")}
    return _FakeTQ(
        {
            **full,
            "response_mask": _nested([torch.ones(r, dtype=torch.long) for r in RESPONSE_LENS]),
            "rm_scores": _nested([torch.tensor([0.0] * (r - 1) + [float(i)]) for i, r in enumerate(RESPONSE_LENS)]),
            "uid": torch.tensor([0, 0]),
        }
    )


@pytest.mark.parametrize("calculate_sum_pi_squared", [True, False])
def test_old_log_prob_writes_response_aligned_sum_pi_squared(monkeypatch, calculate_sum_pi_squared):
    fake_tq = _fake_tq_with_actor_outputs()
    full_sum_pi_squared = fake_tq.store["sum_pi_squared"]
    monkeypatch.setattr(trainer_base, "tq", fake_tq)
    trainer = _make_trainer(calculate_sum_pi_squared)
    batch = _Batch(keys=["a", "b"], partition_id="train", extra_info={})

    trainer._compute_old_log_prob(batch, metrics={})

    assert batch.extra_info["calculate_sum_pi_squared"] is calculate_sum_pi_squared
    stored = fake_tq.store["sum_pi_squared"]
    if not calculate_sum_pi_squared:
        assert stored is full_sum_pi_squared
        return
    for row, full_row, prompt_len, response_len in zip(
        stored.unbind(), full_sum_pi_squared.unbind(), PROMPT_LENS, RESPONSE_LENS, strict=True
    ):
        # Same left shift as old_log_probs: the response tokens are predicted from positions [P-1, P+R-1).
        torch.testing.assert_close(row, full_row[prompt_len - 1 : prompt_len + response_len - 1])


@pytest.mark.parametrize("adv_estimator", ["optimal_token_baseline", "tir_optimal_token_baseline"])
def test_optimal_token_baseline_receives_sum_pi_squared(monkeypatch, adv_estimator):
    fake_tq = _fake_tq_with_actor_outputs()
    monkeypatch.setattr(trainer_base, "tq", fake_tq)
    trainer = _make_trainer(calculate_sum_pi_squared=True, adv_estimator=adv_estimator)
    batch = _Batch(keys=["a", "b"], partition_id="train", extra_info={})

    trainer._compute_old_log_prob(batch, metrics={})
    trainer._compute_advantage(batch, metrics={})

    advantages = fake_tq.store["advantages"]
    assert [len(row) for row in advantages.unbind()] == RESPONSE_LENS
    assert all(torch.isfinite(row).all() for row in advantages.unbind())


@pytest.mark.parametrize("ref_in_actor", [True, False])
def test_ref_log_prob_keeps_actor_sum_pi_squared(monkeypatch, ref_in_actor):
    fake_tq = _fake_tq_with_actor_outputs()
    monkeypatch.setattr(trainer_base, "tq", fake_tq)
    trainer = _make_trainer(calculate_sum_pi_squared=True)
    trainer.ref_in_actor = ref_in_actor
    trainer.ref_policy_wg = MagicMock()
    trainer.ref_policy_wg.compute_ref_log_prob.side_effect = lambda batch: list(batch.keys)
    batch = _Batch(keys=["a", "b"], partition_id="train", extra_info={})

    trainer._compute_old_log_prob(batch, metrics={})
    trainer._compute_ref_log_prob(batch, metrics={})

    # The reference pass must not compute (and so overwrite) sum_pi_squared.
    assert batch.extra_info["calculate_sum_pi_squared"] is False


def test_bypass_mode_rejects_sum_pi_squared(monkeypatch):
    monkeypatch.setattr(trainer_base, "tq", _fake_tq_with_actor_outputs())
    trainer = _make_trainer(calculate_sum_pi_squared=True)
    trainer.config.algorithm.rollout_correction = {"bypass_mode": True}
    batch = _Batch(keys=["a", "b"], partition_id="train", extra_info={})

    with pytest.raises(ValueError, match="bypass_mode"):
        trainer._compute_old_log_prob(batch, metrics={})
