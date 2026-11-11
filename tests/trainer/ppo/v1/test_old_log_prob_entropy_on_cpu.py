# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""CPU tests: V1 ``_compute_old_log_prob`` only computes entropy when it is used."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.v1 import trainer_base
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


class _Batch:
    def __init__(self, n: int):
        self.keys = [f"k{i}" for i in range(n)]
        self.partition_id = "train"
        self.extra_info = {}

    def __len__(self):
        return len(self.keys)


class _Fields(dict):
    def select(self, *names):
        return _Fields({name: self[name] for name in names})

    def to_padded_tensor(self):
        return dict(self)


def _stub(*, calculate_entropy: bool, entropy_coeff: float):
    stub = SimpleNamespace(
        config=OmegaConf.create(
            {
                "algorithm": {},
                "actor_rollout_ref": {
                    "actor": {
                        "calculate_entropy": calculate_entropy,
                        "entropy_coeff": entropy_coeff,
                        "loss_agg_mode": "token-mean",
                        "loss_scale_factor": None,
                    },
                    "rollout": {"temperature": 1.0, "calculate_log_probs": False},
                },
            }
        )
    )
    stub.actor_rollout_wg = MagicMock()
    stub.actor_rollout_wg.compute_log_prob.side_effect = lambda batch: batch
    stub._compute_old_log_prob = PPOTrainer._compute_old_log_prob.__get__(stub)
    return stub


@pytest.fixture
def fake_tq(monkeypatch):
    mask = torch.ones(2, 3)
    tq = MagicMock()
    tq.kv_batch_get.side_effect = lambda keys, partition_id, select_fields: _Fields(
        {name: torch.full((2, 3), 0.5) if name != "response_mask" else mask for name in select_fields}
    )
    monkeypatch.setattr(trainer_base, "tq", tq)
    monkeypatch.setattr(trainer_base, "response_from_nested", lambda tensor, mask: tensor)
    monkeypatch.setattr(trainer_base, "DataProto", lambda batch: SimpleNamespace(batch=batch))
    return tq


def test_entropy_skipped_when_unused(fake_tq):
    stub = _stub(calculate_entropy=False, entropy_coeff=0.0)
    batch, metrics = _Batch(2), {}

    stub._compute_old_log_prob(batch, metrics)

    assert batch.extra_info["calculate_entropy"] is False
    assert fake_tq.kv_batch_get.call_args.kwargs["select_fields"] == ["log_probs", "response_mask"]
    assert set(fake_tq.kv_batch_put.call_args.kwargs["fields"]) == {"old_log_probs"}
    assert "actor/entropy" not in metrics


@pytest.mark.parametrize(
    "calculate_entropy, entropy_coeff",
    [(True, 0.0), (False, 0.01)],
    ids=["calculate_entropy", "entropy_coeff"],
)
def test_entropy_computed_when_requested(fake_tq, calculate_entropy, entropy_coeff):
    stub = _stub(calculate_entropy=calculate_entropy, entropy_coeff=entropy_coeff)
    batch, metrics = _Batch(2), {}

    stub._compute_old_log_prob(batch, metrics)

    assert batch.extra_info["calculate_entropy"] is True
    assert fake_tq.kv_batch_get.call_args.kwargs["select_fields"] == ["entropy", "log_probs", "response_mask"]
    assert set(fake_tq.kv_batch_put.call_args.kwargs["fields"]) == {"old_log_probs", "entropy"}
    assert metrics["actor/entropy"] == pytest.approx(0.5)
