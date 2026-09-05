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
"""CPU tests for V1 ``trainer.balance_batch`` gating.

Upsample (mini-batch / DP LCM) must always run. The seqlen reorder is the
documented optional load-balance and must honor ``trainer.balance_batch``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from omegaconf import OmegaConf

from verl.trainer.ppo.v1 import trainer_base
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


class _FakeBatch:
    def __init__(self):
        self.tags = [{"seq_len": 4}, {"seq_len": 8}]
        self.reordered = None

    def reorder(self, idx):
        self.reordered = list(idx)


class _FakeWorkerGroup:
    def __init__(self):
        self._dispatch_info = {"actor": [0, 1]}


def _stub(*, balance_batch: bool):
    stub = SimpleNamespace(
        config=OmegaConf.create(
            {
                "trainer": {
                    "balance_batch": balance_batch,
                    "critic_warmup": 0,
                },
                "critic": {"ppo_mini_batch_size": 1},
                "actor_rollout_ref": {
                    "actor": {"ppo_mini_batch_size": 1},
                    "rollout": {"n": 1},
                },
            }
        ),
        actor_rollout_wg=_FakeWorkerGroup(),
        tokenizer=SimpleNamespace(eos_token_id=0),
        use_critic=False,
        global_steps=1,
    )
    stub._get_required_batch_multiple = PPOTrainer._get_required_batch_multiple.__get__(stub)
    stub._balance_batch = PPOTrainer._balance_batch.__get__(stub)
    return stub


def test_balance_batch_false_upsamples_but_does_not_reorder(monkeypatch):
    stub = _stub(balance_batch=False)
    batch = _FakeBatch()
    upsample = MagicMock(side_effect=lambda b, *_args, **_kwargs: b)
    reorder_partitions = MagicMock()
    monkeypatch.setattr(trainer_base, "upsample_batch_to_divisible_size", upsample)
    monkeypatch.setattr(trainer_base, "get_seqlen_balanced_partitions", reorder_partitions)

    out = stub._balance_batch(batch, metrics={})

    upsample.assert_called_once()
    reorder_partitions.assert_not_called()
    assert out is batch
    assert batch.reordered is None


def test_balance_batch_true_reorders(monkeypatch):
    stub = _stub(balance_batch=True)
    batch = _FakeBatch()
    upsample = MagicMock(side_effect=lambda b, *_args, **_kwargs: b)
    monkeypatch.setattr(trainer_base, "upsample_batch_to_divisible_size", upsample)
    monkeypatch.setattr(trainer_base, "calculate_workload", lambda _seq: [1, 2])
    monkeypatch.setattr(trainer_base, "get_seqlen_balanced_partitions", lambda *_args, **_kwargs: [[0], [1]])
    monkeypatch.setattr(trainer_base, "log_seqlen_unbalance", lambda **_kwargs: {"global_seqlen/minmax_diff": 0})

    out = stub._balance_batch(batch, metrics={})

    upsample.assert_called_once()
    assert out is batch
    assert batch.reordered == [0, 1]
