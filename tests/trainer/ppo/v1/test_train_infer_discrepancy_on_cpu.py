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

"""CPU-only tests for ``PPOTrainer._log_train_infer_discrepancy``.

Bypass mode sets ``old_log_probs = rollout_log_probs``, so the gap between the training and
rollout engines is only observable through an extra forward pass. These tests pin down that the
pass is metrics-only -- it must not write ``old_log_probs`` back -- and that the partial /
non-partial split survives, since the weight versions live in ``batch.tags`` under V1.
"""

import math

import pytest
import torch
from omegaconf import OmegaConf

import verl.utils.tensordict_utils as tu
from verl.trainer.ppo.v1 import trainer_base
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


class _FakeWorkerGroup:
    def __init__(self):
        self.calls = []

    def compute_log_prob(self, batch):
        self.calls.append(batch)
        return batch


def _nested(rows):
    return torch.nested.as_nested_tensor([torch.tensor(row) for row in rows], layout=torch.jagged)


def _make_stub(tags):
    stub = type("Stub", (), {})()
    stub.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"temperature": 1.0}}})
    stub.actor_rollout_wg = _FakeWorkerGroup()
    stub._log_train_infer_discrepancy = PPOTrainer._log_train_infer_discrepancy.__get__(stub)
    batch = type("Batch", (), {})()
    batch.keys = ["a", "b"]
    batch.partition_id = "train"
    batch.extra_info = {}
    batch.tags = tags
    return stub, batch


def _patch_queue(monkeypatch, log_probs, rollout_log_probs, response_mask):
    gets = []

    def fake_get(**kwargs):
        gets.append(kwargs)
        return tu.get_tensordict(
            {
                # log_probs covers prompt + response; response_from_nested slices the response off.
                "log_probs": _nested(log_probs),
                "rollout_log_probs": _nested(rollout_log_probs),
                "response_mask": _nested(response_mask),
            }
        )

    monkeypatch.setattr(trainer_base.tq, "kv_batch_get", fake_get)
    puts = []
    monkeypatch.setattr(trainer_base.tq, "kv_batch_put", lambda **kwargs: puts.append(kwargs))
    return gets, puts


def test_discrepancy_pass_is_metrics_only_and_splits_partial_trajectories(monkeypatch):
    # Row 0 spans two weight versions, row 1 does not.
    tags = [
        {"min_global_steps": 3, "max_global_steps": 5},
        {"min_global_steps": 7, "max_global_steps": 7},
    ]
    stub, batch = _make_stub(tags)
    # Two response tokens per row; the actor is off by log(4) on the partial row only.
    gets, puts = _patch_queue(
        monkeypatch,
        # response_from_nested left-shifts by one, so the response tokens are the first two.
        log_probs=[[math.log(4.0), math.log(4.0), 0.0], [0.0, 0.0, 0.0]],
        rollout_log_probs=[[0.0, 0.0], [0.0, 0.0]],
        response_mask=[[1, 1], [1, 1]],
    )

    metrics = stub._log_train_infer_discrepancy(batch)

    # The loss anchor must be untouched: nothing is written back to the queue.
    assert puts == []
    assert len(stub.actor_rollout_wg.calls) == 1
    assert batch.extra_info["compute_loss"] is False
    assert gets[0]["select_fields"] == ["log_probs", "rollout_log_probs", "response_mask"]

    k3 = 4.0 - 1 - math.log(4.0)
    assert metrics["training/train_infer_kl_k3_partial"] == pytest.approx(k3, rel=1e-5)
    assert metrics["training/train_infer_kl_k3_nonpartial"] == pytest.approx(0.0)
    assert metrics["training/train_infer_tokens_sum_partial"] == 2
    assert metrics["training/train_infer_tokens_sum_nonpartial"] == 2


def test_discrepancy_pass_without_weight_versions_reports_no_split(monkeypatch):
    # Rollout backends that do not report versions must not be reported as single-version.
    stub, batch = _make_stub([{}, {}])
    _patch_queue(
        monkeypatch,
        log_probs=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        rollout_log_probs=[[0.0, 0.0], [0.0, 0.0]],
        response_mask=[[1, 1], [1, 1]],
    )

    metrics = stub._log_train_infer_discrepancy(batch)

    assert "training/train_infer_kl_k3" in metrics
    assert not [key for key in metrics if key.endswith(("_partial", "_nonpartial"))]
