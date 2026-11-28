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

import pytest
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.v1 import trainer_base
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


def _captured_select_fields(use_reference_policy: bool, monkeypatch) -> list[str]:
    """Run _compute_advantage until the kv_batch_get call and return its select_fields."""
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.use_reference_policy = use_reference_policy
    trainer.use_critic = False

    captured = {}

    def fake_kv_batch_get(keys=None, partition_id=None, select_fields=None):
        captured["fields"] = select_fields
        raise RuntimeError("stop-before-advantage-math")

    monkeypatch.setattr(trainer_base.tq, "kv_batch_get", fake_kv_batch_get)
    batch = KVBatchMeta(partition_id="train", keys=[], tags=[])
    with pytest.raises(RuntimeError, match="stop-before-advantage-math"):
        trainer._compute_advantage(batch, metrics={})
    return captured["fields"]


def test_compute_advantage_skips_ref_log_prob_without_ref_policy(monkeypatch):
    # ref_log_prob is produced only when a reference policy is enabled
    # (algorithm.use_kl_in_reward or actor_rollout_ref.actor.use_kl_loss);
    # requesting it unconditionally crashes every no-ref recipe on step 1 (#8891).
    fields = _captured_select_fields(use_reference_policy=False, monkeypatch=monkeypatch)
    assert "ref_log_prob" not in fields


def test_compute_advantage_requests_ref_log_prob_with_ref_policy(monkeypatch):
    fields = _captured_select_fields(use_reference_policy=True, monkeypatch=monkeypatch)
    assert "ref_log_prob" in fields
