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

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode, apply_bypass_mode_to_tq_batch
from verl.utils import config as config_utils
from verl.utils.config import _validate_rollout_log_probs_for_bypass, validate_config


class _FakeActorConfig:
    def validate(self, *_args, **_kwargs):
        return None


def _bypass_config(calculate_log_probs: bool):
    return OmegaConf.create(
        {
            "algorithm": {"rollout_correction": {"bypass_mode": True}},
            "actor_rollout_ref": {"rollout": {"calculate_log_probs": calculate_log_probs}},
        }
    )


def test_apply_bypass_mode_sets_old_log_probs_from_rollout_log_probs():
    rollout_log_probs = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])
    batch = DataProto.from_dict(tensors={"rollout_log_probs": rollout_log_probs})
    policy_loss_config = OmegaConf.create({"loss_mode": "vanilla"})
    rollout_corr_config = OmegaConf.create({"bypass_mode": True, "loss_type": "ppo_clip"})

    apply_bypass_mode(
        batch=batch,
        rollout_corr_config=rollout_corr_config,
        policy_loss_config=policy_loss_config,
    )

    assert torch.equal(batch.batch["old_log_probs"], rollout_log_probs)
    assert policy_loss_config.loss_mode == "bypass_mode"
    assert policy_loss_config.rollout_correction.bypass_mode is True
    assert policy_loss_config.rollout_correction.loss_type == "ppo_clip"


def test_apply_bypass_mode_requires_rollout_log_probs():
    batch = DataProto.from_dict(tensors={"response_mask": torch.ones(2, 2)})
    policy_loss_config = OmegaConf.create({"loss_mode": "vanilla"})

    with pytest.raises(ValueError, match="calculate_log_probs=true"):
        apply_bypass_mode(
            batch=batch,
            rollout_corr_config=OmegaConf.create({"bypass_mode": True}),
            policy_loss_config=policy_loss_config,
        )


def test_tq_bypass_returns_updated_kv_batch_meta():
    rollout_log_probs = torch.tensor([[-1.0, -2.0]])
    input_batch = SimpleNamespace(keys=["sample-0"], partition_id="train")
    updated_batch = SimpleNamespace(
        keys=input_batch.keys,
        partition_id=input_batch.partition_id,
        fields=["rollout_log_probs", "old_log_probs"],
    )
    put_calls = []

    class FakeTQ:
        @staticmethod
        def kv_batch_get(keys, partition_id, select_fields):
            assert keys == input_batch.keys
            assert partition_id == input_batch.partition_id
            assert select_fields == ["rollout_log_probs"]
            return TensorDict({"rollout_log_probs": rollout_log_probs.clone()}, batch_size=[1])

        @staticmethod
        def kv_batch_put(keys, partition_id, fields):
            assert keys == input_batch.keys
            assert partition_id == input_batch.partition_id
            assert "rollout_log_probs" not in fields
            assert torch.equal(fields["old_log_probs"], rollout_log_probs)
            put_calls.append(fields.clone())
            return updated_batch

    result = apply_bypass_mode_to_tq_batch(input_batch, FakeTQ)

    assert result is updated_batch
    assert len(put_calls) == 1


def test_validate_bypass_mode_requires_rollout_log_probs():
    with pytest.raises(ValueError, match="actor_rollout_ref.rollout.calculate_log_probs=true"):
        _validate_rollout_log_probs_for_bypass(_bypass_config(calculate_log_probs=False))


def test_validate_bypass_mode_accepts_rollout_log_probs():
    _validate_rollout_log_probs_for_bypass(_bypass_config(calculate_log_probs=True))


def test_validate_bypass_mode_ignores_decoupled_mode_without_rollout_log_probs():
    config = OmegaConf.create(
        {
            "algorithm": {"rollout_correction": {"bypass_mode": False}},
            "actor_rollout_ref": {"rollout": {"calculate_log_probs": False}},
        }
    )

    _validate_rollout_log_probs_for_bypass(config)


def test_validate_config_rejects_bypass_without_rollout_log_probs(monkeypatch):
    config = OmegaConf.create(
        {
            "trainer": {"n_gpus_per_node": 1, "nnodes": 1},
            "data": {"train_batch_size": 1},
            "algorithm": {
                "rollout_correction": {"bypass_mode": True},
                "use_kl_in_reward": False,
            },
            "actor_rollout_ref": {
                "actor": {"use_dynamic_bsz": True, "use_kl_loss": False},
                "rollout": {
                    "calculate_log_probs": False,
                    "val_kwargs": {"do_sample": False},
                    "name": "vllm",
                },
                "model": {},
            },
        }
    )
    monkeypatch.setattr(config_utils, "omega_conf_to_dataclass", lambda _config: _FakeActorConfig())

    with pytest.raises(ValueError, match="actor_rollout_ref.rollout.calculate_log_probs=true"):
        validate_config(config=config, use_reference_policy=False, use_critic=False)


def test_validate_config_accepts_bypass_with_rollout_log_probs(monkeypatch):
    config = OmegaConf.create(
        {
            "trainer": {"n_gpus_per_node": 1, "nnodes": 1},
            "data": {"train_batch_size": 1},
            "algorithm": {
                "rollout_correction": {"bypass_mode": True},
                "use_kl_in_reward": False,
            },
            "actor_rollout_ref": {
                "actor": {"use_dynamic_bsz": True, "use_kl_loss": False},
                "rollout": {
                    "calculate_log_probs": True,
                    "val_kwargs": {"do_sample": False},
                    "name": "vllm",
                },
                "model": {},
            },
        }
    )
    monkeypatch.setattr(config_utils, "omega_conf_to_dataclass", lambda _config: _FakeActorConfig())

    validate_config(config=config, use_reference_policy=False, use_critic=False)
