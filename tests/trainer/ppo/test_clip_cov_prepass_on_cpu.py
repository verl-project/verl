# Copyright 2026 Individual Contributor: LOGO127
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

"""Exercise prepass compatibility checks before an actor forward can run."""

from unittest.mock import Mock

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.config import RolloutCorrectionConfig
from verl.utils import tensordict_utils as tu
from verl.workers.config import ActorConfig
from verl.workers.config.actor import PolicyLossConfig
from verl.workers.utils.clip_cov import compute_global_clip_cov_mask, prepare_clip_cov_batch


@pytest.fixture(autouse=True)
def _cpu_device(monkeypatch):
    """The production helper defaults to CUDA; exercise its CPU RNG branch."""
    monkeypatch.setattr("verl.workers.utils.clip_cov.get_device_name", lambda: "cpu")


def _config(score_centering=False, config_type=dict):
    """Create the FSDP actor configuration used by the Clip-Cov callback."""
    return ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        policy_loss=PolicyLossConfig(
            loss_mode="clip_cov",
            clip_cov_ratio=0.5,
            rollout_correction=config_type(
                score_centering=score_centering, bypass_mode=True, loss_type="reinforce", rollout_is="token"
            ),
        ),
    )


@pytest.mark.parametrize("config_type", [dict, RolloutCorrectionConfig])
@pytest.mark.parametrize("config_enabled,batch_enabled", [(True, False), (False, True), (True, True)])
def test_score_centering_is_rejected_before_inference(config_enabled, batch_enabled, config_type):
    """Do not silently change the policy distribution or call a missing processor."""
    data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(data, score_centering=batch_enabled, distillation_use_topk=True)
    engine = Mock()
    engine.infer_batch.side_effect = AssertionError("unexpected inference before compatibility check")
    rng_before = torch.random.get_rng_state().clone()
    config = _config(config_enabled, config_type)
    with pytest.raises(NotImplementedError, match="Clip-Cov.*score centering"):
        prepare_clip_cov_batch(engine, data, config)
    engine.infer_batch.assert_not_called()
    engine.get_data_parallel_group.assert_not_called()
    assert tu.get(data, "score_centering") == batch_enabled
    assert tu.get(data, "distillation_use_topk") is True
    assert "clip_cov_mask" not in data
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_distillation_only_still_skips_policy_selection():
    """No policy mask is needed when the policy objective is disabled."""
    data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(data, distillation_only=True, score_centering=True)
    engine = Mock()
    prepare_clip_cov_batch(engine, data, _config(True))
    engine.infer_batch.assert_not_called()
    assert "clip_cov_mask" not in data


@pytest.mark.parametrize("explicit_disabled", [False, True])
def test_supported_prepass_preserves_metadata_rng_and_selects_mask(explicit_disabled):
    """The ordinary prepass keeps its original distribution and batch semantics."""
    advantages = torch.tensor([[8.0, -8.0], [8.0, -8.0]])
    old_log_probs = torch.zeros_like(advantages)
    response_mask = torch.ones_like(advantages, dtype=torch.bool)
    log_probs = torch.tensor([0.15, -0.15, 0.0, 0.15, -0.15, 0.0])
    data = TensorDict(
        {
            "prompts": torch.ones(2, 1, dtype=torch.long),
            "responses": torch.ones(2, 2, dtype=torch.long),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
            "advantages": advantages,
            "old_log_probs": old_log_probs,
            "response_mask": response_mask,
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(data, distillation_use_topk=True)
    if explicit_disabled:
        tu.assign_non_tensor(data, score_centering=False)
    engine = Mock()
    engine.get_data_parallel_group.return_value = None

    def infer_batch(batch, loss_function):
        assert loss_function is None
        assert tu.get(batch, "distillation_use_topk") is False
        assert not tu.get(batch, "score_centering", False)
        torch.rand(4)  # The prepass must not consume the training RNG stream.
        return {"model_output": {"log_probs": log_probs}}

    engine.infer_batch.side_effect = infer_batch
    rng_before = torch.random.get_rng_state().clone()
    prepare_clip_cov_batch(engine, data, _config())
    expected = compute_global_clip_cov_mask(
        old_log_probs,
        torch.sign(advantages) * 0.15,
        advantages,
        response_mask,
        0.5,
        1.0,
        5.0,
        0.2,
        0.2,
    )
    torch.testing.assert_close(data["clip_cov_mask"], expected)
    assert expected.sum() == 2
    assert tu.get(data, "distillation_use_topk") is True
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    engine.infer_batch.assert_called_once()
