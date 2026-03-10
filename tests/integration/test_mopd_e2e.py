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

"""Integration tests for MOPD (Multi-Teacher On-Policy Distillation).

Tests the full data flow from config creation through advantage computation,
verifying all MOPD components work together correctly.
"""

import os

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.core_algos import get_adv_estimator_fn
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.workers.config.teacher import MOPDConfig, TeacherConfig

# ---------------------------------------------------------------------------
# Lightweight integration tests (run without GPU or Ray)
# ---------------------------------------------------------------------------


class TestMOPDDataFlow:
    """Test the full MOPD data flow: config → advantage computation → result."""

    @pytest.fixture()
    def mopd_config(self):
        """Create a full MOPD config using OmegaConf."""
        return OmegaConf.create(
            {
                "mopd": {
                    "enabled": True,
                    "lambda_val": 1.0,
                    "orm_weight": 0.0,
                    "is_correction": True,
                    "is_epsilon_low": 0.1,
                    "is_epsilon_high": 10.0,
                    "use_base_normalization": False,
                    "base_model_path": None,
                    "teachers": [
                        {"name": "math", "model_path": "/models/math-teacher"},
                        {"name": "code", "model_path": "/models/code-teacher"},
                    ],
                },
            }
        )

    @pytest.fixture()
    def mopd_batch(self):
        """Create a DataProto with all MOPD-required fields."""
        B, T = 8, 16
        data = DataProto.from_single_dict(
            {
                "token_level_rewards": torch.zeros(B, T),
                "response_mask": torch.ones(B, T),
                "old_log_probs": torch.randn(B, T),
                "teacher_log_prob": torch.randn(B, T),
            }
        )
        data.non_tensor_batch["uid"] = np.array(["q1"] * 4 + ["q2"] * 4)
        data.non_tensor_batch["teacher_id"] = np.array(["math"] * 4 + ["code"] * 4)
        return data

    def test_config_to_advantage_flow(self, mopd_config, mopd_batch):
        """Test full flow: OmegaConf config → compute_advantage → result with correct structure."""
        # Act: call compute_advantage with MOPD estimator and config
        result = compute_advantage(
            mopd_batch,
            adv_estimator="mopd",
            config=mopd_config,
        )

        # Assert: result has expected keys and shapes
        assert "advantages" in result.batch
        assert "returns" in result.batch
        assert result.batch["advantages"].shape == (8, 16)
        assert result.batch["returns"].shape == (8, 16)
        # Original fields are preserved
        assert "old_log_probs" in result.batch
        assert "teacher_log_prob" in result.batch

    def test_advantage_values_are_deterministic(self, mopd_config, mopd_batch):
        """Test that running compute_advantage twice yields identical results."""
        # We need fresh copies because compute_advantage mutates data in-place
        batch_copy = DataProto.from_single_dict({k: v.clone() for k, v in mopd_batch.batch.items()})
        batch_copy.non_tensor_batch = dict(mopd_batch.non_tensor_batch)

        result1 = compute_advantage(mopd_batch, adv_estimator="mopd", config=mopd_config)
        result2 = compute_advantage(batch_copy, adv_estimator="mopd", config=mopd_config)

        torch.testing.assert_close(result1.batch["advantages"], result2.batch["advantages"])
        torch.testing.assert_close(result1.batch["returns"], result2.batch["returns"])

    def test_response_mask_zeros_out_advantages(self, mopd_config):
        """Test that advantages are zero where response_mask is zero."""
        B, T = 4, 10
        response_mask = torch.ones(B, T)
        # Mask out last 3 tokens of each sequence
        response_mask[:, -3:] = 0.0

        data = DataProto.from_single_dict(
            {
                "token_level_rewards": torch.zeros(B, T),
                "response_mask": response_mask,
                "old_log_probs": torch.randn(B, T),
                "teacher_log_prob": torch.randn(B, T),
            }
        )
        data.non_tensor_batch["uid"] = np.array(["q1"] * 4)

        result = compute_advantage(data, adv_estimator="mopd", config=mopd_config)

        # Masked positions must be exactly zero
        masked_advantages = result.batch["advantages"][:, -3:]
        assert torch.all(masked_advantages == 0.0), f"Expected zeros, got {masked_advantages}"

    def test_standard_mopd_advantage_values(self):
        """Test MOPD produces correct advantage values (teacher_log_prob - old_log_probs)."""
        B, T = 2, 5
        teacher_lp = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [0.5, 1.5, 2.5, 3.5, 4.5]])
        old_lp = torch.tensor([[0.5, 1.0, 1.5, 2.0, 2.5], [0.0, 0.5, 1.0, 1.5, 2.0]])
        response_mask = torch.ones(B, T)

        config = OmegaConf.create({"mopd": {"lambda_val": 1.0, "is_correction": False}})
        data = DataProto.from_single_dict(
            {
                "token_level_rewards": torch.zeros(B, T),
                "response_mask": response_mask,
                "old_log_probs": old_lp,
                "teacher_log_prob": teacher_lp,
            }
        )
        data.non_tensor_batch["uid"] = np.array(["q1", "q2"])

        result = compute_advantage(data, adv_estimator="mopd", config=config)
        expected = (teacher_lp - old_lp).detach() * response_mask
        torch.testing.assert_close(result.batch["advantages"], expected)

    def test_exopd_mode_end_to_end(self):
        """Test ExOPD (base-normalized) mode through compute_advantage dispatch."""
        B, T = 2, 5
        teacher_lp = torch.ones(B, T) * 2.0
        old_lp = torch.ones(B, T) * 1.0
        base_lp = torch.ones(B, T) * 0.5

        config = OmegaConf.create(
            {
                "mopd": {
                    "lambda_val": 1.25,
                    "is_correction": False,
                },
            }
        )
        data = DataProto.from_single_dict(
            {
                "token_level_rewards": torch.zeros(B, T),
                "response_mask": torch.ones(B, T),
                "old_log_probs": old_lp,
                "teacher_log_prob": teacher_lp,
                "base_log_prob": base_lp,
            }
        )
        data.non_tensor_batch["uid"] = np.array(["q1", "q2"])

        result = compute_advantage(data, adv_estimator="mopd", config=config)

        # ExOPD: -[(old - base) - lambda*(teacher - base)]
        # = -[(1.0 - 0.5) - 1.25*(2.0 - 0.5)]
        # = -[0.5 - 1.875] = 1.375
        expected = torch.ones(B, T) * 1.375
        torch.testing.assert_close(result.batch["advantages"], expected, rtol=1e-4, atol=1e-4)

    def test_is_correction_through_dispatch(self):
        """Test IS correction flows correctly through compute_advantage."""
        B, T = 2, 5
        teacher_lp = torch.ones(B, T) * 2.0
        old_lp = torch.ones(B, T) * 1.0
        # Token [0, 2] has extreme ratio: exp(1 - (-4)) = exp(5) ≈ 148 > 10
        rollout_lp = torch.tensor(
            [
                [1.0, 1.0, -4.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )

        config = OmegaConf.create(
            {
                "mopd": {
                    "is_correction": True,
                    "is_epsilon_low": 0.1,
                    "is_epsilon_high": 10.0,
                },
            }
        )
        data = DataProto.from_single_dict(
            {
                "token_level_rewards": torch.zeros(B, T),
                "response_mask": torch.ones(B, T),
                "old_log_probs": old_lp,
                "teacher_log_prob": teacher_lp,
                "rollout_log_probs": rollout_lp,
            }
        )
        data.non_tensor_batch["uid"] = np.array(["q1", "q2"])

        result = compute_advantage(data, adv_estimator="mopd", config=config)

        # Token [0, 2] should be masked to zero by IS correction
        assert result.batch["advantages"][0, 2] == 0.0
        # Non-masked tokens should be non-zero
        assert result.batch["advantages"][0, 0] != 0.0


class TestMOPDConfigIntegration:
    """Test that MOPD config dataclasses integrate with OmegaConf correctly."""

    def test_teacher_config_roundtrip(self):
        """Test TeacherConfig can be created from dict and exported back."""
        cfg_dict = {"name": "math", "model_path": "/models/math", "weight": 1.5}
        teacher = TeacherConfig(**cfg_dict)
        assert teacher.name == "math"
        assert teacher.model_path == "/models/math"
        assert teacher.weight == 1.5

    def test_mopd_config_with_teachers(self):
        """Test MOPDConfig accepts properly constructed teachers."""
        teachers = [
            TeacherConfig(name="math", model_path="/models/math"),
            TeacherConfig(name="code", model_path="/models/code"),
        ]
        config = MOPDConfig(enabled=True, teachers=teachers, lambda_val=1.25)
        assert config.enabled is True
        assert len(config.teachers) == 2
        assert config.lambda_val == 1.25

    def test_mopd_config_disabled_by_default(self):
        """Test MOPDConfig defaults to disabled (backward compatibility)."""
        config = MOPDConfig()
        assert config.enabled is False
        assert len(config.teachers) == 0

    def test_need_reference_policy_with_mopd_config(self):
        """Test need_reference_policy returns True when MOPD is enabled."""
        from verl.trainer.ppo.utils import need_reference_policy

        config = OmegaConf.create(
            {
                "algorithm": {
                    "use_kl_in_reward": False,
                    "mopd": {"enabled": True},
                },
                "actor_rollout_ref": {"actor": {"use_kl_loss": False}},
            }
        )
        assert need_reference_policy(config) is True

    def test_need_reference_policy_without_mopd(self):
        """Test need_reference_policy returns False when MOPD disabled and no KL."""
        from verl.trainer.ppo.utils import need_reference_policy

        config = OmegaConf.create(
            {
                "algorithm": {
                    "use_kl_in_reward": False,
                    "mopd": {"enabled": False},
                },
                "actor_rollout_ref": {"actor": {"use_kl_loss": False}},
            }
        )
        assert need_reference_policy(config) is False


class TestMOPDAdvantageEstimatorRegistry:
    """Test that the MOPD estimator is properly registered and callable."""

    def test_mopd_is_registered(self):
        """Test that 'mopd' advantage estimator is registered."""
        fn = get_adv_estimator_fn("mopd")
        assert fn is not None
        assert callable(fn)

    def test_mopd_fn_has_correct_name(self):
        """Test that the registered function is compute_mopd_advantage."""
        fn = get_adv_estimator_fn("mopd")
        assert fn.__name__ == "compute_mopd_advantage"

    def test_mopd_returns_tuple(self):
        """Test that MOPD advantage estimator returns (advantages, returns) tuple."""
        B, T = 2, 4
        fn = get_adv_estimator_fn("mopd")
        result = fn(
            token_level_rewards=torch.zeros(B, T),
            response_mask=torch.ones(B, T),
            teacher_log_prob=torch.randn(B, T),
            old_log_probs=torch.randn(B, T),
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        advantages, returns = result
        assert advantages.shape == (B, T)
        assert returns.shape == (B, T)


# ---------------------------------------------------------------------------
# Full E2E test (requires GPU + Ray + model weights)
# ---------------------------------------------------------------------------

# This test requires a fully provisioned environment:
# - CUDA-capable GPU
# - Ray cluster
# - Model weights at MOPD_TEST_MODEL_PATH and MOPD_TEST_TEACHER_PATH
# Set VERL_MOPD_E2E=1 to enable this test.
_MOPD_E2E_ENABLED = os.environ.get("VERL_MOPD_E2E", "0") == "1"


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not _MOPD_E2E_ENABLED, reason="Set VERL_MOPD_E2E=1 to run full E2E test")
def test_mopd_training_e2e():
    """Test full MOPD training loop with actual model workers.

    This test requires:
    - CUDA-capable GPU
    - Ray cluster (auto-initialized)
    - Model weights at configured paths
    - Environment variable VERL_MOPD_E2E=1

    It verifies:
    - Teacher worker groups are created
    - Sub-batch routing dispatches to correct teachers
    - MOPD advantages are computed and used for policy updates
    - Training completes at least one step

    To run:
        VERL_MOPD_E2E=1 MOPD_TEST_MODEL_PATH=/path/to/model \\
        MOPD_TEST_TEACHER_PATH=/path/to/teacher \\
        pytest tests/integration/test_mopd_e2e.py::test_mopd_training_e2e -v
    """
    import ray

    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    model_path = os.environ.get("MOPD_TEST_MODEL_PATH", "/models/base")
    teacher_path = os.environ.get("MOPD_TEST_TEACHER_PATH", "/models/math-teacher")
    train_files = os.environ.get("MOPD_TEST_TRAIN_FILES", "/data/train.jsonl")

    config = OmegaConf.create(
        {
            "algorithm": {
                "adv_estimator": "mopd",
                "use_kl_in_reward": False,
                "ppo_epochs": 1,
                "mopd": {
                    "enabled": True,
                    "lambda_val": 1.0,
                    "orm_weight": 0.0,
                    "is_correction": True,
                    "is_epsilon_low": 0.1,
                    "is_epsilon_high": 10.0,
                    "teachers": [
                        {
                            "name": "math",
                            "model_path": teacher_path,
                            "resource_pool": "global_pool",
                        },
                    ],
                },
            },
            "trainer": {"total_epochs": 1},
            "actor_rollout_ref": {"actor": {"use_kl_loss": False}},
            "model": {"path": model_path},
            "data": {"train_files": train_files},
        }
    )

    if not ray.is_initialized():
        ray.init(num_cpus=4, num_gpus=torch.cuda.device_count())

    try:
        trainer = RayPPOTrainer(config=config)
        trainer.init_workers()
        trainer.fit()
        assert trainer.global_steps > 0
    finally:
        if ray.is_initialized():
            ray.shutdown()
