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

import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import patch

from omegaconf import OmegaConf

from verl.base_config import BaseConfig
from verl.utils import omega_conf_to_dataclass
from verl.utils.config import validate_config


@dataclass
class TestDataclass(BaseConfig):
    hidden_size: int = 0
    activation: str = "relu"


@dataclass
class TestTrainConfig(BaseConfig):
    batch_size: int = 0
    model: TestDataclass = field(default_factory=TestDataclass)
    override_config: dict = field(default_factory=dict)


_cfg_str = """train_config:
  _target_: tests.utils.test_config_on_cpu.TestTrainConfig
  batch_size: 32
  model:
    hidden_size: 768
    activation: relu
  override_config: {}"""


class TestConfigOnCPU(unittest.TestCase):
    """Test cases for configuration utilities on CPU.

    Test Plan:
    1. Test basic OmegaConf to dataclass conversion for simple nested structures
    2. Test nested OmegaConf to dataclass conversion for complex hierarchical configurations
    3. Verify all configuration values are correctly converted and accessible
    """

    def setUp(self):
        self.config = OmegaConf.create(_cfg_str)

    def test_omega_conf_to_dataclass(self):
        sub_cfg = self.config.train_config.model
        cfg = omega_conf_to_dataclass(sub_cfg, TestDataclass)
        self.assertEqual(cfg.hidden_size, 768)
        self.assertEqual(cfg.activation, "relu")
        assert isinstance(cfg, TestDataclass)

    def test_nested_omega_conf_to_dataclass(self):
        cfg = omega_conf_to_dataclass(self.config.train_config, TestTrainConfig)
        self.assertEqual(cfg.batch_size, 32)
        self.assertEqual(cfg.model.hidden_size, 768)
        self.assertEqual(cfg.model.activation, "relu")
        assert isinstance(cfg, TestTrainConfig)
        assert isinstance(cfg.model, TestDataclass)


class TestPrintCfgCommand(unittest.TestCase):
    """Test suite for the print_cfg.py command-line tool."""

    def test_command_with_override(self):
        """Test that the command runs without error when overriding config values."""
        import subprocess

        # Run the command
        result = subprocess.run(
            ["python3", "scripts/print_cfg.py"],
            capture_output=True,
            text=True,
        )

        # Verify the command exited successfully
        self.assertEqual(result.returncode, 0, f"Command failed with stderr: {result.stderr}")

        # Verify the output contains expected config information
        self.assertIn("critic", result.stdout)
        self.assertIn("profiler", result.stdout)


class TestStaticMicroBatchValidation(unittest.TestCase):
    @staticmethod
    def _config(train_batch_size, sp_size=1):
        return OmegaConf.create(
            {
                "trainer": {"n_gpus_per_node": 8, "nnodes": 1},
                "data": {"train_batch_size": train_batch_size},
                "actor_rollout_ref": {
                    "actor": {
                        "strategy": "fsdp",
                        "use_dynamic_bsz": False,
                        "ppo_micro_batch_size": None,
                        "ppo_micro_batch_size_per_gpu": 3,
                        "ulysses_sequence_parallel_size": sp_size,
                        "fsdp_config": {"ulysses_sequence_parallel_size": sp_size},
                        "use_kl_loss": False,
                    },
                    "rollout": {
                        "n": 1,
                        "name": "sglang",
                        "log_prob_micro_batch_size": None,
                        "log_prob_micro_batch_size_per_gpu": 1,
                        "val_kwargs": {"do_sample": False},
                    },
                    "ref": {
                        "log_prob_micro_batch_size": None,
                        "log_prob_micro_batch_size_per_gpu": 1,
                    },
                    "model": {"lora": {}, "lora_rank": 0},
                },
                "algorithm": {"use_kl_in_reward": False},
            }
        )

    @patch("verl.utils.config.omega_conf_to_dataclass")
    def test_rejects_batch_smaller_than_per_gpu_micro_batch_product(self, to_dataclass):
        to_dataclass.return_value = SimpleNamespace(validate=lambda *args: None)

        with self.assertRaisesRegex(AssertionError, r"minimal possible batch size \(24\)"):
            validate_config(self._config(train_batch_size=8), use_reference_policy=False, use_critic=False)

    @patch("verl.utils.config.omega_conf_to_dataclass")
    def test_accepts_batch_divisible_by_per_gpu_micro_batch_product(self, to_dataclass):
        to_dataclass.return_value = SimpleNamespace(validate=lambda *args: None)

        validate_config(self._config(train_batch_size=24), use_reference_policy=False, use_critic=False)

    @patch("verl.utils.config.omega_conf_to_dataclass")
    def test_accounts_for_ulysses_sequence_parallel_size(self, to_dataclass):
        to_dataclass.return_value = SimpleNamespace(validate=lambda *args: None)

        validate_config(self._config(train_batch_size=12, sp_size=2), use_reference_policy=False, use_critic=False)


if __name__ == "__main__":
    unittest.main()
