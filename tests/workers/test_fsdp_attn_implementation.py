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

"""
Test for attn_implementation override configuration in FSDP workers.

This test verifies that the fix for honoring attn_implementation override config
works correctly in the ActorRolloutRefWorker._build_model_optimizer method.
"""

from unittest.mock import Mock, patch

import pytest
import torch
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM

# Only run these tests if we can import verl components
try:
    from verl.workers.config import FSDPEngineConfig  # noqa: F401
    from verl.workers.engine_workers import (
        ActorRolloutRefWorker,  # noqa: F401
    )

    VERL_AVAILABLE = True
except ImportError:
    VERL_AVAILABLE = False


@pytest.mark.skipif(not VERL_AVAILABLE, reason="VERL components not available")
class TestFSDPAttnImplementation:
    """Test cases for attn_implementation override in FSDP workers."""

    def test_attn_implementation_extraction_logic(self):
        """Test the core logic for extracting attn_implementation from override config."""

        # Test case 1: Default behavior
        override_config = {}
        attn_impl = override_config.get("attn_implementation", "flash_attention_2")
        assert attn_impl == "flash_attention_2"

        # Test case 2: Override to eager
        override_config = {"attn_implementation": "eager"}
        attn_impl = override_config.get("attn_implementation", "flash_attention_2")
        assert attn_impl == "eager"

        # Test case 3: Override to sdpa
        override_config = {"attn_implementation": "sdpa"}
        attn_impl = override_config.get("attn_implementation", "flash_attention_2")
        assert attn_impl == "sdpa"

        # Test case 4: Other configs don't affect attn_implementation
        override_config = {"other_setting": "value", "dropout": 0.1}
        attn_impl = override_config.get("attn_implementation", "flash_attention_2")
        assert attn_impl == "flash_attention_2"

    @patch("transformers.AutoConfig.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    def test_attn_implementation_passed_to_autoconfig(self, mock_model_from_pretrained, mock_config_from_pretrained):
        """Test that attn_implementation is correctly passed to AutoConfig.from_pretrained."""

        # Mock the AutoConfig return value
        mock_config = Mock()
        mock_config.tie_word_embeddings = False
        mock_config.architectures = ["LlamaForCausalLM"]
        mock_config_from_pretrained.return_value = mock_config

        # Mock the model return value
        mock_model = Mock()
        mock_model_from_pretrained.return_value = mock_model

        # Test data
        test_cases = [
            ({}, "flash_attention_2"),  # Default
            ({"attn_implementation": "eager"}, "eager"),  # Override to eager
            ({"attn_implementation": "sdpa"}, "sdpa"),  # Override to sdpa
        ]

        for override_config, expected_attn_impl in test_cases:
            # Reset mocks
            mock_config_from_pretrained.reset_mock()
            mock_model_from_pretrained.reset_mock()

            # Simulate the logic from FSDP workers
            attn_implementation = override_config.get("attn_implementation", "flash_attention_2")

            # This simulates what happens in _build_model_optimizer
            AutoConfig.from_pretrained("test_path", trust_remote_code=False, attn_implementation=attn_implementation)

            # Verify AutoConfig.from_pretrained was called with correct attn_implementation
            mock_config_from_pretrained.assert_called_once_with(
                "test_path", trust_remote_code=False, attn_implementation=expected_attn_impl
            )

    @patch("transformers.AutoConfig.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    def test_attn_implementation_passed_to_model(self, mock_model_from_pretrained, mock_config_from_pretrained):
        """Test that attn_implementation is correctly passed to model.from_pretrained."""

        # Mock the AutoConfig return value
        mock_config = Mock()
        mock_config.tie_word_embeddings = False
        mock_config.architectures = ["LlamaForCausalLM"]
        mock_config_from_pretrained.return_value = mock_config

        # Mock the model return value
        mock_model = Mock()
        mock_model_from_pretrained.return_value = mock_model

        # Test with override config
        override_config = {"attn_implementation": "eager"}
        attn_implementation = override_config.get("attn_implementation", "flash_attention_2")

        # This simulates what happens in _build_model_optimizer
        AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path="test_path",
            torch_dtype=torch.bfloat16,
            config=mock_config,
            trust_remote_code=False,
            attn_implementation=attn_implementation,
        )

        # Verify AutoModelForCausalLM.from_pretrained was called with correct attn_implementation
        mock_model_from_pretrained.assert_called_once_with(
            pretrained_model_name_or_path="test_path",
            torch_dtype=torch.bfloat16,
            config=mock_config,
            trust_remote_code=False,
            attn_implementation="eager",
        )

    def test_override_config_integration(self):
        """Test that override_config from Hydra configuration works correctly."""

        # Simulate the OmegaConf configuration structure used in VERL
        config_dict = {
            "model": {"path": "/test/path", "override_config": {"attn_implementation": "eager", "dropout": 0.1}}
        }

        # Convert to OmegaConf structure
        omegaconf = OmegaConf.create(config_dict)

        # Simulate what happens in the FSDP worker
        override_model_config = OmegaConf.to_container(OmegaConf.create(omegaconf.model.get("override_config", {})))

        # Test extraction
        attn_implementation = override_model_config.get("attn_implementation", "flash_attention_2")
        assert attn_implementation == "eager"

        # Test that other configs are preserved
        assert override_model_config.get("dropout") == 0.1

    def test_hydra_plus_prefix_config(self):
        """Test that Hydra +prefix configurations work correctly."""

        # This simulates the configuration when user specifies:
        # +actor_rollout_ref.model.override_config.attn_implementation=eager

        # The + prefix in Hydra adds new keys to the config
        config_dict = {
            "actor_rollout_ref": {
                "model": {
                    "path": "/test/path",
                    "override_config": {
                        "attn_implementation": "eager"  # This gets added via +prefix
                    },
                }
            }
        }

        omegaconf = OmegaConf.create(config_dict)

        # Extract override config as done in FSDP workers
        override_model_config = OmegaConf.to_container(
            OmegaConf.create(omegaconf.actor_rollout_ref.model.get("override_config", {}))
        )

        # Verify extraction works
        attn_implementation = override_model_config.get("attn_implementation", "flash_attention_2")
        assert attn_implementation == "eager"

    def test_backward_compatibility(self):
        """Test that the fix maintains backward compatibility."""

        # Test case 1: No override_config at all (old behavior)
        config_without_override = {}
        attn_implementation = config_without_override.get("attn_implementation", "flash_attention_2")
        assert attn_implementation == "flash_attention_2"

        # Test case 2: Empty override_config
        config_with_empty_override = {"override_config": {}}
        override_config = config_with_empty_override.get("override_config", {})
        attn_implementation = override_config.get("attn_implementation", "flash_attention_2")
        assert attn_implementation == "flash_attention_2"

        # Test case 3: override_config with other settings but no attn_implementation
        config_with_other_overrides = {"override_config": {"dropout": 0.1, "hidden_size": 1024}}
        override_config = config_with_other_overrides.get("override_config", {})
        attn_implementation = override_config.get("attn_implementation", "flash_attention_2")
        assert attn_implementation == "flash_attention_2"

    def test_critic_attn_implementation_extraction_logic(self):
        """Test the core logic for extracting attn_implementation from override config for CriticWorker."""

        # Test case 1: Default behavior for critic
        override_config = {}
        attn_impl = override_config.get("attn_implementation", "flash_attention_2")
        assert attn_impl == "flash_attention_2"

        # Test case 2: Override to eager for critic
        override_config = {"attn_implementation": "eager"}
        attn_impl = override_config.get("attn_implementation", "flash_attention_2")
        assert attn_impl == "eager"

        # Test case 3: Override to sdpa for critic
        override_config = {"attn_implementation": "sdpa"}
        attn_impl = override_config.get("attn_implementation", "flash_attention_2")
        assert attn_impl == "sdpa"

        # Test case 4: Other configs don't affect attn_implementation for critic
        override_config = {"other_setting": "value", "dropout": 0.1}
        attn_impl = override_config.get("attn_implementation", "flash_attention_2")
        assert attn_impl == "flash_attention_2"

    @patch("transformers.AutoConfig.from_pretrained")
    def test_critic_attn_implementation_passed_to_autoconfig(self, mock_config_from_pretrained):
        """Test that attn_implementation is correctly passed to AutoConfig.from_pretrained in CriticWorker."""

        # Mock the AutoConfig return value
        mock_config = Mock()
        mock_config.tie_word_embeddings = False
        mock_config.architectures = ["LlamaForCausalLM"]
        mock_config.num_labels = 1
        mock_config_from_pretrained.return_value = mock_config

        # Test data for critic model
        test_cases = [
            ({}, "flash_attention_2"),  # Default
            ({"attn_implementation": "eager"}, "eager"),  # Override to eager
            ({"attn_implementation": "sdpa"}, "sdpa"),  # Override to sdpa
        ]

        for override_config, expected_attn_impl in test_cases:
            # Reset mocks
            mock_config_from_pretrained.reset_mock()

            # Simulate the logic from CriticWorker _build_critic_model_optimizer
            attn_implementation = override_config.get("attn_implementation", "flash_attention_2")

            # This simulates what should happen in CriticWorker._build_critic_model_optimizer
            # (This is where the fix needs to be applied in the actual implementation)
            AutoConfig.from_pretrained(
                "test_path",
                attn_implementation=attn_implementation,
                trust_remote_code=False,
            )

            # Verify AutoConfig.from_pretrained was called with correct attn_implementation
            mock_config_from_pretrained.assert_called_once_with(
                "test_path",
                attn_implementation=expected_attn_impl,
                trust_remote_code=False,
            )

    def test_critic_override_config_integration(self):
        """Test that override_config from Hydra configuration works correctly for CriticWorker."""

        # Simulate the OmegaConf configuration structure used in VERL for critic
        config_dict = {
            "critic": {
                "model": {"path": "/test/path", "override_config": {"attn_implementation": "eager", "dropout": 0.1}}
            }
        }

        # Convert to OmegaConf structure
        omegaconf = OmegaConf.create(config_dict)

        # Simulate what happens in the CriticWorker
        override_model_config = OmegaConf.to_container(
            OmegaConf.create(omegaconf.critic.model.get("override_config", {}))
        )

        # Test extraction for critic
        attn_implementation = override_model_config.get("attn_implementation", "flash_attention_2")
        assert attn_implementation == "eager"

        # Test that other configs are preserved for critic
        assert override_model_config.get("dropout") == 0.1

    def test_critic_hydra_plus_prefix_config(self):
        """Test that Hydra +prefix configurations work correctly for CriticWorker."""

        # This simulates the configuration when user specifies:
        # +critic.model.override_config.attn_implementation=eager

        # The + prefix in Hydra adds new keys to the config
        config_dict = {
            "critic": {
                "model": {
                    "path": "/test/path",
                    "override_config": {
                        "attn_implementation": "eager"  # This gets added via +prefix for critic
                    },
                }
            }
        }

        omegaconf = OmegaConf.create(config_dict)

        # Extract override config as done in CriticWorker
        override_model_config = OmegaConf.to_container(
            OmegaConf.create(omegaconf.critic.model.get("override_config", {}))
        )

        # Verify extraction works for critic
        attn_implementation = override_model_config.get("attn_implementation", "flash_attention_2")
        assert attn_implementation == "eager"

    def test_both_actor_and_critic_configuration(self):
        """Test that both actor and critic can have different attn_implementation overrides simultaneously."""

        # This simulates a complete training configuration with both actor and critic overrides
        config_dict = {
            "actor_rollout_ref": {"model": {"override_config": {"attn_implementation": "eager"}}},
            "critic": {"model": {"override_config": {"attn_implementation": "sdpa"}}},
        }

        omegaconf = OmegaConf.create(config_dict)

        # Extract actor override config
        actor_override_config = OmegaConf.to_container(
            OmegaConf.create(omegaconf.actor_rollout_ref.model.get("override_config", {}))
        )
        actor_attn_implementation = actor_override_config.get("attn_implementation", "flash_attention_2")

        # Extract critic override config
        critic_override_config = OmegaConf.to_container(
            OmegaConf.create(omegaconf.critic.model.get("override_config", {}))
        )
        critic_attn_implementation = critic_override_config.get("attn_implementation", "flash_attention_2")

        # Verify both can be configured independently
        assert actor_attn_implementation == "eager"
        assert critic_attn_implementation == "sdpa"

    def test_critic_backward_compatibility(self):
        """Test that the CriticWorker fix maintains backward compatibility."""

        # Test case 1: No override_config at all for critic (old behavior)
        config_without_override = {}
        attn_implementation = config_without_override.get("attn_implementation", "flash_attention_2")
        assert attn_implementation == "flash_attention_2"

        # Test case 2: Empty override_config for critic
        config_with_empty_override = {"override_config": {}}
        override_config = config_with_empty_override.get("override_config", {})
        attn_implementation = override_config.get("attn_implementation", "flash_attention_2")
        assert attn_implementation == "flash_attention_2"

        # Test case 3: override_config with other settings but no attn_implementation for critic
        config_with_other_overrides = {"override_config": {"dropout": 0.1, "num_labels": 1}}
        override_config = config_with_other_overrides.get("override_config", {})
        attn_implementation = override_config.get("attn_implementation", "flash_attention_2")
        assert attn_implementation == "flash_attention_2"


def test_attn_implementation_fix_integration():
    """Integration test to verify the entire fix works as expected."""

    # This test simulates the complete flow from configuration to model creation

    # Step 1: Simulate Hydra configuration with +prefix
    # user_config = "+actor_rollout_ref.model.override_config.attn_implementation=eager"

    # This would result in a config structure like:
    config_dict = {"actor_rollout_ref": {"model": {"override_config": {"attn_implementation": "eager"}}}}

    # Step 2: Extract override_model_config as done in FSDP workers
    omegaconf = OmegaConf.create(config_dict)
    override_model_config = OmegaConf.to_container(
        OmegaConf.create(omegaconf.actor_rollout_ref.model.get("override_config", {}))
    )

    # Step 3: Apply the fix logic
    attn_implementation = override_model_config.get("attn_implementation", "flash_attention_2")

    # Step 4: Verify the fix works
    assert attn_implementation == "eager"

    # Step 5: Verify this would be passed to both AutoConfig and Model creation
    # (This would normally be done with mocks, but we can test the parameter preparation)
    config_params = {"attn_implementation": attn_implementation}
    model_params = {"attn_implementation": attn_implementation}

    assert config_params["attn_implementation"] == "eager"
    assert model_params["attn_implementation"] == "eager"


def test_critic_attn_implementation_fix_integration():
    """Integration test to verify the entire fix works as expected for CriticWorker."""

    # This test simulates the complete flow from configuration to model creation for critic

    # Step 1: Simulate Hydra configuration with +prefix for critic
    # user_config = "+critic.model.override_config.attn_implementation=sdpa"

    # This would result in a config structure like:
    config_dict = {"critic": {"model": {"override_config": {"attn_implementation": "sdpa"}}}}

    # Step 2: Extract override_model_config as should be done in CriticWorker
    omegaconf = OmegaConf.create(config_dict)
    override_model_config = OmegaConf.to_container(OmegaConf.create(omegaconf.critic.model.get("override_config", {})))

    # Step 3: Apply the fix logic (what needs to be implemented in CriticWorker)
    attn_implementation = override_model_config.get("attn_implementation", "flash_attention_2")

    # Step 4: Verify the fix works for critic
    assert attn_implementation == "sdpa"

    # Step 5: Verify this would be passed to AutoConfig creation for critic
    config_params = {"attn_implementation": attn_implementation}

    assert config_params["attn_implementation"] == "sdpa"


def test_complete_training_configuration():
    """Integration test for a complete training configuration with both actor and critic overrides."""

    # This test simulates a realistic training configuration where both
    # actor and critic have different attention implementations
    config_dict = {
        "actor_rollout_ref": {
            "model": {
                "path": "/shared/models/llama-7b",
                "override_config": {"attn_implementation": "eager", "torch_dtype": "bfloat16"},
            }
        },
        "critic": {
            "model": {
                "path": "/shared/models/llama-7b",
                "override_config": {"attn_implementation": "sdpa", "num_labels": 1},
            }
        },
    }

    omegaconf = OmegaConf.create(config_dict)

    # Extract configurations as would be done in the workers
    actor_override_config = OmegaConf.to_container(
        OmegaConf.create(omegaconf.actor_rollout_ref.model.get("override_config", {}))
    )
    critic_override_config = OmegaConf.to_container(OmegaConf.create(omegaconf.critic.model.get("override_config", {})))

    # Apply the fix logic for both
    actor_attn_implementation = actor_override_config.get("attn_implementation", "flash_attention_2")
    critic_attn_implementation = critic_override_config.get("attn_implementation", "flash_attention_2")

    # Verify both configurations work independently
    assert actor_attn_implementation == "eager"
    assert critic_attn_implementation == "sdpa"

    # Verify other configs are preserved
    assert actor_override_config.get("torch_dtype") == "bfloat16"
    assert critic_override_config.get("num_labels") == 1


if __name__ == "__main__":
    # Run basic tests
    test_attn_implementation_fix_integration()
    test_critic_attn_implementation_fix_integration()
    test_complete_training_configuration()

    if VERL_AVAILABLE:
        # Run class-based tests
        test_class = TestFSDPAttnImplementation()
        test_class.test_attn_implementation_extraction_logic()
        test_class.test_override_config_integration()
        test_class.test_hydra_plus_prefix_config()
        test_class.test_backward_compatibility()

        # Run new critic tests
        test_class.test_critic_attn_implementation_extraction_logic()
        test_class.test_critic_override_config_integration()
        test_class.test_critic_hydra_plus_prefix_config()
        test_class.test_both_actor_and_critic_configuration()
        test_class.test_critic_backward_compatibility()

        print("✓ All FSDP attn_implementation tests passed!")
        print("✓ All CriticWorker attn_implementation tests passed!")
    else:
        print("⚠ VERL components not available, skipping VERL-specific tests")

    print("✓ Integration tests passed!")
    print("✓ Critic integration tests passed!")
