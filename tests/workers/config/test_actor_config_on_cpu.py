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

import os
import unittest

from omegaconf import OmegaConf

from verl.utils.config import _validate_v1_fsdp_strategy, omega_conf_to_dataclass
from verl.workers.config import (
    ActorConfig,
    FSDPActorConfig,
    McoreActorConfig,
    OptimizerConfig,
)


class TestActorConfig(unittest.TestCase):
    """Test the ActorConfig dataclass and its variants."""

    def test_config_inheritance(self):
        """Test that the inheritance hierarchy works correctly."""
        megatron_dict = {
            "_target_": "verl.workers.config.McoreActorConfig",
            "strategy": "megatron",
            "ppo_mini_batch_size": 256,
            "ppo_micro_batch_size_per_gpu": 256,
            "clip_ratio": 0.2,
            "optim": {
                "_target_": "verl.workers.config.McoreOptimizerConfig",
                "lr": 0.1,
            },
            "rollout_n": 1,
        }
        fsdp_dict = {
            "_target_": "verl.workers.config.FSDPActorConfig",
            "strategy": "fsdp",
            "ppo_mini_batch_size": 256,
            "ppo_micro_batch_size_per_gpu": 256,
            "clip_ratio": 0.2,
            "optim": {
                "_target_": "verl.workers.config.FSDPOptimizerConfig",
                "lr": 0.1,
            },
            "rollout_n": 1,
        }

        megatron_config = omega_conf_to_dataclass(megatron_dict)
        fsdp_config = omega_conf_to_dataclass(fsdp_dict)

        self.assertIsInstance(megatron_config, ActorConfig)
        self.assertIsInstance(fsdp_config, ActorConfig)

        self.assertEqual(megatron_config.ppo_mini_batch_size, fsdp_config.ppo_mini_batch_size)
        self.assertEqual(megatron_config.clip_ratio, fsdp_config.clip_ratio)

    def test_actor_config_from_yaml(self):
        """Test creating ActorConfig from YAML file."""
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/actor")):
            cfg = compose(config_name="actor", overrides=["strategy=fsdp", "ppo_micro_batch_size_per_gpu=128"])

        config = omega_conf_to_dataclass(cfg)

        self.assertIsInstance(config, ActorConfig)
        self.assertEqual(config.strategy, "fsdp")

    def test_fsdp_actor_config_from_yaml(self):
        """Test that supported top-level strategies reach the FSDP engine config."""
        from hydra import compose, initialize_config_dir

        for strategy in ("fsdp", "fsdp2", "fsdp_turbo"):
            with self.subTest(strategy=strategy):
                with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/actor")):
                    cfg = compose(
                        config_name="dp_actor",
                        overrides=[f"strategy={strategy}", "ppo_micro_batch_size_per_gpu=128"],
                    )

                config = omega_conf_to_dataclass(cfg)

                self.assertIsInstance(config, FSDPActorConfig)
                self.assertEqual(config.strategy, strategy)
                self.assertEqual(config.engine.strategy, strategy)

    def test_fsdp_actor_config_accepts_matching_nested_strategy(self):
        """Test compatibility with configs that redundantly set the same nested strategy."""
        from hydra import compose, initialize_config_dir

        for strategy in ("fsdp", "fsdp2", "fsdp_turbo"):
            with self.subTest(strategy=strategy):
                with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/actor")):
                    cfg = compose(
                        config_name="dp_actor",
                        overrides=[
                            f"strategy={strategy}",
                            f"fsdp_config.strategy={strategy}",
                            "ppo_micro_batch_size_per_gpu=128",
                        ],
                    )

                config = omega_conf_to_dataclass(cfg)

                self.assertEqual(config.strategy, strategy)
                self.assertEqual(config.engine.strategy, strategy)

    def test_v1_fsdp_actor_config_rejects_nested_strategy_override(self):
        """Test that V1 rejects a nested strategy that conflicts with the top-level key."""
        from hydra import compose, initialize_config_dir

        for nested_strategy in ("fsdp2", "fsdp_turbo"):
            with self.subTest(nested_strategy=nested_strategy):
                with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/actor")):
                    cfg = compose(
                        config_name="dp_actor",
                        overrides=[
                            f"fsdp_config.strategy={nested_strategy}",
                            "ppo_micro_batch_size_per_gpu=128",
                        ],
                    )

                config = OmegaConf.create(
                    {
                        "trainer": {"use_v1": True},
                        "actor_rollout_ref": {"actor": cfg},
                    }
                )
                with self.assertRaisesRegex(ValueError, "top-level `strategy`.*source of truth"):
                    _validate_v1_fsdp_strategy(config)

    def test_legacy_fsdp_actor_config_preserves_nested_strategy_compatibility(self):
        """Test that deprecated runners keep their existing nested-selector behavior."""
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/actor")):
            cfg = compose(
                config_name="dp_actor",
                overrides=[
                    "fsdp_config.strategy=fsdp2",
                    "ppo_micro_batch_size_per_gpu=128",
                ],
            )

        global_config = OmegaConf.create(
            {
                "trainer": {"use_v1": False},
                "actor_rollout_ref": {"actor": cfg},
            }
        )
        _validate_v1_fsdp_strategy(global_config)

        config = omega_conf_to_dataclass(cfg)
        self.assertEqual(config.strategy, "fsdp")
        self.assertEqual(config.engine.strategy, "fsdp")

    def test_fsdp_actor_config_rejects_unsupported_top_level_strategy(self):
        """Test that an FSDP actor cannot overwrite its engine with an invalid backend."""
        for strategy in ("megatron", "typo"):
            with self.subTest(strategy=strategy):
                with self.assertRaisesRegex(ValueError, "Unsupported FSDP strategy"):
                    FSDPActorConfig(strategy=strategy, use_dynamic_bsz=True, rollout_n=1)

    def test_fsdp_actor_config_direct_constructor_uses_top_level_strategy(self):
        """Test direct construction keeps the legacy nested default as an implicit value."""
        for strategy in ("fsdp2", "fsdp_turbo"):
            with self.subTest(strategy=strategy):
                config = FSDPActorConfig(strategy=strategy, use_dynamic_bsz=True, rollout_n=1)

                self.assertEqual(config.strategy, strategy)
                self.assertEqual(config.engine.strategy, strategy)

    def test_megatron_actor_config_from_yaml(self):
        """Test creating McoreActorConfig from YAML file."""
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/actor")):
            cfg = compose(config_name="megatron_actor", overrides=["ppo_micro_batch_size_per_gpu=128"])

        config = omega_conf_to_dataclass(cfg)

        self.assertIsInstance(config, McoreActorConfig)
        self.assertEqual(config.strategy, "megatron")

    def test_config_get_method(self):
        """Test the get method for backward compatibility."""
        config_dict = {
            "_target_": "verl.workers.config.ActorConfig",
            "strategy": "fsdp",
            "ppo_mini_batch_size": 256,
            "ppo_micro_batch_size_per_gpu": 256,
            "optim": {
                "_target_": "verl.workers.config.OptimizerConfig",
                "lr": 0.1,
            },
            "rollout_n": 1,
        }
        config = omega_conf_to_dataclass(config_dict)

        self.assertEqual(config.get("strategy"), "fsdp")
        self.assertEqual(config.get("ppo_mini_batch_size"), 256)

        self.assertIsNone(config.get("non_existing"))
        self.assertEqual(config.get("non_existing", "default"), "default")

    def test_config_dict_like_access(self):
        """Test dictionary-like access to config fields."""
        config_dict = {
            "_target_": "verl.workers.config.ActorConfig",
            "strategy": "fsdp",
            "ppo_mini_batch_size": 256,
            "ppo_micro_batch_size_per_gpu": 256,
            "optim": {
                "_target_": "verl.workers.config.OptimizerConfig",
                "lr": 0.1,
            },
            "rollout_n": 1,
        }
        config = omega_conf_to_dataclass(config_dict)

        self.assertEqual(config["strategy"], "fsdp")
        self.assertEqual(config["ppo_mini_batch_size"], 256)

        field_names = list(config)
        self.assertIn("strategy", field_names)
        self.assertIn("ppo_mini_batch_size", field_names)

        self.assertGreater(len(config), 0)

    def test_frozen_fields_modification_raises_exception(self):
        """Test that modifying frozen fields raises an exception."""
        config_dict = {
            "_target_": "verl.workers.config.ActorConfig",
            "strategy": "fsdp",
            "ppo_mini_batch_size": 256,
            "ppo_micro_batch_size_per_gpu": 256,
            "optim": {
                "_target_": "verl.workers.config.OptimizerConfig",
                "lr": 0.1,
            },
            "rollout_n": 1,
        }
        config = omega_conf_to_dataclass(config_dict)

        with self.assertRaises(AttributeError):
            config.strategy = "megatron"

        with self.assertRaises(AttributeError):
            config.clip_ratio = 0.5

        config.ppo_mini_batch_size = 512  # This should work since it's not in frozen fields anymore
        self.assertEqual(config.ppo_mini_batch_size, 512)

    def test_actor_config_validation_exceptions(self):
        """Test that ActorConfig.__post_init__ raises appropriate validation exceptions."""
        optim = OptimizerConfig(lr=0.1)
        with self.assertRaises((ValueError, AssertionError)) as cm:
            ActorConfig(
                strategy="fsdp",
                loss_agg_mode="invalid-mode",
                use_dynamic_bsz=True,
                optim=optim,
                ppo_micro_batch_size_per_gpu=4,
                rollout_n=1,
            )
        self.assertIn("Invalid loss_agg_mode", str(cm.exception))

        with self.assertRaises((ValueError, AssertionError)) as cm:
            ActorConfig(
                strategy="fsdp",
                use_dynamic_bsz=False,
                ppo_micro_batch_size=4,
                ppo_micro_batch_size_per_gpu=2,
                optim=optim,
                rollout_n=1,
            )
        self.assertIn("You have set both", str(cm.exception))

        with self.assertRaises((ValueError, AssertionError)) as cm:
            ActorConfig(
                strategy="fsdp",
                use_dynamic_bsz=False,
                ppo_micro_batch_size=None,
                ppo_micro_batch_size_per_gpu=None,
                optim=optim,
                rollout_n=1,
            )
        self.assertIn("Please set at least one", str(cm.exception))

        config = ActorConfig(
            strategy="fsdp",
            use_dynamic_bsz=True,
            ppo_micro_batch_size=None,
            ppo_micro_batch_size_per_gpu=None,
            optim=optim,
            rollout_n=1,
        )
        self.assertIsNotNone(config)  # Should not raise an exception

    def test_fsdp_actor_config_validation_exceptions(self):
        """Test that FSDPActorConfig.validate() raises appropriate validation exceptions."""
        optim = OptimizerConfig(lr=0.1)
        config = FSDPActorConfig(
            strategy="fsdp",
            ulysses_sequence_parallel_size=2,
            use_dynamic_bsz=True,  # Skip batch size validation to focus on FSDP validation
            optim=optim,
            rollout_n=1,
        )

        model_config = {"use_remove_padding": False}
        with self.assertRaises(ValueError) as cm:
            config.validate(n_gpus=8, train_batch_size=256, model_config=model_config)
        self.assertIn("you must enable `use_remove_padding`", str(cm.exception))

    def test_actor_config_validate_method_exceptions(self):
        """Test that ActorConfig.validate() raises appropriate validation exceptions."""
        optim = OptimizerConfig(lr=0.1)
        config = ActorConfig(
            strategy="fsdp",
            use_dynamic_bsz=False,
            ppo_mini_batch_size=256,
            ppo_micro_batch_size=8,
            ppo_micro_batch_size_per_gpu=None,  # Ensure only one batch size setting is used
            optim=optim,
            rollout_n=1,
        )

        with self.assertRaises(ValueError) as cm:
            config.validate(n_gpus=8, train_batch_size=128)
        self.assertIn("train_batch_size", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            config.validate(n_gpus=16, train_batch_size=512)
        self.assertIn("must be >= n_gpus", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
