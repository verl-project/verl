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
import warnings

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

_BREAKING_CHANGES = [
    "critic.optim.lr",  # mcore critic lr init value 1e-6 -> 1e-5
    "actor_rollout_ref.actor.optim.lr_warmup_steps",  # None -> -1
    "critic.optim.lr_warmup_steps",  # None -> -1
    "actor_rollout_ref.rollout.name",  # vllm -> ???
    "actor_rollout_ref.actor.megatron.expert_tensor_parallel_size",
    "actor_rollout_ref.ref.megatron.expert_tensor_parallel_size",
    "critic.megatron.expert_tensor_parallel_size",
    "reward_model.megatron.expert_tensor_parallel_size",
]


class TestConfigComparison(unittest.TestCase):
    """Test that current configs match their legacy counterparts exactly."""

    ignored_keys = [
        "enable_gradient_checkpointing",
        "gradient_checkpointing_kwargs",
        "activations_checkpoint_method",
        "activations_checkpoint_granularity",
        "activations_checkpoint_num_layers",
        "discrete",
        "profiler",
        "profile",
        "use_profile",
        "npu_profile",
        "profile_steps",
        "worker_nsight_options",
        "controller_nsight_options",
    ]
    ignored_paths = ["reward_model"]

    def _compare_configs_recursively(
        self, current_config, legacy_config, path="", legacy_allow_missing=True, current_allow_missing=False
    ):
        """Recursively compare two OmegaConf configs and assert they are identical.

        Args:
            legacy_allow_missing (bool): sometimes the legacy megatron config contains fewer keys and
              we allow that to happen
        """
        if path in self.ignored_paths:
            return

        if isinstance(current_config, dict) and isinstance(legacy_config, dict):
            current_keys = set(current_config.keys())
            legacy_keys = set(legacy_config.keys())

            missing_in_current = legacy_keys - current_keys
            missing_in_legacy = current_keys - legacy_keys

            # Ignore specific keys that are allowed to be missing
            for key in self.ignored_keys:
                if key in missing_in_current:
                    missing_in_current.remove(key)
                if key in missing_in_legacy:
                    missing_in_legacy.remove(key)

            if missing_in_current:
                msg = f"Keys missing in current config at {path}: {missing_in_current}"
                if current_allow_missing:
                    warnings.warn(msg, stacklevel=1)
                else:
                    self.fail(f"Keys missing in current config at {path}: {missing_in_current}")
            if missing_in_legacy:
                # if the legacy
                msg = f"Keys missing in legacy config at {path}: {missing_in_legacy}"
                if legacy_allow_missing:
                    warnings.warn(msg, stacklevel=1)
                else:
                    self.fail(msg)

            for key in current_keys:
                current_path = f"{path}.{key}" if path else key
                if key in legacy_config:
                    self._compare_configs_recursively(current_config[key], legacy_config[key], current_path)
        elif isinstance(current_config, list) and isinstance(legacy_config, list):
            self.assertEqual(
                len(current_config),
                len(legacy_config),
                f"List lengths differ at {path}: current={len(current_config)}, legacy={len(legacy_config)}",
            )
            for i, (current_item, legacy_item) in enumerate(zip(current_config, legacy_config, strict=True)):
                self._compare_configs_recursively(current_item, legacy_item, f"{path}[{i}]")
        elif path not in _BREAKING_CHANGES:
            self.assertEqual(
                current_config,
                legacy_config,
                f"Values differ at {path}: current={current_config}, legacy={legacy_config}",
            )

    def test_ppo_trainer_config_matches_legacy(self):
        """Test that ppo_trainer.yaml matches legacy_ppo_trainer.yaml exactly."""
        import os

        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()

        try:
            with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
                current_config = compose(config_name="ppo_trainer")

            legacy_config = OmegaConf.load("tests/trainer/config/legacy_ppo_trainer.yaml")
            current_dict = OmegaConf.to_container(current_config, resolve=True)
            legacy_dict = OmegaConf.to_container(legacy_config, resolve=True)

            if "defaults" in current_dict:
                del current_dict["defaults"]

            self._compare_configs_recursively(current_dict, legacy_dict)
        finally:
            GlobalHydra.instance().clear()

    def test_ppo_megatron_trainer_config_matches_legacy(self):
        """Test that ppo_megatron_trainer.yaml matches legacy_ppo_megatron_trainer.yaml exactly."""

        GlobalHydra.instance().clear()

        try:
            with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
                current_config = compose(config_name="ppo_megatron_trainer")

            legacy_config = OmegaConf.load("tests/trainer/config/legacy_ppo_megatron_trainer.yaml")
            current_dict = OmegaConf.to_container(current_config, resolve=True)
            legacy_dict = OmegaConf.to_container(legacy_config, resolve=True)

            if "defaults" in current_dict:
                del current_dict["defaults"]

            self._compare_configs_recursively(
                current_dict, legacy_dict, legacy_allow_missing=True, current_allow_missing=False
            )
        finally:
            GlobalHydra.instance().clear()

    def test_load_component(self):
        """Test that ppo_megatron_trainer.yaml matches legacy_ppo_megatron_trainer.yaml exactly."""

        GlobalHydra.instance().clear()
        configs_to_load = [
            ("verl/trainer/config/actor", "dp_actor"),
            ("verl/trainer/config/actor", "megatron_actor"),
            ("verl/trainer/config/ref", "dp_ref"),
            ("verl/trainer/config/ref", "megatron_ref"),
            ("verl/trainer/config/rollout", "rollout"),
        ]
        for config_dir, config_file in configs_to_load:
            try:
                with initialize_config_dir(config_dir=os.path.abspath(config_dir)):
                    compose(config_name=config_file)
            finally:
                GlobalHydra.instance().clear()


if __name__ == "__main__":
    unittest.main()
