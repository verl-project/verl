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

import unittest

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import (
    FSDPActorConfig,
    FSDPCriticConfig,
    FSDPOptimizerConfig,
    OptimizerConfig,
    VeOmniOptimizerConfig,
)


def _fsdp_actor_dict(**overrides):
    cfg = {
        "_target_": "verl.workers.config.FSDPActorConfig",
        "strategy": "fsdp",
        "ppo_mini_batch_size": 256,
        "ppo_micro_batch_size_per_gpu": 256,
        "rollout_n": 1,
        "optim": {"_target_": "verl.workers.config.FSDPOptimizerConfig", "lr": 0.1},
    }
    cfg.update(overrides)
    return cfg


def _fsdp_critic_dict(**overrides):
    cfg = {
        "_target_": "verl.workers.config.FSDPCriticConfig",
        "strategy": "fsdp",
        "ppo_mini_batch_size": 256,
        "ppo_micro_batch_size_per_gpu": 256,
        "optim": {"_target_": "verl.workers.config.FSDPOptimizerConfig", "lr": 0.1},
    }
    cfg.update(overrides)
    return cfg


class TestClipGradConfig(unittest.TestCase):
    """optim.clip_grad is forwarded to clip_grad_norm_ by the FSDP/VeOmni engines and must be a usable max_norm;
    the deprecated worker-level grad_clip must reach optim.clip_grad instead of being ignored."""

    def test_clip_grad_bounds(self):
        # torch clip_grad_norm_ needs max_norm > 0 (0 zeroes gradients, negative flips them; inf disables
        # clipping). Megatron treats 0 as "no clipping", so the base config only rejects negatives and NaN.
        cases = [
            (FSDPOptimizerConfig, -1.0, False),
            (FSDPOptimizerConfig, 0.0, False),
            (FSDPOptimizerConfig, 0.5, True),
            (FSDPOptimizerConfig, float("inf"), True),
            (VeOmniOptimizerConfig, 0.0, False),
            (VeOmniOptimizerConfig, 0.5, True),
            (OptimizerConfig, -1.0, False),
            (OptimizerConfig, float("nan"), False),
            (OptimizerConfig, 0.0, True),
        ]
        for cls, value, ok in cases:
            with self.subTest(cls=cls.__name__, clip_grad=value):
                if ok:
                    self.assertEqual(cls(lr=1e-3, clip_grad=value).clip_grad, value)
                else:
                    with self.assertRaises(ValueError):
                        cls(lr=1e-3, clip_grad=value)

    def test_deprecated_paths_are_validated(self):
        with self.assertWarns(DeprecationWarning), self.assertRaises(ValueError):
            FSDPOptimizerConfig(lr=1e-3, grad_clip=-1.0)
        # hydra wraps errors raised in __post_init__ in InstantiationException; match on the message.
        with self.assertRaises(Exception) as ctx:
            omega_conf_to_dataclass(_fsdp_critic_dict(grad_clip=-1.0))
        self.assertIn("clip_grad must be > 0", str(ctx.exception))

    def test_worker_level_grad_clip_applies_to_optim(self):
        for build, cls in ((_fsdp_actor_dict, FSDPActorConfig), (_fsdp_critic_dict, FSDPCriticConfig)):
            with self.subTest(cls=cls.__name__):
                with self.assertWarns(DeprecationWarning):
                    cfg = omega_conf_to_dataclass(build(grad_clip=0.5))
                self.assertIsInstance(cfg, cls)
                self.assertEqual(cfg.optim.clip_grad, 0.5)

    def test_unset_grad_clip_keeps_optim_default(self):
        cfg = omega_conf_to_dataclass(_fsdp_actor_dict())
        self.assertIsNone(cfg.grad_clip)
        self.assertEqual(cfg.optim.clip_grad, 1.0)
