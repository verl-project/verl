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
from verl.workers.config import RolloutConfig


def _actor_dict(**overrides):
    cfg = {
        "_target_": "verl.workers.config.FSDPActorConfig",
        "strategy": "fsdp",
        "ppo_mini_batch_size": 256,
        "ppo_micro_batch_size_per_gpu": 8,
        "rollout_n": 1,
        "optim": {"_target_": "verl.workers.config.FSDPOptimizerConfig", "lr": 0.1},
    }
    cfg.update(overrides)
    return cfg


def _critic_dict(**overrides):
    cfg = {
        "_target_": "verl.workers.config.FSDPCriticConfig",
        "strategy": "fsdp",
        "ppo_mini_batch_size": 256,
        "ppo_micro_batch_size_per_gpu": 8,
        "optim": {"_target_": "verl.workers.config.FSDPOptimizerConfig", "lr": 0.1},
    }
    cfg.update(overrides)
    return cfg


class TestRolloutConfigPositiveValues(unittest.TestCase):
    """Zero parallelism sizes or n=0 used to surface as ZeroDivisionError deep in worker init."""

    def test_rejects_non_positive(self):
        for key in ("n", "tensor_model_parallel_size", "data_parallel_size", "pipeline_model_parallel_size"):
            for bad in (0, -1):
                with self.assertRaises(ValueError, msg=f"{key}={bad}"):
                    RolloutConfig(name="vllm", **{key: bad})

    def test_accepts_positive(self):
        cfg = RolloutConfig(name="vllm", n=4, tensor_model_parallel_size=1, data_parallel_size=1)
        self.assertEqual(cfg.n, 4)


class TestBatchSizePositiveValues(unittest.TestCase):
    """ppo_micro_batch_size_per_gpu=0 used to reach the engine's divisibility check as a modulo by zero."""

    def test_zero_micro_batch_rejected(self):
        # hydra wraps errors raised in __post_init__ in InstantiationException; match on the message.
        for build in (_actor_dict, _critic_dict):
            with self.subTest(config=build.__name__), self.assertRaises(Exception) as ctx:
                omega_conf_to_dataclass(build(ppo_micro_batch_size_per_gpu=0))
            self.assertIn("must be a positive integer", str(ctx.exception))

    def test_valid_batch_sizes_pass(self):
        self.assertEqual(omega_conf_to_dataclass(_actor_dict()).ppo_micro_batch_size_per_gpu, 8)
        self.assertEqual(omega_conf_to_dataclass(_critic_dict()).ppo_micro_batch_size_per_gpu, 8)
