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

from hydra import compose, initialize_config_dir

from verl.utils.config import validate_config

BASE_OVERRIDES = [
    "actor_rollout_ref.rollout.name=vllm",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4",
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",
    "critic.ppo_micro_batch_size_per_gpu=4",
    "data.train_batch_size=512",
]


def _compose(overrides):
    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config"), version_base=None):
        return compose(config_name="ppo_trainer", overrides=BASE_OVERRIDES + overrides)


class TestValidateConfigGpuCount(unittest.TestCase):
    """n_gpus_per_node=0 used to crash validate_config itself with ZeroDivisionError."""

    # The check runs before validate_config instantiates the model configs, so this test does not
    # need model weights on disk (the positive path is covered by the e2e CI scripts).
    def test_zero_gpus_rejected_with_message(self):
        cfg = _compose(["trainer.n_gpus_per_node=0"])
        with self.assertRaises(ValueError) as ctx:
            validate_config(cfg, use_reference_policy=False, use_critic=True)
        self.assertIn("trainer.n_gpus_per_node", str(ctx.exception))
