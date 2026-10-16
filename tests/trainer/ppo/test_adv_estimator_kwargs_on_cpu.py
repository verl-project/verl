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
"""A registered advantage estimator can read the per-sample non-tensor fields."""

import numpy as np
import torch

from verl.protocol import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import _accepts_kwarg, compute_advantage


def _batch() -> DataProto:
    tensors = {
        "token_level_rewards": torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
        "response_mask": torch.ones(2, 2, dtype=torch.int64),
    }
    non_tensors = {
        "uid": np.array(["g", "g"], dtype=object),
        "sample_is_valid": np.array([True, False], dtype=object),
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def test_accepts_kwarg_detects_declared_and_catch_all_parameters():
    def declared(token_level_rewards, response_mask, non_tensor_batch=None):
        return None

    def catch_all(token_level_rewards, response_mask, **kwargs):
        return None

    def fixed(token_level_rewards, response_mask, index=None):
        return None

    assert _accepts_kwarg(declared, "non_tensor_batch")
    assert _accepts_kwarg(catch_all, "non_tensor_batch")
    assert not _accepts_kwarg(fixed, "non_tensor_batch")


def test_custom_estimator_receives_non_tensor_batch():
    seen = {}

    @core_algos.register_adv_est("test_non_tensor_aware")
    def _estimator(token_level_rewards, response_mask, index=None, config=None, **kwargs):
        seen["non_tensor_batch"] = kwargs.get("non_tensor_batch")
        advantages = token_level_rewards * response_mask
        return advantages, advantages

    data = compute_advantage(_batch(), adv_estimator="test_non_tensor_aware")

    assert seen["non_tensor_batch"] is not None
    assert list(seen["non_tensor_batch"]["sample_is_valid"]) == [True, False]
    assert "advantages" in data.batch


def test_estimator_with_a_fixed_signature_is_unaffected():
    """An estimator that cannot accept the extra keys must still be callable."""
    calls = {"n": 0}

    @core_algos.register_adv_est("test_fixed_signature")
    def _estimator(token_level_rewards, response_mask, index=None, config=None):
        calls["n"] += 1
        advantages = token_level_rewards * response_mask
        return advantages, advantages

    compute_advantage(_batch(), adv_estimator="test_fixed_signature")

    assert calls["n"] == 1
