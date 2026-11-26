# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import pytest
import torch

from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

WEIGHTS = {"model.layers.0.mlp.down_proj.weight": torch.ones(2, 2), "model.norm.weight": torch.ones(2)}


def _engine(peft_config=None, is_lora=False):
    """An FSDPEngine whose export is stubbed, so only the rollout quantizer wiring runs."""
    engine = FSDPEngine.__new__(FSDPEngine)
    engine._rollout_weight_quantizer = None
    engine._is_lora = is_lora
    engine._get_per_tensor_param = lambda **kwargs: (iter(WEIGHTS.items()), peft_config)
    return engine


def _tag_quantizer(weights):
    for name, tensor in weights:
        yield f"quantized.{name}", tensor


def test_weights_pass_through_without_quantizer():
    weights, peft_config = _engine().get_per_tensor_param()
    assert dict(weights).keys() == WEIGHTS.keys()
    assert peft_config is None


def test_quantizer_applies_to_rollout_sync():
    engine = _engine()
    engine.set_rollout_weight_quantizer(_tag_quantizer)
    weights, _ = engine.get_per_tensor_param()
    assert list(dict(weights)) == [f"quantized.{name}" for name in WEIGHTS]


def test_checkpoint_export_skips_quantizer():
    engine = _engine()
    engine.set_rollout_weight_quantizer(_tag_quantizer)
    weights, _ = engine._get_hf_export_per_tensor_param()
    assert dict(weights).keys() == WEIGHTS.keys()


def test_unmerged_lora_adapter_sync_is_rejected():
    engine = _engine(peft_config={"r": 8}, is_lora=True)
    engine.set_rollout_weight_quantizer(_tag_quantizer)
    with pytest.raises(NotImplementedError, match="lora.merge=True"):
        engine.get_per_tensor_param(base_sync_done=True)
