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
"""FP8 tensor selection follows checkpoint dtypes instead of name patterns."""

from types import SimpleNamespace

import torch
from safetensors.torch import save_file


def _checkpoint(tmp_path):
    save_file(
        {
            "model.layers.0.self_attn.wkv.weight": torch.randn(8, 8).to(torch.float8_e4m3fn),
            "model.layers.0.self_attn.wq_b.weight": torch.randn(8, 8).to(torch.float8_e4m3fn),
            "model.layers.0.self_attn.compressor.wkv.weight": torch.randn(8, 8, dtype=torch.bfloat16),
            "model.layers.0.self_attn.wo_a.weight": torch.randn(8, 8, dtype=torch.bfloat16),
            "model.layers.0.self_attn.wo_b.weight": torch.randn(8, 8).to(torch.float8_e4m3fn),
        },
        tmp_path / "model.safetensors",
    )
    return str(tmp_path)


def test_checkpoint_predicate_uses_stored_dtype(tmp_path):
    from verl.utils.fp8_ckpt_dtypes import build_ckpt_fp8_predicate

    pred = build_ckpt_fp8_predicate(_checkpoint(tmp_path))
    assert pred is not None
    assert pred("model.layers.0.self_attn.wkv.weight") is True
    assert pred("model.layers.0.self_attn.compressor.wkv.weight") is False
    assert pred("model.layers.0.self_attn.wo_a.weight") is False
    assert pred("model.layers.0.self_attn.wo_b.weight") is True


def test_checkpoint_predicate_matches_bridge_attention_spelling(tmp_path):
    from verl.utils.fp8_ckpt_dtypes import build_ckpt_fp8_predicate

    pred = build_ckpt_fp8_predicate(_checkpoint(tmp_path))
    assert pred("model.layers.0.attn.wkv.weight") is True
    assert pred("model.layers.0.attn.compressor.wkv.weight") is False


def test_missing_checkpoint_returns_none(tmp_path):
    from verl.utils.fp8_ckpt_dtypes import build_ckpt_fp8_predicate

    assert build_ckpt_fp8_predicate(str(tmp_path / "missing")) is None


def test_delta_spec_uses_training_engine_model_path(tmp_path, monkeypatch):
    from verl.checkpoint_engine.delta_checkpoint_engine import DeltaShardedCheckpointEngine

    helper = SimpleNamespace(
        quant_config={"weight_block_size": [128, 128]},
        should_quantize_param=lambda _name: False,
    )
    monkeypatch.setattr(DeltaShardedCheckpointEngine, "_fp8_helper", lambda _self, _engine: helper)

    checkpoint_engine = object.__new__(DeltaShardedCheckpointEngine)
    training_engine = SimpleNamespace(model_config=SimpleNamespace(local_path=_checkpoint(tmp_path)))
    spec = checkpoint_engine._fp8_spec(training_engine)

    assert spec.should_quantize("model.layers.0.self_attn.wq_b.weight") is True
