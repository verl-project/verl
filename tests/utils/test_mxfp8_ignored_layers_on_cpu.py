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
"""Regression tests: lm_head must never be MXFP8-quantized.

Quantizing ``lm_head`` under MXFP8 makes the rollout logits ``nan``. The failure
is silent — the run still exits 0 — so it needs a test that pins the contract
rather than an assertion at runtime. Reproduced on 2xB200 (Qwen3-8B / gsm8k):
reward stayed 0.0 for all 20 steps and every ``rollout_corr/*`` metric was
``nan`` until ``lm_head`` was excluded.
"""

from verl.utils.mxfp8_quant import MXFP8_KEEP_HIGH_PRECISION_LAYERS


def test_keep_high_precision_layers_covers_lm_head():
    assert "lm_head" in MXFP8_KEEP_HIGH_PRECISION_LAYERS


def test_sglang_mxfp8_config_excludes_lm_head():
    from verl.utils.sglang.sglang_mxfp8_utils import build_sglang_mxfp8_quant_config

    cfg = build_sglang_mxfp8_quant_config()
    assert cfg["quant_method"] == "mxfp8"
    assert cfg["weight_block_size"] == [1, 32]
    ignored = cfg.get("ignored_layers") or []
    for name in MXFP8_KEEP_HIGH_PRECISION_LAYERS:
        assert name in ignored, f"{name} must be in ignored_layers, got {ignored}"


def test_sglang_mxfp8_config_preserves_caller_ignored_layers():
    """The high-precision layers are additive, not a replacement."""
    from verl.utils.sglang.sglang_mxfp8_utils import build_sglang_mxfp8_quant_config

    cfg = build_sglang_mxfp8_quant_config(ignored_layers=["model.layers.0.mlp.gate"])
    ignored = cfg.get("ignored_layers") or []
    assert "model.layers.0.mlp.gate" in ignored
    assert "lm_head" in ignored


def test_no_duplicates_when_caller_already_excludes_lm_head():
    from verl.utils.sglang.sglang_mxfp8_utils import build_sglang_mxfp8_quant_config

    cfg = build_sglang_mxfp8_quant_config(ignored_layers=["lm_head"])
    ignored = cfg.get("ignored_layers") or []
    assert ignored.count("lm_head") == 1, ignored


def test_vllm_mxfp8_quant_kwargs_excludes_lm_head():
    """Mirror the dict the vLLM launch path builds (see vllm_async_server.py).

    Kept as an explicit construction rather than importing the server module,
    which pulls in vLLM at import time and is not available on CPU CI.
    """
    all_mlp_gate_layers = [f"model.layers.{i}.mlp.gate" for i in range(4)]
    quant_kwargs = {
        "quant_method": "mxfp8",
        "ignored_layers": all_mlp_gate_layers + list(MXFP8_KEEP_HIGH_PRECISION_LAYERS),
    }
    assert "lm_head" in quant_kwargs["ignored_layers"]
    assert "model.layers.0.mlp.gate" in quant_kwargs["ignored_layers"]
