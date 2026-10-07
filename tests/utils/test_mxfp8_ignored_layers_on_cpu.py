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
