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
"""Regression tests: which layers an MXFP8 rollout keeps in BF16.

Both contracts below are the trainer-side exporter's default ignore list, which the
rollout's compressed-tensors config carries as well; a preset or a user ``ignore``
list adds to it and cannot remove it. Both failures were silent on 2xB200:

- Quantizing ``lm_head`` makes the rollout logits ``nan`` (Qwen3-8B / gsm8k: reward
  0.0 for all 20 steps, every ``rollout_corr/*`` metric ``nan``, exit code 0).
- vLLM 0.24's ModelOpt MXFP8 config matched the router exclusion ``mlp.gate`` as a
  prefix of ``mlp.gate_up_proj``, so the dense MLPs were served in BF16 while training
  ran them in FP8 (56 of 311 parameters of Qwen3-1.7B).
"""

import importlib.util

import pytest
import torch

from verl.utils.qat.compressed_tensors import DEFAULT_IGNORE, _matches

requires_ct = pytest.mark.skipif(
    importlib.util.find_spec("compressed_tensors") is None, reason="compressed-tensors is not installed"
)


@pytest.mark.parametrize(
    "module", ["lm_head", "model.embed_tokens", "model.language_model.lm_head", "model.language_model.embed_tokens"]
)
def test_default_ignore_keeps_lm_head_and_the_embedding(module):
    assert _matches(module, DEFAULT_IGNORE)


def test_default_ignore_keeps_routers_but_not_gate_up_proj():
    # The router (mlp.gate) and the shared-expert gate stay in BF16; the gated MLP projections do not.
    assert _matches("model.layers.0.mlp.gate", DEFAULT_IGNORE)
    assert _matches("model.layers.0.mlp.shared_expert_gate", DEFAULT_IGNORE)
    assert not _matches("model.layers.0.mlp.gate_up_proj", DEFAULT_IGNORE)
    assert not _matches("model.layers.0.mlp.gate_proj", DEFAULT_IGNORE)
    assert not _matches("model.layers.0.mlp.shared_expert.gate_proj", DEFAULT_IGNORE)


@requires_ct
@pytest.mark.parametrize(("linear", "moe"), [("MXFP8", "MXFP8"), ("MXFP8", None)])
def test_mxfp8_export_leaves_lm_head_and_the_embedding_in_bf16(linear, moe):
    from verl.utils.qat.compressed_tensors import CompressedTensorsWeightQuantizer, build_compressed_tensors_config

    config = build_compressed_tensors_config(linear=linear, moe=moe, ignore=[r"re:.*layers\.0\..*"], config_file=None)
    assert config["ignore"][: len(DEFAULT_IGNORE)] == DEFAULT_IGNORE  # the user list is added, not substituted
    weights = [
        ("lm_head.weight", torch.randn(256, 64, dtype=torch.bfloat16)),
        ("model.embed_tokens.weight", torch.randn(256, 64, dtype=torch.bfloat16)),
    ]
    out = dict(CompressedTensorsWeightQuantizer(config)(weights))
    assert set(out) == {"lm_head.weight", "model.embed_tokens.weight"}
    assert all(tensor.dtype == torch.bfloat16 for tensor in out.values())
