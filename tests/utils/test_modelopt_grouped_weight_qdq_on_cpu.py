# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pytest.importorskip("compressed_tensors")
pytest.importorskip("modelopt")
pytest.importorskip("transformer_engine")

# Hide an unusable optional FlashAttention-4 package before ModelOpt imports Megatron.
# isort: off
import verl.models.mcore  # noqa: F401
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.plugins.transformer_engine import _QuantTEGroupedLinear
from verl.utils.qat.modelopt import _batch_grouped_expert_weight_qdq, build_quantize_config

# isort: on

NUM_GEMMS = 4


def _expert_weight_quantizer(qat_mode):
    """The routed-expert weight quantizer ModelOpt builds for ``qat_mode``."""
    model = nn.Module()
    model.mlp = nn.Module()
    model.mlp.experts = nn.Module()
    model.mlp.experts.linear_fc1 = nn.Linear(256, 64, bias=False, dtype=torch.bfloat16)
    mtq.quantize(model, build_quantize_config(qat_mode))
    return model.mlp.experts.linear_fc1.weight_quantizer


def _expert_weights():
    torch.manual_seed(0)
    weights = [torch.randn(64, 256, dtype=torch.bfloat16) * scale for scale in (1e-3, 0.05, 1.0, 30.0)]
    weights[1][3] = 0  # an all-zero row
    weights[2][5, :32] = 6.0  # a group sitting on the largest E2M1 value
    return weights


class _Recorder:
    """An enabled pass-through quantizer that records the shapes it sees (the MXFP8 input quantizer needs CUDA)."""

    is_enabled = True

    def __init__(self, backend=None):
        self.backend = backend
        self.calls = []

    def __call__(self, tensor):
        self.calls.append(tensor.shape)
        return tensor


def _count_calls(module):
    calls = []
    module.register_forward_hook(lambda mod, inputs, output: calls.append(inputs[0].shape))
    return calls


def _grouped_call(weight_quantizer, input_quantizer, weights):
    """Run the (patched) grouped linear functional on TE's _GroupedLinear.apply argument layout."""
    _batch_grouped_expert_weight_qdq()
    seen = {}

    def apply(inp, non_tensor_args, *weights_and_biases):
        seen["inp"], seen["weights_and_biases"] = inp, weights_and_biases
        return inp

    module = SimpleNamespace(
        num_gemms=NUM_GEMMS,
        input_quantizer=input_quantizer,
        weight_quantizer=weight_quantizer,
        output_quantizer=nn.Identity(),
    )
    inp = torch.randn(8, 256, dtype=torch.bfloat16)
    biases = [torch.zeros(64, dtype=torch.bfloat16) for _ in range(NUM_GEMMS)]
    non_tensor_args = ([2] * NUM_GEMMS,)
    _QuantTEGroupedLinear._quantized_linear_fn(
        SimpleNamespace(_apply=apply), "_apply", module, inp, non_tensor_args, *weights, *biases
    )
    return seen, biases


def test_stacked_mxfp4_weight_qdq_matches_per_expert():
    weight_quantizer = _expert_weight_quantizer("mxfp4")
    weights = [w.requires_grad_(True) for w in _expert_weights()]
    per_expert = torch.stack([weight_quantizer(w) for w in weights])
    stacked = weight_quantizer(torch.stack(weights))
    assert torch.equal(per_expert, stacked)

    # the straight-through gradient reaches every expert weight unchanged
    upstream = torch.randn_like(stacked)
    grads_loop = torch.autograd.grad((per_expert * upstream).sum(), weights)
    grads_stacked = torch.autograd.grad((weight_quantizer(torch.stack(weights)) * upstream).sum(), weights)
    for g_loop, g_stacked, g_ref in zip(grads_loop, grads_stacked, upstream.unbind(0), strict=True):
        assert torch.equal(g_loop, g_stacked)
        assert torch.equal(g_stacked, g_ref)


def test_stacked_call_leaves_the_quantizer_unchanged_for_single_experts():
    # TensorQuantizer derives some state from the input's rank; a stacked call first must not change later 2-D calls
    weight_quantizer = _expert_weight_quantizer("mxfp4")
    weights = _expert_weights()
    stacked = weight_quantizer(torch.stack(weights))
    per_expert = torch.stack([weight_quantizer(w) for w in weights])
    assert torch.equal(stacked, per_expert)
    assert torch.equal(weight_quantizer(torch.stack(weights)), stacked)


def test_mxfp4_grouped_linear_quantizes_all_experts_in_one_call():
    weight_quantizer, input_quantizer = _expert_weight_quantizer("mxfp4"), _Recorder()
    weights = _expert_weights()
    expected = [weight_quantizer(w) for w in weights]
    weight_calls = _count_calls(weight_quantizer)

    seen, biases = _grouped_call(weight_quantizer, input_quantizer, weights)

    assert weight_calls == [torch.Size([NUM_GEMMS, 64, 256])]
    assert input_quantizer.calls == [torch.Size([8, 256])]
    quantized, passed_biases = seen["weights_and_biases"][:NUM_GEMMS], seen["weights_and_biases"][NUM_GEMMS:]
    for got, want in zip(quantized, expected, strict=True):
        assert torch.equal(got, want)
    assert all(got is want for got, want in zip(passed_biases, biases, strict=True))


def test_other_weight_backends_keep_the_per_expert_loop():
    weight_quantizer = _Recorder(backend="verl_fp8_block")

    _grouped_call(weight_quantizer, _Recorder(), _expert_weights())

    assert weight_quantizer.calls == [torch.Size([64, 256])] * NUM_GEMMS


def test_patch_is_applied_once():
    _batch_grouped_expert_weight_qdq()
    patched = _QuantTEGroupedLinear.__dict__["_quantized_linear_fn"]
    _batch_grouped_expert_weight_qdq()
    assert _QuantTEGroupedLinear.__dict__["_quantized_linear_fn"] is patched
