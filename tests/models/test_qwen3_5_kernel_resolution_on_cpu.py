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

"""Where the Qwen3.5 patch looks for the delta-net kernels and the block type.

Both moved in transformers: the kernels from GatedDeltaNet attributes to module
scope, and the decoder layer's `layer_type` to `block_type`. These pin the
resolution so a rename upstream fails loudly here instead of at the first step.
"""

import inspect

import pytest
import torch

from verl.models.transformers import qwen3_5
from verl.models.transformers.qwen3_5 import _call_accepts_kwarg, _delta_net_kernel, qwen3_5_decoder_layer_forward

KERNELS = [
    "chunk_gated_delta_rule",
    "recurrent_gated_delta_rule",
    "causal_conv1d_fn",
    "causal_conv1d_update",
]


class _Bare:
    """A module that carries none of the kernels, like current transformers."""


@pytest.fixture
def without_fast_kernels(monkeypatch):
    """Resolve as if neither FLA nor causal-conv1d were installed."""
    monkeypatch.setattr(qwen3_5, "_fast_kernel", lambda name: None)


@pytest.mark.parametrize("name", ["chunk_gated_delta_rule", "recurrent_gated_delta_rule", "causal_conv1d_update"])
def test_a_kernel_absent_from_the_instance_is_found_anyway(name):
    assert callable(_delta_net_kernel(_Bare(), name))


def test_an_attribute_on_the_instance_still_wins():
    sentinel = object()

    class WithAttr:
        chunk_gated_delta_rule = sentinel

    assert _delta_net_kernel(WithAttr(), "chunk_gated_delta_rule") is sentinel


def test_a_none_attribute_on_the_instance_is_honoured():
    # Older transformers resolved the conv kernel on the instance and set it to
    # None when causal-conv1d was missing; that answer must not be second-guessed.
    class WithNone:
        causal_conv1d_fn = None

    assert _delta_net_kernel(WithNone(), "causal_conv1d_fn") is None


def test_the_chunk_rule_keeps_the_kwargs_this_file_introspects():
    # _packed_chunk_gated_delta_rule asks the resolved function whether it accepts
    # cu_seqlens and cp_context. transformers' decorated wrapper re-exports the torch
    # fallback's signature and hides both, which silently disables the packed fast
    # path and makes ulysses SP raise NotImplementedError.
    fla = pytest.importorskip("fla.ops.gated_delta_rule")
    params = inspect.signature(_delta_net_kernel(_Bare(), "chunk_gated_delta_rule")).parameters
    assert "cu_seqlens" in params
    assert "cp_context" in params
    assert _delta_net_kernel(_Bare(), "chunk_gated_delta_rule") is fla.chunk_gated_delta_rule


def test_a_missing_conv_kernel_resolves_to_none_so_the_slow_conv_runs(without_fast_kernels):
    # transformers exposes a torch `causal_conv1d_fn` even without causal-conv1d,
    # but it takes `hidden_states`, not the `x=` this file passes, and it knows
    # nothing about packed sequences. The callers' own per-sequence conv1d does.
    assert _delta_net_kernel(_Bare(), "causal_conv1d_fn") is None


@pytest.mark.parametrize("name", ["chunk_gated_delta_rule", "recurrent_gated_delta_rule"])
def test_a_missing_delta_rule_does_not_claim_packed_sequence_support(without_fast_kernels, name):
    # The torch reference swallows unknown kwargs, so through it `cu_seqlens` looks
    # accepted while being ignored, and packed examples would be processed as one
    # continuous sequence. The closed signature makes the packed path split instead.
    fn = _delta_net_kernel(_Bare(), name)
    assert callable(fn)
    assert not _call_accepts_kwarg(fn, "cu_seqlens")
    assert not _call_accepts_kwarg(fn, "cp_context")
    assert _call_accepts_kwarg(fn, "initial_state")


@pytest.mark.parametrize("name", ["chunk_gated_delta_rule", "recurrent_gated_delta_rule"])
def test_the_closed_reference_computes_what_transformers_reference_computes(without_fast_kernels, name):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 8, 2, 4) for _ in range(3))
    g, beta = torch.randn(1, 8, 2), torch.rand(1, 8, 2)
    kwargs = dict(g=g, beta=beta, initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True)

    out, state = _delta_net_kernel(_Bare(), name)(q, k, v, **kwargs)
    ref_out, ref_state = getattr(hf, f"torch_{name}")(q, k, v, **kwargs)
    torch.testing.assert_close(out, ref_out)
    torch.testing.assert_close(state, ref_state)


def test_an_unknown_kernel_name_is_refused():
    with pytest.raises(AttributeError, match="not a delta-net kernel"):
        _delta_net_kernel(_Bare(), "no_such_kernel")


def test_a_decoder_layer_with_no_recognisable_block_type_refuses_to_run():
    # Without this the if/elif fell through and the layer returned its input, i.e.
    # training proceeded with no attention at all and no error.
    class Layer:
        block_type = "something_else"

        def input_layernorm(self, x):
            return x

    with pytest.raises(ValueError, match="skip attention"):
        qwen3_5_decoder_layer_forward(Layer(), hidden_states=None, position_embeddings=None)
