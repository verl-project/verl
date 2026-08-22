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

from verl.models.transformers.qwen3_5 import _delta_net_kernel, qwen3_5_decoder_layer_forward

KERNELS = [
    "chunk_gated_delta_rule",
    "recurrent_gated_delta_rule",
    "causal_conv1d_fn",
    "causal_conv1d_update",
]


class _Bare:
    """A module that carries none of the kernels, like current transformers."""


@pytest.mark.parametrize("name", KERNELS)
def test_a_kernel_absent_from_the_instance_is_found_anyway(name):
    assert callable(_delta_net_kernel(_Bare(), name))


def test_an_attribute_on_the_instance_still_wins():
    sentinel = object()

    class WithAttr:
        chunk_gated_delta_rule = sentinel

    assert _delta_net_kernel(WithAttr(), "chunk_gated_delta_rule") is sentinel


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


def test_an_unknown_kernel_name_is_refused():
    with pytest.raises(AttributeError, match="delta-net kernels live"):
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
