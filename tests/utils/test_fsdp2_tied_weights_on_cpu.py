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
"""Weight tying must survive the meta-init path used by ``fsdp2_load_full_state_dict``.

Ranks other than 0 build the module on meta and are materialized with
``Module.to_empty()``, which hands every parameter fresh storage and therefore
drops the aliasing that makes ``lm_head.weight`` *be* the input embedding. That
failure is silent: the model still runs and still converges, but the embedding
only receives the input-side half of its gradient, so the broadcast ranks train a
different model from rank 0.
"""

import pytest
import torch
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM, LlamaConfig


def _tied_config():
    return LlamaConfig(
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        hidden_size=128,
        intermediate_size=64,
        vocab_size=64,
        tie_word_embeddings=True,
    )


def _is_tied(model) -> bool:
    return model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()


def test_rank0_path_is_tied():
    """The rank-0 branch materializes on CPU directly and keeps the tie."""
    model = AutoModelForCausalLM.from_config(_tied_config())
    assert _is_tied(model)


def test_to_empty_drops_the_tie_and_tie_weights_restores_it():
    """This is the hazard the meta-init path has to compensate for."""
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(_tied_config())
    assert _is_tied(model), "tie should hold while still on meta"

    model = model.to_empty(device="cpu")
    assert not _is_tied(model), (
        "to_empty is expected to drop the tie; if this ever starts passing, "
        "the re-tie in fsdp2_load_full_state_dict can go"
    )

    model.tie_weights()
    assert _is_tied(model)


@pytest.mark.parametrize("retie", [True, False])
def test_embedding_gradient_matches_rank0_only_when_retied(retie: bool):
    """A dropped tie halves the embedding gradient -- silently."""
    torch.manual_seed(0)
    reference = AutoModelForCausalLM.from_config(_tied_config())

    with init_empty_weights():
        broadcast_rank = AutoModelForCausalLM.from_config(_tied_config())
    broadcast_rank = broadcast_rank.to_empty(device="cpu")
    # stands in for set_model_state_dict(..., broadcast_from_rank0=True): rank 0's
    # state dict carries lm_head.weight, so values agree either way -- only the
    # aliasing differs.
    broadcast_rank.load_state_dict(reference.state_dict(), strict=False)
    # buffers (e.g. rotary inv_freq) are not in the state dict and to_empty leaves
    # them uninitialized, which is why fsdp2_load_full_state_dict broadcasts them
    # separately. Mirror that here so the only difference under test is the tie.
    ref_buffers = dict(reference.named_buffers())
    for name, buf in broadcast_rank.named_buffers():
        buf.copy_(ref_buffers[name])
    if retie:
        broadcast_rank.tie_weights()

    ids = torch.randint(0, 64, (2, 8))
    grads = []
    for model in (reference, broadcast_rank):
        model.zero_grad()
        model(input_ids=ids, labels=ids).loss.backward()
        grads.append(model.model.embed_tokens.weight.grad.clone())

    if retie:
        torch.testing.assert_close(grads[0], grads[1])
    else:
        assert not torch.allclose(grads[0], grads[1]), (
            "an untied lm_head must change the embedding gradient, otherwise this test proves nothing"
        )
