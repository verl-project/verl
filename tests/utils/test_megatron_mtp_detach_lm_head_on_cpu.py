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

"""The MTP loss must train the MTP block but never the LM head (or a tied input embedding)."""

from inspect import signature
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron")

from megatron.core.tensor_parallel.layers import ColumnParallelLinear  # noqa: E402

import verl.models.mcore.mtp_patch as mtp_patch  # noqa: E402

if not mtp_patch._HAS_PROCESS_MTP_LOSS:
    pytest.skip("installed megatron-core has no process_mtp_loss", allow_module_level=True)
process_mtp_loss = mtp_patch._process_mtp_loss

SEQ, BATCH, HIDDEN, VOCAB = 4, 1, 8, 16


class _OutputLayer(torch.nn.Module):
    """CPU stand-in for ColumnParallelLinear (TP=1) with the same forward contract."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(VOCAB, HIDDEN))

    def forward(self, input_, weight=None, runtime_gather_output=None):
        weight = self.weight if weight is None else weight
        return input_ @ weight.t(), None


def _config():
    return SimpleNamespace(mtp_num_layers=1, mtp_loss_scaling_factor=0.1, calculate_per_token_loss=False)


def _cross_entropy(labels, logits):
    # labels: [b, s]; logits: [s, b, v] -> per-token loss [b, s], as Megatron's compute_language_model_loss
    return torch.nn.functional.cross_entropy(
        logits.transpose(0, 1).reshape(-1, VOCAB), labels.reshape(-1), reduction="none"
    ).view_as(labels)


def _run_mtp_loss(output_layer, output_weight=None):
    """Backpropagate only the MTP loss through the real process_mtp_loss; return the stacked hidden."""
    hidden = torch.randn(2 * SEQ, BATCH, HIDDEN, requires_grad=True)  # [main; mtp_1]
    labels = torch.randint(0, VOCAB, (BATCH, SEQ))
    out = process_mtp_loss(
        hidden_states=hidden,
        labels=labels,
        loss_mask=None,
        output_layer=output_layer,
        output_weight=output_weight,
        runtime_gather_output=None,
        is_training=False,
        compute_language_model_loss=_cross_entropy,
        config=_config(),
    )
    # zero upstream grad: whatever reaches the parameters comes from MTPLossAutoScaler alone
    (out * 0).sum().backward()
    return hidden


def test_fake_output_layer_matches_column_parallel_linear_forward():
    assert list(signature(_OutputLayer.forward).parameters) == list(signature(ColumnParallelLinear.forward).parameters)


def test_without_detach_the_mtp_loss_updates_the_lm_head():
    """Negative control: proves the fixture can observe the leak the fix removes."""
    layer = _OutputLayer()
    _run_mtp_loss(layer)
    assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0


def test_detached_output_layer_keeps_the_untied_lm_head_out_of_the_mtp_loss():
    layer = _OutputLayer()
    hidden = _run_mtp_loss(mtp_patch.detached_output_layer(layer))

    assert layer.weight.grad is None
    assert hidden.grad[SEQ:].abs().sum() > 0, "the MTP branch itself must still receive gradient"


def test_detached_output_layer_keeps_the_tied_embedding_out_of_the_mtp_loss():
    layer = _OutputLayer()
    shared = torch.nn.Parameter(torch.randn(VOCAB, HIDDEN))
    hidden = _run_mtp_loss(mtp_patch.detached_output_layer(layer), output_weight=shared)

    assert shared.grad is None
    assert layer.weight.grad is None
    assert hidden.grad[SEQ:].abs().sum() > 0


def test_postprocess_hands_process_mtp_loss_a_detached_output_layer(monkeypatch):
    layer = _OutputLayer()
    model = SimpleNamespace(
        share_embeddings_and_output_weights=False,
        post_process=True,
        config=SimpleNamespace(**vars(_config()), use_mup=False),
        output_layer=layer,
        training=False,
        compute_language_model_loss=_cross_entropy,
    )
    captured = {}

    def fake_process(**kwargs):
        captured.update(kwargs)
        return kwargs["hidden_states"][:SEQ]

    monkeypatch.setattr(mtp_patch, "_HAS_PROCESS_MTP_LOSS", True)
    monkeypatch.setattr(mtp_patch, "_PROCESS_MTP_LOSS_PARAMS", {"hidden_states", "output_layer", "output_weight"})
    monkeypatch.setattr(mtp_patch, "_process_mtp_loss", fake_process, raising=False)
    mtp_patch._megatron_gptmodel_postprocess(
        model,
        hidden_states=torch.randn(2 * SEQ, BATCH, HIDDEN),
        input_ids=None,
        position_ids=None,
        labels=torch.zeros(BATCH, SEQ, dtype=torch.long),
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
    )

    mtp_logits, _ = captured["output_layer"](
        torch.randn(SEQ, BATCH, HIDDEN, requires_grad=True), weight=captured["output_weight"]
    )
    mtp_logits.sum().backward()
    assert layer.weight.grad is None
