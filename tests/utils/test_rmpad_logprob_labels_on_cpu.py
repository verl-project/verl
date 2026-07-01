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
import sys
import types

import torch

from verl.utils import torch_functional


def _pad_input(hidden_states, indices, batch, seqlen):
    output = torch.zeros(
        batch * seqlen,
        *hidden_states.shape[1:],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    output[indices] = hidden_states
    return output.view(batch, seqlen, *hidden_states.shape[1:])


def _unpad_input(hidden_states, attention_mask):
    indices = torch.nonzero(attention_mask.reshape(-1), as_tuple=False).squeeze(-1)
    hidden_states = hidden_states.reshape(attention_mask.numel(), *hidden_states.shape[2:])
    return hidden_states[indices], indices, None, None, None


def _install_fake_flash_attn_padding(monkeypatch):
    flash_attn = types.ModuleType("flash_attn")
    bert_padding = types.ModuleType("flash_attn.bert_padding")
    bert_padding.pad_input = _pad_input
    bert_padding.unpad_input = _unpad_input
    flash_attn.bert_padding = bert_padding
    monkeypatch.setitem(sys.modules, "flash_attn", flash_attn)
    monkeypatch.setitem(sys.modules, "flash_attn.bert_padding", bert_padding)


def _return_labels_as_log_probs(monkeypatch):
    def fake_logprobs_from_logits(logits, labels, inplace_backward=True):
        return labels.to(dtype=torch.float32)

    monkeypatch.setattr(torch_functional, "logprobs_from_logits", fake_logprobs_from_logits)


def test_shifted_labels_do_not_wrap_within_or_across_sequences():
    input_ids = torch.tensor(
        [
            [10, 11, 12],
            [20, 21, 22],
        ]
    )
    indices = torch.arange(input_ids.numel())

    labels = torch_functional.get_unpad_sequence_shifted_labels(input_ids, indices)

    expected = torch.tensor([11, 12, 12, 21, 22, 22])
    torch.testing.assert_close(labels, expected)


def test_response_rmpad_labels_do_not_cross_sequence_boundaries(monkeypatch):
    _install_fake_flash_attn_padding(monkeypatch)
    _return_labels_as_log_probs(monkeypatch)

    input_ids = torch.tensor(
        [
            [10, 11, 12, 0, 0],
            [20, 21, 0, 0, 0],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    logits_rmpad = torch.zeros((5, 30))

    output = torch_functional.log_probs_from_logits_response_rmpad(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_rmpad=logits_rmpad,
        response_length=4,
    )

    expected = torch.tensor(
        [
            [11, 12, 0, 0],
            [21, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(output, expected)


def test_all_rmpad_labels_do_not_cross_sequence_boundaries(monkeypatch):
    _install_fake_flash_attn_padding(monkeypatch)
    _return_labels_as_log_probs(monkeypatch)
    monkeypatch.setattr(torch_functional, "get_device_name", lambda: "cuda")

    input_ids_rmpad = torch.tensor([[10, 11, 12, 20, 21]])
    indices = torch.tensor([0, 1, 2, 5, 6])
    logits_rmpad = torch.zeros((5, 30))

    output = torch_functional.log_probs_from_logits_all_rmpad(
        input_ids_rmpad=input_ids_rmpad,
        logits_rmpad=logits_rmpad,
        indices=indices,
        batch_size=2,
        seqlen=5,
        response_length=4,
    )

    expected = torch.tensor(
        [
            [11, 12, 0, 0],
            [21, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(output, expected)
