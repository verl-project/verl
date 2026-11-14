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

import pytest
import torch

import verl.utils.torch_functional as verl_F


def test_liger_is_skipped_for_cpu_tensors(monkeypatch):
    # Liger's Triton kernel cannot run on CPU tensors even when liger-kernel is installed.
    monkeypatch.setattr(verl_F, "FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE", False)
    monkeypatch.setattr(verl_F, "NPU_CROSS_ENTROPY_LOSS_AVAILABLE", False)
    monkeypatch.setattr(verl_F, "LIGER_CROSS_ENTROPY_LOSS_AVAILABLE", True)

    def unexpected(*args, **kwargs):
        raise AssertionError("Liger must not be dispatched for CPU tensors")

    monkeypatch.setattr(verl_F, "logprobs_from_logits_liger", unexpected)

    logits = torch.randn(4, 7, 32)
    labels = torch.randint(0, 32, (4, 7))
    output = verl_F.logprobs_from_logits(logits, labels)
    torch.testing.assert_close(output, verl_F.logprobs_from_logits_naive(logits, labels))


@pytest.mark.parametrize("calculate_entropy", [False, True])
def test_prefix_grouper_keeps_logits_for_entropy(monkeypatch, calculate_entropy):
    pytest.importorskip("prefix_grouper")
    from verl.trainer.ppo import prefix_grouper_utils

    seen = {}

    def fake_logprobs_from_logits(logits, labels, inplace_backward=True):
        seen["inplace_backward"] = inplace_backward
        return torch.zeros(labels.shape)

    monkeypatch.setattr(prefix_grouper_utils, "logprobs_from_logits", fake_logprobs_from_logits)

    logits = torch.randn(2, 5, 16)

    class Model:
        def __call__(self, **kwargs):
            return type("Output", (), {"logits": logits})()

    class Grouper:
        def split_output(self, logits, include_prefix_last):
            return logits, None, logits, torch.ones(logits.shape[:2], dtype=torch.bool)

        def convert_padding(self, ids, mask, padding_mode):
            return ids

    prefix_grouper_utils.pg_forward(
        Model(),
        Grouper(),
        concat_input_ids=None,
        attention_mask=None,
        position_ids=None,
        completion_ids=torch.zeros(2, 4, dtype=torch.long),
        completion_mask=None,
        calculate_entropy=calculate_entropy,
        entropy_fn=verl_F.entropy_from_logits,
    )
    assert seen["inplace_backward"] is (not calculate_entropy)
