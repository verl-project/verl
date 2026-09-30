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

import torch

from verl.utils.torch_functional import logprobs_from_logits_sampler, logprobs_from_logits_v2


def test_sampler_formula_is_fp32_log_softmax_gather():
    torch.manual_seed(0)
    logits = torch.randn(3, 5, 11, dtype=torch.bfloat16)
    labels = torch.randint(0, 11, (3, 5))
    out = logprobs_from_logits_sampler(logits, labels, inplace_backward=False)
    ref = torch.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    assert out.dtype == torch.float32
    assert out.shape == labels.shape
    assert torch.equal(out, ref)


def test_sampler_formula_differs_from_bf16_path():
    torch.manual_seed(0)
    logits = torch.randn(4, 64, dtype=torch.bfloat16) * 8
    labels = torch.randint(0, 64, (4,))
    native = logprobs_from_logits_v2(logits, labels)
    sampler = logprobs_from_logits_sampler(logits, labels)
    assert native.dtype == torch.bfloat16
    assert not torch.equal(native.float(), sampler)


def test_chunked_rows_match_whole_tensor_bitwise():
    torch.manual_seed(0)
    logits = torch.randn(50, 4096, dtype=torch.bfloat16) * 4
    labels = torch.randint(0, 4096, (50,))
    ref = torch.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    for chunk_size in (1, 7, 50, 2048):
        out = logprobs_from_logits_sampler(logits, labels, chunk_size=chunk_size)
        assert torch.equal(out, ref)
