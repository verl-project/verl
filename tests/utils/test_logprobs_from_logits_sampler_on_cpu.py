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


def test_backward_matches_reference_without_keeping_fp32_log_softmax():
    torch.manual_seed(0)
    logits = (torch.randn(40, 4096, dtype=torch.bfloat16) * 4).requires_grad_()
    labels = torch.randint(0, 4096, (40,))
    ref_logits = logits.detach().clone().requires_grad_()
    grad_out = torch.randn(40)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append((t.dtype, tuple(t.shape))) or t, lambda t: t):
        out = logprobs_from_logits_sampler(logits, labels, chunk_size=16)
    out.backward(grad_out)
    ref = torch.log_softmax(ref_logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    ref.backward(grad_out)
    assert torch.equal(out, ref.detach())
    assert torch.equal(logits.grad, ref_logits.grad)
    assert (torch.float32, (16, 4096)) not in saved, saved


def test_temperature_divides_the_fp32_logits_like_the_vllm_sampler():
    # vLLM casts the logits to fp32, then divides by the fp32 temperature.
    torch.manual_seed(0)
    logits = torch.randn(50, 4096, dtype=torch.bfloat16) * 4
    labels = torch.randint(0, 4096, (50,))
    temperature = torch.full((50,), 0.7)
    ref = torch.log_softmax(logits.float() / temperature.unsqueeze(-1), dim=-1)
    ref = ref.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    for chunk_size in (7, 2048):
        out = logprobs_from_logits_sampler(logits, labels, temperature=temperature, chunk_size=chunk_size)
        assert torch.equal(out, ref)
    bf16_scaled = logits / temperature.unsqueeze(-1).to(logits.dtype)
    assert not torch.equal(logprobs_from_logits_sampler(bf16_scaled, labels), ref)


def test_greedy_temperature_reports_unscaled_log_probs_like_the_vllm_sampler():
    # vLLM samples greedily below a temperature of 1e-5 and reports the log-probs of the unscaled fp32 logits.
    torch.manual_seed(0)
    logits = torch.randn(6, 64, dtype=torch.bfloat16) * 4
    labels = torch.randint(0, 64, (6,))
    temperature = torch.tensor([0.0, 1e-6, 0.0, 1.0, 0.0, 0.0])
    ref = torch.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    assert torch.equal(logprobs_from_logits_sampler(logits, labels, temperature=temperature), ref)
