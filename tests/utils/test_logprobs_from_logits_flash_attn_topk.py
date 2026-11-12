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
"""flash-attn's in-place CE backward vs top-k distillation on the same logits.

FSDP engine order: `_chunked_topk_log_probs` runs before the log-prob call. For
fp32 logits its `.float()` does not copy, so its logsumexp keeps the logits
buffer for backward, and flash-attn's in-place backward overwrites it first.
bf16 logits are copied by `.float()` and are not affected.
"""

import pytest
import torch

import verl.utils.torch_functional as verl_F
from verl.trainer.distillation.fsdp.losses import _chunked_topk_log_probs
from verl.utils.device import get_device_name

pytestmark = pytest.mark.skipif(
    not verl_F.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE or get_device_name() != "cuda",
    reason="needs flash-attn cross entropy on CUDA",
)

N, V, K = 256, 151936, 8


def _student_grad(dtype, logprob_fn):
    gen = torch.Generator().manual_seed(0)
    base = (torch.randn(N, V, generator=gen) * 3).to("cuda", dtype)
    labels = torch.randint(0, V, (N,), generator=gen).to("cuda")
    topk_ids = torch.randint(0, V, (1, N, K), generator=gen).to("cuda")
    w_logprob = torch.randn(N, generator=gen).to("cuda")
    w_distill = torch.randn(N, generator=gen).to("cuda")

    leaf = base.clone().requires_grad_(True)
    logits = leaf * 1.0  # non-leaf, like an lm_head output
    distill = _chunked_topk_log_probs(logits.unsqueeze(0), topk_ids).squeeze(0).float().sum(-1)
    log_probs = logprob_fn(logits, labels).float()
    (-(log_probs * w_logprob).sum() - (distill * w_distill).sum()).backward()
    return leaf.grad.float()


def _rel_err(dtype, inplace_backward, monkeypatch):
    monkeypatch.setenv("VERL_DISABLE_FLASH_ATTN_CE", "0")
    grad = _student_grad(dtype, lambda x, y: verl_F.logprobs_from_logits(x, y, inplace_backward=inplace_backward))
    ref = _student_grad(dtype, verl_F.logprobs_from_logits_naive)
    return ((grad - ref).norm() / ref.norm()).item()


def test_flash_ce_keep_logits_with_fp32_topk_distillation(monkeypatch):
    # what the FSDP engine passes for fp32 logits with top-k distillation
    assert _rel_err(torch.float32, inplace_backward=False, monkeypatch=monkeypatch) < 1e-5


def test_flash_ce_inplace_corrupts_fp32_topk_distillation(monkeypatch):
    # pins why the engine guard exists: no error is raised, the gradient is just wrong
    assert _rel_err(torch.float32, inplace_backward=True, monkeypatch=monkeypatch) > 1e-2


def test_flash_ce_inplace_safe_with_bf16_topk_distillation(monkeypatch):
    # pins why the guard is fp32-only: `.float()` copies bf16 logits
    keep = _rel_err(torch.bfloat16, inplace_backward=False, monkeypatch=monkeypatch)
    inplace = _rel_err(torch.bfloat16, inplace_backward=True, monkeypatch=monkeypatch)
    assert inplace == pytest.approx(keep, rel=1e-3)
