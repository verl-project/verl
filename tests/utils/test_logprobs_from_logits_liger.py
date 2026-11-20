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
"""Liger branch of logprobs_from_logits, checked against the pure-PyTorch reference.

Liger writes the gradient into its input during the forward pass, so the cases
that matter are the ones where the caller still reads `logits` afterwards:
entropy computed from the same tensor, either before the log-prob call (FSDP
engine order) or after it (prefix_grouper_utils.pg_forward order).
"""

import pytest
import torch

import verl.utils.torch_functional as verl_F
from verl.utils.device import get_device_name

pytestmark = pytest.mark.skipif(
    not verl_F.LIGER_CROSS_ENTROPY_LOSS_AVAILABLE
    or verl_F.NPU_CROSS_ENTROPY_LOSS_AVAILABLE
    or get_device_name() == "cpu",
    reason="needs liger-kernel on an accelerator",
)

N, V = 256, 151936
ENTROPY_COEFF = 0.01


@pytest.fixture
def liger_calls(monkeypatch):
    # flash-attn's CE comes first in the dispatch; disable it so CUDA reaches Liger too
    monkeypatch.setenv("VERL_DISABLE_FLASH_ATTN_CE", "1")
    monkeypatch.delenv("VERL_DISABLE_LIGER_CE", raising=False)
    # records whether the tensor handed to the Liger kernel requires grad,
    # i.e. whether Liger is allowed to write its gradient into it
    calls = []
    original = verl_F.liger_cross_entropy

    def recording(logits, *args, **kwargs):
        calls.append(logits.requires_grad)
        return original(logits, *args, **kwargs)

    monkeypatch.setattr(verl_F, "liger_cross_entropy", recording)
    return calls


def _inputs(dtype=torch.float32):
    gen = torch.Generator().manual_seed(0)
    base = (torch.randn(N, V, generator=gen) * 3).to(get_device_name(), dtype)
    labels = torch.randint(0, V, (N,), generator=gen).to(get_device_name())
    w_logprob = torch.randn(N, generator=gen).to(get_device_name())
    w_entropy = torch.randn(N, generator=gen).to(get_device_name())
    return base, labels, w_logprob, w_entropy


def _forward_backward(base, labels, w_logprob, w_entropy, with_entropy, logprob_fn, entropy_after=False):
    leaf = base.clone().requires_grad_(True)
    logits = leaf * 1.0  # non-leaf, like an lm_head output
    entropy = verl_F.entropy_from_logits(logits) if with_entropy and not entropy_after else None
    log_probs = logprob_fn(logits)
    if with_entropy and entropy_after:
        entropy = verl_F.entropy_from_logits(logits)
    loss = -(log_probs * w_logprob).sum()
    if with_entropy:
        loss = loss - ENTROPY_COEFF * (entropy * w_entropy).sum()
    loss.backward()
    return log_probs.detach(), leaf.grad


@pytest.mark.parametrize("with_entropy,entropy_after", [(False, False), (True, False), (True, True)])
def test_liger_matches_reference(liger_calls, with_entropy, entropy_after):
    base, labels, w_logprob, w_entropy = _inputs()
    ref_lp, ref_grad = _forward_backward(
        base,
        labels,
        w_logprob,
        w_entropy,
        with_entropy,
        lambda x: verl_F.logprobs_from_logits_naive(x, labels),
        entropy_after,
    )
    lp, grad = _forward_backward(
        base,
        labels,
        w_logprob,
        w_entropy,
        with_entropy,
        lambda x: verl_F.logprobs_from_logits(x, labels, inplace_backward=not with_entropy),
        entropy_after,
    )
    # in-place Liger only when nothing else reads the logits
    assert liger_calls == [not with_entropy]
    torch.testing.assert_close(lp, ref_lp, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(grad, ref_grad, atol=1e-5, rtol=1e-4)


def test_liger_keep_logits_leaves_logits_untouched(liger_calls):
    base, labels, _, _ = _inputs()
    logits = base.clone().requires_grad_(True) * 1.0
    before = logits.detach().clone()
    verl_F.logprobs_from_logits(logits, labels, inplace_backward=False)
    assert liger_calls == [False]
    assert torch.equal(logits.detach(), before)


def test_liger_inplace_overwrites_logits(liger_calls):
    # pins why callers must pass inplace_backward=False when they read logits afterwards
    base, labels, _, _ = _inputs()
    logits = base.clone().requires_grad_(True) * 1.0
    before = logits.detach().clone()
    verl_F.logprobs_from_logits(logits, labels, inplace_backward=True)
    assert liger_calls == [True]
    assert not torch.equal(logits.detach(), before)


def test_liger_no_grad_leaves_logits_untouched(liger_calls):
    base, labels, _, _ = _inputs()
    logits = base.clone()
    with torch.no_grad():
        lp = verl_F.logprobs_from_logits(logits, labels, inplace_backward=False)
    assert liger_calls == [False]
    assert torch.equal(logits, base)
    torch.testing.assert_close(lp, verl_F.logprobs_from_logits_naive(base, labels), atol=1e-4, rtol=1e-5)


def test_liger_is_bitwise_deterministic(liger_calls):
    base, labels, w_logprob, w_entropy = _inputs(torch.bfloat16)
    runs = [
        _forward_backward(
            base,
            labels,
            w_logprob,
            w_entropy,
            True,
            lambda x: verl_F.logprobs_from_logits(x, labels, inplace_backward=False),
        )
        for _ in range(2)
    ]
    assert torch.equal(runs[0][0], runs[1][0])
    assert torch.equal(runs[0][1], runs[1][1])


def test_disable_liger_env_falls_back_to_v2(liger_calls, monkeypatch):
    monkeypatch.setenv("VERL_DISABLE_LIGER_CE", "1")
    base, labels, _, _ = _inputs()
    lp = verl_F.logprobs_from_logits(base, labels)
    assert liger_calls == []
    assert torch.equal(lp, verl_F.logprobs_from_logits_v2(base, labels))


def test_liger_keep_logits_with_topk_distillation(liger_calls, monkeypatch):
    # FSDP engine order: top-k distillation reads the same fp32 logits first and keeps
    # them for its logsumexp backward. A small backward chunk also covers the chunk loop.
    from verl.trainer.distillation.fsdp.losses import _chunked_topk_log_probs

    monkeypatch.setattr(verl_F, "_LIGER_KEEP_LOGITS_BWD_CHUNK_SIZE", 100)
    base, labels, w_logprob, w_distill = _inputs()
    topk_ids = torch.randint(0, V, (1, N, 8), generator=torch.Generator().manual_seed(1)).to(get_device_name())

    def forward_backward(logprob_fn):
        leaf = base.clone().requires_grad_(True)
        logits = leaf * 1.0  # non-leaf, like an lm_head output
        distill = _chunked_topk_log_probs(logits.unsqueeze(0), topk_ids).squeeze(0).sum(-1)
        log_probs = logprob_fn(logits)
        (-(log_probs * w_logprob).sum() - (distill * w_distill).sum()).backward()
        return log_probs.detach(), leaf.grad

    ref_lp, ref_grad = forward_backward(lambda x: verl_F.logprobs_from_logits_naive(x, labels))
    lp, grad = forward_backward(lambda x: verl_F.logprobs_from_logits(x, labels, inplace_backward=False))
    assert liger_calls == [False]
    torch.testing.assert_close(lp, ref_lp, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(grad, ref_grad, atol=1e-5, rtol=1e-4)
