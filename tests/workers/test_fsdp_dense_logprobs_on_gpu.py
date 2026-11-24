# Copyright 2026 Individual Contributor
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

from verl.utils import torch_functional as vf
from verl.workers.engine.fsdp.transformer_impl import _logprobs_from_dense_logits

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("flash", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("lengths", [[31], [19, 7, 0]])
@pytest.mark.parametrize("shared_loss", [False, True])
@pytest.mark.parametrize("layout", ["contiguous", "token_stride", "vocab_stride"])
def test_cuda_dense_views_preserve_logits_and_gradients(monkeypatch, flash, dtype, lengths, shared_loss, layout):
    if flash and not vf.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE:
        pytest.skip("flash-attn cross entropy is not installed")
    monkeypatch.setenv("VERL_DISABLE_FLASH_ATTN_CE", "0" if flash else "1")
    torch.manual_seed(41)
    # Strided token views with contiguous vocabulary rows exercise actual CUDA
    # kernels without assuming dense batch/sequence storage.
    shape = (len(lengths), 64 if layout == "token_stride" else max(lengths), 514 if layout == "vocab_stride" else 257)
    base = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    reference = base.detach().clone().requires_grad_()
    # Model operations may save their logits even with no engine auxiliary loss.
    logits = base.softmax(-1)
    ref_logits = reference.softmax(-1)
    if layout == "token_stride":
        logits, ref_logits = logits[:, ::2], ref_logits[:, ::2]
    elif layout == "vocab_stride":
        logits, ref_logits = logits[..., ::2], ref_logits[..., ::2]
    original = logits.detach().clone()
    labels = torch.arange(sum(lengths), device="cuda") % 257
    actual = _logprobs_from_dense_logits(logits, labels, torch.tensor(lengths, device="cuda"))
    packed = torch.cat([row[:n] for row, n in zip(ref_logits, lengths, strict=True)])
    expected = packed.float().log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)
    weights = torch.randn_like(expected)
    loss = (actual * weights).sum()
    expected_loss = (expected * weights).sum()
    if shared_loss:
        # These operate on the *same* dense storage the new helper consumes.
        loss = loss + 0.2 * vf.entropy_from_logits(logits).sum()
        loss = loss + 0.3 * vf.calculate_sum_pi_squared_from_logits(logits).sum()
        expected_loss = expected_loss + 0.2 * vf.entropy_from_logits(ref_logits).sum()
        expected_loss = expected_loss + 0.3 * vf.calculate_sum_pi_squared_from_logits(ref_logits).sum()
    loss.backward()
    expected_loss.backward()
    tolerance = {"atol": 0.04, "rtol": 0.02} if dtype == torch.bfloat16 else {"atol": 2e-6, "rtol": 2e-5}
    torch.testing.assert_close(actual.float(), expected, **tolerance)
    grad_tol = {"atol": 2e-5, "rtol": 0.03} if dtype == torch.bfloat16 else {"atol": 2e-7, "rtol": 2e-5}
    torch.testing.assert_close(base.grad, reference.grad, **grad_tol)
    torch.testing.assert_close(logits, original, atol=0, rtol=0)
