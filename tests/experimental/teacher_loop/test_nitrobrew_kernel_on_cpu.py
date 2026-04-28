# Copyright 2026 Tilde Research
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

"""CPU correctness tests for the nitrobrew chunked-vocab KL kernels.

Compares the chunked online-softmax forward (and the autograd backward) against
a naive reference that materialises the full ``[N, V]`` teacher logits.
"""

import os

# Ensure the @torch.compile decorators are no-ops on CPU runners.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import torch
import torch.nn.functional as F

from verl.trainer.distillation.fsdp.nitrobrew_loss import (
    _chunked_kl_forward,
    _chunked_reverse_kl_forward,
    _NitrobrewKL,
    _NitrobrewReverseKL,
)


def _naive_forward_kl(z: torch.Tensor, w: torch.Tensor, s: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """KL(p_T || p_S) per token, computed by materialising teacher logits."""
    zt = (z @ w.T).float() / temperature
    s = s.float() / temperature
    log_pt = F.log_softmax(zt, dim=-1)
    log_ps = F.log_softmax(s, dim=-1)
    pt = log_pt.exp()
    return (pt * (log_pt - log_ps)).sum(dim=-1)


def _naive_reverse_kl(z: torch.Tensor, w: torch.Tensor, s: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """KL(p_S || p_T) per token, computed by materialising teacher logits."""
    zt = (z @ w.T).float() / temperature
    s = s.float() / temperature
    log_pt = F.log_softmax(zt, dim=-1)
    log_ps = F.log_softmax(s, dim=-1)
    ps = log_ps.exp()
    return (ps * (log_ps - log_pt)).sum(dim=-1)


def _make_inputs(seed: int = 0, n: int = 4, v: int = 32, d: int = 8):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, d, generator=g)
    w = torch.randn(v, d, generator=g) * 0.1
    s = torch.randn(n, v, generator=g)
    return z, w, s


def test_chunked_forward_kl_matches_naive():
    z, w, s = _make_inputs()
    kl_chunked, _ = _chunked_kl_forward(z, w, s, chunk_V=4)
    kl_naive = _naive_forward_kl(z, w, s)
    assert torch.allclose(kl_chunked, kl_naive, atol=1e-5, rtol=1e-4), (
        f"max abs diff = {(kl_chunked - kl_naive).abs().max().item()}"
    )


def test_chunked_reverse_kl_matches_naive():
    z, w, s = _make_inputs(seed=1)
    kl_chunked, _, _ = _chunked_reverse_kl_forward(z, w, s, chunk_V=4)
    kl_naive = _naive_reverse_kl(z, w, s)
    assert torch.allclose(kl_chunked, kl_naive, atol=1e-5, rtol=1e-4), (
        f"max abs diff = {(kl_chunked - kl_naive).abs().max().item()}"
    )


def test_chunked_forward_kl_with_temperature():
    z, w, s = _make_inputs(seed=2)
    kl_chunked, _ = _chunked_kl_forward(z, w, s, chunk_V=8, temperature=0.7)
    kl_naive = _naive_forward_kl(z, w, s, temperature=0.7)
    assert torch.allclose(kl_chunked, kl_naive, atol=1e-5, rtol=1e-4)


def test_forward_kl_backward_matches_autograd():
    """_NitrobrewKL.backward must match a reference autograd through the naive forward."""
    z, w, s = _make_inputs(seed=3)
    s_chunked = s.clone().requires_grad_(True)
    s_naive = s.clone().requires_grad_(True)

    kl_chunked = _NitrobrewKL.apply(z, w, s_chunked, 4, 1.0, None)
    kl_naive = _naive_forward_kl(z, w, s_naive)

    grad_out = torch.randn_like(kl_chunked)
    kl_chunked.backward(grad_out)
    kl_naive.backward(grad_out)

    assert torch.allclose(s_chunked.grad, s_naive.grad, atol=1e-5, rtol=1e-4), (
        f"grad max abs diff = {(s_chunked.grad - s_naive.grad).abs().max().item()}"
    )


def test_reverse_kl_backward_matches_autograd():
    z, w, s = _make_inputs(seed=4)
    s_chunked = s.clone().requires_grad_(True)
    s_naive = s.clone().requires_grad_(True)

    kl_chunked = _NitrobrewReverseKL.apply(z, w, s_chunked, 4, 1.0)
    kl_naive = _naive_reverse_kl(z, w, s_naive)

    grad_out = torch.randn_like(kl_chunked)
    kl_chunked.backward(grad_out)
    kl_naive.backward(grad_out)

    assert torch.allclose(s_chunked.grad, s_naive.grad, atol=1e-5, rtol=1e-4), (
        f"grad max abs diff = {(s_chunked.grad - s_naive.grad).abs().max().item()}"
    )
