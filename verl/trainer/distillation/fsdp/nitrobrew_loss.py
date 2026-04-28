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

"""Nitrobrew: fused, constant-memory KL divergence from hidden states (FSDP path).

Computes KL(teacher || student) or KL(student || teacher) without materialising
the full [N, V] teacher logit tensor. Teacher logits are reconstructed on-the-fly
as z @ W.T in vocabulary chunks of size C using single-pass online-softmax.

Peak extra memory: O(N * C) per chunk, instead of O(N * V).
"""

import torch

from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size, slice_input_tensor
from verl.workers.config import DistillationConfig

_CHUNK_V: int = 1024


# ---------------------------------------------------------------------------
# Forward KL: KL(p_T || p_S)
# ---------------------------------------------------------------------------


@torch.compile
def _fwd_chunk_update(
    z_f: torch.Tensor,       # (N, D_t) float32
    W_chunk: torch.Tensor,   # (C, D_t) float32
    zs_chunk: torch.Tensor,  # (N, C)   float32
    mt: torch.Tensor,        # (N,) running teacher max
    st: torch.Tensor,        # (N,) sum exp(zt - mt)
    tt: torch.Tensor,        # (N,) sum exp(zt - mt) * zt
    ut: torch.Tensor,        # (N,) sum exp(zt - mt) * zs_clamped
    s_min: torch.Tensor,     # (N, 1) floor for student logits
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Online-softmax update for one teacher vocab chunk."""
    zt = z_f @ W_chunk.T
    tile_mt = zt.max(dim=1).values
    new_mt = torch.maximum(mt, tile_mt)
    alpha = (mt - new_mt).exp()
    pt = (zt - new_mt.unsqueeze(1)).exp()
    st = st * alpha + pt.sum(dim=1)
    tt = tt * alpha + (pt * zt).sum(dim=1)
    zs_clamped = torch.maximum(zs_chunk, s_min)
    ut = ut * alpha + (pt * zs_clamped).sum(dim=1)
    return new_mt, st, tt, ut


def _chunked_kl_forward(
    z: torch.Tensor,              # (N, D_t)
    W: torch.Tensor,              # (V, D_t)
    student_logits: torch.Tensor, # (N, V)
    chunk_V: int = _CHUNK_V,
    temperature: float = 1.0,
    log_prob_min_clamp: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-pass chunked forward. Returns (kl, t_lse) both (N,) float32."""
    N = z.shape[0]
    V = W.shape[0]
    device = z.device

    z_f = z.float()
    W_f = W.to(device=device, dtype=torch.float32)
    s_f = student_logits.float()

    if temperature != 1.0:
        inv_T = 1.0 / temperature
        z_f = z_f * inv_T
        s_f = s_f * inv_T

    s_lse = s_f.logsumexp(dim=-1)

    if log_prob_min_clamp is not None:
        s_min = (s_lse + log_prob_min_clamp).unsqueeze(1)
    else:
        s_min = torch.full((1, 1), float("-inf"), dtype=torch.float32, device=device)

    mt = torch.full((N,), float("-inf"), dtype=torch.float32, device=device)
    st = torch.zeros(N, dtype=torch.float32, device=device)
    tt = torch.zeros(N, dtype=torch.float32, device=device)
    ut = torch.zeros(N, dtype=torch.float32, device=device)

    for v in range(0, V, chunk_V):
        mt, st, tt, ut = _fwd_chunk_update(
            z_f, W_f[v : v + chunk_V], s_f[:, v : v + chunk_V],
            mt, st, tt, ut, s_min,
        )

    t_lse = mt + st.log()
    kl = tt / st - t_lse - ut / st + s_lse
    return kl, t_lse


# ---------------------------------------------------------------------------
# Backward: standard d(KL)/d(zs_v) = p_S(v) - p_T(v)
# ---------------------------------------------------------------------------


@torch.compile
def _bwd_chunk(
    z_f: torch.Tensor,       # (N, D_t) float32
    W_chunk: torch.Tensor,   # (C, D_t) float32
    zs_chunk: torch.Tensor,  # (N, C)   float32
    t_lse: torch.Tensor,     # (N,)
    s_lse: torch.Tensor,     # (N,)
    grad: torch.Tensor,      # (N,)
) -> torch.Tensor:
    """d(KL)/d(student_logits) for one vocab chunk. Returns (N, C)."""
    zt = z_f @ W_chunk.T
    p_T = (zt - t_lse.unsqueeze(1)).exp()
    p_S = (zs_chunk - s_lse.unsqueeze(1)).exp()
    return (p_S - p_T) * grad.unsqueeze(1)


# ---------------------------------------------------------------------------
# autograd.Function -- wires forward & backward
# ---------------------------------------------------------------------------


class _NitrobrewKL(torch.autograd.Function):
    """KL(p_T || p_S) with chunked forward and backward over vocab axis."""

    @staticmethod
    def forward(ctx, z, W, student_logits, chunk_V, temperature=1.0,
                log_prob_min_clamp=None):
        kl, t_lse = _chunked_kl_forward(
            z, W, student_logits, chunk_V, temperature, log_prob_min_clamp,
        )
        s_f = student_logits.float()
        if temperature != 1.0:
            s_f = s_f / temperature
        s_lse = s_f.logsumexp(dim=-1)
        ctx.save_for_backward(z, W, student_logits, t_lse, s_lse)
        ctx.chunk_V = chunk_V
        ctx.temperature = temperature
        return kl

    @staticmethod
    def backward(ctx, grad_output):
        z, W, student_logits, t_lse, s_lse = ctx.saved_tensors
        chunk_V = ctx.chunk_V
        temperature = ctx.temperature
        N, V = student_logits.shape
        device = z.device

        z_f = z.float()
        W_f = W.to(device=device, dtype=torch.float32)
        s_f = student_logits.float()
        if temperature != 1.0:
            inv_T = 1.0 / temperature
            z_f = z_f * inv_T
            s_f = s_f * inv_T
        grad = grad_output.float()

        grad_s = torch.empty(N, V, dtype=torch.float32, device=device)
        for v in range(0, V, chunk_V):
            grad_s[:, v : v + chunk_V] = _bwd_chunk(
                z_f, W_f[v : v + chunk_V], s_f[:, v : v + chunk_V],
                t_lse, s_lse, grad,
            )

        if temperature != 1.0:
            grad_s = grad_s / temperature

        return None, None, grad_s.to(student_logits.dtype), None, None, None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_nitrobrew_kl(
    student_logits: torch.Tensor,
    teacher_hidden_states: torch.Tensor,
    teacher_unembed: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> dict[str, torch.Tensor]:
    """Compute Nitrobrew forward KL loss KL(p_T || p_S) for the FSDP path."""
    z, T_sp = _unpack_hidden_states(teacher_hidden_states, student_logits)
    z_flat = z.view(T_sp, -1)
    s_flat = student_logits.view(T_sp, -1)

    loss_config = config.distillation_loss
    per_token_kl = _NitrobrewKL.apply(
        z_flat, teacher_unembed, s_flat, _CHUNK_V,
        loss_config.kd_temperature, loss_config.log_prob_min_clamp,
    )
    return {"distillation_losses": per_token_kl.view(1, T_sp)}


# ---------------------------------------------------------------------------
# Reverse KL: KL(p_S || p_T)
# ---------------------------------------------------------------------------


@torch.compile
def _rev_fwd_chunk(
    z_f: torch.Tensor,       # (N, D_t)
    W_chunk: torch.Tensor,   # (C, D_t)
    zs_chunk: torch.Tensor,  # (N, C)
    s_lse: torch.Tensor,     # (N,) pre-computed student log-partition
    mt: torch.Tensor,        # (N,) running teacher max
    st: torch.Tensor,        # (N,) running sum exp(z_t - mt)
    ut: torch.Tensor,        # (N,) running sum p_S * z_t
    et: torch.Tensor,        # (N,) running sum p_S * z_s
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Online-softmax update for one teacher vocab chunk of reverse KL."""
    zt = z_f @ W_chunk.T

    tile_mt = zt.max(dim=1).values
    new_mt = torch.maximum(mt, tile_mt)
    alpha = (mt - new_mt).exp()
    st = st * alpha + (zt - new_mt.unsqueeze(1)).exp().sum(dim=1)

    ps_chunk = (zs_chunk - s_lse.unsqueeze(1)).exp()
    ut = ut + (ps_chunk * zt).sum(dim=1)
    et = et + (ps_chunk * zs_chunk).sum(dim=1)

    return new_mt, st, ut, et


def _chunked_reverse_kl_forward(
    z: torch.Tensor,              # (N, D_t)
    W: torch.Tensor,              # (V, D_t)
    student_logits: torch.Tensor, # (N, V)
    chunk_V: int = _CHUNK_V,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chunked reverse KL forward. Returns (kl, t_lse, s_lse) all (N,) float32."""
    N = z.shape[0]
    V = W.shape[0]
    device = z.device

    z_f = z.float()
    W_f = W.to(device=device, dtype=torch.float32)
    s_f = student_logits.float()

    if temperature != 1.0:
        inv_T = 1.0 / temperature
        z_f = z_f * inv_T
        s_f = s_f * inv_T

    s_lse = s_f.logsumexp(dim=-1)

    mt = torch.full((N,), float("-inf"), dtype=torch.float32, device=device)
    st = torch.zeros(N, dtype=torch.float32, device=device)
    ut = torch.zeros(N, dtype=torch.float32, device=device)
    et = torch.zeros(N, dtype=torch.float32, device=device)

    for v in range(0, V, chunk_V):
        mt, st, ut, et = _rev_fwd_chunk(
            z_f, W_f[v : v + chunk_V], s_f[:, v : v + chunk_V],
            s_lse, mt, st, ut, et,
        )

    t_lse = mt + st.log()
    kl = et - ut - s_lse + t_lse
    return kl, t_lse, s_lse


@torch.compile
def _rev_bwd_chunk(
    z_f: torch.Tensor,       # (N, D_t)
    W_chunk: torch.Tensor,   # (C, D_t)
    zs_chunk: torch.Tensor,  # (N, C)
    t_lse: torch.Tensor,     # (N,)
    s_lse: torch.Tensor,     # (N,)
    kl: torch.Tensor,        # (N,)
    grad: torch.Tensor,      # (N,)
) -> torch.Tensor:
    """dKL(p_S||p_T)/dz_s(v) = p_S(v) * [log(p_S(v)/p_T(v)) - KL]. Returns (N, C)."""
    zt = z_f @ W_chunk.T
    log_ps = zs_chunk - s_lse.unsqueeze(1)
    log_pt = zt - t_lse.unsqueeze(1)
    ps = log_ps.exp()
    return ps * (log_ps - log_pt - kl.unsqueeze(1)) * grad.unsqueeze(1)


class _NitrobrewReverseKL(torch.autograd.Function):
    """KL(p_S || p_T) with chunked forward and backward over vocab axis."""

    @staticmethod
    def forward(ctx, z, W, student_logits, chunk_V, temperature=1.0):
        kl, t_lse, s_lse = _chunked_reverse_kl_forward(z, W, student_logits, chunk_V, temperature)
        ctx.save_for_backward(z, W, student_logits, t_lse, s_lse, kl)
        ctx.chunk_V = chunk_V
        ctx.temperature = temperature
        return kl

    @staticmethod
    def backward(ctx, grad_output):
        z, W, student_logits, t_lse, s_lse, kl = ctx.saved_tensors
        chunk_V = ctx.chunk_V
        temperature = ctx.temperature
        N, V = student_logits.shape

        z_f = z.float()
        W_f = W.to(device=z.device, dtype=torch.float32)
        s_f = student_logits.float()
        if temperature != 1.0:
            inv_T = 1.0 / temperature
            z_f = z_f * inv_T
            s_f = s_f * inv_T
        grad = grad_output.float()

        grad_s = torch.empty(N, V, dtype=torch.float32, device=z.device)
        for v in range(0, V, chunk_V):
            grad_s[:, v : v + chunk_V] = _rev_bwd_chunk(
                z_f, W_f[v : v + chunk_V], s_f[:, v : v + chunk_V],
                t_lse, s_lse, kl, grad,
            )

        if temperature != 1.0:
            grad_s = grad_s / temperature

        return None, None, grad_s.to(student_logits.dtype), None, None


def compute_nitrobrew_reverse_kl(
    student_logits: torch.Tensor,
    teacher_hidden_states: torch.Tensor,
    teacher_unembed: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> dict[str, torch.Tensor]:
    """Compute Nitrobrew reverse KL loss KL(p_S || p_T) for the FSDP path."""
    z, T_sp = _unpack_hidden_states(teacher_hidden_states, student_logits)
    z_flat = z.view(T_sp, -1)
    s_flat = student_logits.view(T_sp, -1)

    loss_config = config.distillation_loss
    per_token_kl = _NitrobrewReverseKL.apply(
        z_flat, teacher_unembed, s_flat, _CHUNK_V, loss_config.kd_temperature,
    )
    return {"distillation_losses": per_token_kl.view(1, T_sp)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unpack_hidden_states(
    teacher_hidden_states: torch.Tensor,
    student_logits: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Convert nested teacher hidden states to flat packed tensor. Returns (z, T_sp)."""
    if teacher_hidden_states.is_nested:
        z = teacher_hidden_states.values().unsqueeze(0)
    else:
        z = teacher_hidden_states.flatten(0, 1).unsqueeze(0)

    if get_ulysses_sequence_parallel_world_size() > 1:
        z = slice_input_tensor(z, dim=1)

    assert z.shape[:2] == student_logits.shape[:2], (
        f"Shape mismatch: z {z.shape[:2]} vs student_logits {student_logits.shape[:2]}"
    )
    return z, z.shape[1]
