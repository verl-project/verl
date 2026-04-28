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

"""Nitrobrew: fused, constant-memory KL divergence from hidden states (Megatron path).

Same online-softmax chunking as the FSDP kernel, but adapted for vocab-parallel
(TP-sharded) student logits. Each TP rank computes local accumulators over its
V/tp vocab shard, then merges via all-gather for exact global log-partition values.

Forward only -- reverse KL is not yet implemented for Megatron.
"""

import torch

from verl.models.mcore.util import preprocess_bshd_engine, preprocess_thd_engine
from verl.workers.config import DistillationConfig

_CHUNK_V: int = 1024


@torch.compile
def _fwd_chunk_update(
    z_f: torch.Tensor,       # (N, D_t)
    W_chunk: torch.Tensor,   # (C, D_t)
    zs_chunk: torch.Tensor,  # (N, C)
    mt: torch.Tensor,        # (N,)
    st: torch.Tensor,        # (N,) sum exp(zt - mt)
    tt: torch.Tensor,        # (N,) sum exp(zt - mt) * zt
    ut: torch.Tensor,        # (N,) sum exp(zt - mt) * zs_clamped
    s_min: torch.Tensor,     # (N, 1) floor for student logits
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Online-softmax update for one local vocab chunk. Returns updated (mt, st, tt, ut)."""
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


def _chunked_local_lse_and_kl_accumulators(
    z: torch.Tensor,                  # (N, D_t)
    W_vp: torch.Tensor,              # (V/tp, D_t)
    student_logits_vp: torch.Tensor,  # (N, V/tp)
    chunk_V: int,
    s_min: torch.Tensor,             # (N, 1)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute local (this TP rank) online-softmax accumulators. Returns (mt, st, tt, ut)."""
    N = z.shape[0]
    V_tp = W_vp.shape[0]
    device = z.device

    z_f = z.float()
    W_f = W_vp.to(device=device, dtype=torch.float32)
    s_f = student_logits_vp.float()

    mt = torch.full((N,), float("-inf"), dtype=torch.float32, device=device)
    st = torch.zeros(N, dtype=torch.float32, device=device)
    tt = torch.zeros(N, dtype=torch.float32, device=device)
    ut = torch.zeros(N, dtype=torch.float32, device=device)

    for v in range(0, V_tp, chunk_V):
        mt, st, tt, ut = _fwd_chunk_update(
            z_f, W_f[v : v + chunk_V], s_f[:, v : v + chunk_V],
            mt, st, tt, ut, s_min,
        )
    return mt, st, tt, ut


def _merge_tp_accumulators(
    mt: torch.Tensor,
    st: torch.Tensor,
    tt: torch.Tensor,
    ut: torch.Tensor,
    group: torch.distributed.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge online-softmax accumulators across TP ranks (exact)."""
    ws = torch.distributed.get_world_size(group)
    if ws == 1:
        return mt, st, tt, ut

    N = mt.shape[0]
    device = mt.device
    local = torch.stack([mt, st, tt, ut], dim=1)
    gathered = [torch.empty_like(local) for _ in range(ws)]
    torch.distributed.all_gather(gathered, local, group=group)

    global_mt = torch.stack([g[:, 0] for g in gathered], dim=1).max(dim=1).values
    global_st = torch.zeros(N, dtype=torch.float32, device=device)
    global_tt = torch.zeros(N, dtype=torch.float32, device=device)
    global_ut = torch.zeros(N, dtype=torch.float32, device=device)
    for g in gathered:
        alpha = (g[:, 0] - global_mt).exp()
        global_st += g[:, 1] * alpha
        global_tt += g[:, 2] * alpha
        global_ut += g[:, 3] * alpha

    return global_mt, global_st, global_tt, global_ut


class _VocabParallelNitrobrewKL(torch.autograd.Function):
    """KL(p_T || p_S) with TP-sharded vocab: chunked forward + all-gather merge."""

    @staticmethod
    def forward(ctx, vp_student_logits, teacher_z, W_vp, chunk_V, log_prob_min_clamp):
        from megatron.core.parallel_state import get_tensor_model_parallel_group

        N, V_tp = vp_student_logits.shape
        device = teacher_z.device

        s_f = vp_student_logits.float()

        tp_group = get_tensor_model_parallel_group()
        s_local_max = s_f.max(dim=-1).values
        s_local_sumexp = (s_f - s_local_max.unsqueeze(-1)).exp().sum(-1)
        s_max_all = s_local_max.clone()
        torch.distributed.all_reduce(s_max_all, op=torch.distributed.ReduceOp.MAX, group=tp_group)
        s_sumexp_all = s_local_sumexp * (s_local_max - s_max_all).exp()
        torch.distributed.all_reduce(s_sumexp_all, op=torch.distributed.ReduceOp.SUM, group=tp_group)
        s_lse_global = s_max_all + s_sumexp_all.log()

        if log_prob_min_clamp is not None:
            s_min = (s_lse_global + log_prob_min_clamp).unsqueeze(1)
        else:
            s_min = torch.full((1, 1), float("-inf"), dtype=torch.float32, device=device)

        mt, st, tt, ut = _chunked_local_lse_and_kl_accumulators(
            teacher_z, W_vp, vp_student_logits, chunk_V, s_min,
        )
        mt, st, tt, ut = _merge_tp_accumulators(mt, st, tt, ut, tp_group)

        t_lse = mt + st.log()
        per_token_kl = tt / st - t_lse - ut / st + s_lse_global

        ctx.save_for_backward(teacher_z, W_vp, vp_student_logits, t_lse, s_lse_global)
        ctx.chunk_V = chunk_V

        return per_token_kl

    @staticmethod
    def backward(ctx, grad_output):
        teacher_z, W_vp, vp_student_logits, t_lse, s_lse = ctx.saved_tensors
        N, V_tp = vp_student_logits.shape
        chunk_V = ctx.chunk_V
        device = teacher_z.device

        z_f = teacher_z.float()
        W_f = W_vp.to(device=device, dtype=torch.float32)
        s_f = vp_student_logits.float()
        grad = grad_output.float()

        grad_s = torch.empty(N, V_tp, dtype=torch.float32, device=device)
        for v in range(0, V_tp, chunk_V):
            zt = z_f @ W_f[v : v + chunk_V].T
            p_T = (zt - t_lse.unsqueeze(1)).exp()
            p_S = (s_f[:, v : v + chunk_V] - s_lse.unsqueeze(1)).exp()
            grad_s[:, v : v + chunk_V] = (p_S - p_T) * grad.unsqueeze(1)

        return grad_s.to(vp_student_logits.dtype), None, None, None, None


def compute_nitrobrew_kl(
    student_logits: torch.Tensor,
    teacher_hidden_states: torch.Tensor,
    teacher_unembed: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> dict[str, torch.Tensor]:
    """Compute Nitrobrew forward KL using teacher hidden states (Megatron vocab-parallel path)."""
    assert teacher_hidden_states.is_nested, "teacher_hidden_states must be nested after left_right_2_no_padding"

    if data_format == "thd":
        teacher_z_cp, *_ = preprocess_thd_engine(teacher_hidden_states, pre_process=True)
    else:
        teacher_z_cp, *_ = preprocess_bshd_engine(teacher_hidden_states, pre_process=True)

    B, T_cp, V_tp = student_logits.shape
    assert teacher_z_cp.shape[:2] == (B, T_cp), (
        f"Shape mismatch: teacher_z {teacher_z_cp.shape[:2]} vs student {(B, T_cp)}"
    )

    N = B * T_cp
    z_flat = teacher_z_cp.reshape(N, -1)
    s_flat = student_logits.reshape(N, V_tp)

    loss_config = config.distillation_loss
    per_token_kl = _VocabParallelNitrobrewKL.apply(
        s_flat, z_flat, teacher_unembed, _CHUNK_V, loss_config.log_prob_min_clamp,
    )

    return {"distillation_losses": per_token_kl.view(B, T_cp)}
