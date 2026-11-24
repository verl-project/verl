"""Opt-in Kimi BF16 alignment reference: one recurrence per token/request.

Forward is shared by independent training and serving calls. Training keeps
the native chunk operator's mathematical VJP; it does not reuse rollout data.
This deliberately slow reference is for alignment, not a throughput claim.
"""
import os
import torch


def enabled():
    return os.environ.get("KIMI_ALIGNMENT_RECURRENT_KDA") == "1"


def _host_list(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    return [int(x) for x in value]


@torch.no_grad()
def canonical_kda(q, k, v, g, beta, a_log, dt_bias, *, cu_seqlens=None,
                  initial_state=None, lower_bound=-5.0, beta_is_preprocessed=True):
    from fla_npu.ops.ascendc import recurrent_kda
    if q.ndim != 4 or q.shape[0] != 1 or q.dtype != torch.bfloat16:
        raise ValueError("Kimi alignment expects packed B=1 BF16 BSND operands")
    _, length, heads, dim = q.shape
    if not (q.shape == k.shape == v.shape == g.shape and dim == 128):
        raise ValueError("Kimi alignment requires matching Q/K/V/g heads and D=128")
    if beta.shape != q.shape[:3]:
        raise ValueError("Kimi alignment beta shape mismatch")
    bounds = [0, length] if cu_seqlens is None else _host_list(cu_seqlens)
    if (len(bounds) < 2 or bounds[0] != 0 or bounds[-1] != length
            or any(a > b for a, b in zip(bounds, bounds[1:]))):
        raise ValueError("Invalid packed sequence boundaries")
    count = len(bounds) - 1
    state_shape = (count, heads, dim, dim)
    if initial_state is None:
        states = torch.zeros(state_shape, dtype=torch.float32, device=q.device)
    else:
        if tuple(initial_state.shape) != state_shape or initial_state.dtype != torch.float32:
            raise ValueError("Kimi recurrence requires FP32 [N,H,V,K] states")
        states = initial_state.detach().clone().contiguous()
    a_log = a_log.float().reshape(-1).contiguous()
    dt_bias = dt_bias.float().reshape(-1).contiguous()
    if a_log.numel() != heads or dt_bias.numel() != heads * dim:
        raise ValueError("Kimi recurrence parameter/head mismatch")
    beta = beta.float() if beta_is_preprocessed else beta.float().sigmoid()
    cu = torch.tensor([0, 1], dtype=torch.int32, device=q.device)
    index = torch.tensor([0], dtype=torch.int32, device=q.device)
    output = torch.empty_like(v)
    for sequence, (begin, end) in enumerate(zip(bounds, bounds[1:])):
        state = states[sequence:sequence + 1]
        for pos in range(begin, end):
            operands = [x[:, pos:pos + 1].contiguous() for x in (q, k, v, g, beta)]
            value, _ = recurrent_kda(
                *operands, state, cu_seqlens=cu, ssm_state_indices=index,
                A_log=a_log, dt_bias=dt_bias, num_accepted_tokens=None,
                layout="BSND", scale=dim ** -0.5, output_final_state=False,
                inplace_final_state=True, state_v_first=True,
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
                safe_gate=lower_bound is not None,
                lower_bound=lower_bound if lower_bound is not None else -5.0,
            )
            output[:, pos:pos + 1].copy_(value)
    return output, states


class _ServingValueNativeVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, native_value, serving_value):
        return serving_value

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def training_kda(native_fn, q, k, v, g, beta, args, kwargs):
    # Match the native AscendC training contract: no recurrent cache/state VJP.
    if args or kwargs.get("initial_state") is not None or kwargs.get("use_cache", False):
        raise ValueError("Aligned training KDA only supports keyword options and no KV cache")
    for flag in ("use_qk_l2norm_in_kernel", "use_gate_in_kernel", "use_beta_sigmoid_in_kernel"):
        if kwargs.get(flag) is not True:
            raise ValueError(f"Aligned Kimi KDA requires {flag}=True")
    if kwargs.get("allow_neg_eigval", False) or kwargs.get("return_intermediate_states", False):
        raise ValueError("Unsupported alignment KDA option")
    if kwargs.get("scale") not in (None, q.shape[-1] ** -0.5):
        raise ValueError("Aligned Kimi KDA requires the model's default scale")
    value, _ = canonical_kda(
        q, k, v, g, beta, kwargs["A_log"], kwargs["dt_bias"],
        cu_seqlens=kwargs.get("cu_seqlens_cpu", kwargs.get("cu_seqlens")),
        lower_bound=kwargs.get("lower_bound"), beta_is_preprocessed=False,
    )
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v, g, beta, kwargs["A_log"], kwargs["dt_bias"])):
        native, native_state = native_fn(q, k, v, g, beta, **kwargs)
        if native_state is not None:
            raise RuntimeError("Expected the native training operator's no-cache contract")
        value = _ServingValueNativeVJP.apply(native, value)
    return value, None


@torch.no_grad()
def serving_decode(q, k, v, g, beta, state, cu_seqlens, state_indices,
                   a_log, dt_bias, *, lower_bound, num_accepted_tokens=None):
    if num_accepted_tokens is not None:
        raise ValueError("Canonical Kimi alignment does not cover speculative decoding")
    bounds, indices = _host_list(cu_seqlens), _host_list(state_indices)
    if len(indices) != len(bounds) - 1 or bounds[0] != 0 or bounds[-1] != q.shape[1]:
        raise ValueError("Decode cache mapping does not cover the packed input")
    valid = [idx for a, b, idx in zip(bounds, bounds[1:], indices) if a < b and idx >= 0]
    if len(set(valid)) != len(valid):
        raise ValueError("Overlapping live decode cache slots")
    output = torch.zeros_like(v)
    for begin, end, slot in zip(bounds, bounds[1:], indices):
        if begin == end or slot < 0:
            continue
        if slot >= state.shape[0]:
            raise ValueError("Decode state index out of range")
        out, final = canonical_kda(
            *(x[:, begin:end] for x in (q, k, v, g, beta)), a_log, dt_bias,
            initial_state=state[slot:slot + 1], lower_bound=lower_bound,
        )
        output[:, begin:end].copy_(out)
        state[slot:slot + 1].copy_(final)
    return output
