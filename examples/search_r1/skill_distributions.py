"""Frozen top-k collection and forward KL on top-k plus a vocabulary-tail bin.

All engine outputs are indexed by the *predictor* position. Targets stored by
the supervisor are indexed by response token and shifted here exactly once.
"""

import torch
from torch.utils.checkpoint import checkpoint


def topk_distribution(logits, k, chunk_size=32):
    """FP32, globally normalized top-k probabilities and exact remaining mass."""
    if not 0 < k < logits.shape[-1]:
        raise ValueError("top_k must be positive and smaller than vocabulary size")
    ids, logps, tails = [], [], []
    for chunk in logits.detach().split(chunk_size):
        values = chunk.float()
        z = values.logsumexp(-1, keepdim=True)
        selected, indices = values.topk(k, dim=-1)
        tail = values.scatter(-1, indices, -torch.inf).logsumexp(-1) - z.squeeze(-1)
        ids.append(indices)
        logps.append(selected - z)
        tails.append(tail)
    return torch.cat(ids), torch.cat(logps), torch.cat(tails)


def overlap_prefix_gate(student_ids, teacher_ids, watch_mask, min_overlap):
    """Exclude the first bad watched token and the entire following trajectory.

    Tool/padding tokens never trigger the gate. No reset at assistant turn or
    XML boundaries. Returned prefix is applied to each teacher's own loss mask.
    """
    if student_ids.shape != teacher_ids.shape or watch_mask.shape != student_ids.shape[:-1]:
        raise ValueError("Misaligned top-k gate inputs")
    if not 0 <= min_overlap <= 1:
        raise ValueError("min_overlap must be in [0, 1]")
    overlap = (teacher_ids.unsqueeze(-1) == student_ids.unsqueeze(-2)).any(-1).float().mean(-1)
    bad = watch_mask.bool() & (overlap < min_overlap)
    prefix = bad.long().cumsum(-1).eq(0)
    first = bad.nonzero().flatten()
    return prefix, overlap, int(first[0]) if first.numel() else -1


def forward_kl_topk_tail(logits, teacher_ids, teacher_logps, teacher_tail_logp):
    """KL(P_teacher || P_student), on teacher top-k singleton bins + tail.

    Unlike a truncated sum, this is a normalized categorical KL, is nonnegative
    up to rounding, and trains student mass outside the teacher top-k as well.
    It is a coarsening/lower bound of full-vocabulary KL, not exact full KL.
    """
    values = logits.float()
    ids = teacher_ids.detach()
    p_log = teacher_logps.detach().float()
    p_tail = teacher_tail_logp.detach().float()
    z = values.logsumexp(-1)
    q_log = values.gather(-1, ids) - z.unsqueeze(-1)
    q_tail = values.scatter(-1, ids, -torch.inf).logsumexp(-1) - z
    return (p_log.exp() * (p_log - q_log)).sum(-1) + p_tail.exp() * (p_tail - q_tail)


def skill_model_outputs(logits, data):
    """Return packed full-sequence outputs for the FSDP engine's no-SP path."""
    from verl.utils import tensordict_utils as tu

    k = tu.get(data, "skill_collect_topk", default=0)
    if k:
        ids, probs, tail = topk_distribution(logits, int(k))
        return {"skill_topk_ids": ids, "skill_topk_logps": probs, "skill_tail_logp": tail}
    names = tu.get(data, "skill_forward_kl_names", default=[])
    outputs = {}
    prompt_lens = [len(p) for p in data["prompts"].unbind()]
    response_lens = [len(r) for r in data["responses"].unbind()]
    assert sum(p + r for p, r in zip(prompt_lens, response_lens, strict=True)) == len(logits)
    for name in names:
        masks = data[f"skill_{name}_mask"].unbind()
        ids_rows = data[f"skill_{name}_ids"].unbind()
        prob_rows = data[f"skill_{name}_logps"].unbind()
        tail_rows = data[f"skill_{name}_tail"].unbind()
        positions, ids, probs, tails = [], [], [], []
        offset = 0
        for plen, rlen, mask, indices, logps, tail in zip(
            prompt_lens, response_lens, masks, ids_rows, prob_rows, tail_rows, strict=True
        ):
            assert len(mask) == len(indices) == len(logps) == len(tail) == rlen
            selected = mask.bool().nonzero().flatten()
            positions.append(selected + offset + plen - 1)
            ids.append(indices[selected])
            probs.append(logps[selected])
            tails.append(tail[selected])
            offset += plen + rlen
        positions = torch.cat(positions).to(logits.device)
        # Gives a differentiable zero even when every target was gated off.
        losses = logits[:, 0].float() * 0
        if positions.numel():
            ids = torch.cat(ids).to(logits.device)
            probs = torch.cat(probs).to(logits.device)
            tails = torch.cat(tails).to(logits.device)
            chunks = []
            for pos, indices, logps, tail in zip(
                positions.split(32), ids.split(32), probs.split(32), tails.split(32), strict=True
            ):
                # Recompute the FP32 vocabulary reductions during backward.
                def chunk_loss(source, positions, indices, logps, tail):
                    return forward_kl_topk_tail(source.index_select(0, positions), indices, logps, tail)

                chunks.append(checkpoint(chunk_loss, logits, pos, indices, logps, tail, use_reentrant=False))
            losses = losses.index_copy(0, positions, torch.cat(chunks))
        outputs[f"skill_{name}_forward_kl"] = losses
    return outputs
