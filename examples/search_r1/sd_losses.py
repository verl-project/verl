"""Differentiable numerical core of equations (6)--(8)."""

import math

import torch


def topk_union_jsd(student: torch.Tensor, teacher: torch.Tensor, k: int = 50) -> torch.Tensor:
    """Per-position JSD, renormalized on UNIQUE union of teacher/student top-k.

    Inputs are logits [..., vocabulary]. Teacher is detached here as a second
    line of defense. Gradients flow through BOTH occurrences of the student in
    the mixture, not just the student KL term. k=0 means full vocabulary.
    """
    if student.shape != teacher.shape or student.ndim < 2:
        raise ValueError("student and teacher need identical [..., vocabulary] shapes")
    if k < 0:
        raise ValueError("k must be nonnegative")
    dtype = torch.float64 if student.dtype == torch.float64 else torch.float32
    s, t = student.to(dtype), teacher.detach().to(dtype)
    vocab = s.shape[-1]
    if 0 < k < vocab:
        # Sorting the concatenation lets us mask duplicates without allocating
        # a vocabulary-sized boolean scatter tensor.
        ids = torch.cat((s.detach().topk(k).indices, t.topk(k).indices), -1).sort(-1).values
        valid = torch.ones_like(ids, dtype=torch.bool)
        valid[..., 1:] = ids[..., 1:] != ids[..., :-1]
        s, t = s.gather(-1, ids), t.gather(-1, ids)
        s, t = s.masked_fill(~valid, -torch.inf), t.masked_fill(~valid, -torch.inf)
    else:
        valid = torch.ones_like(s, dtype=torch.bool)
    ls, lt = s.log_softmax(-1), t.log_softmax(-1)
    # Use finite placeholders at duplicate slots BEFORE arithmetic: -inf - -inf
    # can otherwise poison backward, even if the final result is masked.
    ls_safe, lt_safe = ls.masked_fill(~valid, 0), lt.masked_fill(~valid, 0)
    lm = torch.logaddexp(ls_safe, lt_safe) - math.log(2)
    terms = 0.5 * (ls.exp() * (ls_safe - lm) + lt.exp() * (lt_safe - lm))
    return terms.sum(-1)


def trajectory_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Eq. 6: mean within each trajectory, THEN mean over trajectories.

    Empty-query trajectories contribute zero and stay in the batch denominator.
    """
    if values.shape != mask.shape or values.ndim != 2:
        raise ValueError("expected [batch, query_positions]")
    return ((values * mask).sum(-1) / mask.sum(-1).clamp_min(1)).mean()


def sd_coefficient(completed_steps: int, warmup_steps=50, alpha=1e-3) -> float:
    return 0.0 if completed_steps < warmup_steps else alpha
