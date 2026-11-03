"""Clipped GRPO/PPO loss shared by the JAX step and the torch parity test.

The arithmetic matches ``compute_policy_loss_vanilla`` with ``token-mean``
aggregation and ``dp_size == 1``. ``xp`` is ``numpy`` or ``jax.numpy``.
"""

from __future__ import annotations

from typing import Any


def clipped_policy_loss(
    xp,
    log_prob,
    old_log_prob,
    advantages,
    response_mask,
    clip_ratio_low: float,
    clip_ratio_high: float,
    clip_ratio_c: float,
):
    """Return the scalar token-mean clipped policy loss and a few metrics."""
    if clip_ratio_c <= 1.0:
        raise ValueError(f"clip_ratio_c must be greater than 1, got {clip_ratio_c}")
    mask = response_mask.astype(log_prob.dtype)
    token_count = xp.sum(mask)
    negative_kl = xp.clip(log_prob - old_log_prob, -20.0, 20.0)
    ratio = xp.exp(negative_kl)
    unclipped = -advantages * ratio
    clipped = -advantages * xp.clip(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    clip_upper = xp.maximum(unclipped, clipped)
    clip_lower = xp.minimum(-advantages * clip_ratio_c, clip_upper)
    losses = xp.where(advantages < 0, clip_lower, clip_upper)
    loss = xp.sum(losses * mask) / xp.maximum(token_count, xp.asarray(1.0, dtype=log_prob.dtype))
    clipfrac = xp.sum((clipped > unclipped).astype(log_prob.dtype) * mask) / xp.maximum(token_count, xp.asarray(1.0))
    ppo_kl = xp.sum((-negative_kl) * mask) / xp.maximum(token_count, xp.asarray(1.0))
    return loss, {"pg_clipfrac": clipfrac, "ppo_kl": ppo_kl, "token_count": token_count}


def clip_from_actor(config: Any) -> tuple[float, float, float]:
    """Read the three clip bounds from a verl actor config or a stand-in."""
    clip_ratio = float(config.clip_ratio)
    low = config.clip_ratio_low
    high = config.clip_ratio_high
    getter = getattr(config, "get", None)
    clip_c = getter("clip_ratio_c", 3.0) if getter is not None else getattr(config, "clip_ratio_c", 3.0)
    return (
        clip_ratio if low is None else float(low),
        clip_ratio if high is None else float(high),
        float(clip_c),
    )
