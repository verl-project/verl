# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Opt-in paired log-prob evidence, without changing sampling or optimization."""

import hashlib
import json
from pathlib import Path

import torch


def dump_paired_log_probs(batch, directory: str, step: int, model_path: str, *, actor_log_probs_source: str) -> dict:
    """Compare actor/rollout on exactly the same masked trajectory positions.

    Step 1 is before the first optimizer update. Later steps exercise live refit.
    Different arms may generate different answers: this is NOT a four-way
    BF16/FP4 fixed-answer ablation, nor a dense distribution KL calculation.
    """
    if actor_log_probs_source != "recomputed":
        raise ValueError("paired validation requires independently recomputed actor log-probs")
    mask = batch.batch["response_mask"].bool().cpu()
    actor = batch.batch["old_log_probs"].detach().cpu()
    rollout = batch.batch["rollout_log_probs"].detach().cpu()
    responses = batch.batch["responses"].detach().cpu()
    if actor.shape != mask.shape or rollout.shape != mask.shape or responses.shape != mask.shape:
        raise ValueError("paired log-probs, response tokens and masks must have identical shapes")
    actor = actor[mask].float()
    rollout = rollout[mask].float()
    if not actor.numel() or not torch.isfinite(actor).all() or not torch.isfinite(rollout).all():
        raise ValueError("empty or non-finite paired log-probs")
    delta = actor.double() - rollout.double()
    absolute = delta.abs()
    k3 = torch.expm1(delta) - delta
    tokens = responses[mask].contiguous()
    summary = {
        "step": step,
        "model_path": model_path,
        "actor_log_probs_source": actor_log_probs_source,
        "checkpoint_position": "initial_before_first_update" if step == 1 else "after_live_updates",
        "temperature": float(batch.meta_info["temperature"]),
        "sequences": len(mask),
        "valid_tokens": actor.numel(),
        "response_tokens_sha256": hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
        "actor_minus_rollout_mean": delta.mean().item(),
        "abs_mean": absolute.mean().item(),
        "abs_p95": torch.quantile(absolute, 0.95).item(),
        "abs_p99": torch.quantile(absolute, 0.99).item(),
        "abs_max": absolute.max().item(),
        "k3_mean": k3.mean().item(),
        "k3_finite": bool(torch.isfinite(k3).all()),
    }
    # Exclusive outputs prevent a rerun from silently mixing checkpoint evidence.
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"step_{step}.pt").open("xb") as stream:
        torch.save(
            {
                "actor_log_probs": actor,
                "rollout_log_probs": rollout,
                "response_tokens": tokens,
                "response_lengths": mask.sum(-1),
                "prompt_ids": batch.batch["prompts"].detach().cpu(),
                "summary": summary,
            },
            stream,
        )
    with (root / f"step_{step}.json").open("x") as stream:
        json.dump(summary, stream, allow_nan=False, indent=2)
    print("VERL_PAIRED_LOGPROB_EVIDENCE " + json.dumps(summary, allow_nan=False), flush=True)
    return summary
