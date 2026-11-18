# Copyright 2025 Individual Contributor: TomQunChaoA
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

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import torch

from verl.protocol import DataProto

logger = logging.getLogger(__file__)


def calculate_token_list_diff(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # verify inputs
    if tensor1.numel() == 0 or tensor2.numel() == 0:
        return torch.zeros(tensor1.shape[0], dtype=torch.long, device=tensor1.device)
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        print(
            f"<WARN> dim of tensor1, tensor2, mask is not equal, {(tensor1.shape)=},{(tensor2.shape)=}, {(mask.shape)=}"
        )
        return torch.ones_like(tensor1)
    # transfer to same device
    if tensor2.device != tensor1.device:
        tensor2 = tensor2.to(tensor1.device)
    if mask.device != tensor1.device:
        mask = mask.to(tensor1.device)

    # calculate diff
    diff_mask = tensor1 != tensor2

    valid_diff_mask = diff_mask & (mask == 1)

    diff_counts = valid_diff_mask.sum(dim=1)

    return diff_counts


def pearson_correlation_coefficient(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # implemention of https://arxiv.org/pdf/2506.13585
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        return 0
    mt1 = torch.masked_select(tensor1, mask)
    mt2 = torch.masked_select(tensor2, mask)
    result = torch.corrcoef(torch.stack([mt1, mt2], dim=0))
    return result[0][1].detach().item()


def calculate_log_prob_diff(log_probs1: torch.Tensor, log_probs2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    full_diff = torch.abs(log_probs1 - log_probs2)
    return torch.masked_select(full_diff, mask)


def calculate_train_infer_discrepancy(
    actor_log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    partial_mask: torch.Tensor | None = None,
    tau: float = 2.0,
) -> dict:
    """Quantify how far the training engine's distribution is from the rollout engine's.

    Both quantities come from R3 (arXiv:2510.11370): the k3 estimate of the
    training-inference KL divergence (Eq. 3) and the extreme token fraction
    F(tau) (Eq. 4), the share of tokens whose probability ratio between the two
    engines exceeds ``tau`` in either direction.

    Args:
        actor_log_probs: Training engine log probs of the rollout tokens.
        rollout_log_probs: Rollout engine log probs of the same tokens.
        response_mask: 1 for tokens that contribute to the loss.
        partial_mask: Per-sequence flag marking trajectories that span a weight sync.
            When given, the metrics are also reported for partial and non-partial
            trajectories separately, which is what isolates the cost of resuming a
            trajectory under weights (and, with R3, routing) it was not generated with.
        tau: Threshold of F(tau).

    Returns:
        dict: metrics under the ``training/`` prefix, suffixed ``_partial`` /
        ``_nonpartial`` for the per-group values. The ``_sum`` metrics keep the
        token-weighted averages exact once a caller adds up several steps, whose
        token counts differ. Groups with no tokens are omitted, not reported as nan.
    """
    actor_log_probs = actor_log_probs.float()
    rollout_log_probs = rollout_log_probs.float()
    log_ratio = actor_log_probs - rollout_log_probs
    k3 = torch.exp(log_ratio) - 1 - log_ratio
    extreme = log_ratio.abs() > math.log(tau)
    # An engine reporting -inf for a token it considers impossible leaves k3 at inf - inf, so one
    # such token would make the whole average nan. Count them instead; being told that a token that
    # was actually sampled is impossible is itself a discrepancy worth watching.
    finite = torch.isfinite(k3)
    causes = {
        "impossible_under_rollout": torch.isneginf(rollout_log_probs),
        "impossible_under_actor": torch.isneginf(actor_log_probs),
        "nan_input": torch.isnan(actor_log_probs) | torch.isnan(rollout_log_probs),
    }

    groups = {"": response_mask.bool()}
    if partial_mask is not None:
        partial_mask = partial_mask.to(response_mask.device).bool().unsqueeze(-1)
        groups["_partial"] = response_mask.bool() & partial_mask
        groups["_nonpartial"] = response_mask.bool() & ~partial_mask

    metrics = {}
    for suffix, mask in groups.items():
        num_tokens = int(mask.sum().item())
        if not num_tokens:
            # A step with no partial trajectories is normal, and a nan here would be averaged
            # into the whole sync cycle, discarding the steps that did have them.
            continue
        scored = mask & finite
        num_scored = int(scored.sum().item())
        num_extreme = int(torch.masked_select(extreme, mask).sum().item())
        k3_sum = torch.masked_select(k3, scored).sum().item() if num_scored else 0.0
        metrics[f"training/train_infer_tokens_sum{suffix}"] = num_tokens
        metrics[f"training/train_infer_scored_tokens_sum{suffix}"] = num_scored
        metrics[f"training/train_infer_extreme_tokens_sum{suffix}"] = num_extreme
        metrics[f"training/train_infer_kl_k3_sum{suffix}"] = k3_sum
        metrics[f"training/train_infer_extreme_token_frac{suffix}"] = num_extreme / num_tokens
        metrics[f"training/train_infer_nonfinite_token_frac{suffix}"] = (num_tokens - num_scored) / num_tokens
        if num_scored:
            metrics[f"training/train_infer_kl_k3{suffix}"] = k3_sum / num_scored
        if num_scored < num_tokens:
            for cause, hit in causes.items():
                metrics[f"training/train_infer_{cause}_tokens_sum{suffix}"] = int(
                    torch.masked_select(hit, mask).sum().item()
                )
    return metrics


def partial_rollout_mask(tags: Sequence[Mapping[str, Any]]) -> torch.Tensor | None:
    """Flag trajectories generated across more than one weight version, or None if unknown."""
    starts = [tag.get("min_global_steps") for tag in tags]
    ends = [tag.get("max_global_steps") for tag in tags]
    if not starts or any(v is None for v in starts) or any(v is None for v in ends):
        return None
    return torch.tensor([int(a) != int(b) for a, b in zip(starts, ends, strict=True)], dtype=torch.bool)


def calculate_debug_metrics(data: DataProto, partial_mask: Optional[torch.Tensor] = None) -> dict:
    """
    calculate rollout vs actor logprobs diff, for debugging purpose

    Args:
        data: DataProto
            the data batch to calculate
            rollout_log_probs: log_probs record when rollout forward tokens
            old_log_probs(actor log probs): log_probs record when actor forward tokens
            loss_mask or attention_mask: to mask unrelated token
            responses: the response tokens, for calculating size
    Returns:
        dict: metrics
            "training/rollout_probs_diff_valid": 1->input is valid, 0->input is invalid
            "training/rollout_probs_diff_max": max value of logprob diff of rollout vs. actor
            "training/rollout_probs_diff_mean": mean value of logprob diff of rollout vs. actor
            "training/rollout_probs_diff_std": std value of logprob diff of rollout vs. actor
            "training/rollout_actor_probs_pearson_corr": logprob's pearson corrcoef of rollout vs. actor, reference to https://arxiv.org/pdf/2506.13585
            "training/rollout_logprobs_mismatch_count": number of valid tokens whose raw logprobs differ
            "training/rollout_logprobs_diff_max": max value of raw logprob diff of rollout vs. actor
            "training/rollout_logprobs_diff_mean": mean value of raw logprob diff of rollout vs. actor
        The probs_diff metrics compare exp(logprob) and miss 1-ULP logprob differences; the logprobs metrics do not.
    """

    rollout_old_log_probs = data.batch["rollout_log_probs"]
    actor_old_log_probs = data.batch["old_log_probs"]
    if "response_mask" in data.batch:
        logger.debug("response mask found, use it to mask log probs")
        log_prob_mask = data.batch["response_mask"]
    elif "attention_mask" in data.batch:
        log_prob_mask = data.batch["attention_mask"]
    else:
        logger.warning(f"no mask info found, use all log probs, {(data.batch.keys())=}")
        log_prob_mask = torch.ones_like(rollout_old_log_probs)
    responses = data.batch["responses"]
    response_length = responses.size(1)

    response_mask = log_prob_mask[:, -response_length:]
    # calculate pearson corrcoef
    actor_probs = torch.exp(actor_old_log_probs)
    rollout_probs = torch.exp(rollout_old_log_probs)
    response_mask_bool = response_mask.bool()

    # check if there are any valid tokens before computing metrics
    if not response_mask_bool.any():
        logger.warning("response_mask is all False, returning default metrics")
        return {
            "training/rollout_probs_diff_valid": 0,
            "training/rollout_probs_diff_max": float("nan"),
            "training/rollout_probs_diff_mean": float("nan"),
            "training/rollout_probs_diff_std": float("nan"),
            "training/rollout_actor_probs_pearson_corr": float("nan"),
            "training/rollout_logprobs_mismatch_count": 0,
            "training/rollout_logprobs_diff_max": float("nan"),
            "training/rollout_logprobs_diff_mean": float("nan"),
        }

    pearson_corrcoef = pearson_correlation_coefficient(actor_probs, rollout_probs, response_mask_bool)
    rollout_probs_diff = calculate_log_prob_diff(actor_probs, rollout_probs, response_mask_bool)
    rollout_logprobs_diff = calculate_log_prob_diff(
        actor_old_log_probs.float(), rollout_old_log_probs.float(), response_mask_bool
    )
    logprobs_mismatch = rollout_logprobs_diff != 0
    return {
        **calculate_train_infer_discrepancy(actor_old_log_probs, rollout_old_log_probs, response_mask, partial_mask),
        "training/rollout_probs_diff_valid": 1,
        "training/rollout_probs_diff_max": torch.max(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_mean": torch.mean(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_std": torch.std(rollout_probs_diff).detach().item(),
        "training/rollout_actor_probs_pearson_corr": pearson_corrcoef,
        "training/rollout_logprobs_mismatch_count": int(logprobs_mismatch.sum().item()),
        "training/rollout_logprobs_diff_max": torch.max(rollout_logprobs_diff).detach().item(),
        "training/rollout_logprobs_diff_mean": torch.mean(rollout_logprobs_diff).detach().item(),
    }
