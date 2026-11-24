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


def calculate_debug_metrics(data: DataProto) -> dict:
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
    import os
    audit_root = os.environ.get("KIMI_ALIGNMENT_AUDIT_DIR")
    if audit_root:
        from pathlib import Path
        import json, time
        root = Path(audit_root)
        root.mkdir(parents=True, exist_ok=True)
        stamp = str(time.time_ns())
        left = actor_old_log_probs.detach().cpu().contiguous()
        right = rollout_old_log_probs.detach().cpu().contiguous()
        mask = response_mask_bool.detach().cpu()
        finite = torch.isfinite(left) & torch.isfinite(right)
        delta = (left.double() - right.double()).abs()[mask]
        different = (left != right) & mask
        where = different.nonzero()
        summary = {
            "comparison": "independent_actor_old_logprob_vs_rollout",
            "valid_tokens": int(mask.sum()), "nonfinite_tokens": int((~finite & mask).sum()),
            "different_tokens": int(different.sum()),
            "mean_abs_diff": float(delta.mean()), "max_abs_diff": float(delta.max()),
            "first_difference": where[0].tolist() if len(where) else None,
            "actor_dtype": str(left.dtype), "rollout_dtype": str(right.dtype),
            "passed": bool(mask.any() and finite[mask].all() and not different.any()),
        }
        artifact = {"actor_logprobs": left, "rollout_logprobs": right, "mask": mask,
                    "responses": responses.detach().cpu()}
        for key in ("input_ids", "attention_mask", "position_ids"):
            if key in data.batch:
                artifact[key] = data.batch[key].detach().cpu()
        torch.save(artifact, root / (stamp + ".pt"))
        (root / (stamp + ".json")).write_text(json.dumps(summary, indent=2) + "\n")
        print("KIMI_ALIGNMENT_AUDIT " + json.dumps(summary), flush=True)
    return {
        "training/rollout_probs_diff_valid": 1,
        "training/rollout_probs_diff_max": torch.max(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_mean": torch.mean(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_std": torch.std(rollout_probs_diff).detach().item(),
        "training/rollout_actor_probs_pearson_corr": pearson_corrcoef,
        "training/rollout_logprobs_mismatch_count": int(logprobs_mismatch.sum().item()),
        "training/rollout_logprobs_diff_max": torch.max(rollout_logprobs_diff).detach().item(),
        "training/rollout_logprobs_diff_mean": torch.mean(rollout_logprobs_diff).detach().item(),
    }
