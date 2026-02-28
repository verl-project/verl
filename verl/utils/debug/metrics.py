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


def _find_contiguous_segments(mask_1d: torch.Tensor) -> list[tuple[int, int]]:
    """Find contiguous segments of 1s in a 1D mask tensor.

    Each contiguous segment of 1s represents one round of model generation
    in a multi-turn trajectory. Segments are separated by 0s (environment
    tokens like images, or padding).

    Example:
        mask = [1,1,1,1, 0,0,0,0,0, 1,1,1, 0,0,0,0,0, 1,1,1, 0,0,0]
                |--R0--| |--env--| |--R1-| |--env--| |--R2-| |pad|
        Returns: [(0, 4), (9, 12), (17, 20)]

    Args:
        mask_1d: 1D tensor with 0s and 1s

    Returns:
        List of (start, end) tuples for each contiguous segment of 1s.
        end is exclusive (Python slice convention).
    """
    segments = []
    in_segment = False
    start = 0

    for i in range(len(mask_1d)):
        val = mask_1d[i].item() if isinstance(mask_1d[i], torch.Tensor) else mask_1d[i]
        if val == 1 and not in_segment:
            in_segment = True
            start = i
        elif val == 0 and in_segment:
            in_segment = False
            segments.append((start, i))

    if in_segment:
        segments.append((start, len(mask_1d)))

    return segments


def _calculate_per_round_metrics(
    train_log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict:
    """Calculate per-round logprob mismatch metrics for multi-turn trajectories.

    Identifies rounds by finding contiguous segments of 1s in response_mask,
    then computes mean absolute logprob difference per round.

    This is useful for multi-turn RL training where different rounds may have
    different attention mask behavior (e.g., image window attention), causing
    mismatch between training and rollout engines to vary across rounds.

    Args:
        train_log_probs: Log probs from training engine (batch_size, seq_len)
        rollout_log_probs: Log probs from rollout engine (batch_size, seq_len)
        response_mask: Mask for valid positions (batch_size, seq_len),
            1=model generated token, 0=environment token or padding

    Returns:
        Dictionary with per-round metrics:
            - per_round/total_rounds: Max number of rounds across batch
            - per_round/round_{i}_abs_diff_mean: Mean |logprob_train - logprob_rollout| for round i
            - per_round/round_{i}_token_count: Number of tokens in round i
            - per_round/max_round_diff: Which round has the largest mean diff
            - per_round/max_diff_value: The largest mean diff value
    """
    batch_size = train_log_probs.shape[0]

    # round_idx -> list of (train_vals, rollout_vals)
    all_round_data: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    max_rounds = 0

    for b in range(batch_size):
        segments = _find_contiguous_segments(response_mask[b])
        max_rounds = max(max_rounds, len(segments))

        for round_idx, (start, end) in enumerate(segments):
            if round_idx not in all_round_data:
                all_round_data[round_idx] = []
            all_round_data[round_idx].append((train_log_probs[b, start:end], rollout_log_probs[b, start:end]))

    if not all_round_data:
        return {"per_round/total_rounds": 0}

    metrics: dict = {"per_round/total_rounds": max_rounds}
    max_diff = -1.0
    max_diff_round = -1

    for round_idx in sorted(all_round_data.keys()):
        train_all = torch.cat([t for t, _ in all_round_data[round_idx]])
        rollout_all = torch.cat([r for _, r in all_round_data[round_idx]])

        if train_all.numel() == 0:
            continue

        abs_diff = torch.abs(train_all - rollout_all)
        mean_diff = abs_diff.mean().item()

        metrics[f"per_round/round_{round_idx}_abs_diff_mean"] = mean_diff
        metrics[f"per_round/round_{round_idx}_token_count"] = train_all.numel()

        if mean_diff > max_diff:
            max_diff = mean_diff
            max_diff_round = round_idx

    metrics["per_round/max_round_diff"] = max_diff_round
    if max_diff_round >= 0:
        metrics["per_round/max_diff_value"] = max_diff

    return metrics


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
    pearson_corrcoef = pearson_correlation_coefficient(actor_probs, rollout_probs, response_mask_bool)
    rollout_probs_diff = calculate_log_prob_diff(actor_probs, rollout_probs, response_mask_bool)

    metrics = {
        "training/rollout_probs_diff_valid": 1,
        "training/rollout_probs_diff_max": torch.max(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_mean": torch.mean(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_std": torch.std(rollout_probs_diff).detach().item(),
        "training/rollout_actor_probs_pearson_corr": pearson_corrcoef,
    }

    # Per-round logprob mismatch metrics for multi-turn trajectories
    per_round_metrics = _calculate_per_round_metrics(actor_old_log_probs, rollout_old_log_probs, response_mask)
    metrics.update(per_round_metrics)

    return metrics
