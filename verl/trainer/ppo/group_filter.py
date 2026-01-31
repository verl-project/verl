# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Group filtering utilities for GRPO training.

Implements DAPO-style filtering to remove groups where all samples have
identical rewards (no contrastive signal for GRPO).

Reference:
- DAPO: https://arxiv.org/abs/2503.14476
"""

from collections import defaultdict

import numpy as np
import torch

from verl import DataProto


def filter_zero_variance_groups(
    batch: DataProto,
    epsilon: float = 1e-8,
    use_rm_scores: bool = False,
) -> tuple[DataProto, dict]:
    """Filter out groups where all samples have the same reward (DAPO-style).

    For GRPO training, groups where all n rollouts have identical rewards
    provide no contrastive signal and waste training compute. This function
    identifies such groups and DIRECTLY REMOVES them from the batch to save
    compute in subsequent forward/backward passes.

    NOTE: NCCL divisibility padding should be handled separately (e.g., in _balance_batch).

    The filtering works by:
    1. Computing pre-filter reward metrics (for logging)
    2. Grouping samples by uid (prompt-level grouping)
    3. Checking if the reward variance within the group is below epsilon
    4. If yes, directly removing samples from batch using select_idxs

    Args:
        batch: DataProto containing batch data with:
            - batch["token_level_rewards"]: reward tensor [bsz, seq_len] (if use_rm_scores=False)
            - batch["rm_scores"]: reward tensor [bsz, seq_len] (if use_rm_scores=True)
            - batch["response_mask"]: mask tensor [bsz, seq_len]
            - non_tensor_batch["uid"]: prompt-level group identifiers
        epsilon: Threshold for zero variance detection. Groups with
            reward std < epsilon are considered zero-variance.
        use_rm_scores: If True, use rm_scores directly instead of token_level_rewards.
            This allows filtering earlier (after batch.union, before reward computation).
            rm_scores contains the final_score from reward model at the last token position.

    Returns:
        batch: Modified batch with filtered samples REMOVED (not just masked).
        metrics: Dict containing filtering statistics:
            - "filter_groups/pre_filter_reward_mean": Pre-filter reward mean (excl. aborted)
            - "filter_groups/pre_filter_reward_max": Pre-filter reward max
            - "filter_groups/pre_filter_reward_min": Pre-filter reward min
            - "filter_groups/total_groups": Total number of groups
            - "filter_groups/zero_variance_groups": Number of filtered groups
            - "filter_groups/zero_variance_ratio": Ratio of filtered groups
            - "filter_groups/filtered_samples": Number of filtered samples
            - "filter_groups/filtered_sample_ratio": Ratio of filtered samples
            - "filter_groups/all_success_groups": Number of all-success groups
            - "filter_groups/all_fail_groups": Number of all-fail groups
            - "filter_groups/original_batch_size": Batch size before filtering
            - "filter_groups/filtered_batch_size": Batch size after filtering
    """
    metrics = {}
    original_batch_size = len(batch)

    # Get uid for grouping
    uids = batch.non_tensor_batch.get("uid")
    if uids is None:
        # No grouping info, cannot filter
        metrics["filter_groups/skipped"] = 1.0
        return batch, metrics

    # Get response_mask (needed for pre-filter metrics and aborted detection)
    response_mask = batch.batch.get("response_mask")
    if response_mask is None:
        metrics["filter_groups/skipped"] = 1.0
        return batch, metrics

    # Compute sequence-level rewards based on mode
    if use_rm_scores:
        # Early filtering mode: use rm_scores (from reward model's score)
        # rm_scores has reward value only at the last response token position
        rm_scores = batch.batch.get("rm_scores")
        if rm_scores is None:
            metrics["filter_groups/skipped"] = 1.0
            return batch, metrics
        # Sum to get sequence-level reward (rm_scores has reward only at last token)
        sequence_rewards = rm_scores.sum(dim=-1)  # [bsz]
    else:
        # Late filtering mode: use token_level_rewards (after reward computation)
        token_level_rewards = batch.batch.get("token_level_rewards")
        if token_level_rewards is None:
            metrics["filter_groups/skipped"] = 1.0
            return batch, metrics
        # Compute sequence-level rewards (sum of token rewards)
        # This is what GRPO uses for advantage normalization
        sequence_rewards = (token_level_rewards * response_mask).sum(dim=-1)  # [bsz]

    # ========== Pre-filter reward metrics (before any filtering) ==========
    # Compute metrics on all samples (excluding aborted ones with response_length=0)
    # This ensures we capture reward stats before filtering removes samples
    response_length = response_mask.sum(dim=-1)
    non_aborted_mask = response_length > 0
    valid_rewards = sequence_rewards[non_aborted_mask]

    if len(valid_rewards) > 0:
        metrics["filter_groups/pre_filter_reward_mean"] = valid_rewards.mean().item()
        metrics["filter_groups/pre_filter_reward_max"] = valid_rewards.max().item()
        metrics["filter_groups/pre_filter_reward_min"] = valid_rewards.min().item()

    # Group samples by uid
    uid_to_indices = defaultdict(list)
    for i, uid in enumerate(uids):
        if uid is not None and uid != "__PAD__":
            uid_to_indices[uid].append(i)

    total_groups = len(uid_to_indices)
    zero_variance_groups = 0
    all_success_groups = 0
    all_fail_groups = 0
    filtered_sample_indices = []

    # Per-reason drop counts (e.g., "zero_std_0.0", "zero_std_1.0")
    drop_reason_counts = defaultdict(int)

    for uid, indices in uid_to_indices.items():
        indices = np.array(indices)
        group_rewards = sequence_rewards[indices]

        # Compute variance
        if len(group_rewards) > 1:
            reward_std = group_rewards.std().item()
            mean_reward = group_rewards.mean().item()
        else:
            reward_std = 0.0
            mean_reward = group_rewards[0].item() if len(group_rewards) > 0 else 0.0

        # Check if zero variance
        if reward_std < epsilon:
            zero_variance_groups += 1
            filtered_sample_indices.extend(indices.tolist())

            # Track all-success (all rewards ~ 1) vs all-fail (all rewards ~ 0)
            success_threshold = 1.0 - epsilon  # reward ~ 1.0
            fail_threshold = epsilon  # reward ~ 0.0

            if mean_reward >= success_threshold:
                all_success_groups += 1
            elif mean_reward <= fail_threshold:
                all_fail_groups += 1

            # Per-reason drop with actual reward value
            drop_reason = f"zero_std_{round(mean_reward, 1)}"
            drop_reason_counts[drop_reason] += 1

    # ========== Direct sample removal (instead of just masking) ==========
    # This saves compute in subsequent forward/backward passes
    filtered_samples = len(filtered_sample_indices)

    if filtered_sample_indices:
        # Build keep mask
        keep_mask = torch.ones(original_batch_size, dtype=torch.bool)
        keep_mask[filtered_sample_indices] = False
        keep_indices = torch.where(keep_mask)[0].numpy()

        # Check if all samples would be filtered
        if len(keep_indices) == 0:
            raise NotImplementedError(
                "All groups filtered, empty batch not supported yet. "
                f"Original batch size: {original_batch_size}, "
                f"all {original_batch_size} samples filtered."
            )

        # Directly remove filtered samples
        batch = batch.select_idxs(keep_indices)

    # Compute metrics
    total_samples = original_batch_size

    metrics["filter_groups/total_groups"] = total_groups
    metrics["filter_groups/zero_variance_groups"] = zero_variance_groups
    metrics["filter_groups/zero_variance_ratio"] = zero_variance_groups / total_groups if total_groups > 0 else 0.0
    metrics["filter_groups/filtered_samples"] = filtered_samples
    metrics["filter_groups/filtered_sample_ratio"] = filtered_samples / total_samples if total_samples > 0 else 0.0
    # zero_variance_other = groups where all rewards are same but not 0 or 1 (e.g., all 0.5)
    zero_variance_other = zero_variance_groups - all_success_groups - all_fail_groups

    metrics["filter_groups/all_success_groups"] = all_success_groups
    metrics["filter_groups/all_fail_groups"] = all_fail_groups
    metrics["filter_groups/zero_variance_other_groups"] = zero_variance_other
    metrics["filter_groups/all_success_ratio"] = all_success_groups / total_groups if total_groups > 0 else 0.0
    metrics["filter_groups/all_fail_ratio"] = all_fail_groups / total_groups if total_groups > 0 else 0.0

    # Per-reason drop metrics
    for reason, count in drop_reason_counts.items():
        metrics[f"filter_groups/drop_{reason}"] = count

    # Batch size metrics
    metrics["filter_groups/original_batch_size"] = original_batch_size
    metrics["filter_groups/filtered_batch_size"] = len(batch)  # After removal (before padding)

    if zero_variance_groups > 0:
        print(
            f"[filter_groups] Removed {zero_variance_groups}/{total_groups} groups "
            f"({filtered_samples}/{total_samples} samples, batch: {original_batch_size}->{len(batch)}): "
            f"all_success(reward=1)={all_success_groups}, "
            f"all_fail(reward=0)={all_fail_groups}, other={zero_variance_other}"
        )

    return batch, metrics
