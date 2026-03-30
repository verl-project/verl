"""Convert Atropos ScoredData dicts to verl DataProto format.

ScoredData format (from Atropos):
    tokens:     [[int, ...], ...]      — prompt+response sequences (group_size items)
    masks:      [[int, ...], ...]      — -100 for prompt, token IDs for response
    scores:     [float, ...]           — per-sequence scalar rewards
    advantages: [[float, ...], ...]    — optional per-token advantages
    overrides:  [dict, ...]            — optional per-item overrides
"""

import uuid

import numpy as np
import torch

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask


def _find_response_start(mask: list[int]) -> int:
    """Find the first non-(-100) position in a mask, which marks the response start."""
    for i, val in enumerate(mask):
        if val != -100:
            return i
    return len(mask)


def scored_data_to_dataproto(
    scored_data_list: list[dict],
    max_prompt_length: int,
    max_response_length: int,
    pad_token_id: int = 0,
) -> DataProto:
    """Convert Atropos ScoredData dicts into a padded verl DataProto."""
    all_prompts = []
    all_responses = []
    all_attention_masks = []
    all_rm_scores = []
    all_input_ids = []
    all_uids = []
    all_token_level_advantages = []
    has_token_level_advantages = False

    for scored_data in scored_data_list:
        tokens_list = scored_data["tokens"]
        masks_list = scored_data["masks"]
        scores = scored_data["scores"]
        advantages = scored_data.get("advantages", None)
        overrides = scored_data.get("overrides", None)

        # all items in a ScoredData group share the same UID (for GRPO grouping)
        group_uid = str(uuid.uuid4())

        for i, (tokens, mask, score) in enumerate(zip(tokens_list, masks_list, scores, strict=True)):
            override_zero = False
            if overrides and i < len(overrides) and overrides[i]:
                if overrides[i].get("set_advantage_to_zero", False):
                    score = 0.0
                    override_zero = True

            response_start = _find_response_start(mask)
            prompt_tokens = tokens[:response_start]
            response_tokens = tokens[response_start:]

            prompt_tokens = prompt_tokens[-max_prompt_length:]
            response_tokens = response_tokens[:max_response_length]

            actual_prompt_len = len(prompt_tokens)
            actual_response_len = len(response_tokens)

            prompt_padding = max_prompt_length - actual_prompt_len
            padded_prompt = [pad_token_id] * prompt_padding + prompt_tokens

            response_padding = max_response_length - actual_response_len
            padded_response = response_tokens + [pad_token_id] * response_padding

            input_ids = padded_prompt + padded_response

            attention_mask = (
                [0] * prompt_padding + [1] * actual_prompt_len + [1] * actual_response_len + [0] * response_padding
            )

            # rm_scores: scalar reward placed at the last valid response position
            # matches verl's NaiveRewardManager format
            rm_score = [0.0] * max_response_length
            if actual_response_len > 0:
                rm_score[actual_response_len - 1] = score

            all_prompts.append(padded_prompt)
            all_responses.append(padded_response)
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_rm_scores.append(rm_score)
            all_uids.append(group_uid)

            # token-level advantages (zeroed out when set_advantage_to_zero override is active)
            if override_zero:
                if advantages is not None:
                    has_token_level_advantages = True
                all_token_level_advantages.append([0.0] * max_response_length)
            elif advantages is not None and i < len(advantages) and advantages[i] is not None:
                has_token_level_advantages = True
                adv = advantages[i]
                adv_response = adv[response_start:]
                adv_response = adv_response[:max_response_length]
                adv_padding = max_response_length - len(adv_response)
                padded_adv = adv_response + [0.0] * adv_padding
                all_token_level_advantages.append(padded_adv)
            else:
                all_token_level_advantages.append([0.0] * max_response_length)

    if len(all_prompts) == 0:
        raise ValueError("No data to convert: scored_data_list is empty or contains no items")

    bs = len(all_prompts)

    attention_mask = torch.tensor(all_attention_masks, dtype=torch.long)
    tensors = {
        "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
        "prompts": torch.tensor(all_prompts, dtype=torch.long),
        "responses": torch.tensor(all_responses, dtype=torch.long),
        "attention_mask": attention_mask,
        "position_ids": compute_position_id_with_mask(attention_mask),
        "token_level_rewards": torch.tensor(all_rm_scores, dtype=torch.float32),
    }

    if has_token_level_advantages:
        tensors["token_level_advantages"] = torch.tensor(all_token_level_advantages, dtype=torch.float32)

    # non_tensor_batch: uid for GRPO grouping, multi_modal_inputs (required by workers, empty here)
    non_tensors = {
        "uid": np.array(all_uids, dtype=object),
        "multi_modal_inputs": np.array([{} for _ in range(bs)], dtype=object),
    }

    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)
