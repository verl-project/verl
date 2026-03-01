#!/usr/bin/env python3
"""
RubiksCubeTokenRewards: Token-level reward utilities for Rubik's Cube environment

This module provides functions for calculating token-level rewards, which are
important for fine-grained RL training signals that help the model understand
which tokens in its response contribute to success or failure.
"""

import re
from typing import List, Optional


def calculate_token_level_rewards(
    response_text: str,
    is_valid_move: bool,
    parsed_move: Optional[str],
    reward: float,
    token_ids: List[int],
    scale_factor: float = 0.1,
) -> List[float]:
    """
    Calculate token-level rewards based on the response quality

    Args:
        response_text: Full response text from the LLM
        is_valid_move: Whether the parsed move was valid
        parsed_move: The parsed move if any
        reward: The overall reward for the response
        token_ids: List of token IDs in the response
        scale_factor: Scale factor for token rewards

    Returns:
        A list of token-level rewards with the same length as token_ids
    """
    # Initialize with neutral rewards
    token_rewards = [0.0] * len(token_ids)

    if len(token_ids) == 0:
        return token_rewards

    # Extract key parts of the response
    thinking_match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)

    # Find the indices of key tokens
    thinking_start_idx = response_text.find("<think>")
    thinking_end_idx = response_text.find("</think>")

    # Determine approximate character-to-token ratio
    chars_per_token = len(response_text) / len(token_ids)

    # Flag for quality of thinking
    has_good_thinking = False
    if thinking_match and len(thinking_match.group(1).strip()) > 50:
        has_good_thinking = True

    # Process rewards based on response quality
    if is_valid_move and has_good_thinking:
        # Good response with both thinking and valid move
        # Reward distribution: ~60% to tool call, ~40% to thinking
        base_reward = reward * scale_factor

        # Distribute rewards
        for i in range(len(token_ids)):
            # Estimate the character position this token represents
            char_pos = int(i * chars_per_token)

            if thinking_start_idx <= char_pos <= thinking_end_idx:
                # Token is part of thinking section
                token_rewards[i] = base_reward * 0.4
            else:
                # Token is part of other sections
                token_rewards[i] = base_reward * 0.1

    elif is_valid_move and not has_good_thinking:
        # Valid move but poor thinking
        base_reward = reward * scale_factor * 0.7  # Reduced overall reward

        for i in range(len(token_ids)):
            char_pos = int(i * chars_per_token)

            if char_pos >= thinking_start_idx and char_pos <= thinking_end_idx:
                # Token is part of thinking section - still good
                token_rewards[i] = base_reward * 0.8
            else:
                # Token is part of other sections - minimal reward
                token_rewards[i] = base_reward * 0.2

    elif not is_valid_move and has_good_thinking:
        # Good thinking but invalid move
        base_reward = reward * scale_factor * 0.5  # Significantly reduced

        for i in range(len(token_ids)):
            char_pos = int(i * chars_per_token)

            if thinking_start_idx <= char_pos <= thinking_end_idx:
                # Token is part of thinking section - somewhat good
                token_rewards[i] = base_reward * 0.6
            else:
                # Token is part of other sections
                token_rewards[i] = base_reward * 0.3
    else:
        # Poor response overall
        base_reward = reward * scale_factor * 0.3  # Minimal reward

        # Distribute minimal rewards evenly
        for i in range(len(token_ids)):
            token_rewards[i] = base_reward

    # Special handling for move-related tokens when there is a valid move
    if is_valid_move and parsed_move:
        # Try to find the specific tokens that represent the move
        move_pattern = re.escape(parsed_move)
        move_matches = list(re.finditer(move_pattern, response_text))

        for match in move_matches:
            move_start_idx = match.start()
            move_end_idx = match.end()

            # Estimate corresponding token indices
            move_start_token = int(move_start_idx / chars_per_token)
            move_end_token = int(move_end_idx / chars_per_token) + 1

            # Ensure indices are within bounds
            move_start_token = max(0, min(move_start_token, len(token_ids) - 1))
            move_end_token = max(0, min(move_end_token, len(token_ids)))

            # Boost rewards for tokens that directly encode the move
            for i in range(move_start_token, move_end_token):
                token_rewards[i] = (
                    base_reward * 1.5
                )  # Higher reward for the actual move

    return token_rewards


def calculate_advantage_token_weights(
    token_rewards: List[List[float]],
) -> List[List[float]]:
    """
    Calculate token weights for advantage computation

    Args:
        token_rewards: List of token-level rewards for each alternative response

    Returns:
        List of normalized token weights for each alternative
    """
    # Create a copy to avoid modifying the input
    token_weights = [rewards.copy() for rewards in token_rewards]

    # For each alternative
    for i in range(len(token_weights)):
        # Get min and max rewards for this alternative
        min_reward = min(token_weights[i]) if token_weights[i] else 0.0
        max_reward = max(token_weights[i]) if token_weights[i] else 0.0
        reward_range = max_reward - min_reward

        # Normalize to [0.5, 1.0] range to ensure all tokens get some weight
        if reward_range > 0:
            for j in range(len(token_weights[i])):
                normalized = (
                    0.5 + 0.5 * (token_weights[i][j] - min_reward) / reward_range
                )
                token_weights[i][j] = normalized
        else:
            # If all rewards are the same, use uniform weights
            for j in range(len(token_weights[i])):
                token_weights[i][j] = 1.0

    return token_weights
