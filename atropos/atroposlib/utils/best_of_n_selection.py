"""
Greedy selection of the best alternative in a group of alternatives.

For a group of alternatives, select the one with the highest score (raw rewards or advantages).
"""

from typing import List, Union


def select_best_index(
    primary_scores: List[Union[float, int]],
    secondary_scores: List[Union[float, int]],
    primary_higher_is_better: bool = True,
    secondary_lower_is_better: bool = True,
) -> int:
    """
    Selects the index of the best item from a list based on primary and secondary scores.

    Args:
        primary_scores: A list of scores that are the primary criterion for selection.
        secondary_scores: A list of scores used for tie-breaking if primary scores are equal.
        primary_higher_is_better: If True, higher primary scores are considered better.
                                   If False, lower primary scores are considered better.
        secondary_lower_is_better: If True, lower secondary scores are considered better for tie-breaking.
                                    If False, higher secondary scores are considered better.

    Returns:
        The index of the best item.

    Raises:
        ValueError: If primary_scores and secondary_scores have different lengths or are empty.
    """
    if not primary_scores or not secondary_scores:
        raise ValueError("Input score lists cannot be empty.")
    if len(primary_scores) != len(secondary_scores):
        raise ValueError("Primary and secondary score lists must have the same length.")

    num_items = len(primary_scores)
    if num_items == 0:  # Should be caught by the first check, but as a safeguard.
        raise ValueError("Input score lists cannot be empty.")

    best_index = 0

    for i in range(1, num_items):
        # Primary score comparison
        current_primary_is_better = False
        primary_score_i = primary_scores[i]
        primary_score_best = primary_scores[best_index]

        if primary_higher_is_better:
            if primary_score_i > primary_score_best:
                current_primary_is_better = True
        else:  # primary_lower_is_better
            if primary_score_i < primary_score_best:
                current_primary_is_better = True

        if current_primary_is_better:
            best_index = i
            continue

        # If primary scores are effectively equal (within a very small tolerance for floats)
        # or exactly equal for integers, then compare secondary scores.
        # Using a small tolerance for float comparison might be needed if scores are computed.
        # For simplicity here, we'll use direct equality, which is fine for typical int/float rewards.
        if primary_score_i == primary_score_best:
            secondary_score_i = secondary_scores[i]
            secondary_score_best = secondary_scores[best_index]

            current_secondary_is_better_for_tiebreak = False
            if secondary_lower_is_better:
                if secondary_score_i < secondary_score_best:
                    current_secondary_is_better_for_tiebreak = True
            else:  # secondary_higher_is_better
                if secondary_score_i > secondary_score_best:
                    current_secondary_is_better_for_tiebreak = True

            if current_secondary_is_better_for_tiebreak:
                best_index = i

    return best_index
