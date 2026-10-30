from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ThresholdLengthPenaltyConfig:
    """Configuration for length penalty calculations"""

    max_token_length: int
    threshold_percentage: float = 0.5  # Default threshold at 50% of max length


class ThresholdLengthPenaltyCalculator:
    """Handles calculation of length-based penalties for token sequences"""

    def __init__(self, config: ThresholdLengthPenaltyConfig):
        """
        Initialize the length penalty calculator

        Args:
            config: Configuration object containing max_token_length and threshold settings
        """
        self.config = config
        self.length_threshold = (
            self.config.max_token_length * self.config.threshold_percentage
        )

    def apply_length_penalties(
        self, scores: Dict[str, List]
    ) -> Optional[Dict[str, List]]:
        """
        Apply length-based penalties to scores if all responses are correct

        Args:
            scores: Dictionary containing 'scores' and 'tokens' lists

        Returns:
            Modified scores dictionary or None if invalid input
        """
        # Validate input
        if not scores or "scores" not in scores or "tokens" not in scores:
            return None

        # Only apply penalties if all responses are correct
        if not all([score == 1.0 for score in scores["scores"]]):
            return scores

        # Calculate token lengths
        token_lengths = [len(token) for token in scores["tokens"]]
        if max(token_lengths) == 0:
            return None

        # Apply modified length penalty with threshold
        new_scores = []
        for length in token_lengths:
            if length <= self.length_threshold:
                # No penalty for responses under threshold
                new_scores.append(1.0)
            else:
                # Calculate penalty based on how far we are between threshold and max
                percentage_of_range = (length - self.length_threshold) / (
                    self.config.max_token_length - self.length_threshold
                )
                # Cap at 1.0 in case length exceeds max_allowed_length
                percentage_of_range = min(percentage_of_range, 1.0)
                # Apply linear penalty scaling from 1.0 down to 0.0
                new_scores.append(1.0 - percentage_of_range)

        scores["scores"] = new_scores
        return scores


# Example usage:
# config = LengthPenaltyConfig(max_token_length=1024)
# calculator = LengthPenaltyCalculator(config)
# modified_scores = calculator.apply_length_penalties(scores_dict)
