"""Reward function for evaluating crossword puzzle answer formatting."""

import logging
import re
from typing import Any, List, Optional, Pattern

from .registry import registry
from .reward_function import RewardFunction

logger = logging.getLogger(__name__)


@registry.register
class CrosswordFormatReward(RewardFunction):
    """
    Reward function for crossword puzzle game answers.

    Checks if completions follow the expected formatting for crossword puzzle answers:
    - Contains answer patterns like "1-Across: WORD"
    - Uses only valid characters (letters, no numbers or special chars in answers)
    - Follows specified formatting patterns
    """

    def __init__(
        self,
        format_patterns: Optional[List[Pattern]] = None,
        reward_value: float = 1.0,
        penalize_invalid_chars: bool = True,
        valid_chars: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        weight: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the crossword format reward function.

        Args:
            format_patterns: List of regex patterns to match (optional)
            reward_value: Value to award for correct formatting
            penalize_invalid_chars: Whether to penalize invalid characters
            valid_chars: String of valid characters for answers
            weight: Weight for this reward
            **kwargs: Additional configuration
        """
        super().__init__(weight=weight, **kwargs)
        self.reward_value = reward_value
        self.penalize_invalid_chars = penalize_invalid_chars
        self.valid_chars = valid_chars.upper()

        # Default patterns if none provided
        self.format_patterns = format_patterns or [
            re.compile(
                r"\d+-(?:Across|Down):\s+[A-Z\s]+", re.IGNORECASE
            ),  # Basic format pattern
            re.compile(
                r"^(?:\d+-(?:Across|Down):\s+[A-Z\s]+[\s,]*)+$", re.IGNORECASE
            ),  # Full response format
        ]

    def compute(self, completions: List[Any], **kwargs) -> List[float]:
        """
        Check if completions follow crossword answer formatting.

        Args:
            completions: List of completions to evaluate
            **kwargs: Additional context

        Returns:
            List of rewards (reward_value for correct format, 0.0 otherwise)
        """
        # Extract content from different possible formats
        completion_contents = [
            self.get_content(completion) for completion in completions
        ]

        rewards = []
        for content in completion_contents:
            try:
                # Check for format patterns
                format_match = any(
                    pattern.search(content) for pattern in self.format_patterns
                )

                # Look for answers and check for invalid characters
                valid_chars = True
                if self.penalize_invalid_chars:
                    # Extract answers (text after "Across:" or "Down:")
                    answers = re.findall(
                        r"(?:Across|Down):\s+([A-Za-z]+)", content, re.IGNORECASE
                    )
                    for answer in answers:
                        # Check if answer contains only valid characters
                        if not all(c.upper() in self.valid_chars for c in answer):
                            valid_chars = False
                            break

                # Both format and valid chars must be correct for full reward
                correct_format = format_match and valid_chars
                rewards.append(self.reward_value if correct_format else 0.0)

            except Exception as e:
                logger.error(f"Error in crossword format reward calculation: {e}")
                logger.exception(e)
                rewards.append(0.0)

        return rewards


# Legacy function for backward compatibility
def crossword_format_reward(completions: List[Any], **kwargs) -> List[float]:
    """
    Legacy function wrapper for CrosswordFormatReward.

    Args:
        completions: List of completions to evaluate
        **kwargs: Additional parameters

    Returns:
        List of rewards for crossword format quality
    """
    reward_fn = CrosswordFormatReward()
    return reward_fn.compute(completions, **kwargs)
