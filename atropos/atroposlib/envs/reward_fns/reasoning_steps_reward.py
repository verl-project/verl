"""Reward function for evaluating step-by-step reasoning in completions."""

import logging
import re
from typing import Any, Dict, List, Optional

from .registry import registry
from .reward_function import RewardFunction

logger = logging.getLogger(__name__)


@registry.register
class ReasoningStepsReward(RewardFunction):
    r"""
    Reward function that evaluates step-by-step reasoning in completions.

    Looks for several types of step-by-step reasoning indicators:
    1. Numbered step patterns like "Step 1:", "Step 2:"
    2. Numbered lists like "1.", "2." at start of line
    3. Bullet points with hyphens or asterisks
    4. Sequential transition words (First, Second, Next, Finally, etc.)
    """

    def __init__(
        self,
        min_words: int = 10,
        min_steps: int = 3,
        base_score: float = 0.1,
        pattern_weights: Optional[Dict[str, float]] = None,
        weight: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the reasoning steps reward function.

        Args:
            min_words: Minimum number of words to consider for base score
            min_steps: Number of steps needed for full points in each category
            base_score: Base score for having content longer than min_words
            pattern_weights: Custom weights for each pattern type (optional)
            weight: Weight for this reward
            **kwargs: Additional configuration
        """
        super().__init__(weight=weight, **kwargs)
        self.min_words = min_words
        self.min_steps = min_steps
        self.base_score = base_score

        # Default pattern weights
        self.pattern_weights = {
            "numbered_steps": 0.5,  # Strong indicators
            "list_numbers": 0.5,  # Strong indicators
            "bullet_points": 0.4,  # Medium indicators
            "transition_words": 0.3,  # Weaker indicators
        }

        # Override with custom weights if provided
        if pattern_weights:
            self.pattern_weights.update(pattern_weights)

        # Patterns for different types of step indicators
        self.patterns = {
            # Step 1: style numbered steps
            "numbered_steps": r"Step\s+\d+[\s:]+",
            # Numbered lists (1., 2., etc.)
            "list_numbers": r"(?:^|\n)\s*\d+\.\s+",
            # Bullet points
            "bullet_points": r"(?:^|\n)\s*[\-\*•]\s+",
            # Sequential transition words - expanded to include more phrases
            "transition_words": r"\b(?:First|Second|Third|Fourth|Fifth|Next|Then|Finally|"
            r"Subsequently|Afterward|Lastly|Initially|To begin|Let\'s begin|"
            r"I\'ll first|After that|In conclusion|Eventually|Subsequently|"
            r"To solve|begin by|understand|analyze|apply|compute)\b",
        }

    def compute(self, completions: List[Any], **kwargs) -> List[float]:
        """
        Calculate reasoning quality scores based on pattern matching.

        Args:
            completions: List of completions to evaluate
            **kwargs: Additional context

        Returns:
            List of reward scores between 0.0 and 1.0
        """
        # Extract content from different possible formats
        completion_contents = [
            self.get_content(completion) for completion in completions
        ]

        rewards = []
        for content in completion_contents:
            score = 0.0
            pattern_matches = {}

            # Check for each type of pattern
            for pattern_type, pattern in self.patterns.items():
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                pattern_matches[pattern_type] = len(matches)

                # Add score based on matches and pattern weight
                weight = self.pattern_weights.get(
                    pattern_type, 0.3
                )  # Default weight if not specified
                score += min(1.0, len(matches) / self.min_steps) * weight

            # Add a small base score for any content that has more than just an answer
            # This helps differentiate minimal reasoning from no reasoning
            if len(content.split()) > self.min_words:
                score += self.base_score

            # Cap the total score at 1.0
            score = min(1.0, score)
            rewards.append(score)

            logger.info(
                f"Reasoning steps reward for completion: {pattern_matches}, score: {score}"
            )

        return rewards


# Legacy function for backward compatibility
def reasoning_steps_reward(completions: List[Any], **kwargs) -> List[float]:
    """
    Legacy function wrapper for ReasoningStepsReward.

    Args:
        completions: List of completions to evaluate
        **kwargs: Additional parameters

    Returns:
        List of reward scores between 0.0 and 1.0
    """
    reward_fn = ReasoningStepsReward()
    return reward_fn.compute(completions, **kwargs)
