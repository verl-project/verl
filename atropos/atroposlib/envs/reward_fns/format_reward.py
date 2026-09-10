"""Reward function for checking if completions have specific XML-style tags."""

import logging
import re
from typing import Any, List, Optional

from .registry import registry
from .reward_function import RewardFunction

logger = logging.getLogger(__name__)


@registry.register
class FormatReward(RewardFunction):
    """Reward function that checks if completions have XML-style tags."""

    def __init__(
        self,
        preferred_tags: Optional[List[str]] = None,
        require_all_tags: bool = False,
        case_sensitive: bool = False,
        weight: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the format reward function.

        Args:
            preferred_tags: List of tag names to search for (defaults to ['think', 'answer'])
            require_all_tags: If True, require all tags to be present for a reward
            case_sensitive: If True, perform case-sensitive tag matching
            weight: Weight for this reward
            **kwargs: Additional configuration
        """
        super().__init__(weight=weight, **kwargs)
        self.preferred_tags = preferred_tags or ["think", "answer"]
        self.require_all_tags = require_all_tags
        self.case_sensitive = case_sensitive

    def compute(self, completions: List[Any], **kwargs) -> List[float]:
        """
        Check if completions have the expected XML-style tags.

        Args:
            completions: List of completions to evaluate
            **kwargs: Additional context

        Returns:
            List of rewards for each completion (1.0 for good format, 0.0 otherwise)
        """
        # Extract content from different possible formats
        completion_contents = [
            self.get_content(completion) for completion in completions
        ]

        # For each completion, check for the preferred tags
        rewards = []

        flags = 0 if self.case_sensitive else re.IGNORECASE
        flags |= re.DOTALL  # Allow . to match newlines

        for content in completion_contents:
            if self.require_all_tags:
                # All tags must be present
                all_tags_present = True
                for tag in self.preferred_tags:
                    pattern = f"<{tag}>.*?</{tag}>"
                    if not re.search(pattern, content, flags):
                        all_tags_present = False
                        break
                rewards.append(1.0 if all_tags_present else 0.0)
            else:
                # Any tag can be present
                has_tags = False
                for tag in self.preferred_tags:
                    pattern = f"<{tag}>.*?</{tag}>"
                    if re.search(pattern, content, flags):
                        has_tags = True
                        break
                rewards.append(1.0 if has_tags else 0.0)

        # Log the results
        logger.info(
            f"Format reward results: {sum(rewards)}/{len(rewards)} completions match format"
        )
        return rewards


# Legacy function for backward compatibility
def format_reward(
    completions: List[Any], preferred_tags: Optional[List[str]] = None, **kwargs
) -> List[float]:
    """
    Legacy function wrapper for FormatReward.

    Args:
        completions: List of completions to evaluate
        preferred_tags: List of tag names to search for (defaults to ['think', 'answer'])
        **kwargs: Additional keyword arguments

    Returns:
        List of rewards for each completion (1.0 for good format, 0.0 otherwise)
    """
    reward_fn = FormatReward(preferred_tags=preferred_tags)
    return reward_fn.compute(completions, **kwargs)
