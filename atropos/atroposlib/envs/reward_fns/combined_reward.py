"""Combined reward function that combines multiple reward functions."""

import logging
from typing import Any, Dict, List, Union

from .registry import registry
from .reward_function import RewardFunction

logger = logging.getLogger(__name__)


@registry.register
class CombinedReward(RewardFunction):
    """Meta reward function that combines multiple reward functions"""

    def __init__(
        self,
        rewards: List[Union[str, Dict]],
        normalization: str = "none",
        weight: float = 1.0,
        **kwargs,
    ):
        """
        Initialize with a list of reward functions to combine.

        Args:
            rewards: List of reward functions (names or config dicts)
            normalization: How to normalize rewards, one of:
                          - "none": No normalization
                          - "sum": Divide by sum of weights
                          - "minmax": Scale to range [0,1] based on min/max values
            weight: Weight for this combined reward
            **kwargs: Additional parameters
        """
        super().__init__(weight=weight, **kwargs)
        self.normalization = normalization
        self.reward_functions = []

        # Initialize all sub-reward functions
        for reward_config in rewards:
            self.reward_functions.append(registry.create(reward_config))

    @property
    def name(self) -> str:
        """Get a descriptive name for this combined reward"""
        return f"combined({','.join(r.name for r in self.reward_functions)})"

    def set_wandb_logger(self, logger):
        """Propagate the WandB logger to all sub-rewards"""
        super().set_wandb_logger(logger)
        for reward_fn in self.reward_functions:
            reward_fn.set_wandb_logger(logger)

    def compute(self, completions: List[Any], **kwargs) -> List[float]:
        """Compute combined rewards by calling all sub-rewards"""
        if not completions:
            return []

        # Initialize with zeros
        combined_rewards = [0.0] * len(completions)

        # Collect all sub-reward values
        all_rewards = []
        for reward_fn in self.reward_functions:
            try:
                rewards = reward_fn.compute(completions, **kwargs)
                all_rewards.append(rewards)

                # Add to combined total (pre-normalization)
                for i, r in enumerate(rewards):
                    combined_rewards[i] += r
            except Exception as e:
                logger.error(f"Error computing reward for {reward_fn.name}: {e}")
                logger.exception(e)

        # Apply normalization if needed
        if self.normalization == "sum":
            total_weight = sum(r.weight for r in self.reward_functions)
            if total_weight > 0:
                combined_rewards = [r / total_weight for r in combined_rewards]
        elif self.normalization == "minmax":
            # Avoid division by zero
            reward_min = min(combined_rewards) if combined_rewards else 0
            reward_max = max(combined_rewards) if combined_rewards else 0
            if reward_max > reward_min:
                combined_rewards = [
                    (r - reward_min) / (reward_max - reward_min)
                    for r in combined_rewards
                ]

        return combined_rewards
