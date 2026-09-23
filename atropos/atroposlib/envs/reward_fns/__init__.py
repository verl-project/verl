"""
Reward functions for evaluating model outputs in various environments.

This module provides a framework for creating, composing, and applying reward functions
to evaluate model outputs. Reward functions can be used for both dataset environments
and online/gymnasium environments.

Key components:
- RewardFunction: Abstract base class for all reward functions
- RewardRegistry: Registry for registering and loading reward functions
- CombinedReward: Meta reward function that combines multiple reward functions

Usage:
    # Define a reward function
    @registry.register
    class MyReward(RewardFunction):
        def compute(self, completions, **kwargs):
            # Implementation
            return [score for completion in completions]

    # Create and use a reward function
    reward_fn = registry.create("my_reward", weight=1.5)
    scores = reward_fn(completions, **kwargs)
"""

from .combined_reward import CombinedReward
from .registry import registry
from .reward_function import RewardFunction

__all__ = ["RewardFunction", "registry", "CombinedReward"]
