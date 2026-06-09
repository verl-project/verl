"""Reward function that combines reasoning format and accuracy rewards."""

import logging
import re
from typing import Any, Dict, List, Optional, Union

from .registry import registry
from .reward_function import RewardFunction

logger = logging.getLogger(__name__)


def parse_reasoning_response(text: str) -> Dict[str, Any]:
    """
    Parse text to extract thinking section and response section.

    Args:
        text: Text to parse for thinking and response sections

    Returns:
        Dictionary with thinking_content, response, and multiple_thinking flag
    """
    # Check if text is actually a string
    if not isinstance(text, str):
        logger.warning(f"Expected string but got {type(text)}: {text}")
        return {
            "thinking_content": "",
            "response": str(text),
            "multiple_thinking": False,
        }

    # Find all thinking blocks
    thinking_blocks = re.findall(r"<think>.*?</think>", text, re.DOTALL)

    # If there's more than one thinking block, fail
    if len(thinking_blocks) > 1:
        return {"thinking_content": "", "response": text, "multiple_thinking": True}

    # Match the single thinking block if it exists
    pattern = r"<think>\s*(.*?)\s*</think>\s*(.*)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return {"thinking_content": "", "response": text, "multiple_thinking": False}

    return {
        "thinking_content": match.group(1).strip(),
        "response": match.group(2).strip(),
        "multiple_thinking": False,
    }


@registry.register
class FormatReasoningReward(RewardFunction):
    """
    Reward function that checks for proper reasoning format.

    Checks if completions have:
    1. A thinking section in <think> tags
    2. A response section after the thinking tags
    3. No multiple thinking sections (only one <think> block)
    """

    def __init__(
        self,
        reward_value: float = 0.5,
        require_thinking: bool = True,
        require_response: bool = True,
        allow_multiple_thinking: bool = False,
        weight: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the format reasoning reward function.

        Args:
            reward_value: Value to award for correct formatting
            require_thinking: Whether to require thinking content
            require_response: Whether to require response content
            allow_multiple_thinking: Whether to allow multiple thinking sections
            weight: Weight for this reward
            **kwargs: Additional configuration
        """
        super().__init__(weight=weight, **kwargs)
        self.reward_value = reward_value
        self.require_thinking = require_thinking
        self.require_response = require_response
        self.allow_multiple_thinking = allow_multiple_thinking

    def compute(self, completions: List[Any], **kwargs) -> List[float]:
        """
        Check if completions have proper reasoning format.

        Args:
            completions: List of completions to evaluate
            **kwargs: Additional context

        Returns:
            List of rewards (reward_value for correct format, 0.0 otherwise)
        """
        parsed = []
        for completion in completions:
            try:
                content = self.get_content(completion)
                parsed.append(parse_reasoning_response(content))
            except Exception as e:
                logger.error(f"Error parsing response: {e}")
                logger.exception(e)
                parsed.append(
                    {"thinking_content": "", "response": "", "multiple_thinking": False}
                )

        rewards = []
        for p in parsed:
            try:
                # Check if response meets format requirements
                valid_format = True

                if self.require_thinking and not p["thinking_content"]:
                    valid_format = False

                if self.require_response and not p["response"]:
                    valid_format = False

                if not self.allow_multiple_thinking and p["multiple_thinking"]:
                    valid_format = False

                rewards.append(self.reward_value if valid_format else 0.0)
            except Exception as e:
                logger.error(f"Error in format reward calculation: {e}")
                logger.exception(e)
                rewards.append(0.0)

        return rewards


@registry.register
class AccuracyXReward(RewardFunction):
    """
    Reward function that checks if completion responses contain the solution.

    First parses completions to extract the response part after thinking tags,
    then checks if the solution string is contained in the response.
    """

    def __init__(
        self,
        exact_match: bool = False,
        case_sensitive: bool = False,
        reward_value: float = 1.0,
        weight: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the accuracy reward function.

        Args:
            exact_match: Whether to require exact match vs. contained
            case_sensitive: Whether to do case-sensitive matching
            reward_value: Value to award for correct answer
            weight: Weight for this reward
            **kwargs: Additional configuration
        """
        super().__init__(weight=weight, **kwargs)
        self.exact_match = exact_match
        self.case_sensitive = case_sensitive
        self.reward_value = reward_value

    def compute(
        self,
        completions: List[Any],
        solution: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> List[float]:
        """
        Check if completion responses contain the solution.

        Args:
            completions: List of completions to evaluate
            solution: The solution to check for
            **kwargs: Additional context

        Returns:
            List of rewards (reward_value for correct, 0.0 otherwise)
        """
        if solution is None:
            logger.warning("No solution provided for accuracy reward")
            return [0.0] * len(completions)

        parsed_responses = []
        for completion in completions:
            try:
                content = self.get_content(completion)
                parsed_responses.append(parse_reasoning_response(content))
            except Exception as e:
                logger.error(f"Error parsing response: {e}")
                logger.exception(e)
                parsed_responses.append(
                    {"thinking_content": "", "response": "", "multiple_thinking": False}
                )

        rewards = []

        # Ensure solution is in the right format
        if not isinstance(solution, list):
            solution = [solution] * len(parsed_responses)

        for resp, sol in zip(parsed_responses, solution):
            try:
                # Extract solution content if needed
                sol_content = self.get_content(sol) if not isinstance(sol, str) else sol
                resp_content = resp["response"]

                # Do the matching based on settings
                if not self.case_sensitive:
                    sol_content = sol_content.lower()
                    resp_content = resp_content.lower()

                if self.exact_match:
                    match = resp_content == sol_content
                else:
                    match = sol_content in resp_content

                rewards.append(self.reward_value if match else 0.0)
            except Exception as e:
                logger.error(f"Error in accuracy reward calculation: {e}")
                logger.exception(e)
                rewards.append(0.0)

        return rewards


@registry.register
class R1Reward(RewardFunction):
    """
    Combined reward function that rewards both reasoning format and accuracy.

    This reward function combines:
    1. FormatReasoningReward - rewards for proper <think> and response formatting
    2. AccuracyXReward - rewards for having the solution in the response
    """

    def __init__(
        self,
        format_weight: float = 0.5,
        accuracy_weight: float = 1.0,
        weight: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the R1 reward function.

        Args:
            format_weight: Weight for the format component
            accuracy_weight: Weight for the accuracy component
            weight: Weight for the overall reward
            **kwargs: Additional configuration
        """
        super().__init__(weight=weight, **kwargs)
        self.format_weight = format_weight
        self.accuracy_weight = accuracy_weight

        # Create component reward functions
        self.format_reward_fn = FormatReasoningReward(
            reward_value=1.0,  # Use 1.0 here, we'll apply weight in compute
            weight=1.0,  # This will be overridden
        )

        self.accuracy_reward_fn = AccuracyXReward(
            reward_value=1.0,  # Use 1.0 here, we'll apply weight in compute
            weight=1.0,  # This will be overridden
        )

    def compute(
        self,
        completions: List[Any],
        solution: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> List[float]:
        """
        Calculate combined format and accuracy rewards.

        Args:
            completions: List of completions to evaluate
            solution: The solution to check for accuracy
            **kwargs: Additional context

        Returns:
            List of combined rewards
        """
        try:
            # Calculate component rewards
            format_rewards = self.format_reward_fn.compute(completions, **kwargs)
            accuracy_rewards = self.accuracy_reward_fn.compute(
                completions, solution=solution, **kwargs
            )

            # Apply component weights and combine
            rewards = [
                (f * self.format_weight) + (a * self.accuracy_weight)
                for f, a in zip(format_rewards, accuracy_rewards)
            ]

            logger.info(
                f"R1 rewards: accuracy={accuracy_rewards}, format={format_rewards}, combined={rewards}"
            )
            return rewards
        except Exception as e:
            logger.error(f"Error in r1_reward: {e}")
            logger.exception(e)
            # Return zero rewards for all completions
            return [0.0] * len(completions)


# Legacy function for backward compatibility
def format_reasoning_reward(completions: List[Any], **kwargs) -> List[float]:
    """
    Legacy function wrapper for FormatReasoningReward.

    Args:
        completions: List of completions to evaluate
        **kwargs: Additional parameters

    Returns:
        List of rewards for format quality
    """
    reward_fn = FormatReasoningReward(reward_value=0.5)
    return reward_fn.compute(completions, **kwargs)


def accuracy_reward(
    completions: List[Any], solution: Union[str, List[str]], **kwargs
) -> List[float]:
    """
    Legacy function wrapper for AccuracyXReward.

    Args:
        completions: List of completions to evaluate
        solution: The solution to check for
        **kwargs: Additional parameters

    Returns:
        List of rewards for accuracy
    """
    reward_fn = AccuracyXReward()
    return reward_fn.compute(completions, solution=solution, **kwargs)


def r1_reward(
    completions: List[Any], solution: Union[str, List[str]], **kwargs
) -> List[float]:
    """
    Legacy function wrapper for R1Reward.

    Args:
        completions: List of completions to evaluate
        solution: The solution to check for
        **kwargs: Additional parameters

    Returns:
        List of combined rewards
    """
    reward_fn = R1Reward()
    return reward_fn.compute(completions, solution=solution, **kwargs)
