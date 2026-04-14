"""Reward function for checking if completions match ground truth answers."""

import logging
import re
from typing import Any, List, Optional, Union

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from .registry import registry
from .reward_function import RewardFunction

logger = logging.getLogger(__name__)


def _normalize_numerical_value(value_str: str) -> float:
    """Convert a string representation of a number to float, handling formatting."""
    return float(value_str.replace(",", "").strip())


def _extract_final_answer(text: str) -> str:
    """
    Extract the final answer from text that might include a full solution.

    Handles formats like:
    - "#### 42" (GSM8K style)
    - "The answer is 42"
    - "\\boxed{42}"

    Returns the extracted answer or the original text if no pattern is found.
    """
    # Check for GSM8K style answers (#### 42)
    if "####" in text:
        match = re.search(r"####\s*(.*?)(?:\s*$|\n)", text)
        if match:
            return match.group(1).strip()

    # Check for boxed answers
    if "\\boxed{" in text:
        match = re.search(r"\\boxed\{([^}]+)\}", text)
        if match:
            return match.group(1).strip()

    # If no special format is found, return the original text
    return text


def _verify_answer(
    content: str, gold_answer: Union[float, int, str], tolerance: float = 1e-6
) -> bool:
    """
    Verifies if the provided content contains an answer matching the gold answer.
    Uses a robust approach with multiple fallback strategies.

    Args:
        content: The model's response content to evaluate
        gold_answer: The correct answer to compare against
        tolerance: Tolerance for floating point comparisons

    Returns:
        Boolean indicating whether the answer is correct
    """
    # Extract the final answer from the gold answer if it has a special format
    if isinstance(gold_answer, str):
        # Check for GSM8K style answers (#### number)
        if "####" in gold_answer:
            gold_answer = _extract_final_answer(gold_answer)
            logger.warning(f"Extracted gold answer: {gold_answer}")

    # Convert gold_answer to numerical if it's not already and if possible
    gold_value = None
    if isinstance(gold_answer, (int, float)):
        gold_value = gold_answer
    elif isinstance(gold_answer, str):
        # Try to extract numerical value if it's in boxed format
        if "\\boxed{" in gold_answer:
            try:
                gold_value = _normalize_numerical_value(
                    gold_answer.replace("\\boxed{", "").replace("}", "")
                )
            except ValueError:
                # Not a numerical value, keep as string for LaTeX parsing
                pass
        else:
            # Try to convert to float if possible
            try:
                gold_value = _normalize_numerical_value(gold_answer)
            except ValueError:
                # Not a numerical value, keep as string for LaTeX parsing
                pass

    # First attempt: Try to parse with math_verify
    try:
        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )

        logger.warning(f"Answer parsed result: {answer_parsed}")

        # If we got a valid parse, verify it against the gold answer
        if answer_parsed:
            # Format gold answer for verification
            gold_str = (
                f"\\boxed{{{gold_answer}}}"
                if not isinstance(gold_answer, str) or "\\boxed" not in gold_answer
                else gold_answer
            )

            gold_parsed = parse(
                gold_str,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            logger.warning(f"Gold parsed result: {gold_parsed}")

            if gold_parsed:
                return verify(answer_parsed, gold_parsed)
    except Exception as e:
        logger.warning(f"Exception in primary parsing: {e}")

    # Fallback: Use regex to extract boxed content for numerical comparison
    if gold_value is not None:  # Only try numerical comparison if gold is a number
        try:
            # Try to extract a boxed answer first
            boxed_matches = re.findall(r"\\boxed\{([^}]+)\}", content)
            if boxed_matches:
                logger.warning(f"Regex boxed matches: {boxed_matches}")
                # Try to extract a numerical value from the boxed content
                try:
                    extracted_value = _normalize_numerical_value(boxed_matches[0])
                    logger.warning(
                        f"Extracted value: {extracted_value}, Gold value: {gold_value}"
                    )
                    # Allow for small floating point differences
                    return abs(extracted_value - gold_value) < tolerance
                except ValueError:
                    logger.warning(f"Could not convert '{boxed_matches[0]}' to float")

            # If no boxed answer, check for a final answer after ####
            if "####" in content:
                match = re.search(r"####\s*([\d\.]+)", content)
                if match:
                    extracted_value = _normalize_numerical_value(match.group(1))
                    logger.warning(
                        f"Extracted value from ####: {extracted_value}, Gold value: {gold_value}"
                    )
                    return abs(extracted_value - gold_value) < tolerance
        except Exception as e:
            logger.warning(f"Exception in regex parsing: {e}")

    return False


@registry.register
class AccuracyReward(RewardFunction):
    """
    Reward function that checks if completions match ground truth answers.

    Works with boxed LaTeX answers, GSM8K-style answers, and other formats.
    Uses a robust approach with multiple fallback strategies for parsing and verification.
    """

    def __init__(
        self,
        tolerance: float = 1e-6,
        split_on_think_tag: bool = True,
        max_boxed_threshold: int = 6,
        weight: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the accuracy reward function.

        Args:
            tolerance: Tolerance for floating point comparisons
            split_on_think_tag: Whether to use only the text after </think> tag
            max_boxed_threshold: Maximum number of boxed expressions before marking as incorrect
            weight: Weight for this reward
            **kwargs: Additional configuration
        """
        super().__init__(weight=weight, **kwargs)
        self.tolerance = tolerance
        self.split_on_think_tag = split_on_think_tag
        self.max_boxed_threshold = max_boxed_threshold

    def compute(
        self,
        completions: List[Any],
        solution: Optional[Union[str, List[str]]] = None,
        ground_truth: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> List[float]:
        """
        Check if completions match ground truth answers.

        Args:
            completions: List of model completions to evaluate
            solution: Ground truth solution(s) - can be a single value or list of values
            ground_truth: Optional canonical ground truth answers (used instead of solution if provided)
            **kwargs: Additional context

        Returns:
            List of reward values (1.0 for correct, 0.0 for incorrect)
        """
        rewards = []

        # Check if we have a solution or ground truth
        if solution is None and ground_truth is None:
            logger.warning("No solution or ground_truth provided to accuracy_reward")
            return [0.0] * len(completions)

        # Use ground_truth instead of solution if available
        gold_answers = ground_truth if ground_truth is not None else solution

        if isinstance(gold_answers, list):
            answers = gold_answers
        else:
            answers = [gold_answers] * len(completions)

        for completion, ans in zip(completions, answers):
            try:
                content = self.get_content(completion)

                if (
                    self.split_on_think_tag
                    and "</think>" in content
                    and content.split("</think>")[-1].count("\\boxed")
                    > self.max_boxed_threshold
                ):
                    logger.warning(
                        "Too many \\boxed commands in response, marking as incorrect"
                    )
                    reward = 0.0
                else:
                    if self.split_on_think_tag and "</think>" in content:
                        answer_part = content.split("</think>")[-1]
                    else:
                        answer_part = content

                    reward = float(_verify_answer(answer_part, ans, self.tolerance))

            except Exception as e:
                logger.warning(f"Error in accuracy_reward: {e}")
                logger.exception(e)
                reward = 0.0

            rewards.append(reward)

        # Calculate statistics
        if rewards:
            logger.info(
                f"Accuracy: {sum(rewards)}/{len(rewards)} ({sum(rewards)/len(rewards):.2f})"
            )

        return rewards


# Legacy function for backward compatibility
def accuracy_reward(
    completions: List[Any],
    solution: Union[str, List[str]] = None,
    ground_truth: Union[str, List[str]] = None,
    **kwargs,
) -> List[float]:
    """
    Legacy function wrapper for AccuracyReward.

    Args:
        completions: List of model completions to evaluate
        solution: Ground truth solution(s) - can be a single value or list of values
        ground_truth: Optional canonical ground truth answers (used instead of solution if provided)
        **kwargs: Additional parameters

    Returns:
        List of reward values (1.0 for correct, 0.0 for incorrect)
    """
    reward_fn = AccuracyReward()
    return reward_fn.compute(
        completions, solution=solution, ground_truth=ground_truth, **kwargs
    )
