import logging
import re
from typing import Any, List, Union

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

logger = logging.getLogger(__name__)


def get_completion_content(completion) -> str:
    """Extract content from completion in various formats."""
    if isinstance(completion, str):
        return completion
    elif isinstance(completion, dict):
        if "content" in completion:
            return completion["content"]
        elif isinstance(completion.get("message", {}), dict):
            return completion["message"].get("content", "")
    elif isinstance(completion, list) and len(completion) > 0:
        if isinstance(completion[0], dict) and "content" in completion[0]:
            return completion[0]["content"]

    logger.warning(f"Could not extract content from completion: {completion}")
    return str(completion)


def _normalize_numerical_value(value_str: str) -> float:
    """Convert a string representation of a number to float, handling formatting."""
    return float(value_str.replace(",", "").strip())


def _extract_final_answer(text: str) -> str:
    """Extract the final answer from text with various formats (GSM8K, boxed, etc)."""
    if "####" in text:
        match = re.search(r"####\s*(.*?)(?:\s*$|\n)", text)
        if match:
            return match.group(1).strip()

    if "\\boxed{" in text:
        match = re.search(r"\\boxed\{([^}]+)\}", text)
        if match:
            return match.group(1).strip()

    return text


def _verify_answer(content: str, gold_answer: Union[float, int, str]) -> bool:
    """Verifies if the content matches the gold answer using multiple strategies."""
    if isinstance(gold_answer, str):
        if "####" in gold_answer:
            gold_answer = _extract_final_answer(gold_answer)
            logger.warning(f"Extracted gold answer: {gold_answer}")

    gold_value = None
    if isinstance(gold_answer, (int, float)):
        gold_value = gold_answer
    elif isinstance(gold_answer, str):
        if "\\boxed{" in gold_answer:
            try:
                gold_value = _normalize_numerical_value(
                    gold_answer.replace("\\boxed{", "").replace("}", "")
                )
            except ValueError:
                pass
        else:
            try:
                gold_value = _normalize_numerical_value(gold_answer)
            except ValueError:
                pass

    # Try math_verify parsing first
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
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )

        if answer_parsed:
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
            if gold_parsed:
                return verify(answer_parsed, gold_parsed)
    except Exception as e:
        logger.warning(f"Exception in primary parsing: {e}")

    # Fallback to numerical comparison
    if gold_value is not None:
        try:
            boxed_matches = re.findall(r"\\boxed\{([^}]+)\}", content)
            if boxed_matches:
                try:
                    extracted_value = _normalize_numerical_value(boxed_matches[0])
                    return abs(extracted_value - gold_value) < 1e-6
                except ValueError:
                    pass

            if "####" in content:
                match = re.search(r"####\s*([\d\.]+)", content)
                if match:
                    extracted_value = _normalize_numerical_value(match.group(1))
                    return abs(extracted_value - gold_value) < 1e-6
        except Exception as e:
            logger.warning(f"Exception in regex parsing: {e}")

    return False


def format_reward(completions, reward_value=0.5, **kwargs):
    """Checks if completion has proper think tag formatting."""
    pattern = r"^<think>[^<]*</think>[^<]*$"

    try:
        completion_contents = [get_completion_content(c) for c in completions]
        matches = [
            re.match(pattern, content, re.DOTALL) for content in completion_contents
        ]
        return [reward_value if match else 0.0 for match in matches]
    except Exception as e:
        logger.error(f"Error in format reward calculation: {e}")
        return [0.0] * len(completions)


def accuracy_reward(
    completions: List[Any], solution: Union[str, List[str]], **kwargs
) -> List[float]:
    """Checks answer accuracy using sophisticated verification."""
    rewards = []

    if not isinstance(solution, list):
        solution = [solution] * len(completions)

    for completion, sol in zip(completions, solution):
        try:
            content = get_completion_content(completion)

            if (
                "</think>" in content
                and content.split("</think>")[-1].count("\\boxed") > 6
            ):
                logger.warning(
                    "Too many \\boxed commands in response, marking as incorrect"
                )
                reward = 0.0
            else:
                answer_part = (
                    content.split("</think>")[-1] if "</think>" in content else content
                )
                reward = float(_verify_answer(answer_part, sol))
        except Exception as e:
            logger.warning(f"Error in accuracy reward: {e}")
            reward = 0.0

        rewards.append(reward)

    return rewards


def cascading_r1_math_reward(completions, solution, **kwargs) -> list[float]:
    """Combines sophisticated accuracy checking with format verification."""
    try:
        accuracy_rewards = accuracy_reward(completions, solution, **kwargs)
        format_rewards = format_reward(completions)

        combined_rewards = []
        for accuracy_score, format_score in zip(accuracy_rewards, format_rewards):
            # Only add format bonus if answer is correct
            format_bonus = format_score if accuracy_score > 0 else 0.0
            total_reward = accuracy_score + format_bonus
            combined_rewards.append(total_reward)

        logger.info(
            f"Teknium rewards: accuracy={accuracy_rewards}, format={format_rewards}, combined={combined_rewards}"
        )
        return combined_rewards
    except Exception as e:
        logger.error(f"Error in teknium_reward: {e}")
        return [0.0] * len(completions)
