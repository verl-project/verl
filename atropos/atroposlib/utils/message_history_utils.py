"""
Trajectory utils

Utils for managing trajectory sizing, formatting, compression, etc.
"""

import logging
from typing import List

from transformers import PreTrainedTokenizer

from atroposlib.envs.base import ScoredDataGroup
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

logger = logging.getLogger(__name__)


def strip_thinking(response_text: str) -> str:
    """Helper to strip the <think> block of a response entirely.

    Args:
        response_text: The response text to strip.

    Returns:
        The stripped response text.
    """
    think_start_tag = "<think>"
    think_end_tag = "</think>"

    think_start_idx = response_text.find(think_start_tag)
    think_end_idx = response_text.find(think_end_tag)

    if think_start_idx != -1 and think_end_idx != -1:
        return (
            response_text[:think_start_idx]
            + response_text[think_end_idx + len(think_end_tag) :]
        )
    else:
        return response_text


def truncate_thinking(
    response_text: str, tokenizer: PreTrainedTokenizer, max_think_tokens: int
) -> str:
    """Helper to truncate the <think> block of a response for message history based on token count.

    Args:
        response_text: The response text to truncate.
        tokenizer: The tokenizer to use for counting tokens.
        max_think_tokens: The maximum number of tokens to keep in the <think> block.

    Returns:
        The truncated response text.
    """
    try:
        think_start_tag = "<think>"
        think_end_tag = "</think>"

        think_start_idx = response_text.find(think_start_tag)
        think_end_idx = response_text.find(think_end_tag)

        if not (
            think_start_idx != -1
            and think_end_idx != -1
            and think_start_idx < think_end_idx
        ):
            return response_text

        part_before_content = response_text[: think_start_idx + len(think_start_tag)]
        original_think_content_raw = response_text[
            think_start_idx + len(think_start_tag) : think_end_idx
        ]
        part_after_content = response_text[think_end_idx:]

        original_think_content_stripped = original_think_content_raw.strip()

        if not original_think_content_stripped:
            # Normalize empty or whitespace-only think blocks to <think></think>
            return f"{part_before_content.rstrip()}{part_after_content.lstrip()}"

        all_think_tokens = tokenizer.encode(
            original_think_content_stripped, add_special_tokens=False
        )

        is_truncated_internally = False
        final_think_tokens: List[int]

        if len(all_think_tokens) <= max_think_tokens:
            final_think_tokens = all_think_tokens
            is_truncated_internally = False
        else:
            is_truncated_internally = (
                True  # Mark as truncated if len(all_think_tokens) > max_think_tokens
            )
            paragraphs = [
                p.strip()
                for p in original_think_content_stripped.split("\n\n")
                if p.strip()
            ]

            attempted_paragraph_truncation = False
            if paragraphs:
                last_paragraph_text = paragraphs[-1]
                # Check if last paragraph is genuinely shorter than the whole content
                # (i.e., there was content before it)
                if len(last_paragraph_text) < len(original_think_content_stripped):
                    last_paragraph_tokens = tokenizer.encode(
                        last_paragraph_text, add_special_tokens=False
                    )
                    if len(last_paragraph_tokens) <= max_think_tokens:
                        final_think_tokens = last_paragraph_tokens
                        attempted_paragraph_truncation = True

            if (
                not attempted_paragraph_truncation
            ):  # Default to truncating the whole content from the end
                # Ensure max_think_tokens is not negative, though practically it shouldn't be.
                slice_start = max(0, len(all_think_tokens) - max_think_tokens)
                final_think_tokens = all_think_tokens[slice_start:]

        # Decode the tokens to string
        decoded_think_content = tokenizer.decode(
            final_think_tokens, skip_special_tokens=True
        )

        # Add "..." prefix if truncated and content remains
        final_internal_content_str = decoded_think_content
        if is_truncated_internally and decoded_think_content.strip():
            final_internal_content_str = "... " + decoded_think_content.lstrip()

        # Determine the final block content (empty or with newlines)
        final_internal_content_str_stripped = final_internal_content_str.strip()
        final_content_for_block: str
        if (
            not final_internal_content_str_stripped
            or final_internal_content_str_stripped == "..."
        ):
            final_content_for_block = ""
        else:
            final_content_for_block = f"\n{final_internal_content_str_stripped}\n"

        return f"{part_before_content.rstrip()}{final_content_for_block}{part_after_content.lstrip()}"

    except Exception as e:
        logger.error(
            f"Error in truncate_thinking for text '{response_text[:200]}...': {e}",
            exc_info=True,
        )
        return response_text


def ensure_trajectory_token_limit(
    trajectory: List[ScoredDataGroup],
    tokenizer: PreTrainedTokenizer,
    max_trajectory_tokens: int,
) -> List[ScoredDataGroup]:
    """
    Ensure token sequences in a trajectory don't exceed max_trajectory_tokens.
    Attempts to uniformly truncate older messages (preferably paired turns) from all alternatives within a step.
    The system prompt, last environment observation, and last agent response are preserved as a minimum.
    If a step still exceeds the limit after maximum possible truncation, it is discarded.

    Args:
        trajectory: List of ScoredDataGroup from an episode

    Returns:
        The trajectory with potentially truncated messages/tokens/masks or filtered steps
    """
    if not trajectory:
        return trajectory

    filtered_trajectory: List[ScoredDataGroup] = []

    for step_idx, original_step_data in enumerate(trajectory):
        if not (
            original_step_data.get("messages")
            and original_step_data.get("tokens")
            and original_step_data.get("masks")
            and original_step_data.get("seed") is not None
            and original_step_data.get("parsed_actions") is not None
        ):
            logger.warning(
                f"[_ensure_trajectory_token_limit] Step {step_idx} in MC env "
                f"is missing critical data. Skipping."
            )
            continue

        max_initial_tokens = 0
        if original_step_data["tokens"]:
            max_initial_tokens = (
                max(
                    len(alt_tokens)
                    for alt_tokens in original_step_data["tokens"]
                    if isinstance(alt_tokens, list)
                )
                if any(
                    isinstance(alt_tokens, list)
                    for alt_tokens in original_step_data["tokens"]
                )
                else 0
            )

        if max_initial_tokens <= max_trajectory_tokens:
            filtered_trajectory.append(original_step_data)
            logger.info(
                f"[_ensure_trajectory_token_limit] Step {step_idx} compliant in MC env. "
                f"Max tokens: {max_initial_tokens}"
            )
            continue

        logger.info(
            f"[_ensure_trajectory_token_limit] Step {step_idx} in MC env (max tokens: {max_initial_tokens}) "
            f"exceeds limit ({max_trajectory_tokens}). Attempting truncation."
        )

        working_messages = [
            msgs_list.copy() for msgs_list in original_step_data["messages"] or []
        ]
        working_tokens = [
            tkns_list.copy() for tkns_list in original_step_data["tokens"] or []
        ]
        working_masks = [
            msks_list.copy() for msks_list in original_step_data["masks"] or []
        ]
        max_current_tokens = max_initial_tokens
        num_alternatives = len(working_messages)

        if num_alternatives == 0:
            logger.warning(
                f"[_ensure_trajectory_token_limit] Step {step_idx} in MC env has no alternatives"
                " after copying. Skipping."
            )
            continue

        retokenization_error_this_step = False
        while max_current_tokens > max_trajectory_tokens:
            target_pop_counts_per_alt = []
            for alt_idx in range(num_alternatives):
                alt_msg_list = working_messages[alt_idx]
                num_preserved_at_end = 0
                if len(alt_msg_list) > 1 and alt_msg_list[-1]["role"] in [
                    "agent",
                    "assistant",
                ]:
                    num_preserved_at_end = 1
                    if (
                        len(alt_msg_list) > 2
                        and alt_msg_list[-2]["role"] == "environment"
                    ):
                        num_preserved_at_end = 2

                available_to_pop = len(alt_msg_list) - 1 - num_preserved_at_end

                if available_to_pop <= 0:
                    target_pop_counts_per_alt.append(0)
                else:
                    can_pop_pair = (
                        available_to_pop >= 2
                        and len(alt_msg_list) > 2
                        and alt_msg_list[1]["role"] == "environment"
                        and alt_msg_list[2]["role"] in ["agent", "assistant"]
                    )
                    if can_pop_pair:
                        target_pop_counts_per_alt.append(2)
                    else:
                        target_pop_counts_per_alt.append(1)

            positive_pop_counts = [c for c in target_pop_counts_per_alt if c > 0]
            if not positive_pop_counts:
                break

            min_pop_this_round = min(positive_pop_counts)
            temp_new_alt_tokens = []
            temp_new_alt_masks = []
            max_tokens_after_this_trunc = 0

            for alt_idx in range(num_alternatives):
                for _ in range(min_pop_this_round):
                    if len(working_messages[alt_idx]) > 1:
                        working_messages[alt_idx].pop(1)
                    else:
                        logger.error(
                            f"[_ensure_trajectory_token_limit] MC env: Critical error during pop for "
                            f"alt {alt_idx}, step {step_idx}. List too short."
                        )
                        retokenization_error_this_step = True
                        break
                if retokenization_error_this_step:
                    break

                try:
                    tokenized_alt = tokenize_for_trainer(
                        tokenizer, working_messages[alt_idx]
                    )
                    temp_new_alt_tokens.append(tokenized_alt["tokens"])
                    temp_new_alt_masks.append(tokenized_alt["masks"])
                    max_tokens_after_this_trunc = max(
                        max_tokens_after_this_trunc, len(tokenized_alt["tokens"])
                    )
                except Exception as e:
                    logger.error(
                        f"[_ensure_trajectory_token_limit] MC env: Error re-tokenizing alt {alt_idx} "
                        f"in step {step_idx} after truncation: {e}"
                    )
                    retokenization_error_this_step = True
                    break

            if retokenization_error_this_step:
                break

            working_tokens = temp_new_alt_tokens
            working_masks = temp_new_alt_masks
            max_current_tokens = max_tokens_after_this_trunc
            logger.debug(
                f"[_ensure_trajectory_token_limit] MC env: Step {step_idx}, "
                f"after uniform pop of {min_pop_this_round}, "
                f"max tokens: {max_current_tokens}"
            )

        if (
            not retokenization_error_this_step
            and max_current_tokens <= max_trajectory_tokens
        ):
            updated_step_data: ScoredDataGroup = {
                "seed": original_step_data["seed"],
                "messages": working_messages,
                "tokens": working_tokens,
                "masks": working_masks,
                "scores": original_step_data.get("scores"),
                "parsed_actions": original_step_data.get("parsed_actions"),
            }
            filtered_trajectory.append(updated_step_data)
            logger.info(
                f"[_ensure_trajectory_token_limit] MC env: Step {step_idx} successfully processed. "
                f"Final max tokens: {max_current_tokens}"
            )
        else:
            logger.warning(
                f"[_ensure_trajectory_token_limit] MC env: Discarding step {step_idx}. "
                f"Max tokens ({max_current_tokens}) still exceed limit ({max_trajectory_tokens}) "
                f"or retokenization error occurred ({retokenization_error_this_step})."
            )

    if len(filtered_trajectory) < len(trajectory):
        logger.warning(
            f"[_ensure_trajectory_token_limit] MC env: Filtered out "
            f"{len(trajectory) - len(filtered_trajectory)} steps "
            f"due to token limit constraints. Original: {len(trajectory)}, Filtered: {len(filtered_trajectory)}"
        )
    return filtered_trajectory
