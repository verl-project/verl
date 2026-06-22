# Copyright 2025 Bytedance Ltd. and/or its affiliates
import logging
import os

from transformers import PreTrainedTokenizerBase, ProcessorMixin

from verl.utils.tokenizer import normalize_token_ids

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _apply_chat_template_token_ids(
    tokenizer, messages: list[dict], *, add_generation_prompt: bool, **kwargs
) -> list[int]:
    return normalize_token_ids(
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            **kwargs,
        )
    )


def _common_suffix_len(*token_lists: list[int]) -> int:
    suffix_len = 0
    for tokens in zip(*(reversed(token_ids) for token_ids in token_lists), strict=False):
        if any(token != tokens[0] for token in tokens[1:]):
            break
        suffix_len += 1
    return suffix_len


def _common_prefix_len(first: list[int], second: list[int]) -> int:
    prefix_len = 0
    for first_token, second_token in zip(first, second, strict=False):
        if first_token != second_token:
            break
        prefix_len += 1
    return prefix_len


def _remove_suffix(token_ids: list[int], suffix_len: int) -> list[int]:
    if suffix_len == 0:
        return token_ids
    return token_ids[:-suffix_len]


def _infer_prefix_from_appended_turn_core(
    first_turn: list[int], base: list[int], extended: list[int]
) -> list[int] | None:
    if len(extended) <= len(base) or extended[: len(base)] != base:
        return None

    appended_turn = extended[len(base) :]
    if len(appended_turn) > len(first_turn) or first_turn[-len(appended_turn) :] != appended_turn:
        return None

    return first_turn[: -len(appended_turn)]


def _infer_prefix_from_appended_turn(first_turn: list[int], base: list[int], extended: list[int]) -> list[int] | None:
    suffix_lens = [0, *range(_common_suffix_len(first_turn, base, extended), 0, -1)]
    for suffix_len in suffix_lens:
        system_prompt = _infer_prefix_from_appended_turn_core(
            _remove_suffix(first_turn, suffix_len),
            _remove_suffix(base, suffix_len),
            _remove_suffix(extended, suffix_len),
        )
        if system_prompt is not None:
            return system_prompt

    return None


def _extract_generation_prompt(no_generation: list[int], with_generation: list[int]) -> list[int]:
    prefix_len = _common_prefix_len(no_generation, with_generation)
    return with_generation[prefix_len:]


def _infer_system_prompt(tokenizer, token1: list[int], **apply_chat_template_kwargs) -> list[int]:
    two_users = [{"role": "user", "content": ""}, {"role": "user", "content": ""}]
    user_assistant = [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]
    user_assistant_user = user_assistant + [{"role": "user", "content": ""}]

    # Prefer the historical consecutive-user probe when the template supports it.
    try:
        token2 = _apply_chat_template_token_ids(
            tokenizer,
            two_users,
            add_generation_prompt=False,
            **apply_chat_template_kwargs,
        )
        system_prompt = _infer_prefix_from_appended_turn(token1, token1, token2)
        if system_prompt is not None:
            return system_prompt
    except Exception:
        logger.debug("Failed to render consecutive user messages for system prompt inference.", exc_info=True)

    # Some official templates require alternating user/assistant roles.
    try:
        token2 = _apply_chat_template_token_ids(
            tokenizer,
            user_assistant,
            add_generation_prompt=False,
            **apply_chat_template_kwargs,
        )
        token3 = _apply_chat_template_token_ids(
            tokenizer,
            user_assistant_user,
            add_generation_prompt=False,
            **apply_chat_template_kwargs,
        )
        system_prompt = _infer_prefix_from_appended_turn(token1, token2, token3)
        if system_prompt is not None:
            return system_prompt
    except Exception:
        logger.debug("Failed to render alternating messages for system prompt inference.", exc_info=True)

    return []


def initialize_system_prompt(tokenizer, **apply_chat_template_kwargs) -> list[int]:
    """
    Initialize system prompt tokens for chat templates that support them.

    Args:
        tokenizer: The tokenizer with a chat template
        **apply_chat_template_kwargs: Additional arguments for apply_chat_template

    Returns:
        List of token IDs for the system prompt, or empty list if not supported
    """
    token1 = _apply_chat_template_token_ids(
        tokenizer,
        [{"role": "user", "content": ""}],
        add_generation_prompt=False,
        **apply_chat_template_kwargs,
    )
    return _infer_system_prompt(tokenizer, token1, **apply_chat_template_kwargs)


def extract_system_prompt_and_generation(tokenizer, **apply_chat_template_kwargs):
    token1 = _apply_chat_template_token_ids(
        tokenizer,
        [{"role": "user", "content": ""}],
        add_generation_prompt=False,
        **apply_chat_template_kwargs,
    )
    # get system prompt tokens
    system_prompt = _infer_system_prompt(tokenizer, token1, **apply_chat_template_kwargs)
    # get generate prompt tokens
    token3 = _apply_chat_template_token_ids(
        tokenizer,
        [{"role": "user", "content": ""}],
        add_generation_prompt=True,
        **apply_chat_template_kwargs,
    )
    generate_prompt = _extract_generation_prompt(token1, token3)

    return system_prompt, generate_prompt


def apply_chat_template(
    processor: PreTrainedTokenizerBase | ProcessorMixin,
    messages: list[dict],
    *,
    tokenize: bool = True,
    add_generation_prompt: bool = True,
    tools=None,
    return_dict: bool = False,
    **kwargs,
) -> list[int] | str:
    """apply_chat_template to messages with special attention to template requiring
    at least one user message, e.g. Qwen3.5.

    Args:
        processor: tokenizer or processor.
        messages: list[dict], messages.
        tokenize: bool, whether to tokenize the output.
        add_generation_prompt: bool, whether to add generation prompt.
        tools: list[dict], tools schema.
        return_dict: bool, whether to return a dict.
        **kwargs: additional arguments for apply_chat_template.

    Returns:
        list[int] | str: tokenized ids or text string.
    """
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            return_dict=return_dict,
            **kwargs,
        )
    except Exception:
        # Qwen3.5 apply_chat_template needs messages with at least one user message
        dummy_user_message = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
        dummy_user_prefix = processor.apply_chat_template(
            dummy_user_message,
            tokenize=tokenize,
            add_generation_prompt=False,
            tools=tools,
            return_dict=return_dict,
            **kwargs,
        )
        output = processor.apply_chat_template(
            dummy_user_message + messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            return_dict=return_dict,
            **kwargs,
        )

        if not tokenize:  # tokenize=False
            return output[len(dummy_user_prefix) :]
        elif not return_dict:  # tokenize=True and return_dict=False
            if isinstance(output[0], list):  # transformers>=5
                assert len(output) == 1, "output must be a list[int] or list[list[int]]"
                dummy_user_prefix = dummy_user_prefix[0]
                output = output[0]
            return output[len(dummy_user_prefix) :]
        else:  # tokenize=True and return_dict=True and return_tensors="pt"
            dummy_user_prefix = dict(dummy_user_prefix)
            output = dict(output)
            prefix_len = dummy_user_prefix["input_ids"].shape[1]
            output["input_ids"] = output["input_ids"][:, prefix_len:]
            output["attention_mask"] = output["attention_mask"][:, prefix_len:]
            if "mm_token_type_ids" in output:
                output["mm_token_type_ids"] = output["mm_token_type_ids"][:, prefix_len:]
            return output
