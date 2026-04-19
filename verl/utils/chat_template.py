# Copyright 2025 Bytedance Ltd. and/or its affiliates
import logging
import os

from transformers import PreTrainedTokenizerBase, ProcessorMixin

from verl.utils.tokenizer import normalize_token_ids

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def initialize_system_prompt(tokenizer, **apply_chat_template_kwargs) -> list[int]:
    """
    Initialize system prompt tokens for chat templates that support them.

    Args:
        tokenizer: The tokenizer with a chat template
        **apply_chat_template_kwargs: Additional arguments for apply_chat_template

    Returns:
        List of token IDs for the system prompt, or empty list if not supported
    """
    token1 = normalize_token_ids(
        tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True)
    )
    token2 = normalize_token_ids(
        tokenizer.apply_chat_template([{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True)
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    return system_prompt


def extract_system_prompt_and_generation(tokenizer):
    token1 = normalize_token_ids(
        tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True)
    )
    token2 = normalize_token_ids(
        tokenizer.apply_chat_template([{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True)
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    # get generate prompt tokens
    token3 = normalize_token_ids(
        tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True)
    )
    generate_prompt = token3[len(token1) :]

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
        # Qwen3.5 apply_chat_template needs messages with at least one user message.
        # If the message is a system message it must remain first, so we append the
        # dummy user *after* it and strip the dummy suffix via the difference trick;
        # otherwise we prepend the dummy user and strip the dummy prefix (original behaviour).
        dummy_user_message = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
        has_system = any(m.get("role") == "system" for m in messages)

        if has_system:
            # Compute the token length of one user-message span via the difference trick,
            # so we know how many tokens to strip from the end of the combined output.
            # Always use add_generation_prompt=False here so the dummy user sits at
            # the very end of the token sequence and can be cleanly stripped.
            one_user = processor.apply_chat_template(
                dummy_user_message,
                tokenize=tokenize,
                add_generation_prompt=False,
                tools=None,
                return_dict=return_dict,
                **kwargs,
            )
            two_users = processor.apply_chat_template(
                dummy_user_message * 2,
                tokenize=tokenize,
                add_generation_prompt=False,
                tools=None,
                return_dict=return_dict,
                **kwargs,
            )
            # Force add_generation_prompt=False so the dummy suffix is always at
            # the tail; we re-attach the generation prompt manually if needed.
            output = processor.apply_chat_template(
                messages + dummy_user_message,
                tokenize=tokenize,
                add_generation_prompt=False,
                tools=tools,
                return_dict=return_dict,
                **kwargs,
            )
            # Compute generation prompt tokens separately when requested.
            if add_generation_prompt:
                one_user_with_gen = processor.apply_chat_template(
                    dummy_user_message,
                    tokenize=tokenize,
                    add_generation_prompt=True,
                    tools=None,
                    return_dict=return_dict,
                    **kwargs,
                )
            else:
                one_user_with_gen = None

            if not tokenize:  # tokenize=False
                user_len = len(two_users) - len(one_user)
                result = output[:-user_len]
                if one_user_with_gen is not None:
                    result = result + one_user_with_gen[len(one_user) :]
                return result
            elif not return_dict:  # tokenize=True and return_dict=False
                if isinstance(output[0], list):  # transformers>=5
                    one_user = one_user[0]
                    two_users = two_users[0]
                    output = output[0]
                    if one_user_with_gen is not None:
                        one_user_with_gen = one_user_with_gen[0]
                user_len = len(two_users) - len(one_user)
                result = output[:-user_len]
                if one_user_with_gen is not None:
                    result = result + one_user_with_gen[len(one_user) :]
                return result
            else:  # tokenize=True and return_dict=True and return_tensors="pt"
                import torch

                one_user = dict(one_user)
                two_users = dict(two_users)
                output = dict(output)
                user_len = two_users["input_ids"].shape[1] - one_user["input_ids"].shape[1]
                gen_len = 0
                if one_user_with_gen is not None:
                    one_user_with_gen = dict(one_user_with_gen)
                    gen_len = one_user_with_gen["input_ids"].shape[1] - one_user["input_ids"].shape[1]
                for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
                    if key not in output:
                        continue
                    core = output[key][:, :-user_len]
                    if gen_len > 0:
                        suffix = one_user_with_gen[key][:, -gen_len:]
                        output[key] = torch.cat([core, suffix], dim=1)
                    else:
                        output[key] = core
                return output
        else:
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
