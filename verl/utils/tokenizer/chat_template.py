# Copyright 2025 Bytedance Ltd. and/or its affiliates
import logging
import os

from transformers import PreTrainedTokenizerBase, ProcessorMixin

from .tokenizer import normalize_token_ids

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
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True, **apply_chat_template_kwargs
        )
    )
    token2 = normalize_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}] * 2,
            add_generation_prompt=False,
            tokenize=True,
            **apply_chat_template_kwargs,
        )
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    return system_prompt


def initialize_turn_separator(tokenizer, **apply_chat_template_kwargs) -> list[int]:
    """Tokens a chat template inserts after a message's closing token, before the next turn.

    Multi-turn agent rollouts build the token sequence incrementally. The model stops at the
    assistant close token (e.g. ``<|im_end|>``) and never emits the template's trailing
    turn-separator (e.g. ``"\\n"``, id 198 for Qwen). Rendering the following tool/user turn in
    isolation also omits that separator, so every turn boundary silently drops it and the rollout
    token sequence diverges from ``apply_chat_template`` of the equivalent full conversation.
    This returns the separator so callers can restore it at turn boundaries.

    Derivation: rendering the same (user) turn with empty vs non-empty content only differs in the
    content region, so the maximal common trailing run is exactly ``[close_token, *separator]``.
    A user turn is used deliberately -- probing with an assistant turn would inject reasoning
    scaffolding (e.g. Qwen3's ``<think></think>``) that is not part of the separator. The model
    emits the close token itself (it is the stop token), so the separator is everything after it.

    Returns an empty list when the template has no turn separator or an unexpected structure, so
    callers keep their previous behavior instead of crashing.
    """
    # Render two user turns that differ only in body text; the shared trailing run is the separator.
    # A bare string ``content`` is rejected by some multimodal processors (they iterate ``content``
    # expecting a list of typed parts), so fall back to the list-of-parts form, and return ``[]`` if
    # neither renders. Both probes must use the same form so only the body differs.
    empty = filled = None
    for as_parts in (False, True):
        if as_parts:
            body_empty, body_filled = [{"type": "text", "text": ""}], [{"type": "text", "text": "x"}]
        else:
            body_empty, body_filled = "", "x"
        try:
            empty = normalize_token_ids(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": body_empty}],
                    add_generation_prompt=False,
                    tokenize=True,
                    **apply_chat_template_kwargs,
                )
            )
            filled = normalize_token_ids(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": body_filled}],
                    add_generation_prompt=False,
                    tokenize=True,
                    **apply_chat_template_kwargs,
                )
            )
            break
        except Exception:
            empty = filled = None
    if empty is None or filled is None:
        return []
    # Maximal common trailing run == the message closing token(s) + inter-turn separator (identical
    # regardless of content).
    i = 0
    while i < len(empty) and i < len(filled) and empty[-1 - i] == filled[-1 - i]:
        i += 1
    suffix = empty[len(empty) - i :]
    if not suffix:
        return []
    # Split off the closing token the model already emits; the remainder is the dropped separator.
    # A processor (VLM path) exposes ``eos_token_id`` on its wrapped tokenizer rather than itself,
    # and some tokenizers (e.g. Llama 3) expose it as a list/tuple of ids rather than a single int.
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        eos_id = getattr(getattr(tokenizer, "tokenizer", None), "eos_token_id", None)
    eos_ids = {eos_id} if isinstance(eos_id, int) else set(eos_id or [])
    last_close = max((i for i, tok_id in enumerate(suffix) if tok_id in eos_ids), default=None)
    if last_close is not None:
        return suffix[last_close + 1 :]
    return suffix[1:]


def extract_system_prompt_and_generation(tokenizer, **apply_chat_template_kwargs):
    token1 = normalize_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True, **apply_chat_template_kwargs
        )
    )
    token2 = normalize_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}] * 2,
            add_generation_prompt=False,
            tokenize=True,
            **apply_chat_template_kwargs,
        )
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    # get generate prompt tokens
    token3 = normalize_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True, **apply_chat_template_kwargs
        )
    )
    generate_prompt = token3[len(token1) :]

    return system_prompt, generate_prompt


def _normalize_system_messages(messages: list[dict]) -> list[dict]:
    """Move system content to the first turn when a template requires it."""
    system_messages = [message for message in messages if message.get("role") == "system"]
    if not system_messages or (len(system_messages) == 1 and messages[0].get("role") == "system"):
        return messages

    first_system = dict(system_messages[0])
    contents = [message.get("content", "") for message in system_messages]
    if all(isinstance(content, str) for content in contents):
        first_system["content"] = "\n\n".join(content for content in contents if content)
    else:
        merged_content: list[dict] = []
        for index, content in enumerate(contents):
            if index and merged_content:
                merged_content.append({"type": "text", "text": "\n\n"})
            if isinstance(content, list):
                merged_content.extend(content)
            elif content is not None:
                merged_content.append({"type": "text", "text": str(content)})
        first_system["content"] = merged_content

    return [first_system] + [message for message in messages if message.get("role") != "system"]


def _sequence_length(value) -> int:
    """Return the token-axis length for a tokenizer output value."""
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) == 1:
            return int(shape[0])
        return int(shape[-1])
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            return len(value[0])
        return len(value)
    raise TypeError(f"Unsupported token output type: {type(value)!r}")


def _remove_sequence_span(value, start: int, length: int):
    """Remove a token span while retaining the output array's backend and shape."""
    stop = start + length
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            return [row[:start] + row[stop:] for row in value]
        return value[:start] + value[stop:]

    module = type(value).__module__.split(".", 1)[0]
    head = value[..., :start]
    tail = value[..., stop:]
    if module == "torch":
        import torch

        return torch.cat((head, tail), dim=-1)
    if module == "numpy":
        import numpy as np

        return np.concatenate((head, tail), axis=-1)
    if module == "tensorflow":
        import tensorflow as tf

        return tf.concat((head, tail), axis=-1)
    if module in ("jax", "jaxlib"):
        import jax.numpy as jnp

        return jnp.concatenate((head, tail), axis=-1)
    raise TypeError(f"Unsupported token output backend: {type(value)!r}")


def _unwrap_token_sequence(value):
    """Unwrap the single batch returned by transformers 5 tokenization."""
    if isinstance(value, list) and value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("apply_chat_template must return one tokenized sequence")
        return value[0]
    return value


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
        # A leading system block must stay first, so the dummy user cannot be prepended
        # in that case. It is not appended either: templates that keep the reasoning
        # content of the *last* assistant message only (Qwen3, DeepSeek-R1, ...) would
        # drop it once another message follows. Instead the dummy user is inserted right
        # after the leading system block and its span is cut back out of the middle of
        # the output, which keeps both the system message first and the last assistant
        # message last.
        normalized_messages = _normalize_system_messages(messages)
        if normalized_messages != messages:
            try:
                return processor.apply_chat_template(
                    normalized_messages,
                    tokenize=tokenize,
                    add_generation_prompt=add_generation_prompt,
                    tools=tools,
                    return_dict=return_dict,
                    **kwargs,
                )
            except Exception:
                messages = normalized_messages
        dummy_user_message = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
        num_leading_system = 0
        while num_leading_system < len(messages) and messages[num_leading_system].get("role") == "system":
            num_leading_system += 1

        if num_leading_system:
            head = list(messages[:num_leading_system])
            tail = list(messages[num_leading_system:])
            # The difference trick gives the length of one dummy-user span; subtracting it
            # from the head+dummy rendering gives the length of the head block itself.
            one_user = processor.apply_chat_template(
                head + dummy_user_message,
                tokenize=tokenize,
                add_generation_prompt=False,
                tools=tools,
                return_dict=return_dict,
                **kwargs,
            )
            two_users = processor.apply_chat_template(
                head + dummy_user_message * 2,
                tokenize=tokenize,
                add_generation_prompt=False,
                tools=tools,
                return_dict=return_dict,
                **kwargs,
            )
            output = processor.apply_chat_template(
                head + dummy_user_message + tail,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                tools=tools,
                return_dict=return_dict,
                **kwargs,
            )

            if not tokenize:  # tokenize=False
                user_len = len(two_users) - len(one_user)
                head_len = len(one_user) - user_len
                return output[:head_len] + output[head_len + user_len :]
            elif not return_dict:  # tokenize=True and return_dict=False
                one_user = _unwrap_token_sequence(one_user)
                two_users = _unwrap_token_sequence(two_users)
                output = _unwrap_token_sequence(output)
                user_len = len(two_users) - len(one_user)
                head_len = len(one_user) - user_len
                return output[:head_len] + output[head_len + user_len :]
            else:  # tokenize=True and return_dict=True
                one_user = dict(one_user)
                two_users = dict(two_users)
                output = dict(output)
                user_len = _sequence_length(two_users["input_ids"]) - _sequence_length(one_user["input_ids"])
                head_len = _sequence_length(one_user["input_ids"]) - user_len
                for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
                    if key not in output:
                        continue
                    output[key] = _remove_sequence_span(output[key], head_len, user_len)
                return output

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
            dummy_user_prefix = _unwrap_token_sequence(dummy_user_prefix)
            output = _unwrap_token_sequence(output)
            return output[len(dummy_user_prefix) :]
        else:  # tokenize=True and return_dict=True
            dummy_user_prefix = dict(dummy_user_prefix)
            output = dict(output)
            prefix_len = _sequence_length(dummy_user_prefix["input_ids"])
            for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
                if key not in output:
                    continue
                output[key] = _remove_sequence_span(output[key], 0, prefix_len)
            return output
