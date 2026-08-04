import json
import logging
import os

from transformers import AutoTokenizer

from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

MESSAGES = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant that provides accurate information.",
    },
    {"role": "user", "content": "What's the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "Can you tell me more about Paris?"},
    {
        "role": "assistant",
        "content": "<tool_call>{'tool_name': 'web_search', 'args': {'query': 'Paris'}}</tool_call>",
    },
    {
        "role": "tool",
        "content": (
            "Paris is the capital and most populous city of France. "
            "It has an estimated population of 2,165,423 residents in 2019 "
            "in an area of more than 105 km²."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Paris is indeed the capital of France and its most populous city with over 2 million residents. "
            "It's known for its iconic landmarks like the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. "
            "The city is a global center for art, fashion, gastronomy, and culture."
        ),
    },
]

TEST_MASKS_PATH = os.path.join(os.path.dirname(__file__), "test_masks.json")
with open(TEST_MASKS_PATH) as f:
    TEST_MASKS = json.load(f)


def test_tokenize_for_trainer_mask_len_last_turn_only():
    # random model with chat templates and isn't gated
    try:
        tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        can_run_stop = True
    except (ValueError, EnvironmentError):
        can_run_stop = False
        tok = AutoTokenizer.from_pretrained("Zyphra/Zamba2-1.2B-instruct")
        logging.warning(
            "Could not use gated model, using non-gated model that is bad at tokenizing..."
        )
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    total_toks = tok.apply_chat_template(messages)
    prefix = tok.apply_chat_template(messages[:1], add_generation_prompt=True)
    resp = tokenize_for_trainer(tok, messages, False)
    assert len(resp["tokens"]) == len(total_toks) == len(resp["masks"])
    assert resp["tokens"] == total_toks
    assert all([x == -100 for x in resp["masks"][: len(prefix)]])
    assert all([x != -100 for x in resp["masks"][len(prefix) :]])
    assert resp.get("messages", None is None)
    # This time with add messages
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    resp = tokenize_for_trainer(tok, messages, True)
    assert resp["tokens"] == total_toks
    assert len(resp["tokens"]) == len(total_toks) == len(resp["masks"])
    assert all([x == -100 for x in resp["masks"][: len(prefix)]])
    assert all([x != -100 for x in resp["masks"][len(prefix) :]])
    assert resp["messages"] == messages
    if can_run_stop:
        # now try with finish reason == stop
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        resp = tokenize_for_trainer(tok, messages, False, finish_reason="length")
        assert len(resp["tokens"]) == len(total_toks) - 1 == len(resp["masks"])
        assert resp["tokens"] == total_toks[:-1]
        assert all([x == -100 for x in resp["masks"][: len(prefix)]])
        assert all([x != -100 for x in resp["masks"][len(prefix) :]])
        assert resp.get("messages", None is None)


def test_last_turn_only_masking():
    """
    Test that in last turn only mode, only the tokens from the final assistant turn are unmasked
    (mask != -100) while all tokens from all other messages are masked (mask == -100).
    """
    tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    result = tokenize_for_trainer(
        tok, MESSAGES, include_messages=False, train_on_all_assistant_turns=False
    )

    masks = result["masks"]
    assert masks == TEST_MASKS["last_turn_only"]


def test_all_assistant_turns_masking():
    """
    Test that in all assistant turns mode, tokens for every assistant message are unmasked,
    while tokens for non-assistant messages remain masked.
    """
    tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    result = tokenize_for_trainer(
        tok, MESSAGES, include_messages=False, train_on_all_assistant_turns=True
    )

    masks = result["masks"]
    assert masks == TEST_MASKS["all_assistant_turns"]
