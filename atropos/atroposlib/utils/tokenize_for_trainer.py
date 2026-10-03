import numpy as np
from transformers import PreTrainedTokenizer

from atroposlib.type_definitions import Message

# Roles that should be masked in the loss calculation (not used for training)
UNMASKED_ROLES = ["assistant", "agent"]


def tokenize_for_trainer(
    tokenizer: PreTrainedTokenizer,
    chat: list[Message],
    include_messages: bool = False,
    train_on_all_assistant_turns: bool = False,
    finish_reason: str = "",
) -> dict:
    """
    Tokenize a list of chat messages for the trainer.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use.
        chat (list): A list of chat messages.
        include_messages (bool): Whether to include the messages in the output.
        train_on_all_assistant_turns (bool): If True, mask out system/user/tool roles.
                                            If False, use the original prefix masking.
    Returns:
        dict: A dictionary containing the tokenized chat messages.
    """

    tokens = tokenizer.apply_chat_template(chat)

    if not train_on_all_assistant_turns:
        prefix_len = len(
            tokenizer.apply_chat_template(chat[:-1], add_generation_prompt=True)
        )
        masks = [-100] * prefix_len + tokens[prefix_len:]
    else:
        # NOTE: This implementation will break if the default system prompt is used and depends on world state
        # (e.g. current date). e.g. consider a system prompt that depends on the current date and a run that crosses
        # midnight from 3/9 to 3/10 under a tokenizer that tokenizes 3/9 and 3/10 with a different number of tokens.

        masks = np.ones(len(tokens), dtype=np.int64) * -100

        for i, msg in enumerate(chat):
            if msg["role"] in UNMASKED_ROLES:
                prefix_tokens = tokenizer.apply_chat_template(
                    chat[:i], tokenize=True, add_generation_prompt=True
                )
                unmasked_tokens = tokenizer.apply_chat_template(
                    chat[: i + 1], tokenize=True
                )
                start_idx = len(prefix_tokens)
                end_idx = len(unmasked_tokens)
                masks[start_idx:end_idx] = np.array(unmasked_tokens[start_idx:])

        masks = masks.tolist()
    if finish_reason == "length":
        if tokens[-1] == tokenizer.eos_token_id:
            print("bad token\n")
            # truncate the last token
            tokens = tokens[:-1]
            masks = masks[:-1]

    return {
        "tokens": tokens,
        "masks": masks,
    } | ({"messages": chat} if include_messages else {})


if __name__ == "__main__":

    # Inspired by `preprocess --debug`` of https://github.com/axolotl-ai-cloud/axolotl
    def decode_token_ids(
        token_ids: list, mask, tokenizer, use_rich: bool = False
    ) -> str:
        """Convert a list of token IDs to a formatted string using tokenizer.decode,
        with an option to highlight masked tokens in red using rich markup.

        Each token is represented as decoded(tokenid, mask). If decoding a token returns an empty string
        and the token is a known special token, it is replaced with a descriptive placeholder.
        When use_rich is True, any token whose corresponding mask is -100 is wrapped with red highlighting.

        Args:
            token_ids (list[int]): A list of integer token IDs,
                e.g. [50256, 329].
            mask (list[int]): A list of masks corresponding to token_ids.
                A mask value of -100 indicates the token is masked.
            tokenizer: The Hugging Face tokenizer.
            use_rich (bool): If True, wrap tokens with a mask of -100 in red highlighting.
                Defaults to False.

        Returns:
            str: A space-separated string where each token is represented as decoded(tokenid, mask).

        Raises:
            ValueError: If any element in token_ids is not an integer.

        Example:
            >>> decode_token_ids([50256, 329], mask=[-100, 329], tokenizer=tokenizer, use_rich=True)
            '[red]<|eos|>(50256, -100)[/red] '
            'tokenX(329, 329)'  # (actual output will vary based on the model's tokenizer)
        """
        # Validate that all token_ids are integers.
        if not all(isinstance(t, int) for t in token_ids):
            raise ValueError("All token IDs must be integers.")

        tokens_str_list = []
        for tid, mid in zip(token_ids, mask):
            # Use decode with flags to include special tokens.
            decoded = tokenizer.decode(
                [tid], skip_special_tokens=False, clean_up_tokenization_spaces=False
            ).strip()
            # If the decoded string is empty and it's a special token, replace with a placeholder.
            if not decoded and tid in tokenizer.all_special_ids:
                if tid == tokenizer.eos_token_id:
                    decoded = "<|eos|>"
                else:
                    decoded = f"<SPECIAL_{tid}>"

            # Highlight token in red if use_rich is True and the token is masked (mid == -100)
            if use_rich:
                if mid == -100:
                    token_str = f"[pink3][bold]{decoded}[/bold][/pink3][steel_blue]({tid}, {mid})[/steel_blue]"
                else:
                    token_str = (
                        f"[pale_green3][bold]{decoded}[/bold][/pale_green3]"
                        f"[steel_blue]({tid}, {mid})[/steel_blue]"
                    )
            else:
                token_str = f"{decoded}({tid}, {mid})"

            tokens_str_list.append(token_str)

        return " ".join(tokens_str_list)

    messages = [
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

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    last_turn_only = tokenize_for_trainer(
        tokenizer, messages, train_on_all_assistant_turns=False
    )
    last_turn_only["repr"] = decode_token_ids(
        last_turn_only["tokens"], last_turn_only["masks"], tokenizer, use_rich=True
    )
    all_assistant_turns = tokenize_for_trainer(
        tokenizer, messages, train_on_all_assistant_turns=True
    )
    all_assistant_turns["repr"] = decode_token_ids(
        all_assistant_turns["tokens"],
        all_assistant_turns["masks"],
        tokenizer,
        use_rich=True,
    )

    from rich import print

    print("[bold cyan]last turn only[/]")
    print(last_turn_only["repr"])
    print()
    print("[bold cyan]all assistant turns[/]")
    print(all_assistant_turns["repr"])
