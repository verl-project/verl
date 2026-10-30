import math
import random


# TODO: move this to the server manager
async def generate_with_diverse_first_tokens(
    self, messages, prefill="", n=8, max_tokens=4096, temperature=1.0
):
    """
    Generate diverse completions by sampling different first tokens.

    Parameters:
    - messages: List of message dictionaries for chat completion
    - prefill: Prefix text to add to assistant's message
    - n: Number of diverse completions to generate
    - max_tokens: Maximum tokens per completion
    - temperature: Sampling temperature

    Returns:
    - List of completion strings
    """
    # Step 1: First get the logprobs for just the first token
    first_token_messages = messages + [{"role": "assistant", "content": prefill}]

    first_token_completion = await self.server.chat_completion(
        messages=first_token_messages,
        n=1,
        max_tokens=1,
        temperature=0.0,  # Use 0 temperature to get raw logprobs
        logprobs=True,
        top_logprobs=20,  # Get top 20 logprobs for the first token
    )

    # Extract logprobs from the completion
    try:
        # Get the logprobs for the first token
        logprobs_dict = (
            first_token_completion.choices[0].logprobs.content[0].top_logprobs
        )

        # Convert to list of (token, logprob) tuples
        logprobs_list = [(item.token, item.logprob) for item in logprobs_dict]

        # Convert logprobs to probabilities with temperature
        logprobs_array = [lp for _, lp in logprobs_list]
        probs = [math.exp(lp / temperature) for lp in logprobs_array]
        total = sum(probs)
        probs = [p / total for p in probs]

        # Sample n unique tokens
        sampled_indices = random.choices(
            range(len(logprobs_list)), weights=probs, k=min(n, len(logprobs_list))
        )

        # Ensure unique indices
        sampled_indices = list(set(sampled_indices))

        # If we have fewer than n tokens, sample again to fill
        while len(sampled_indices) < n and len(sampled_indices) < len(logprobs_list):
            remaining = min(
                n - len(sampled_indices), len(logprobs_list) - len(sampled_indices)
            )
            available_indices = [
                i for i in range(len(logprobs_list)) if i not in sampled_indices
            ]
            available_probs = [probs[i] for i in available_indices]
            total = sum(available_probs)
            if total > 0:
                available_probs = [p / total for p in available_probs]
                additional_indices = random.choices(
                    available_indices, weights=available_probs, k=remaining
                )
                sampled_indices.extend(additional_indices)
            else:
                # If all remaining probs are 0, just pick randomly
                additional_indices = random.sample(available_indices, k=remaining)
                sampled_indices.extend(additional_indices)

        # Get the selected first tokens
        first_tokens = [logprobs_list[i][0] for i in sampled_indices]

    except (AttributeError, IndexError, KeyError) as e:
        # Fallback if we can't extract logprobs properly
        print(f"Error extracting logprobs: {e}")
        return await self.fallback_generate(
            messages, prefill, n, max_tokens, temperature
        )

    # Step 2: Generate completions with each selected first token
    completions = []
    for token in first_tokens:
        # Create a prompt with the first token already included
        prompt_with_token = messages + [
            {"role": "assistant", "content": prefill + token}
        ]

        # Generate the rest of the completion
        completion = await self.server.chat_completion(
            messages=prompt_with_token,
            n=1,
            max_tokens=max_tokens - 1,  # Subtract 1 for the token we already used
            temperature=temperature,
            top_p=0.3,
            extra_body={
                "min_p": 0.5,
                "repetition_penalty": 1.05,
            },
        )

        # Extract the completion content and remove the prefill+token
        full_content = completion.choices[0].message.content
        completions.append(token + full_content)
