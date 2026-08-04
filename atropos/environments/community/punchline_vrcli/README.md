# Punchline VR-CLI Environment

This environment trains an LLM to generate humorous punchlines for jokes, utilizing a Reinforcement Learning (RL) technique called Verifiable Rewards via Completion Likelihood Improvement (VR-CLI), from the paper "Learning to Reason for Long-Form Story Generation" (Gurung & Lapata, 2025) [https://arxiv.org/html/2503.22828v1](https://arxiv.org/html/2503.22828v1).

## Design

The core idea is to teach a model not just to produce a punchline, but to first generate "reasoning" that leads to a good punchline. The quality of this reasoning is then "verified" by measuring how much it improves the likelihood (reduces the perplexity) of the *actual* punchline from the dataset, as assessed by a separate, fixed reference model. This greatly reduces overfitting, as the model does not have access to the ground-truth answer. Typical fine-tuning fails at this, as it makes the model memorize the jokes rather than gain an understanding of why they are funny.

## Example

Here's an example of one trajectory:

```
Question: What do you call a herd of cows masturbating?

<think>
Okay, the user is asking, "What do you call a herd of cows masturbating?" Hmm, that's a play on words. Let me think.
The key here is the word "masturbating" and the animal "cows." The answer needs to be a pun or a play on words.

First, I need to connect "herd" with "masturbating." A herd of cows is a group, so maybe something related to a group.
"Masturbating" is a term that's not typically used for animals, but maybe there's a word that combines the two.........
</think>

Beef strokin off!
```

## Impact

How is this applicable to anything useful? Well, VR-CLI can be applied to many domains beyond jokes. The original authors of the paper used it for creative writing, but it could also be applied to code generation without requiring it to be executed in a VM or custom business tasks with existing examples.

## Key Components:

*   **Dataset:** The environment uses the `"SocialGrep/one-million-reddit-jokes"` dataset, filtering for jokes with a question-answer format (setup and punchline) and a minimum number of upvotes.
*   **Task:** Given the setup of a joke (the "question"), the model `Qwen/Qwen3-1.7B` is prompted to generate a thinking process `<think>...</think>` followed by the punchline.
*   **Reward (VR-CLI):**
    1.  A base perplexity of the "golden" punchline is calculated given only the joke's setup, using a reference LLM `Qwen/Qwen3-1.7B-Base`.
    2.  A new perplexity of the golden punchline is calculated, this time conditioned on both the joke's setup AND the model-generated reasoning.
    3.  The reward is proportional to the improvement in perplexity `(base_perplexity - plus_perplexity) / base_perplexity`. A positive reward indicates the reasoning was helpful.

## Metrics

The `evaluate` function selects 64 random jokes, then computes the average reward using greedy decoding. This is a simple and effective way to measure the model's progress over time.

## Quickstart

### Prerequisites

Ensure you have Python installed, then install the dependencies:

```bash
pip install "vllm>=0.8.5" torch transformers datasets wandb tenacity atroposlib pydantic
```

### Processing the Dataset

To process the dataset, use the `punchline_env.py` script with the `process` command:

```bash
python punchline_env.py process \
  --env.data_path_to_save_groups jokes.jsonl \
  --env.wandb_name punchline_vrcli \
  --openai.base_url http://localhost:9001/v1 \
  --openai.api_key x \
  --openai.model_name Qwen/Qwen3-1.7B
```

You will need to have vLLM serving the model on port 9001 for this to work.

[Weights & Biases link](https://wandb.ai/jaboggs-nous-hackathon-nc-state-university/uncategorized/runs/0vly0u4p)

#### Output

[jokes.zip](https://github.com/user-attachments/files/20275345/jokes.zip)
