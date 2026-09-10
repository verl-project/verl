# Understanding the `blackjack_env_thinking` Environment

This document explains the design philosophy behind the `blackjack_env_thinking.py` environment, particularly why its approach to trajectory collection and scoring is necessary for reinforcement learning agents that generate very long sequences, such as those incorporating extensive "thinking" or chain-of-thought reasoning.

## Installation

Before running or using this environment, ensure you have installed the necessary dependencies. Navigate to the `environments/game_environments/gymnasium/blackjack/` directory and run:

```bash
pip install -r requirements.txt
```

## TL;DR

The `blackjack_env_thinking` environment, while based on a simple game, is an example of how to structure RL environments for agents that produce exceptionally long interaction sequences. The core principles are:

1.  **Windowed (Per-Step) Decision Making**: Break down the long episode into a sequence of decisions.
2.  **Local Alternative Generation**: At each decision point (state \(s_t\)), generate multiple (`G`) possible next steps (thought processes and actions).
3.  **Value-Based Pruning/Selection**: Use a value function \(V(s)\) to estimate the long-term quality of states resulting from these alternatives. Select the best alternative to form the actual ongoing trajectory.
4.  **Counterfactual Data for Training**: Package all `G` alternatives (the one chosen and the `G-1` not chosen), along with their calculated advantages relative to \(V(s_t)\), as a training group for the policy optimizer. This data represents a rich set of comparable choices from a single state.

This design pattern allows the agent to be trained on manageable segments that fit within `seq_len`, while still enabling it to make locally optimal decisions that contribute to coherent behavior over episodes that are far longer than the training window. It highlights the importance of integrating some form of value estimation directly into the environment's trajectory generation process when dealing with such scales, especially if a global reward/value model is not being used by the trainer.

## The Problem: Ultra-Long Episode Sequences

Reinforcement learning for language models involves training on sequences of observations, thoughts, and actions. When an agent engages in detailed "thinking" steps (e.g., within `<think>...</think>` blocks), the token length of a single turn can be substantial. Over an entire episode, the total sequence length can easily exceed the maximum sequence length (e.g., 4k, 8k, or even 16k tokens) that contemporary LLMs can process in a single forward pass for training (ie, the maximum sequence length in the trainer). Even a short environment like Blackjack can blow this up in only a couple of steps if it happens to use very long thinking blocks. We could probably accommodate this for Blackjack by increasing the maximum sequence length, at the cost of more resources for the trainer, but this isn't a great solution in general. Many environments can take hundreds or thousands of steps to complete - so a way of managing this is necessary. Blackjack is just used to demonstrate how this works - it can be much more simply implemented without thinking blocks as in `blackjack_env_no_thinking`, but even WITHOUT thinking blocks there's plenty of environments that will run far beyond any reasonable maximum sequence length anyway. Not to mention agents that might make use of RAG or some kind of external planning that will further bloat the token count.

Standard RL training loops assume that an entire episode or a significant, coherent sub-segment (rollout) can be fed into the policy network (ie, the LLM being trained). When episodes are orders of magnitude longer than the `seq_len`, a naive approach of simply truncating or randomly sampling segments breaks the coherence needed for effective policy evaluation and improvement, especially for algorithms like GRPO (**Group Relative Policy Optimization**) that rely on comparing alternative actions from a common state.

## The `blackjack_env_thinking` Solution: Per-Step Windowed Trajectory Generation

The `blackjack_env_thinking.py` environment addresses this by adopting a per-step, windowed approach to trajectory generation and data collection. Instead of trying to manage and score entire, multi-thousand-token episodes as monolithic blocks, it focuses on making an informed decision at each step of the game and packaging the data around this decision point for the trainer.

Key components of this approach:

1.  **Generating Alternatives (`_sample_response`)**: At each state \(s_t\) in an episode, the environment prompts the LLM to generate not one, but `G` (defined by `config.group_size`) different potential continuations (thoughts and actions). This is handled by the `_sample_response` method.

2.  **Value-Guided Greedy Selection**: Playing out all `G` alternatives for the *entire* remaining episode (which could still be very long) is overkill for training and would again hit sequence length limits. Therefore, a greedy search approach is taken here (although it COULD be modified for more sophisticated strategies like beam search or MCTS if desired):
    *   **Value Estimation (`_estimate_value`)**: The environment uses an internal `_estimate_value(s)` function. For Blackjack, this function calculates an accurate expected future reward (value) for any given game state `s`. This acts as a local critic or value function, crucial for evaluating the long-term prospects of immediate actions. It exhaustively explores all possible moves and rewards from the current state. Which, isn't going to work in every environment - but Blackjack has pretty short episodes and a small enough action & state space it's viable to do here. Other strategies to explore and figure out future rewards are necessary for other environments.
    *   **Advantage Calculation**: For each of the `G` alternatives (\(a_i\)) sampled from state \(s_t\), the environment simulates the action, observes the immediate reward (\(R_i\)), and the next state (\(s'_{i}\)). It then calculates an advantage for each alternative, typically using the formula:
        \[ A(s_t, a_i) = R_i + \gamma V(s'_{i}) - V(s_t) \]
        (In `_collect_trajectory`, `gamma` is effectively 1, and \(R_i\) is represented by `alt_combined_rewards[i]`, \(V(s'_{i})\) by `alt_value_next[i]`, and \(V(s_t)\) by `value_t`).

    Note: This has nothing to do with GPRO's internal advantage calculations! Don't get it mixed up, this is just used to help provide some credit for intermediate actions where immediate action rewards aren't available (as well as selecting the next canonical action). Supplementing the actually winning trajectory scores (as in, the canonical trajectory) with the final outcome and a discount factor to assign credit to earlier actions would be an obvious improvement, which has been left off to keep things a little simpler (and will be explored more in another environment with longer trajectories and more sparse rewards where it might matter more to training)

    *   **Choosing the Path (`select_best_index`)**: The `select_best_index` function is then used to pick the alternative with the highest calculated advantage. This chosen alternative's action is what is actually "played" in the environment, advancing the episode to the next state `s_{t+1}`. The other `G-1` alternatives serve as counterfactual data for training. So, we end up with a "canonical" trajectory through the environment. For more comprehensive exploration of alternatives, we'd need to use some more comprehensive form of search like MCTS, which is overkill for something like Blackjack (but we'll demo in some other more complex environments to be added)

3.  **Managing Historical Context Length (`truncate_thinking`, `ensure_trajectory_token_limit`)**: As an episode progresses, the history of observations, thoughts, and actions accumulates. To ensure that the prompt fed to the LLM for generating the *next* `G` alternatives remains within the operational context window (e.g., `max_prompt_tokens`), the environment employs truncation strategies:
    *   Currently, we use simple truncation (e.g., removing the oldest messages or earlier parts of messages that probably don't have the LLMs final thoughts about it's decisions). More sophisticated context management techniques, such as summarization of earlier parts of the episode, could be implemented in the future to retain relevant historical information more effectively within the token constraints.
    *   The `truncate_thinking` utility (from `atroposlib/utils/message_history_utils.py`) is used to shorten the content within `<think>...</think>` blocks in the message history if they exceed a certain token budget. This helps manage the verbosity of past thinking steps.
    *   We use `ensure_trajectory_token_limit` in `blackjack_env_thinking.py` for truncating the overall message history (list of observations, thoughts, and chosen actions from previous turns) before it's used to prompt the LLM for the current step's alternatives. This ensures that the input prompt, including the system message and the current game state, doesn't exceed the LLM's maximum input token limit.

    This step is crucial because, without it, the growing history would quickly make it impossible to generate new actions as the episode continues. It allows the trajectory "windows" in each ScoredDataGroup to be trained on longer, and keep the multiturn nature of the training intact.

## Data Structure for the GRPO Trainer

How data is packaged for the GRPO trainer:

*   At each step `t` of the actual trajectory taken by the agent, the `_collect_trajectory` method compiles a `BlackjackScoredDataGroup`.
*   This single `BlackjackScoredDataGroup` contains the full text (including thoughts), tokenized representations (`tokens`), attention `masks`, and `scores` (which are the \(A(s_t, a_i)\) values) for **all `G` alternatives** that were considered at state \(s_t\).
*   The list `trajectory_data_for_trainer` accumulates these `BlackjackScoredDataGroup` objects, one for each step of the episode.

This means that each element sent to the trainer represents a bundle of `G` closely related sequences. Each sequence in the bundle starts from the *same* parent state \(s_t\) (in terms of message history fed to the LLM for generating the alternatives) and represents one possible thought process and action. Their "scores" (advantages) are all calculated relative to \(V(s_t)\), making them directly comparable.

## Why This is Crucial for GRPO with Long Sequences

**Group Relative Policy Optimization (GRPO)** is a reinforcement learning algorithm that enhances model training by generating multiple responses for each prompt (or state, in our case) and then scoring them. A key aspect of GRPO is how it calculates advantages: instead of relying on a separate value function network (like PPO often does for its baseline), it uses the mean reward of the generated group of responses as a baseline. The advantage for each response \(a_{jk}\) from a state \(s_j\) is then \(A_{jk} = R_{jk} - \bar{R}_j\), where \(R_{jk}\) is the reward of the specific response and \(\bar{R}_j\) is the average reward for all \(K_j\) responses generated from state \(s_j\). This approach is memory-efficient as it avoids the need for a separate value network.

The GRPO trainer typically computes a loss using these advantages. For example, a policy gradient style loss might be structured as:
\[ L = -\sum_{j=1}^{M} \sum_{k=1}^{K_j} \left( \frac{\pi_{\theta}(a_{jk} | s_j)}{\pi_{\theta_{\text{old}}}(a_{jk} | s_j)} A_{jk}^{\text{GRPO}} \right) \]
(often with a KL divergence penalty for stability, ensuring the new policy \(\pi_{\theta}\) doesn\'t deviate too drastically from the old policy \(\pi_{\theta_{\text{old}}}\\)). The `ratio = torch.exp(logp - logp.detach())` and `loss = -reward * ratio` (where `reward` is the \(A_{jk}^{\text{GRPO}}\) advantage) in a typical trainer snippet would align with this principle.

The `blackjack_env_thinking` environment's design is compatible with GRPO's core requirements for input data BUT allowing it to be used across long trajectories. We don't get a nice, well defined reward at every step of every environment - but we want to keep that nice, objective, outcome-oriented RLVR-style reward structure, even in reward-sparse environments.

1.  **Alternative Generation**: From a state \(s_t\), the environment generates `G` alternative continuations (thoughts and actions \(a_1, ..., a_G\)).
2.  **Value-Informed Scoring (within the environment)**: For each alternative \(a_i\), the environment itself calculates a "score" using its internal value estimation: \(S_i = R_i + \gamma V_{\text{env}}(s'_{i}) - V_{\text{env}}(s_t)\). This score, \(S_i\), represents a local, value-informed assessment of that alternative's quality.
    *   This score is used by the environment to select the best alternative to actually execute, guiding the agent along a competent trajectory (the "lookahead" mechanism).
3.  **Data for GRPO Trainer**: The `BlackjackScoredDataGroup` bundles these `G` alternatives. Crucially, the calculated scores \(S_i\) for each alternative are passed to the GRPO trainer as the *input rewards* (let's call them \(R_{tk}\) for alternative \(k\) from state \(s_t\), where \(R_{tk} = S_k\)).
4.  **GRPO's Advantage Calculation**: The GRPO trainer then takes this group of `G` items and their associated input rewards (our \(S_k\) values) and performs its standard advantage calculation:
    *   It computes the mean of these input rewards for the group: \(\bar{R}_t = \frac{1}{G} \sum_{k=1}^{G} S_k\).
    *   It then calculates the GRPO advantage for each alternative: \(A_{tk}^{\text{GRPO}} = S_k - \bar{R}_t\).
5.  **Loss Calculation**: These \(A_{tk}^{\text{GRPO}}\) advantages are what the GRPO trainer uses in its loss function to update the policy (ie, it's regular GPRO stuff).

This two-step process is vital:
*   The environment provides high-quality, comparable, and value-informed *reward signals* (our scores \(S_i\)) for a set of diverse actions originating from the exact same state. The subtraction of \(V_{\text{env}}(s_t)\) in our score calculation helps to center these reward signals, potentially aiding training stability.
*   The GRPO algorithm then applies its group-relative normalization to these rewards to derive the advantages that drive policy learning. This ensures that the policy is updated based on how much better or worse an alternative is compared to the *average quality of alternatives considered at that specific step*.

This structure ensures that the GRPO trainer receives a rich dataset where each group of `G` items represents directly comparable choices from a common decision point (state \(s_t\)), each with a meaningful reward signal attached. The environment's scoring and selection mechanism ensures that the data generated is not only diverse but also reflects locally optimal decision-making. This is particularly handy for long sequences because it allows the model to learn from nuanced differences in multi-step "thinking" paths that all originate from the same immediate context. Greedy selection is faster for the training loop, and probably sufficient for a simple environment like Blackjack, but as mentioned previously you could (somewhat) easily upgrade to beam search or MCTS instead to explore more complex environments thoroughly.

**Why a naive approach of chunking independent long rollouts fails for GRPO:**

Imagine having `G` complete, very long, independent rollouts. If we simply chopped these into, say, 16k token chunks and tried to form a "group" for the trainer using `chunk_k` from each of these `G` rollouts, the problem becomes evident:
*   `rollout_1_chunk_k` starts at some state \(S_{1,k}\).
*   `rollout_2_chunk_k` starts at a completely different state \(S_{2,k}\).
The "advantages" or scores calculated for these chunks would not be comparable in the way GRPO intends, as they don't represent alternative choices from a common decision point. No Bueno! The `blackjack_env_thinking` design ensures this common-state origin for each group of `G` alternatives it sends to the trainer.

## Value Estimation via Local Exhaustive Exploration

The `blackjack_env_thinking` strategy of generating `G` alternatives, simulating each for one step, and then using `_estimate_value` to predict future outcomes allows for a *local exhaustive search* around the current state. This is feasible here because:
1.  Blackjack step simulation is computationally cheap.
2.  `G` is a manageable number (e.g., 16 to 32).
3.  An accurate value function \(V(s)\) can be reasonably implemented for a deterministic, small-state-space game like Blackjack (the `_get_v_star_recursive` in `_estimate_value` gives an exact calculation).

If an accurate value function were not available (e.g., in a more complex environment or if it needed to be learned by a separate model), the quality of the greedy rollouts and the calculated advantages would depend on the accuracy of this learned value estimate or something like VinePPO's Monte Carlo rollouts to get a similar estimate of future rewards.

## Contrast with Simpler, Shorter Episodes (`blackjack_env_no_thinking`)

In the `blackjack_env_no_thinking` environment:
*   Episodes are very short (no long thinking blocks!!)
*   The entire sequence of (observation, action, LLM response) usually fits within the model's `seq_len`. Blackjack is at most a few turns, so this is ok if you JUST want to train on actions, not additional long chains of thought.
*   `collect_trajectory` returns a single `ScoredDataItem` representing the full episode. The "score" is simply the final game outcome (e.g., +1 for a win) and some bonuses for formatting and correct tool calling.
*   The trainer can then process these entire episodes using the normal GRPO method (ie, we're just sending the full alternative trajectories and their scores to be compared, similar to the single-step bandit problems people are commonly using for RLVR). The complexity of per-step alternative generation for windowing and local value estimation isn't needed for fitting within `seq_len`.
