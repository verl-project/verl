# Six-Seat No-Limit Hold'em Poker Environment

Atropos environment for improving an LLM's ability to make optimal decisions in No-Limit Hold'em Poker situations through reinforcement learning on expert hand history data.

## Overview

This environment trains language models to make poker decisions like expert players. It takes a hand situation as input and rewards the model for matching the actions that winning players took in those situations.

## Features

- **Complete Poker Training Environment**: Full implementation of the BaseEnv interface for Atropos
- **Hugging Face Dataset Integration**: Uses `yoniebans/6max-nlh-poker` dataset with train/test splits
- **Specialized Reward Functions**: Custom reward components for action matching and bet sizing accuracy
- **Comprehensive Evaluation**: Tracking by game stage (preflop, flop, turn, river)
- **Configurable Parameters**: Easy customization via YAML configuration

## Files

- **poker_env.py**: Main environment implementation
- **reward_fns/**: Custom reward functions
  - action_match.py: Evaluates correctness of action type
  - bet_sizing.py: Evaluates precision of bet amount
  - combined_poker_reward.py: Combines both reward components
- **DATASET.md**: Detailed documentation of the dataset format


## Dataset

The environment uses a specialized dataset containing poker hand situations and the corresponding actions taken by winning players. Each record includes:

- Game state information (player positions, cards, current bets)
- Previous actions in the hand
- The winning player's action (used as the learning target)

See `source_dataset.md` for detailed dataset documentation.

## Reward System

The reward function uses a dual evaluation approach:

1. **Action Matching (60%)**: Evaluates if the model chose the correct action type
   - Exact match: 1.0 score
   - Action type match: 0.7 score
   - Strategic intent match: 0.5 score

2. **Bet Sizing (40%)**: Evaluates precision in bet amount selection
   - Perfect amount: 1.0 score
   - Linear decay as deviation increases
   - Zero score beyond 50% deviation

This balanced approach ensures the model learns both strategic correctness and numerical precision.

# NOUS HACKATON
- [https://www.loom.com/share/7dda14bfc31b458eaa472a8d34e352c4] a link to a 1 minute youtube video
- an explanation of your env design and motivation

## Quick start

```bash
# run vllm
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-1.7B \
    --gpu-memory-utilization 0.95 \
    --dtype auto \
    --port 9002
```

```bash
# run the enviroment
python poker_env.py process \
    --env.data_path_to_save_groups poker_rollouts.jsonl \
    --openai.base_url https://localhost:9002/v1 \
    --openai.api_key EMPTY \
    --openai.model_name Qwen/Qwen3-1.7B
```
## WanDB runs
[TBD]
