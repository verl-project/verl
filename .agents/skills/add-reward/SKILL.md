---
name: add-reward
description: Guide for adding a new reward function to veRL. Use when user wants to create a reward (compute_score) function.
---

# Add Reward

Add a new reward function to veRL.

## When to Use

This skill is triggered when:

- User asks "how do I add a reward function?"
- User wants to implement custom reward scoring
- User mentions `compute_score` or reward verification

## Overview

veRL separates reward functions into two layers:

1. **`compute_score` function** (`verl/utils/reward_score/<name>.py`) — pure Python,
   takes decoded strings and returns a float score
2. **`RewardManager`** (`verl/workers/reward_manager/`) — wraps compute_score, handles
   batching, decoding, and DataProto interface

For most use cases, you only need to implement a `compute_score` function and register
it. A custom `RewardManager` is only needed for advanced use cases (e.g., remote reward
models, PRIME).

## Step-by-Step Guide

### Step 1: Create the compute_score Function

Create `verl/utils/reward_score/<name>.py`:

```python
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import re
from typing import Any


def compute_score(solution_str: str, ground_truth: Any) -> float:
    """Compute reward score for a single completion.

    Args:
        solution_str: Decoded model output string (prompt + completion).
        ground_truth: Ground truth answer from the dataset.

    Returns:
        Float score, typically in [0.0, 1.0].
    """
    try:
        answer = _extract_answer(solution_str)
        if answer is not None and _is_correct(answer, str(ground_truth)):
            return 1.0
        return 0.0
    except Exception:
        return 0.0


def _extract_answer(solution_str: str) -> str | None:
    """Extract answer from model output. Customize this logic."""
    # Example: extract content from \boxed{}
    match = re.search(r"\\boxed\{([^}]+)\}", solution_str)
    if match:
        return match.group(1).strip()
    return None


def _is_correct(predicted: str, ground_truth: str) -> bool:
    """Check if the predicted answer matches ground truth."""
    return predicted.strip() == ground_truth.strip()
```

### Step 2: Register in default_compute_score

Update `verl/utils/reward_score/__init__.py` to include your new data source:

```python
from verl.utils.reward_score.<name> import compute_score as <name>_compute_score

def default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    # ... existing cases ...
    elif data_source == "<your_dataset_name>":
        return <name>_compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError(f"Unknown data_source: {data_source}")
```

### Step 3: Set data_source in Dataset Preprocessing

In your data preprocessing script, set the `data_source` field to match:

```python
# In data_preprocess/<name>.py
data_source = "<your_dataset_name>"

def make_map_fn(split):
    def process_fn(example, idx):
        return {
            "data_source": data_source,
            "prompt": [...],           # list of chat messages
            "ability": "math",         # task category
            "reward_model": {
                "style": "rule",
                "ground_truth": example["answer"],
            },
            "extra_info": {...},
        }
    return process_fn
```

### Step 4: Wire into Training Config

In your training script (e.g., `examples/grpo_trainer/run_qwen2-7b_math.sh`):

```bash
# Use NaiveRewardManager (default) with your compute_score
reward_model.reward_manager=naive
```

Or in the Python trainer config:

```python
# Trainer will call NaiveRewardManager which calls default_compute_score
# which dispatches to your function based on data_source
```

### Step 5 (Optional): Custom RewardManager

Only needed if `NaiveRewardManager` is insufficient (e.g., remote reward model,
process-level rewards like PRIME). Subclass `AbstractRewardManager`:

```python
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl import DataProto
import torch


@register("<name>")
class MyRewardManager(AbstractRewardManager):
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor:
        # Decode responses, call compute_score, return reward tensor
        ...
```

Then reference it in config: `reward_model.reward_manager=<name>`

## Reference Implementations

| Reward        | File                                     | Description                   |
| ------------- | ---------------------------------------- | ----------------------------- |
| GSM8K         | `verl/utils/reward_score/gsm8k.py`       | Math answer extraction        |
| Math general  | `verl/utils/reward_score/math_reward.py` | LaTeX boxed answer matching   |
| Geo3K         | `verl/utils/reward_score/geo3k.py`       | Geometry answer verification  |
| PRIME         | `verl/workers/reward_manager/prime.py`   | Process reward model          |
| DAPO          | `verl/workers/reward_manager/dapo.py`    | DAPO-style reward shaping     |

## Key Requirements

1. **Return float**: `compute_score` returns a single float per sample
2. **No side effects**: Function must be deterministic and stateless
3. **Handle exceptions**: Return `0.0` on error, do not raise
4. **data_source matches**: The string in dataset must match the dispatch key in `__init__.py`

## Common Mistakes

- ❌ Raising exceptions inside `compute_score` (causes worker crash)
- ❌ `data_source` mismatch between dataset and `default_compute_score`
- ❌ Returning a tensor instead of a float
- ❌ Assuming solution_str contains only the completion (it includes the prompt too)

<!--
================================================================================
                            MAINTAINER GUIDE
================================================================================
Location: .agents/skills/add-reward/SKILL.md

## How to Update
- When reward_score API changes: update Step 1 signature
- When RewardManager API changes: update Step 5
- When new reference implementations added: update table
================================================================================
-->
