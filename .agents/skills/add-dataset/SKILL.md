---
name: add-dataset
description: Guide for adding a new dataset to veRL. Use when user wants to preprocess or integrate a new RL training dataset.
---

# Add Dataset

Add a new dataset for RL training in veRL.

## When to Use

This skill is triggered when:

- User asks "how do I add a dataset?"
- User wants to preprocess a new dataset for GRPO/PPO training
- User mentions creating a data preprocessing script

## Overview

veRL datasets follow a two-step pattern:

1. **Preprocessing script** (`examples/data_preprocess/<name>.py`) — run once offline to
   convert raw data into parquet files with a fixed schema
2. **`RLHFDataset`** (`verl/utils/dataset/rl_dataset.py`) — runtime dataset class that
   reads the parquet files; you usually do NOT need to modify this

## Required Schema

Every preprocessed sample must contain these fields:

```python
{
    "data_source": str,          # Identifies the reward function to use, e.g. "gsm8k"
    "prompt": list[dict],        # Chat messages, e.g. [{"role": "user", "content": "..."}]
    "ability": str,              # Task category, e.g. "math", "coding", "qa"
    "reward_model": {
        "style": "rule",         # "rule" for compute_score, "model" for reward model
        "ground_truth": str,     # Ground truth answer (used by compute_score)
    },
    "extra_info": {
        "split": str,            # "train" or "test"
        "index": int,            # Sample index for reproducibility
    },
}
```

## Step-by-Step Guide

### Step 1: Create Preprocessing Script

Create `examples/data_preprocess/<name>.py`:

```python
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License")

import argparse
import os

import datasets
from verl.utils.hdfs_io import copy, makedirs

data_source = "<name>"     # Must match key in default_compute_score

SYSTEM_PROMPT = "You are a helpful assistant."  # Optional


def make_map_fn(split: str):
    def process_fn(example: dict, idx: int) -> dict:
        # Build the prompt as a chat template
        question = example["question"]  # adapt to your dataset fields
        answer = str(example["answer"])

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        return {
            "data_source": data_source,
            "prompt": messages,
            "ability": "math",          # change as appropriate
            "reward_model": {
                "style": "rule",
                "ground_truth": answer,
            },
            "extra_info": {
                "split": split,
                "index": idx,
            },
        }

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/<name>")
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # Load from HuggingFace hub or local path
    dataset = datasets.load_dataset("<hf_dataset_id>")

    for split in ["train", "test"]:
        if split not in dataset:
            continue
        processed = dataset[split].map(
            function=make_map_fn(split),
            with_indices=True,
            remove_columns=dataset[split].column_names,
        )
        output_path = os.path.join(local_dir, f"{split}.parquet")
        processed.to_parquet(output_path)
        print(f"Saved {len(processed)} samples to {output_path}")

    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
```

### Step 2: Run Preprocessing

```bash
python examples/data_preprocess/<name>.py --local_dir ~/data/<name>
```

### Step 3: Update Training Config

Point the trainer to your new dataset:

```bash
# In your run script or YAML config:
data.train_files=~/data/<name>/train.parquet
data.val_files=~/data/<name>/test.parquet
data.train_data_sources=<name>    # optional, for filtering
```

### Step 4: Register Reward Function

Make sure `verl/utils/reward_score/__init__.py` handles your `data_source`:

```python
elif data_source == "<name>":
    from verl.utils.reward_score.<name> import compute_score
    return compute_score(solution_str, ground_truth)
```

See the `/add-reward` skill for details.

## Reference Implementations

| Dataset      | File                                          | Notes                          |
| ------------ | --------------------------------------------- | ------------------------------ |
| GSM8K        | `examples/data_preprocess/gsm8k.py`           | Math QA baseline               |
| MATH         | `examples/data_preprocess/math_dataset.py`    | Competition math               |
| Geo3K        | `examples/data_preprocess/geo3k.py`           | Geometry                       |
| Multi-turn   | `verl/utils/dataset/multiturn_sft_dataset.py` | SFT with multi-turn dialogues  |

## Key Requirements

1. **Fixed schema**: All required fields must be present (see above)
2. **Parquet format**: veRL's `RLDataset` expects `.parquet` files
3. **data_source matches**: Must align with your reward function's dispatch key
4. **Prompt as chat messages**: Use list-of-dicts format, not a raw string

## Common Mistakes

- ❌ Missing `reward_model.ground_truth` field (reward manager will crash)
- ❌ Prompt as a raw string instead of chat message list
- ❌ `data_source` mismatch with reward function
- ❌ Forgetting to remove original dataset columns after `map()`

<!--
================================================================================
                            MAINTAINER GUIDE
================================================================================
Location: .agents/skills/add-dataset/SKILL.md

## How to Update
- When RLDataset schema changes: update Required Schema section
- When new reference datasets added: update table
================================================================================
-->
