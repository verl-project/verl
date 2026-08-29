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

Reward computation runs through **Reward Loop**
(`verl/experimental/reward_loop/`), the default reward implementation. A
`RewardLoopManager` launches `reward.num_workers` `RewardLoopWorker` actors across
the cluster; each worker owns one `RewardManager` and scores samples concurrently
with `asyncio`. The pre-Reward-Loop implementation under
`verl/workers/reward_manager/` is still shipped for backward compatibility (see
the `legacy_reward_impl` config group), but new reward code should target Reward
Loop.

There are two layers:

1. **`compute_score` function** — pure Python, takes decoded strings and returns
   a float (or a dict carrying a `score` key). It may be **sync or async**; the
   type is detected automatically and sync functions run in an executor.
2. **`RewardManager`** (`verl/experimental/reward_loop/reward_manager/`) — wraps
   `compute_score` and implements `async run_single(data) -> dict`, handling
   decoding and the DataProto interface.

For most use cases you only need a `compute_score` function. A custom
`RewardManager` is only needed for advanced cases (rate-limited APIs, remote
CPU-heavy verifiers, reward models).

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


def compute_score(data_source: str, solution_str: str, ground_truth: Any, extra_info=None) -> float:
    """Compute reward score for a single completion.

    Args:
        data_source: Dataset identifier carried on the sample.
        solution_str: Decoded model response (the completion only, special tokens stripped).
        ground_truth: Ground truth answer from the dataset.
        extra_info: Per-sample dict; Reward Loop also injects `num_turns` and
            `rollout_reward_scores`.

    Returns:
        Float score, typically in [0.0, 1.0]. A dict with a `score` key is also
        accepted; its remaining keys are surfaced as `reward_extra_info`.
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

Use `async def compute_score(...)` when scoring involves external API calls or
sandboxed execution — the worker awaits it directly instead of occupying an
executor thread, which is significantly more efficient under concurrency.

### Step 2: Make the Function Reachable

Two options.

**Option A (no core edit, preferred for project-specific rewards)** — point the
config at your file:

```bash
reward.custom_reward_function.path=/path/to/my_reward.py \
reward.custom_reward_function.name=compute_score
```

The custom function replaces the default dispatch entirely for every sample.

**Option B (contributing a dataset reward upstream)** — register in
`verl/utils/reward_score/__init__.py` so `default_compute_score` dispatches on
`data_source`:

Add an import and one dispatch branch. Leave the existing
`default_compute_score` signature untouched — it carries extra parameters
(`sandbox_fusion_url`, `concurrent_semaphore`, `memory_limit_mb`, `**kwargs`) that
callers rely on:

```python
from verl.utils.reward_score.<name> import compute_score as <name>_compute_score

# inside default_compute_score, alongside the existing branches:
    elif data_source == "<your_dataset_name>":
        return <name>_compute_score(solution_str, ground_truth)
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

Reward settings live under the `reward` config group
(`verl/trainer/config/reward/reward.yaml`):

```bash
reward.reward_manager.name=naive \
reward.num_workers=8
```

`reward.num_workers` sets how many reward workers are launched; raise it when
reward computation, not rollout, is the bottleneck.

### Step 5 (Optional): Custom RewardManager

Only needed when the built-in managers are insufficient. Subclass
`RewardManagerBase` and implement the async `run_single`:

```python
from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase


@register("<name>")
class MyRewardManager(RewardManagerBase):
    def __init__(self, config, tokenizer, compute_score, **kwargs):
        super().__init__(config, tokenizer, compute_score)

    async def run_single(self, data: DataProto) -> dict:
        data_item = data[-1]  # multi-output trajectories: score the last sequence
        # Decode the response, call compute_score, and return the score dict.
        return {"reward_score": score, "reward_extra_info": {}}
```

Then reference it in config: `reward.reward_manager.name=<name>`.

## Reference Implementations

| Reward        | File                                                          | Description                          |
| ------------- | ------------------------------------------------------------- | ------------------------------------ |
| GSM8K         | `verl/utils/reward_score/gsm8k.py`                             | Math answer extraction               |
| Math general  | `verl/utils/reward_score/math_reward.py`                       | LaTeX boxed answer matching          |
| Geo3K         | `verl/utils/reward_score/geo3k.py`                             | Geometry answer verification         |
| naive         | `verl/experimental/reward_loop/reward_manager/naive.py`        | Default manager, sync or async score |
| dapo          | `verl/experimental/reward_loop/reward_manager/dapo.py`         | Overlong reward penalty              |
| limited       | `verl/experimental/reward_loop/reward_manager/limited.py`      | Caps concurrency for rate-limited APIs |
| remote        | `verl/experimental/reward_loop/reward_manager/remote.py`       | Separate process for CPU-heavy verifiers |

## Key Requirements

1. **Return a float or a score dict**: `compute_score` returns one float per
   sample, or a dict whose `score` key holds it
2. **No side effects**: Function must be deterministic and stateless
3. **Handle exceptions**: Return `0.0` on error, do not raise
4. **data_source matches**: The string in the dataset must match the dispatch key
   in `__init__.py` (Option B only)

## Common Mistakes

- ❌ Raising exceptions inside `compute_score` (causes worker crash)
- ❌ `data_source` mismatch between dataset and `default_compute_score`
- ❌ Returning a tensor instead of a float
- ❌ Blocking calls (`requests`, `time.sleep`) inside an `async def compute_score`,
  which stalls every other sample on that worker
- ❌ Targeting the legacy `verl/workers/reward_manager/` registry for new managers

<!--
================================================================================
                            MAINTAINER GUIDE
================================================================================
Location: .agents/skills/add-reward/SKILL.md

## How to Update
- When reward_score API changes: update Step 1 signature
- When the Reward Loop RewardManager API changes: update Step 5
- When the reward config group changes: update Step 2 / Step 4 keys
- When new reference implementations added: update table
================================================================================
-->
