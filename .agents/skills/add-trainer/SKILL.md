---
name: add-trainer
description: Guide for adding a new RL trainer recipe to veRL. Use when user wants to implement a new algorithm or training recipe.
---

# Add Trainer

Add a new RL training recipe (algorithm variant) to veRL.

## When to Use

This skill is triggered when:

- User asks "how do I implement a new algorithm?"
- User wants to add a new trainer on top of veRL's worker infrastructure
- User mentions creating a new `recipe/` entry or trainer script

## Overview

veRL follows a **single-controller + Ray workers** architecture:

```
Your Trainer Script (controller, CPU node)
    │
    ├── ActorRolloutRefWorker  (Ray remote, GPU)  — generates rollouts + computes logprobs
    ├── CriticWorker           (Ray remote, GPU)  — value estimates (PPO only)
    ├── RewardModelWorker      (Ray remote, GPU)  — optional RM scoring
    └── RewardManager          (inline)           — rule-based reward scoring
```

The controller drives the training loop by calling `.generate_sequences()`,
`.compute_ref_log_prob()`, `.update_actor()`, etc. on remote workers via Ray.

For a new algorithm, you typically:
1. Write a trainer class (controller logic)
2. Reuse existing workers or subclass them
3. Add a run script with the config

## Step-by-Step Guide

### Step 1: Study a Reference Trainer

Start by reading the simplest existing trainer that resembles your target algorithm:

| Algorithm | File                                      |
| --------- | ----------------------------------------- |
| GRPO      | `verl/trainer/ppo/ray_trainer.py`         |
| RLOO      | `examples/rloo_trainer/`                  |
| REINFORCE++| `examples/reinforce_plus_plus_trainer/`  |
| DAPO      | `examples/dapo/`                          |
| ReMax     | `examples/remax_trainer/`                 |

### Step 2: Create Trainer Directory

```
examples/<name>_trainer/
├── __init__.py
├── <name>_trainer.py      # Main trainer class
└── run_qwen2-7b.sh        # Example run script
```

### Step 3: Implement the Trainer Class

```python
# examples/<name>_trainer/<name>_trainer.py
import torch
from omegaconf import DictConfig
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer  # or write from scratch


class MyTrainer(RayPPOTrainer):
    """My custom RL trainer."""

    def _compute_advantage(self, data: DataProto) -> DataProto:
        """Override advantage computation for your algorithm."""
        rewards = data.batch["token_level_scores"]  # shape: [bs, seqlen]
        # ... your advantage computation
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        return data

    def fit(self):
        """Main training loop."""
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                # 1. Generate rollouts
                data: DataProto = self.actor_rollout_ref.generate_sequences(batch_dict)

                # 2. Compute rewards
                reward_tensor = self.reward_fn(data)
                data.batch["token_level_scores"] = reward_tensor

                # 3. Compute advantages (your algorithm logic here)
                data = self._compute_advantage(data)

                # 4. Update actor
                actor_output = self.actor_rollout_ref.update_actor(data)

                # 5. Log metrics
                self._log_metrics(actor_output)
```

### Step 4: Write Run Script

```bash
#!/bin/bash
# examples/<name>_trainer/run_qwen2-7b.sh

set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    reward_model.reward_manager=naive \
    trainer.total_epochs=15 \
    trainer.project_name=<name>_trainer \
    trainer.experiment_name=qwen2-7b-gsm8k
```

### Step 5: Key DataProto Fields

`DataProto` is veRL's data container. Common fields in the batch:

| Field                      | Shape              | Description                            |
| -------------------------- | ------------------ | -------------------------------------- |
| `input_ids`                | `[bs, seqlen]`     | Prompt + response token IDs            |
| `attention_mask`           | `[bs, seqlen]`     | 1 for real tokens, 0 for padding       |
| `position_ids`             | `[bs, seqlen]`     | Position indices                       |
| `responses`                | `[bs, resp_len]`   | Response token IDs only                |
| `response_mask`            | `[bs, resp_len]`   | 1 for response tokens                  |
| `token_level_scores`       | `[bs, resp_len]`   | Per-token rewards from reward manager  |
| `advantages`               | `[bs, resp_len]`   | Computed advantages                    |
| `returns`                  | `[bs, resp_len]`   | Computed returns                       |
| `old_log_probs`            | `[bs, resp_len]`   | Log probs from rollout policy          |
| `ref_log_prob`             | `[bs, resp_len]`   | Log probs from reference policy        |

### Step 6: Core Algorithm Utilities

veRL provides registered advantage estimators and policy loss functions:

```python
from verl.trainer.ppo.core_algos import get_adv_estimator_fn, register_adv_est

# Use built-in estimators: "gae", "grpo", "reinforce", "rloo", "remax"
adv_fn = get_adv_estimator_fn("grpo")
advantages, returns = adv_fn(token_level_rewards, response_mask, config)

# Register your own estimator
@register_adv_est("my_estimator")
def my_estimator(token_level_rewards, response_mask, config, **kwargs):
    ...
    return advantages, returns
```

## Key Requirements

1. **Ray-based**: Workers must be Ray remote actors
2. **DataProto**: Use DataProto as the data exchange format between controller and workers
3. **OmegaConf config**: Use Hydra/OmegaConf for configuration
4. **No hardcoded paths**: All paths via config

## Common Mistakes

- ❌ Modifying `DataProto.batch` tensors in-place without `.clone()` when needed
- ❌ Forgetting to apply `response_mask` when computing per-token losses
- ❌ Mixing up `token_level_scores` (raw rewards) vs `advantages` (normalized)
- ❌ Calling blocking Ray ops inside a loop without `.get()` at the right time

<!--
================================================================================
                            MAINTAINER GUIDE
================================================================================
Location: .agents/skills/add-trainer/SKILL.md

## How to Update
- When DataProto fields change: update Step 5 table
- When core_algos API changes: update Step 6
- When new reference trainers added: update Step 1 table
================================================================================
-->
