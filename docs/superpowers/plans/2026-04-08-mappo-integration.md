# MAPPO Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `RayMAPPOTrainer`, `RayRiskAverseTrainer`, and the `main_mappo.py` entrypoint from the root-level research file into `verl/trainer/ppo/mappo_trainer.py` and `verl/trainer/main_mappo.py`, with a Hydra config at `verl/trainer/config/mappo_trainer.yaml`.

**Architecture:** `mappo_trainer.py` holds dataset util functions and both trainer classes, breaking the circular import that existed when the trainer imported from the entrypoint. `main_mappo.py` is moved into `verl/trainer/` so its `config_path="config"` resolves to the existing `verl/trainer/config/` directory, matching `main_ppo.py`. `ray_trainer.py` is not touched.

**Tech Stack:** Python, Ray, Hydra/OmegaConf, PyTorch, verl package conventions.

---

## File Map

| Action | Path |
|--------|------|
| Create | `verl/trainer/ppo/mappo_trainer.py` |
| Create | `verl/trainer/main_mappo.py` |
| Create | `verl/trainer/config/mappo_trainer.yaml` |
| Delete | `main_mappo.py` (repo root) |
| Delete | `ray_trainer_self_version.py` (repo root) |

---

## Task 1: Create `verl/trainer/ppo/mappo_trainer.py` — header and dataset utilities

**Files:**
- Create: `verl/trainer/ppo/mappo_trainer.py`

- [ ] **Step 1: Write the file header with imports and dataset utility functions**

Create `verl/trainer/ppo/mappo_trainer.py` with this exact content as the top of the file (imports come from lines 20–65 of `ray_trainer_self_version.py`, with `ResourcePoolManager` imported from the package rather than defined locally, and without the now-unneeded `from verl.trainer.main_mappo import ...` that will be fixed in Task 2):

```python
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
MAPPO Trainer with Ray-based single controller.
Provides RayMAPPOTrainer and RayRiskAverseTrainer for multi-agent PPO training.
"""

import csv
import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
import verl.utils.torch_functional as verl_F
```

Then append the two dataset utility functions. Copy them **verbatim** from root `main_mappo.py` lines 369–457:

```python
def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True):
    """Create a dataset for a single agent.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer: The tokenizer.
        processor: The processor.

    Returns:
        dataset: The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.utils.import_utils import load_extern_type

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"The custom dataset class '{data_config.custom_cls.name}' from "
                f"'{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )
    elif "datagen" in data_config and data_config.datagen.get("path", None) is not None and is_train:
        from verl.utils.dataset.dynamicgen_dataset import DynamicGenDataset

        dataset_cls = DynamicGenDataset
        print("Using DynamicGenDataset for data generation.")
    else:
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )
    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset: The dataset.

    Returns:
        sampler: The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    from verl.experimental.dataset.sampler import AbstractSampler
    from verl.utils.import_utils import load_extern_type

    if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
        curriculum_class = load_extern_type(
            data_config.sampler.class_path,
            data_config.sampler.class_name,
        )
        sampler = curriculum_class(
            data_source=dataset,
            data_config=data_config,
        )
        assert isinstance(sampler, AbstractSampler)
        assert data_config.get("dataloader_num_workers", 8) == 0, (
            "If using curriculum, num_workers must be 0 to prevent data caching."
        )
    elif data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)
    return sampler
```

- [ ] **Step 2: Verify the header imports cleanly**

```bash
cd /weka/scratch/lshi40_llm/mallm/SRPO
module load anaconda3/2024.02-1 && conda run -n srpo python -c "
import verl.trainer.ppo.mappo_trainer as m
print('create_rl_dataset:', m.create_rl_dataset)
print('create_rl_sampler:', m.create_rl_sampler)
"
```

Expected: prints the two function objects without `ImportError`.

- [ ] **Step 3: Commit**

```bash
git add verl/trainer/ppo/mappo_trainer.py
git commit -m "feat(mappo): add mappo_trainer.py with dataset utilities"
```

---

## Task 2: Add `RayMAPPOTrainer` to `mappo_trainer.py`

**Files:**
- Modify: `verl/trainer/ppo/mappo_trainer.py` (append)

- [ ] **Step 1: Copy `RayMAPPOTrainer` from `ray_trainer_self_version.py`**

Append to `verl/trainer/ppo/mappo_trainer.py` the class `RayMAPPOTrainer` extracted from `ray_trainer_self_version.py` **lines 2630–6090** (stop before `class RayMAPPOShareTrainer` on line 6091, and stop before `class RayRiskAverseTrainer` on line 5742 — do `RayMAPPOTrainer` only in this task).

That is: copy lines **2630–5741** verbatim, appending them to the file.

- [ ] **Step 2: Fix the circular import inside `_create_dataloader`**

In the newly appended `RayMAPPOTrainer._create_dataloader` method, remove the lazy import line. The functions now live in the same module.

Find and remove this exact line (it was line 2719 in `ray_trainer_self_version.py`, and will appear near the top of `_create_dataloader`):

```python
        from verl.trainer.main_mappo import create_rl_dataset, create_rl_sampler
```

The method body already calls `create_rl_dataset(...)` and `create_rl_sampler(...)` — those calls remain unchanged. After removing the lazy import, the module-level definitions handle them.

- [ ] **Step 3: Verify `RayMAPPOTrainer` is importable**

```bash
cd /weka/scratch/lshi40_llm/mallm/SRPO
module load anaconda3/2024.02-1 && conda run -n srpo python -c "
from verl.trainer.ppo.mappo_trainer import RayMAPPOTrainer
print('RayMAPPOTrainer:', RayMAPPOTrainer)
"
```

Expected: prints the class object without `ImportError` or `NameError`.

- [ ] **Step 4: Commit**

```bash
git add verl/trainer/ppo/mappo_trainer.py
git commit -m "feat(mappo): add RayMAPPOTrainer class"
```

---

## Task 3: Add `RayRiskAverseTrainer` to `mappo_trainer.py`

**Files:**
- Modify: `verl/trainer/ppo/mappo_trainer.py` (append)

- [ ] **Step 1: Copy `RayRiskAverseTrainer` from `ray_trainer_self_version.py`**

Append to `verl/trainer/ppo/mappo_trainer.py` the class `RayRiskAverseTrainer` extracted from `ray_trainer_self_version.py` **lines 5742–6090** verbatim. No changes needed — it has no lazy imports of its own and inherits from `RayMAPPOTrainer` which is already defined above it in the same file.

- [ ] **Step 2: Verify both classes are importable**

```bash
cd /weka/scratch/lshi40_llm/mallm/SRPO
module load anaconda3/2024.02-1 && conda run -n srpo python -c "
from verl.trainer.ppo.mappo_trainer import RayMAPPOTrainer, RayRiskAverseTrainer
print('RayMAPPOTrainer:', RayMAPPOTrainer)
print('RayRiskAverseTrainer:', RayRiskAverseTrainer)
assert issubclass(RayRiskAverseTrainer, RayMAPPOTrainer)
print('Inheritance OK')
"
```

Expected: all three prints succeed, no errors.

- [ ] **Step 3: Commit**

```bash
git add verl/trainer/ppo/mappo_trainer.py
git commit -m "feat(mappo): add RayRiskAverseTrainer class"
```

---

## Task 4: Create `verl/trainer/main_mappo.py`

**Files:**
- Create: `verl/trainer/main_mappo.py`

- [ ] **Step 1: Copy root `main_mappo.py` into `verl/trainer/main_mappo.py`**

```bash
cp /weka/scratch/lshi40_llm/mallm/SRPO/main_mappo.py \
   /weka/scratch/lshi40_llm/mallm/SRPO/verl/trainer/main_mappo.py
```

- [ ] **Step 2: Update the trainer import**

In `verl/trainer/main_mappo.py`, find and replace:

```python
from verl.trainer.ppo.ray_trainer import RayMAPPOTrainer
```

with:

```python
from verl.trainer.ppo.mappo_trainer import RayMAPPOTrainer
```

- [ ] **Step 3: Remove the local `create_rl_dataset` and `create_rl_sampler` definitions**

Remove lines 369–457 of the original root `main_mappo.py` — specifically the two function definitions `def create_rl_dataset(...)` and `def create_rl_sampler(...)`. These are now in `mappo_trainer.py`.

In `verl/trainer/main_mappo.py`, add this import near the top (alongside the other `verl.trainer.ppo` imports):

```python
from verl.trainer.ppo.mappo_trainer import create_rl_dataset, create_rl_sampler
```

- [ ] **Step 4: Verify the `config_path` is correct**

Confirm the Hydra decorator reads:

```python
@hydra.main(config_path="config", config_name="mappo_trainer", version_base=None)
```

`config_path="config"` is correct — when the file lives in `verl/trainer/`, Hydra resolves this relative to the file's location, pointing to `verl/trainer/config/`. No change needed here.

- [ ] **Step 5: Verify the entrypoint imports cleanly**

```bash
cd /weka/scratch/lshi40_llm/mallm/SRPO
module load anaconda3/2024.02-1 && conda run -n srpo python -c "
import verl.trainer.main_mappo as m
print('run_mappo:', m.run_mappo)
print('TaskRunner:', m.TaskRunner)
print('create_rl_dataset:', m.create_rl_dataset)
print('create_rl_sampler:', m.create_rl_sampler)
"
```

Expected: all four objects print without error.

- [ ] **Step 6: Commit**

```bash
git add verl/trainer/main_mappo.py
git commit -m "feat(mappo): add main_mappo.py entrypoint under verl/trainer"
```

---

## Task 5: Create `verl/trainer/config/mappo_trainer.yaml`

**Files:**
- Create: `verl/trainer/config/mappo_trainer.yaml`

- [ ] **Step 1: Write the config file**

Create `verl/trainer/config/mappo_trainer.yaml` with the following content. The `defaults` list inherits the full PPO config; `multi_agent` is the only new section:

```yaml
# Format checks enforced on CI:
# 1. Comments must appear above each field.
# 2. There must be a blank line between each field.
# 3. Inline comments (after a field on the same line) are not allowed.
# 4. Indentation level is respected for nested fields.

defaults:
  - ppo_trainer

  - _self_

# Multi-agent training configuration for MAPPO.
# num_agents and agents list are required; all other PPO fields are inherited above.
multi_agent:

  # Number of agents participating in the multi-agent rollout.
  num_agents: 2

  # Number of discussion rounds per training step.
  num_rounds: 3

  # Risk coefficient used by RayRiskAverseTrainer.
  risk_coef: 1.0

  # Prompt prepended to discussion history for each agent turn.
  discussion_prompt: "The discussion history is as follows:"

  # Per-agent resource and model overrides.
  # Length must equal num_agents. Each entry can override actor.model.path,
  # n_gpus_per_node, nnodes, and any actor/critic sub-config.
  agents:
    - actor:
        model:
          path: ???
      n_gpus_per_node: 8
      nnodes: 1
    - actor:
        model:
          path: ???
      n_gpus_per_node: 8
      nnodes: 1
```

- [ ] **Step 2: Verify Hydra can load the config**

```bash
cd /weka/scratch/lshi40_llm/mallm/SRPO
module load anaconda3/2024.02-1 && conda run -n srpo python -c "
from hydra import compose, initialize_config_dir
import os
cfg_dir = os.path.abspath('verl/trainer/config')
with initialize_config_dir(config_dir=cfg_dir, version_base=None):
    cfg = compose(config_name='mappo_trainer', overrides=[
        'multi_agent.agents[0].actor.model.path=dummy0',
        'multi_agent.agents[1].actor.model.path=dummy1',
    ])
    print('multi_agent.num_agents:', cfg.multi_agent.num_agents)
    print('Config load OK')
"
```

Expected: prints `multi_agent.num_agents: 2` and `Config load OK`.

- [ ] **Step 3: Commit**

```bash
git add verl/trainer/config/mappo_trainer.yaml
git commit -m "feat(mappo): add mappo_trainer.yaml Hydra config"
```

---

## Task 6: Delete root-level files and verify end-to-end import chain

**Files:**
- Delete: `main_mappo.py` (repo root)
- Delete: `ray_trainer_self_version.py` (repo root)

- [ ] **Step 1: Verify the full import chain is clean before deleting**

```bash
cd /weka/scratch/lshi40_llm/mallm/SRPO
module load anaconda3/2024.02-1 && conda run -n srpo python -c "
# Full chain: entrypoint → trainer → dataset utils → verl package
from verl.trainer.ppo.mappo_trainer import (
    create_rl_dataset,
    create_rl_sampler,
    RayMAPPOTrainer,
    RayRiskAverseTrainer,
)
import verl.trainer.main_mappo as entrypoint
print('All imports OK')
print('  create_rl_dataset:', create_rl_dataset)
print('  RayMAPPOTrainer:', RayMAPPOTrainer)
print('  RayRiskAverseTrainer:', RayRiskAverseTrainer)
print('  entrypoint.run_mappo:', entrypoint.run_mappo)
"
```

Expected: all lines print without error.

- [ ] **Step 2: Delete the root-level files**

```bash
git rm /weka/scratch/lshi40_llm/mallm/SRPO/main_mappo.py
git rm /weka/scratch/lshi40_llm/mallm/SRPO/ray_trainer_self_version.py
```

- [ ] **Step 3: Final verification after deletion**

```bash
cd /weka/scratch/lshi40_llm/mallm/SRPO
module load anaconda3/2024.02-1 && conda run -n srpo python -c "
from verl.trainer.ppo.mappo_trainer import RayMAPPOTrainer, RayRiskAverseTrainer
import verl.trainer.main_mappo
print('Post-deletion import OK')
"
```

Expected: `Post-deletion import OK` with no errors.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore(mappo): remove root-level main_mappo.py and ray_trainer_self_version.py"
```

---

## Self-Review

**Spec coverage:**
- ✅ `verl/trainer/ppo/mappo_trainer.py` with `create_rl_dataset`, `create_rl_sampler`, `RayMAPPOTrainer`, `RayRiskAverseTrainer` — Tasks 1–3
- ✅ Circular import broken — Task 2 Step 2 removes the lazy import
- ✅ `verl/trainer/main_mappo.py` moved and imports updated — Task 4
- ✅ `verl/trainer/config/mappo_trainer.yaml` extending `ppo_trainer.yaml` — Task 5
- ✅ Root files deleted — Task 6
- ✅ `ray_trainer.py` not touched — confirmed by plan (no task modifies it)

**Placeholder scan:** No TBDs. All steps have exact commands or code. Line ranges are exact. ✅

**Type consistency:** `create_rl_dataset` and `create_rl_sampler` signatures are identical in Task 1 (definition), Task 2 (call sites unchanged), and Task 4 (imported by entrypoint). ✅
