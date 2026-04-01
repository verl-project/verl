---
name: add-unit-tests
description: Guide for adding unit tests to veRL. Use when user wants to add tests for new functionality or increase test coverage.
---

# Add Unit Tests

Add unit tests to veRL following the project's testing conventions.

## When to Use

This skill is triggered when:

- User asks "how do I add tests?"
- User wants to increase test coverage
- User needs to write tests for new functionality

## Step-by-Step Guide

### Step 1: Understand Test Categories

| Category              | Location                     | How It Runs              | Trigger                      |
| --------------------- | ---------------------------- | ------------------------ | ---------------------------- |
| **CPU unit tests**    | `tests/**/test_*_on_cpu.py`  | `pytest` on CPU          | `cpu_unit_tests.yml`         |
| **GPU unit tests**    | `tests/**/test_*.py`         | `pytest` on GPU          | `gpu_unit_tests.yml`         |
| **Distributed tests** | `tests/special_distributed/` | Multi-GPU pytest         | Separate CI workflow         |
| **E2E tests**         | `tests/special_e2e/`         | Full training run        | `e2e_*.yml`                  |
| **Sanity tests**      | `tests/special_sanity/`      | Quick smoke tests        | Always triggered             |

**Naming rules:**
- CPU-only tests: filename must end with `_on_cpu.py`
- Default tests run on GPU — skip gracefully if CUDA unavailable
- Follow the directory structure matching the source: `tests/trainer/` for `verl/trainer/`

### Step 2: Create the Test File

```python
# tests/<namespace>/test_<module>_on_cpu.py  (CPU)
# tests/<namespace>/test_<module>.py          (GPU)

import pytest
import torch

from verl.<namespace>.<module> import MyClass
```

### Step 3: Write Test Functions

Follow the Arrange-Act-Assert pattern, with descriptive names:

```python
def test_compute_score_correct_answer_returns_one():
    """compute_score returns 1.0 when model output matches ground truth."""
    # Arrange
    solution = "The answer is \\boxed{42}"
    ground_truth = "42"

    # Act
    score = compute_score(solution, ground_truth)

    # Assert
    assert score == 1.0


def test_compute_score_wrong_answer_returns_zero():
    solution = "The answer is \\boxed{99}"
    score = compute_score(solution, "42")
    assert score == 0.0


def test_compute_score_exception_returns_zero():
    """compute_score must not raise; return 0.0 on malformed input."""
    score = compute_score(None, "42")  # type: ignore
    assert score == 0.0
```

### Step 4: GPU / CUDA Guards

Always skip gracefully when GPU is not available:

```python
import torch

CUDA_AVAILABLE = torch.cuda.is_available()

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_gpu_forward_pass():
    model = MyModel().cuda()
    x = torch.randn(2, 16, device="cuda")
    out = model(x)
    assert out.shape == (2, 16)
```

### Step 5: Tensor Assertions

Use `torch.testing.assert_close` instead of bare `assert`:

```python
import torch

def test_advantage_normalization():
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    mask = torch.ones_like(rewards)
    adv = normalize_advantages(rewards, mask)

    # Prefer assert_close over .equal() — gives useful diff on failure
    torch.testing.assert_close(adv.mean(), torch.tensor(0.0), atol=1e-5, rtol=0)
    torch.testing.assert_close(adv.std(), torch.tensor(1.0), atol=1e-5, rtol=0)
```

### Step 6: DataProto Tests

When testing code that uses `DataProto`:

```python
from verl import DataProto
import torch

def make_dummy_batch(batch_size=4, seq_len=16):
    return DataProto.from_dict({
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "responses": torch.randint(0, 1000, (batch_size, 8)),
        "response_mask": torch.ones(batch_size, 8, dtype=torch.long),
    })


def test_reward_manager_output_shape():
    data = make_dummy_batch()
    manager = NaiveRewardManager(tokenizer=mock_tokenizer, num_examine=0)
    reward = manager(data)
    assert reward.shape == (4, 8)  # [batch_size, response_len]
```

### Step 7: Parametrize for Multiple Cases

```python
@pytest.mark.parametrize("batch_size,seq_len", [(1, 8), (4, 16), (8, 32)])
def test_seqlen_balancing(batch_size, seq_len):
    seqlens = [seq_len] * batch_size
    result = balance_sequences(seqlens, n_gpus=2)
    assert len(result) == 2
```

### Step 8: CI Registration

After writing your test:

1. **CPU test** (`test_*_on_cpu.py`): automatically picked up by `cpu_unit_tests.yml`
2. **GPU test** (`test_*.py`): automatically picked up by `gpu_unit_tests.yml`
3. **If your test is slow or requires special hardware**: manually exclude it in the
   relevant CI yamls per `tests/README.md` instructions

## Reference Implementations

| File                                      | Description                               |
| ----------------------------------------- | ----------------------------------------- |
| `tests/test_protocol_on_cpu.py`           | DataProto CPU tests (good starting point) |
| `tests/test_base_config_on_cpu.py`        | Config dataclass tests                    |
| `tests/trainer/`                          | Trainer unit tests                        |
| `tests/workers/`                          | Worker unit tests                         |
| `tests/utils/`                            | Utility function tests                    |
| `tests/special_distributed/`             | Multi-GPU distributed tests               |

## Common Mistakes

- ❌ GPU test without `@pytest.mark.skipif(not CUDA_AVAILABLE, ...)` guard
- ❌ `assert tensor.equal(other)` — use `torch.testing.assert_close` instead
- ❌ Missing `_on_cpu.py` suffix for CPU-only tests (they'll be scheduled on GPU nodes)
- ❌ Raising inside `compute_score` under test — always test the 0.0-on-error path
- ❌ Forgetting to exclude slow/special tests from `cpu_unit_tests.yml` / `gpu_unit_tests.yml`

<!--
================================================================================
                            MAINTAINER GUIDE
================================================================================
Location: .agents/skills/add-unit-tests/SKILL.md

## How to Update
- When test layout changes: update Step 1 table
- When DataProto schema changes: update Step 6 helper
- When CI yamls renamed: update Step 8
================================================================================
-->
