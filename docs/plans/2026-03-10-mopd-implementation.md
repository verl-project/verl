# MOPD (Multi-Teacher On-Policy Distillation) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement N-teacher MOPD algorithm in verl framework, supporting domain-specific teacher routing, token-level reverse KL advantages, IS correction, and G-OPD ExOPD compatibility.

**Architecture:** Teachers are deployed as separate RefPolicy worker groups (frozen inference-only). Training loop groups samples by teacher_id, forwards sub-batches to respective teachers, computes per-sample MOPD advantages in core_algos.py, and applies standard policy gradient updates.

**Tech Stack:** Python 3.10+ | PyTorch | Ray | FSDP2 | Hydra | verl framework

**Scope:** ~550 LOC production code + ~180 LOC tests

---

## Design Decisions (All 10 Fixes Incorporated)

### Critical Fixes Applied
1. ✅ Teachers in RefPolicy workers (not Actor workers)
2. ✅ Stacked tensor storage (not individual keys)
3. ✅ Sub-batch forwarding (not full-batch-then-mask)
4. ✅ Correct broadcasting with `.unsqueeze(-1)`
5. ✅ IS correction degenerate case fallback
6. ✅ Explicit backward compatibility guards
7. ✅ `batch.select_idxs()` uses integer tensor indices
8. ✅ Teacher configs validated in `__post_init__`
9. ✅ `rollout_log_probs` optional with None-check
10. ✅ Advantage normalization respects response_mask

### Refinements Applied
- Teacher ID converted to integer indices for performance
- Config validation with unique names, positive lambda, valid epsilon bounds
- Realistic LOC estimate: ~500 production + ~150 tests

---

## Task Breakdown

### Task 1: Create Teacher Configuration Module

**Files:**
- Create: `verl/workers/config/teacher.py`
- Test: `tests/unit/test_teacher_config.py`

**Step 1: Write failing test for TeacherConfig validation**

Create `tests/unit/test_teacher_config.py`:

```python
import pytest
from verl.workers.config.teacher import TeacherConfig, MOPDConfig


def test_teacher_config_requires_name():
    with pytest.raises(ValueError, match="name must be non-empty"):
        TeacherConfig(name="", model_path="/models/test")


def test_teacher_config_requires_model_path():
    with pytest.raises(ValueError, match="model_path must be non-empty"):
        TeacherConfig(name="test", model_path="")


def test_mopd_config_rejects_duplicate_teacher_names():
    teachers = [
        TeacherConfig(name="math", model_path="/models/math"),
        TeacherConfig(name="math", model_path="/models/math2"),
    ]
    with pytest.raises(ValueError, match="Duplicate teacher names"):
        MOPDConfig(enabled=True, teachers=teachers)


def test_mopd_config_validates_lambda():
    with pytest.raises(ValueError, match="lambda_val must be positive"):
        MOPDConfig(enabled=True, lambda_val=0.0)


def test_mopd_config_validates_epsilon_bounds():
    with pytest.raises(ValueError, match="is_epsilon_low must be < is_epsilon_high"):
        MOPDConfig(enabled=True, is_epsilon_low=10.0, is_epsilon_high=1.0)
```

**Step 2: Run test to verify it fails**

```bash
cd /home/scbjtfy/verl/.worktrees/mopd-implementation
pytest tests/unit/test_teacher_config.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'verl.workers.config.teacher'"

**Step 3: Implement TeacherConfig and MOPDConfig**

Create `verl/workers/config/teacher.py`:

```python
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License")

from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig


@dataclass
class TeacherConfig(BaseConfig):
    """Configuration for a single teacher model in MOPD.

    Args:
        name: Unique teacher identifier (e.g., "math", "code")
        model_path: HuggingFace model path or local checkpoint
        weight: Teacher weight for weighted composition (unused in current impl)
        resource_pool: Ray resource pool name (default: "global_pool")
        log_prob_micro_batch_size: Micro-batch size for teacher forward pass
        base_model_path: Optional base model for ExOPD normalization
    """
    name: str = ""
    model_path: str = ""
    weight: float = 1.0
    resource_pool: str = "global_pool"
    log_prob_micro_batch_size: int = 8
    base_model_path: Optional[str] = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("TeacherConfig.name must be non-empty")
        if not self.model_path:
            raise ValueError("TeacherConfig.model_path must be non-empty")
        if self.weight <= 0:
            raise ValueError(f"TeacherConfig.weight must be positive: {self.weight}")


@dataclass
class MOPDConfig(BaseConfig):
    """Configuration for Multi-Teacher On-Policy Distillation.

    Implements MiMo paper (arXiv:2601.02780) Eq. 7-9 + G-OPD ExOPD mode.

    Args:
        enabled: Enable MOPD training (default: False for backward compat)
        teachers: List of teacher configurations
        lambda_val: G-OPD scaling coefficient (1.0=standard MOPD, >1.0=extrapolation)
        orm_weight: Weight for outcome reward mixing (α in A_final = A_mopd + α·A_orm)
        is_correction: Enable importance sampling correction for train/inference mismatch
        is_epsilon_low: Lower bound for IS ratio acceptance
        is_epsilon_high: Upper bound for IS ratio acceptance
        use_base_normalization: Enable ExOPD base model normalization
        base_model_path: Path to base model for ExOPD (shared across teachers)
    """
    enabled: bool = False
    teachers: list[TeacherConfig] = field(default_factory=list)
    lambda_val: float = 1.0
    orm_weight: float = 0.0
    is_correction: bool = True
    is_epsilon_low: float = 0.1
    is_epsilon_high: float = 10.0
    use_base_normalization: bool = False
    base_model_path: Optional[str] = None

    def __post_init__(self):
        # Validate unique teacher names
        names = [t.name for t in self.teachers]
        if len(names) != len(set(names)):
            duplicates = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate teacher names: {set(duplicates)}")

        # Validate lambda_val
        if self.lambda_val <= 0:
            raise ValueError(f"lambda_val must be positive: {self.lambda_val}")

        # Validate IS epsilon bounds
        if self.is_epsilon_low >= self.is_epsilon_high:
            raise ValueError(
                f"is_epsilon_low ({self.is_epsilon_low}) must be < "
                f"is_epsilon_high ({self.is_epsilon_high})"
            )

        # Validate base normalization config
        if self.use_base_normalization and not self.base_model_path:
            raise ValueError(
                "use_base_normalization=True requires base_model_path to be set"
            )
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_teacher_config.py -v
```

Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add verl/workers/config/teacher.py tests/unit/test_teacher_config.py
git commit -m "feat(mopd): add TeacherConfig and MOPDConfig with validation

- TeacherConfig: per-teacher model path, resource pool, base model
- MOPDConfig: MOPD algorithm parameters (lambda, IS bounds, ORM weight)
- Validation: unique names, positive lambda, valid epsilon bounds
- Tests: 5 unit tests for config validation

Implements Fix 8: Config validation in __post_init__"
```

---

### Task 2: Implement MOPD Advantage Estimator

**Files:**
- Modify: `verl/trainer/ppo/core_algos.py`
- Test: `tests/unit/test_mopd_advantage.py`

**Step 1: Write failing test for MOPD advantage computation**

Create `tests/unit/test_mopd_advantage.py`:

```python
import torch
import pytest
from verl.trainer.ppo.core_algos import register_adv_est, get_adv_estimator_fn


def test_mopd_advantage_basic():
    """Test basic MOPD advantage computation (lambda=1.0)."""
    B, T = 4, 10
    teacher_log_prob = torch.randn(B, T)
    old_log_probs = torch.randn(B, T)
    response_mask = torch.ones(B, T)
    token_level_rewards = torch.randn(B, T)

    mopd_fn = get_adv_estimator_fn("mopd")
    advantages, returns = mopd_fn(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        teacher_log_prob=teacher_log_prob,
        old_log_probs=old_log_probs,
        lambda_val=1.0,
    )

    # Advantage should be teacher_log_prob - old_log_probs (detached)
    expected = (teacher_log_prob - old_log_probs).detach() * response_mask
    torch.testing.assert_close(advantages, expected)


def test_mopd_advantage_with_is_correction():
    """Test IS correction masks tokens outside epsilon bounds."""
    B, T = 2, 5
    teacher_log_prob = torch.ones(B, T) * 2.0  # Non-zero to verify masking
    old_log_probs = torch.ones(B, T) * 1.0     # Non-zero advantage
    rollout_log_probs = torch.tensor([
        [1.0, 1.0, -4.0, 1.0, 1.0],  # token 2: ratio = exp(1-(-4)) = 148 > 10
        [1.0, 1.0, 1.0, 1.0, 1.0],
    ])
    response_mask = torch.ones(B, T)
    token_level_rewards = torch.zeros(B, T)

    mopd_fn = get_adv_estimator_fn("mopd")
    advantages, _ = mopd_fn(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        teacher_log_prob=teacher_log_prob,
        old_log_probs=old_log_probs,
        rollout_log_probs=rollout_log_probs,
        is_correction=True,
        is_epsilon_low=0.1,
        is_epsilon_high=10.0,
    )

    # Token [0, 2] should be masked to 0 (ratio = exp(1-(-4)) = 148 > 10)
    assert advantages[0, 2].item() == 0.0
    # Non-masked tokens should have non-zero advantage (teacher - old = 2-1 = 1)
    assert advantages[0, 0].item() != 0.0


def test_mopd_advantage_exopd_mode():
    """Test ExOPD mode with base model normalization."""
    B, T = 2, 5
    teacher_log_prob = torch.ones(B, T) * 2.0
    old_log_probs = torch.ones(B, T) * 1.0
    base_log_prob = torch.ones(B, T) * 0.5
    response_mask = torch.ones(B, T)
    token_level_rewards = torch.zeros(B, T)

    mopd_fn = get_adv_estimator_fn("mopd")
    advantages, _ = mopd_fn(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        teacher_log_prob=teacher_log_prob,
        old_log_probs=old_log_probs,
        base_log_prob=base_log_prob,
        lambda_val=1.25,
        is_correction=False,
    )

    # ExOPD: -[(old - base) - lambda*(teacher - base)]
    # = -[(1.0 - 0.5) - 1.25*(2.0 - 0.5)]
    # = -[0.5 - 1.875] = 1.375
    expected = torch.ones(B, T) * 1.375
    torch.testing.assert_close(advantages, expected, rtol=1e-4, atol=1e-4)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_mopd_advantage.py -v
```

Expected: FAIL with "Unknown advantage estimator: mopd"

**Step 3: Implement compute_mopd_advantage in core_algos.py**

Add to `verl/trainer/ppo/core_algos.py` (after existing advantage estimators, around line 900):

```python
@register_adv_est("mopd")
def compute_mopd_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    teacher_log_prob: torch.Tensor,
    old_log_probs: torch.Tensor,
    rollout_log_probs: Optional[torch.Tensor] = None,
    base_log_prob: Optional[torch.Tensor] = None,
    lambda_val: float = 1.0,
    orm_weight: float = 0.0,
    is_correction: bool = True,
    is_epsilon_low: float = 0.1,
    is_epsilon_high: float = 10.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multi-Teacher On-Policy Distillation (MOPD) advantage estimator.

    Implements MiMo paper (arXiv:2601.02780) Eq. 7-9 + G-OPD ExOPD mode.

    Args:
        token_level_rewards: ORM/verifier scores [batch, response_len]
        response_mask: Valid token mask [batch, response_len]
        teacher_log_prob: Per-sample selected teacher log probs [batch, response_len]
        old_log_probs: Training engine log probs [batch, response_len]
        rollout_log_probs: Inference engine log probs (optional) [batch, response_len]
        base_log_prob: Base model log probs for ExOPD (optional) [batch, response_len]
        lambda_val: G-OPD scaling (1.0=standard MOPD, >1.0=extrapolation)
        orm_weight: Weight for outcome reward (α in A_final = A_mopd + α·A_orm)
        is_correction: Enable importance sampling correction
        is_epsilon_low: Lower bound for IS ratio
        is_epsilon_high: Upper bound for IS ratio

    Returns:
        advantages: MOPD advantages [batch, response_len]
        returns: Token-level rewards (for interface consistency)
    """
    # Token-level teacher advantage (stop-gradient)
    if lambda_val == 1.0 or base_log_prob is None:
        # Standard MOPD: reverse KL
        A_mopd = (teacher_log_prob - old_log_probs).detach()
    else:
        # ExOPD: base-normalized reverse KL with scaling
        A_mopd = -((old_log_probs - base_log_prob)
                   - lambda_val * (teacher_log_prob - base_log_prob)).detach()

    # IS correction (training/inference engine mismatch)
    if is_correction and rollout_log_probs is not None:
        ratio = (old_log_probs - rollout_log_probs).exp()
        valid = (ratio >= is_epsilon_low) & (ratio <= is_epsilon_high)
        weights = torch.where(valid, ratio.detach(), torch.zeros_like(ratio))

        # Fix 5: Degenerate case fallback
        valid_tokens = (weights > 0) & (response_mask > 0)
        all_masked = ~valid_tokens.any(dim=-1)  # [batch]
        if all_masked.any():
            import logging
            logger = logging.getLogger(__file__)
            logger.warning(
                "IS correction masked all tokens for %d samples. "
                "Using unweighted advantages as fallback.",
                all_masked.sum().item()
            )
            weights[all_masked] = 1.0
    else:
        weights = torch.ones_like(old_log_probs)

    # Compose with ORM outcome advantage
    if orm_weight > 0:
        # Validate index is available (required by GRPO)
        if "index" not in kwargs or kwargs["index"] is None:
            raise ValueError(
                "MOPD with orm_weight > 0 requires 'index' (uid) in batch. "
                "Ensure 'uid' is in data.non_tensor_batch."
            )
        # Compute GRPO-style outcome advantage
        A_orm = compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=kwargs["index"],
        )[0]  # returns (advantages, returns)
        A_final = weights * (A_mopd + orm_weight * A_orm)
    else:
        A_final = weights * A_mopd

    # Fix 10: Apply response mask
    A_final = A_final * response_mask

    # Returns = token_level_rewards for interface consistency
    returns = token_level_rewards * response_mask

    return A_final, returns
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_mopd_advantage.py -v
```

Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add verl/trainer/ppo/core_algos.py tests/unit/test_mopd_advantage.py
git commit -m "feat(mopd): add MOPD advantage estimator to core_algos

- Implements MiMo Eq. 7-9: token-level reverse KL advantages
- Supports G-OPD ExOPD mode (lambda scaling + base normalization)
- IS correction with degenerate case fallback (Fix 5)
- ORM outcome reward mixing (α coefficient)
- Tests: 3 unit tests (basic, IS correction, ExOPD)

Implements Fixes 5, 9, 10"
```

---

### Task 2.5: Wire MOPD Kwargs into compute_advantage Dispatch

**Files:**
- Modify: `verl/trainer/ppo/ray_trainer.py:190-217`
- Test: `tests/unit/test_mopd_advantage.py` (extend)

**Step 1: Write failing test for dispatch wiring**

Append to `tests/unit/test_mopd_advantage.py`:

```python
def test_mopd_kwargs_received_via_dispatch():
    """Test that compute_mopd_advantage receives correct kwargs from dispatch."""
    import numpy as np
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import compute_advantage
    from verl.trainer.ppo.core_algos import AdvantageEstimator

    B, T = 4, 10
    data = DataProto.from_single_dict({
        "token_level_rewards": torch.randn(B, T),
        "response_mask": torch.ones(B, T),
        "old_log_probs": torch.randn(B, T),
        "teacher_log_prob": torch.randn(B, T),
    })
    data.non_tensor_batch["uid"] = np.array(["a", "a", "b", "b"])

    # Should not crash — verifies kwargs are passed through
    result = compute_advantage(
        data,
        adv_estimator="mopd",
        config=None,
    )
    assert "advantages" in result.batch
    assert "returns" in result.batch
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_mopd_advantage.py::test_mopd_kwargs_received_via_dispatch -v
```

Expected: FAIL (teacher_log_prob/old_log_probs not passed to estimator)

**Step 3: Add MOPD kwargs to compute_advantage dispatch**

Modify `verl/trainer/ppo/ray_trainer.py` in `compute_advantage()` function,
in the `else` branch (around line 192-215), add MOPD-specific kwargs:

```python
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # MOPD-specific kwargs (pass from batch if available)
        if "teacher_log_prob" in data.batch:
            adv_kwargs["teacher_log_prob"] = data.batch["teacher_log_prob"]
        if "old_log_probs" in data.batch:
            adv_kwargs["old_log_probs"] = data.batch["old_log_probs"]
        if "rollout_log_probs" in data.batch:
            adv_kwargs["rollout_log_probs"] = data.batch["rollout_log_probs"]
        if "base_log_prob" in data.batch:
            adv_kwargs["base_log_prob"] = data.batch["base_log_prob"]

        # Pass MOPD config values if available
        mopd_cfg = getattr(config, "mopd", None) if config else None
        if mopd_cfg is not None:
            adv_kwargs["lambda_val"] = getattr(mopd_cfg, "lambda_val", 1.0)
            adv_kwargs["orm_weight"] = getattr(mopd_cfg, "orm_weight", 0.0)
            adv_kwargs["is_correction"] = getattr(mopd_cfg, "is_correction", True)
            adv_kwargs["is_epsilon_low"] = getattr(mopd_cfg, "is_epsilon_low", 0.1)
            adv_kwargs["is_epsilon_high"] = getattr(mopd_cfg, "is_epsilon_high", 10.0)

        # ... existing optimal token baseline code ...
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_mopd_advantage.py -v
```

Expected: All 4 tests PASS (3 original + 1 new dispatch test)

**Step 5: Commit**

```bash
git add verl/trainer/ppo/ray_trainer.py tests/unit/test_mopd_advantage.py
git commit -m "feat(mopd): wire MOPD kwargs into compute_advantage dispatch

- Pass teacher_log_prob, old_log_probs, rollout_log_probs, base_log_prob
- Forward MOPD config values (lambda, IS bounds, orm_weight)
- Backward compatible (only passes if keys exist in batch)

Fixes Issue 2: compute_advantage dispatch gap"
```

---

### Task 3: Initialize Teacher Workers in Trainer

**Files:**
- Modify: `verl/trainer/ppo/ray_trainer.py:674-850`
- Test: `tests/unit/test_teacher_workers.py`

**Step 1: Write failing test for teacher worker initialization**

Create `tests/unit/test_teacher_workers.py`:

```python
import pytest
import ray
from omegaconf import OmegaConf
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role


def test_teacher_workers_created():
    """Test that teacher worker groups are created when MOPD enabled."""
    config = OmegaConf.create({
        "algorithm": {"mopd": {"enabled": True, "teachers": [
            {"name": "math", "model_path": "/models/math"},
            {"name": "code", "model_path": "/models/code"},
        ]}}
    })

    trainer = RayPPOTrainer(config=config, tokenizer=None, ...)
    trainer.init_workers()

    assert hasattr(trainer, "teacher_wgs")
    assert len(trainer.teacher_wgs) == 2
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_teacher_workers.py -v
```

Expected: FAIL with "AttributeError: 'RayPPOTrainer' object has no attribute 'teacher_wgs'"

**Step 3: Implement teacher worker initialization**

Add to `verl/trainer/ppo/ray_trainer.py` in `init_workers()` (after line 737):

```python
# Create teacher workers for MOPD
if self.config.algorithm.get("mopd", {}).get("enabled", False):
    self.teacher_wgs = {}
    for teacher_cfg in self.config.algorithm.mopd.teachers:
        teacher_worker_config = deepcopy(self.config.actor_rollout_ref)
        teacher_worker_config.model.path = teacher_cfg.model_path

        teacher_cls = RayClassWithInitArgs(
            self.role_worker_mapping[Role.RefPolicy],
            config=teacher_worker_config,
            role=f"Teacher_{teacher_cfg.name}",
        )

        # Use the same resource pool as the actor (global_pool by default)
        resource_pool = self.resource_pool_manager.resource_pool_dict.get(
            teacher_cfg.resource_pool,
            list(self.resource_pool_manager.resource_pool_dict.values())[0]
        )
        teacher_wg = self.ray_worker_group_cls(
            resource_pool=resource_pool,
            ray_cls_with_init=teacher_cls,
            **wg_kwargs,
        )
        teacher_wg.spawn()
        teacher_wg.init_model()
        self.teacher_wgs[teacher_cfg.name] = teacher_wg
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_teacher_workers.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add verl/trainer/ppo/ray_trainer.py tests/unit/test_teacher_workers.py
git commit -m "feat(mopd): initialize teacher workers in RayPPOTrainer

- Create RefPolicy worker groups for each teacher
- Load teacher models with proper resource allocation
- Only initialize when MOPD enabled (backward compatible)

Implements Fix 1: Teachers in RefPolicy workers"
```

---

### Task 5: Implement Sub-Batch Teacher Routing

**Files:**
- Modify: `verl/trainer/ppo/ray_trainer.py:1428-1433`
- Test: `tests/unit/test_teacher_routing.py`

**Note:** Depends on Task 4 (dataset provides `teacher_id`).

**Step 1: Write failing test for sub-batch routing**

Create `tests/unit/test_teacher_routing.py`:

```python
import torch
import numpy as np
from unittest.mock import MagicMock
from verl import DataProto


class MockTeacherWG:
    """Mock teacher worker group for unit testing."""
    def compute_ref_log_prob(self, sub_batch):
        batch_size = sub_batch.batch["responses"].shape[0]
        response_len = sub_batch.batch["responses"].shape[1]
        result = DataProto.from_single_dict({
            "ref_log_prob": torch.randn(batch_size, response_len),
        })
        return result


def test_teacher_log_prob_computation():
    """Test teacher log prob computation with sub-batch routing."""
    batch = DataProto.from_single_dict({
        "input_ids": torch.randint(0, 1000, (4, 128)),
        "responses": torch.randint(0, 1000, (4, 64)),
        "attention_mask": torch.ones(4, 192),
    })
    batch.non_tensor_batch["teacher_id"] = np.array(["math", "math", "code", "code"])

    # Mock teacher worker groups
    teacher_wgs = {"math": MockTeacherWG(), "code": MockTeacherWG()}

    teacher_log_prob = compute_teacher_log_probs(batch, teacher_wgs)

    # Should return stacked tensor [batch, response_len]
    assert teacher_log_prob.shape == (4, 64)
    assert teacher_log_prob.dtype == torch.float32
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_teacher_routing.py -v
```

Expected: FAIL with "NameError: name 'compute_teacher_log_probs' is not defined"

**Step 3: Implement sub-batch routing**

Add to `verl/trainer/ppo/ray_trainer.py` (new method after `_compute_ref_log_prob`):

```python
def _compute_teacher_log_probs(self, batch: DataProto) -> torch.Tensor:
    """Compute teacher log probs with sub-batch routing (Fix 3)."""
    teacher_ids = batch.non_tensor_batch["teacher_id"]
    batch_size = len(teacher_ids)
    response_len = batch.batch["responses"].shape[1]

    # Initialize output tensor (Fix 2: stacked storage)
    teacher_log_probs = torch.zeros(
        batch_size, response_len,
        dtype=torch.float32,
        device=batch.batch["responses"].device
    )

    # Group by teacher_id and forward sub-batches (Fix 3)
    for teacher_name, teacher_wg in self.teacher_wgs.items():
        # Get indices for this teacher (Fix 7: integer tensor)
        mask = teacher_ids == teacher_name
        indices = torch.tensor(np.where(mask)[0], dtype=torch.long)

        if len(indices) == 0:
            continue

        # Select sub-batch (Fix 7: use select_idxs, not select)
        sub_batch = batch.select_idxs(indices)

        # Forward to teacher
        teacher_output = teacher_wg.compute_ref_log_prob(sub_batch)
        sub_log_probs = teacher_output.batch["ref_log_prob"]

        # Scatter back to full batch (Fix 4: correct broadcasting)
        teacher_log_probs[indices] = sub_log_probs

    return teacher_log_probs
```

Modify `fit()` method to call this (replace ref_log_prob computation at line 1430):

```python
if self.use_reference_policy:
    with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
        if hasattr(self, "teacher_wgs"):
            # MOPD: compute teacher log probs
            teacher_log_prob = self._compute_teacher_log_probs(batch)
            batch.batch["teacher_log_prob"] = teacher_log_prob
        else:
            # Standard: compute ref log prob
            ref_log_prob = self._compute_ref_log_prob(batch)
            batch = batch.union(ref_log_prob)
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_teacher_routing.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add verl/trainer/ppo/ray_trainer.py tests/unit/test_teacher_routing.py
git commit -m "feat(mopd): implement sub-batch teacher routing

- Group samples by teacher_id for efficient forwarding
- Use stacked tensor storage (not individual keys)
- Forward only relevant samples to each teacher
- Scatter results back with correct indexing

Implements Fixes 2, 3, 4, 7"
```

---

### Task 4: Add Teacher ID to Dataset

**Files:**
- Modify: `verl/utils/dataset/rl_dataset.py:50-100`
- Test: `tests/unit/test_dataset_teacher_id.py`

**Note:** This task comes before sub-batch routing because routing depends on
dataset providing `teacher_id` in `non_tensor_batch`.

**Step 1: Write failing test for teacher_id in dataset**

Create `tests/unit/test_dataset_teacher_id.py`:

```python
from unittest.mock import MagicMock
from verl.utils.dataset.rl_dataset import RLDataset


def test_dataset_includes_teacher_id():
    """Test that dataset includes teacher_id field."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3]

    dataset = RLDataset(
        data_files=["test_data.jsonl"],
        tokenizer=mock_tokenizer,
        config={"teacher_id_field": "domain"}
    )

    sample = dataset[0]
    assert "teacher_id" in sample
    assert isinstance(sample["teacher_id"], str)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_dataset_teacher_id.py -v
```

Expected: FAIL with "KeyError: 'teacher_id'"

**Step 3: Implement teacher_id extraction**

Modify `verl/utils/dataset/rl_dataset.py` in `__getitem__` method:

```python
def __getitem__(self, idx):
    item = self.data[idx]

    # Extract teacher_id if configured
    teacher_id = None
    if hasattr(self.config, "teacher_id_field") and self.config.teacher_id_field:
        teacher_id = item.get(self.config.teacher_id_field, "default")

    result = {
        "input_ids": ...,
        "attention_mask": ...,
    }

    if teacher_id is not None:
        result["teacher_id"] = teacher_id

    return result
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_dataset_teacher_id.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add verl/utils/dataset/rl_dataset.py tests/unit/test_dataset_teacher_id.py
git commit -m "feat(mopd): add teacher_id field to dataset

- Extract teacher_id from configurable field
- Default to 'default' if field missing
- Backward compatible (optional field)

Supports Fix 3: Sub-batch routing by teacher_id"
```

---

### Task 6: Create Hydra Configuration Files

**Files:**
- Create: `verl/trainer/config/algorithm/mopd.yaml`
- Modify: `verl/trainer/config/ppo_trainer.yaml`
- Test: Manual validation

**Step 1: Create MOPD algorithm config**

Create `verl/trainer/config/algorithm/mopd.yaml`:

```yaml
# @package _global_

algorithm:
  adv_estimator: mopd

  mopd:
    enabled: false
    lambda_val: 1.0
    orm_weight: 0.0
    is_correction: true
    is_epsilon_low: 0.1
    is_epsilon_high: 10.0
    use_base_normalization: false
    base_model_path: null

    teachers: []
    # Example:
    # - name: math
    #   model_path: /models/math-teacher
    #   weight: 1.0
    #   resource_pool: global_pool
    #   log_prob_micro_batch_size: 8
```

**Step 2: Add MOPD to trainer defaults**

Modify `verl/trainer/config/ppo_trainer.yaml`:

```yaml
defaults:
  - actor: fsdp_actor
  - critic: fsdp_critic
  - rollout: vllm_rollout
  - reward: default_reward
  - algorithm: ppo
  - algorithm/mopd: mopd  # Add MOPD config
```

**Step 3: Validate config loading**

```bash
python -c "
from omegaconf import OmegaConf
config = OmegaConf.load('verl/trainer/config/ppo_trainer.yaml')
assert 'mopd' in config.algorithm
print('Config validation: PASS')
"
```

Expected: "Config validation: PASS"

**Step 4: Commit**

```bash
git add verl/trainer/config/algorithm/mopd.yaml verl/trainer/config/ppo_trainer.yaml
git commit -m "feat(mopd): add Hydra configuration files

- Create mopd.yaml with all algorithm parameters
- Integrate into ppo_trainer.yaml defaults
- Backward compatible (enabled: false by default)

Implements Fix 6: Explicit backward compatibility"
```

---

### Task 7: Integration Tests

**Files:**
- Create: `tests/integration/test_mopd_e2e.py`

**Step 1: Write E2E integration test**

Create `tests/integration/test_mopd_e2e.py`:

```python
import pytest
import torch
from omegaconf import OmegaConf
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


@pytest.mark.slow
@pytest.mark.gpu
def test_mopd_training_e2e():
    """Test full MOPD training loop."""
    config = OmegaConf.create({
        "algorithm": {
            "adv_estimator": "mopd",
            "mopd": {
                "enabled": True,
                "lambda_val": 1.0,
                "teachers": [
                    {"name": "math", "model_path": "/models/math"},
                ]
            }
        },
        "trainer": {"total_epochs": 1},
    })

    trainer = RayPPOTrainer(config=config, ...)
    trainer.init_workers()
    trainer.fit()

    assert trainer.global_steps > 0
```

**Step 2: Run integration test**

```bash
pytest tests/integration/test_mopd_e2e.py -v
```

Expected: PASS (full training loop completes)

**Step 3: Commit**

```bash
git add tests/integration/test_mopd_e2e.py
git commit -m "test(mopd): add E2E integration test

- Full training loop with MOPD enabled
- Verifies all components work together
- GPU test (requires CUDA)"
```

---

## Implementation Complete

**Total LOC estimate:** ~550 production + ~180 tests

**All 10 fixes incorporated:**
1. ✅ Teachers in RefPolicy workers (Task 3)
2. ✅ Stacked tensor storage (Task 5)
3. ✅ Sub-batch forwarding (Task 5)
4. ✅ Correct broadcasting (Task 5)
5. ✅ IS correction fallback (Task 2)
6. ✅ Backward compatibility (Task 6)
7. ✅ `select_idxs()` with integer tensor indices (Task 5)
8. ✅ Config validation (Task 1)
9. ✅ Optional rollout_log_probs (Task 2)
10. ✅ Advantage normalization (Task 2)

**Additional fixes from review:**
- ✅ compute_advantage dispatch wiring (Task 2.5)
- ✅ Resource pool API corrected (Task 3)
- ✅ GRPO index validation (Task 2)
- ✅ IS correction test uses non-zero values (Task 2)
- ✅ Mock objects defined in test files (Tasks 4, 5)

**Task execution order:** 1 → 2 → 2.5 → 3 → 4 → 5 → 6 → 7

**Next Steps:**
1. Execute plan using `superpowers:executing-plans` skill
2. Run verification tests
3. Request code review
