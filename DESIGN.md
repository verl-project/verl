# Multi-Tenant LoRA Training — Design Document

## Overview

Extend `verl.experimental.fully_async_policy` to support multi-tenant training where multiple users each train their own LoRA adapter on top of a shared base model, within a single SLURM job.

## Architecture Decisions

### Rollout Engine
- **Single vLLM engine with multi-LoRA serving**
- All tenant LoRA adapters are loaded simultaneously in the vLLM engine
- Base model is loaded once and shared across all tenants
- Rollout requests are tagged with `tenant_id` to select the correct adapter

### Training Workers
- **Time-sliced across tenants** on a shared pool of training GPUs
- Whichever tenant's queue has enough samples ready trains next
- LoRA adapter swapping happens on the training GPUs between tenants
- **All adapters kept in GPU memory** (no CPU offload) — rank-32 LoRA on 7B ≈ ~50MB per adapter, negligible overhead

### Message Queues
- **Per-tenant MessageQueue** (one Ray actor per tenant)
- Each tenant has its own independent queue
- Trainer polls all queues and picks the first tenant with enough samples (`require_batches` threshold)

### Weight Sync
- **Per-tenant adapter sync** — only LoRA weights are synced (already the default behavior)
- Base model synced once at startup, never again
- Syncing tenant A's adapter does not block tenant B's rollout
- Uses existing `TensorLoRARequest` mechanism in vLLM

### Staleness
- **Global staleness tracking** — shared across all tenants
- Single `staleness_threshold` applies to the system as a whole

### Hyperparameters & Reward
- **Same hyperparameters** across all tenants (LoRA rank, learning rate, etc.)
- **Same reward function** (`dapo`) across all tenants
- Per-tenant config may be added later

### Datasets
- **Different datasets per tenant** — each tenant specifies their own `train_files` and `val_files`

### Tenants
- **Fixed at launch** — tenant set defined in the sbatch/shell script, cannot change mid-training
- Arbitrary number of tenants supported, starting with 2 for initial implementation

### Tenant Specification Format
Tenants are defined as a comma-separated list in the shell script:
```bash
TENANTS="alice:~/data/alice_train.parquet:~/data/alice_val.parquet,bob:~/data/bob_train.parquet:~/data/bob_val.parquet"
```
Format: `<tenant_name>:<train_file>:<val_file>` separated by commas.

### Checkpointing
- **Per-tenant adapter saves** — each tenant's LoRA adapter checkpointed independently
- Saved to `<CKPTS_DIR>/<tenant_name>/global_step_<version>/`
- Simplest implementation: save after each training step per tenant

### Infrastructure
- **Single SLURM job** launches the multi-tenant coordinator
- One sbatch file, one run script
- Same cluster setup as existing experiments (SCITAS, 4 GPUs, etc.)

## Components to Modify

1. **`fully_async_main.py`** — Parse tenant list, create per-tenant queues and data loaders, pass tenant config to trainer and rollouter
2. **`fully_async_trainer.py`** — Poll multiple queues, swap active LoRA adapter per tenant, per-tenant checkpointing
3. **`fully_async_rollouter.py`** — Tag rollout requests with tenant_id, feed samples to correct per-tenant queue
4. **`message_queue.py`** — No structural change needed (one instance per tenant)
5. **`agent_loop/agent_loop.py`** — Pass tenant_id / LoRA adapter identifier in generation requests
6. **New: experiment shell script (.sh) and sbatch file** — Multi-tenant launch configuration

## Data Flow

```
Tenant A data ──► Rollouter ──► vLLM (adapter A) ──► MessageQueue A ──┐
                                                                       ├──► Trainer (time-sliced)
Tenant B data ──► Rollouter ──► vLLM (adapter B) ──► MessageQueue B ──┘
                                                                          │
                                                              ┌───────────┤
                                                              ▼           ▼
                                                     Sync adapter A   Sync adapter B
                                                     to vLLM          to vLLM
```
