# Dynamic Speculative Decoding for SPEC-RL

Optimize speculative rollout verification through **dynamic window prediction** and **incremental verification**, targeting 20-30% reduction in forward token computation and **~2.8×** training speedup over non-speculative baseline.

> **Paper**: [SPEC-RL: Speculative Decoding for Reinforcement Learning](https://arxiv.org/abs/2509.23232)  
> **Base Code**: [ShopeeLLM/Spec-RL](https://github.com/ShopeeLLM/Spec-RL)

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Experiment Workflow](#experiment-workflow)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Core Modules](#core-modules)
- [Results & Analysis](#results--analysis)

---

## Overview

### What This Project Does

SPEC-RL accelerates RL training by reusing previous epoch rollouts as draft sequences. This project further optimizes it by:

| Optimization | Mechanism | Expected Gain |
|---|---|---|
| **Per-Prompt Window Prediction** | Different prompts get different verification windows based on historical acceptance patterns | ~20-30% fewer forward tokens |
| **Two-Round Incremental Verification** | Verify predicted window first; only continue if all accepted | Early termination for short-lived sequences |
| **Adaptive Safety Factor** | Dynamically adjust acceptance boundary based on training stability | Better trade-off between speed and acceptance |

### Key Insight

Different prompts have dramatically different acceptance rates. A single global EMA wastes compute on short-lived prompts and under-utilizes stable prompts. Per-prompt P75 prediction captures this heterogeneity.

---

## Prerequisites

### Hardware

| Requirement | Spec |
|---|---|
| GPU | NVIDIA H100 80GB (tested on 8× H100) |
| GPU Driver | >= 535.54.03 |
| CUDA | >= 12.4 |
| System RAM | >= 256 GB |

### Software

```bash
# Python environment (Conda recommended)
conda create -n spec_rl python=3.10
conda activate spec_rl

# Install SPEC-RL dependencies (from base repo)
cd /path/to/Spec-RL
pip install -e .

# Additional experiment dependencies
pip install pandas pyarrow matplotlib numpy
```

### Data & Model

Download offline to avoid HuggingFace network issues:

```bash
# Model: Qwen3-1.7B-Base
/path/to/Spec-RL/model/
└── Qwen3-1.7B-base/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json

# Dataset: DeepMath-6K
/path/to/Spec-RL/data/
└── deepmath/
    ├── train_sample_6144.parquet
    └── test.parquet

# Reward function
/path/to/Spec-RL/custom_reward/verl_math_verify.py
```

### Critical: H100 NCCL Environment

**Must** set these before every run:

```bash
export NCCL_IB_DISABLE=1          # Disable InfiniBand (single node)
export NCCL_P2P_DISABLE=1         # Disable P2P (H100 requirement)
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
```

---

## Quick Start

### 1. Verify Setup

```bash
cd /path/to/Spec-RL
chmod +x verify_setup.sh
bash verify_setup.sh
```

Expected output:
```
[OK] Model: /path/to/Spec-RL/model/Qwen3-1.7B-base
[OK] Train data: /path/to/Spec-RL/data/deepmath/train_sample_6144.parquet
[OK] Reward function
[OK] Training script
All required paths found! Ready to run experiments.
```

### 2. Run 1-Step Sanity Check

```bash
chmod +x run_experiment_v4.sh
bash run_experiment_v4.sh short_test
```

This runs **1 training step** with conservative parameters (matching verified working configuration). Should complete in ~3 minutes.

### 3. Run Full Experiment

```bash
# Small scale (60 steps, batch=128, response=512)
bash run_experiment_v4.sh phase1_small

# Full scale (60 steps, batch=512, response=4096) - if small succeeds
bash run_experiment_v4.sh phase1_full
```

---

## Project Structure

```
spec_rl_experiment/
├── run_experiment_v4.sh              # Main launcher (RECOMMENDED)
├── verify_setup.sh                   # Environment verification
│
├── verl/utils/dynamic_spec/          # Core optimization modules
│   ├── __init__.py
│   ├── window_predictor.py           # 4 prediction strategies
│   ├── incremental_verify.py         # Two-round verification
│   ├── experiment_logger.py          # 23-metric logging
│   └── spec_cut_v2.py                # Enhanced spec_cut
│
├── experiments/
│   ├── run_phase2_1.py               # Offline predictor comparison
│   └── analyze_results.py            # Results analysis & report
│
└── README.md                         # This file
```

---

## Experiment Workflow

```
Phase 0: Sanity Check (short_test)
    └── 1 step, conservative params
        └── Verify NCCL/GPU works

Phase 1: Baseline (phase1_small / phase1_full)
    └── 60 steps, SPEC-RL with speculative decoding
        └── Collect per-prompt acceptance history

Phase 2: Offline Analysis (run_phase2_1.py)
    └── Compare 4 window predictors on historical data
        └── Select best predictor

Phase 3: Online Optimization
    └── Integrate dynamic spec into training loop
        └── Measure speedup vs baseline

Phase 4: Analysis (analyze_results.py)
    └── Generate comparison report
```

---

## Configuration Reference

### v4 Script Parameters

The `run_experiment_v4.sh` supports three preset configurations:

| Preset | Steps | Batch | Response | Rollout N | GPUs | Purpose |
|---|---|---|---|---|---|---|
| `short_test` | 1 | 128 | 512 | 4 | 8 | Sanity check (~3 min) |
| `phase1_small` | 60 | 128 | 512 | 4 | 8 | Conservative baseline (~3 hrs) |
| `phase1_full` | 60 | 512 | 4096 | 8 | 8 | Full configuration (~6 hrs) |

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `WORKING_DIR` | `/home/user/Spec-RL` | Project root path |
| `MODEL_PATH` | `$WORKING_DIR/model` | Model directory parent |
| `MODEL_NAME` | `Qwen3-1.7B-base` | Model subdirectory name |
| `DATA_PATH` | `$WORKING_DIR/data` | Data directory parent |
| `DATASET_NAME` | `deepmath` | Dataset subdirectory |
| `TRAIN_FILE` | `train_sample_6144` | Training file name (no .parquet) |
| `NUM_GPU` | `8` | Number of GPUs |
| `CHECKPOINT_PATH` | `$WORKING_DIR/checkpoints` | Checkpoint save location |

Override any variable:
```bash
export MODEL_NAME=Qwen3-8B-base
export NUM_GPU=4
bash run_experiment_v4.sh short_test
```

### H100 Required Environment

Always export before running:

```bash
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

---

## Troubleshooting

### NCCL Error: `Cuda failure 1 'invalid argument'`

**Cause**: GPU resource conflict or incompatible NCCL settings for H100.

**Fix**:
```bash
# 1. Kill all lingering processes
pkill -9 -f "python"; pkill -9 -f "ray::"; ray stop --force; sleep 5

# 2. Verify GPU is clean
nvidia-smi  # Should show no python processes

# 3. Set NCCL env
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# 4. Use conservative parameters (v4 short_test)
bash run_experiment_v4.sh short_test
```

### FileNotFoundError: Reward function

**Cause**: `WORKING_DIR` not set, paths resolved incorrectly.

**Fix**:
```bash
export WORKING_DIR=/absolute/path/to/Spec-RL
bash run_experiment_v4.sh short_test
```

### OOM During Validation

**Cause**: `max_response_length` or `batch_size` too large for available GPU memory.

**Fix**: Use `short_test` or `phase1_small` preset with smaller parameters:
```bash
bash run_experiment_v4.sh phase1_small  # batch=128, resp=512
```

### Ray Worker Fails to Start

**Cause**: Stale Ray processes from previous runs.

**Fix**: `run_experiment_v4.sh` includes automatic cleanup. If manual:
```bash
ray stop --force
pkill -9 -f "ray::"
rm -rf /tmp/ray
```

---

## Core Modules

### window_predictor.py

Implements 4 strategies for predicting verification window size:

| Predictor | Method | Best For |
|---|---|---|
| `GlobalEMAPredictor` | Single EMA across all prompts | Baseline |
| `PerPromptEMAPredictor` | Per-prompt EMA | Heterogeneous prompts |
| `PerPromptP75Predictor` | Per-prompt P75 + safety factor | **Recommended** |
| `AdaptiveSFPredictor` | Dynamic safety based on log_ratio trend | Adaptive scenarios |

### incremental_verify.py

Two-round verification protocol:
1. **Round 1**: Forward only predicted window `[0, L*]`
   - If rejected: return early (save `R - L*` tokens)
   - If all accepted: proceed to Round 2
2. **Round 2**: Forward remaining `[L*, R]` with KV cache reuse

### experiment_logger.py

Collects 23 metrics including:
- Timing: `step_time_ms`, `verify_time_ms`, `generation_time_ms`
- Quality: `cut_idx_mean`, `accept_rate`, `window_hit_rate`
- Efficiency: `forward_tokens_saved`, `early_stop_rate`
- Per-prompt: `cut_idx_per_prompt`, `log_ratio_mean/std`

---

## Results & Analysis

After running experiments:

```bash
# Compare results across phases
python experiments/analyze_results.py \
    --results_dir experiments/results \
    --output experiments/results/final_report.md
```

Expected metrics to compare:

| Metric | No-Spec | SPEC-RL | +Dynamic Window (Target) |
|---|---|---|---|
| Training Speedup | 1.0× | ~2.3× | **~2.8×** |
| Forward Tokens | 100% | 100% | **70-80%** |
| Window Hit Rate | N/A | N/A | > 40% |
| Early Stop Rate | N/A | N/A | > 30% |

---

## Citation

```bibtex
@article{specrl2025,
  title={SPEC-RL: Speculative Decoding for Reinforcement Learning},
  author={Shopee LLM Team},
  journal={arXiv preprint arXiv:2509.23232},
  year={2025}
}
```

## License

Same as the base SPEC-RL repository.
