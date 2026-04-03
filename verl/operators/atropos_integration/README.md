# Atropos Integration for verl

This module provides integration between [verl](https://github.com/verl-project/verl) and [Atropos](https://github.com/NousResearch/atropos) for GRPO training with external rollouts.

## Overview

Atropos is an LLM RL environment system that provides distributed rollouts. Instead of verl generating rollouts internally, Atropos environments generate rollouts and send back scored data. This integration allows verl to:

1. Spin up vLLM inference servers and provide endpoints to Atropos
2. Manage policy weight updates to the inference server  
3. Use GRPO with optional token-level advantage overrides from Atropos

## Bounty

This integration is part of [verl#1782](https://github.com/verl-project/verl/issues/1782) - **$2,500 bounty** funded by Nous Research.

## Architecture

```
verl (driver)
  ├── vLLM server (managed by verl)
  │     └── /generate endpoint → Atropos env calls this
  └── Atropos API client
        ├── POST /register → announce training step/batch_size
        ├── GET /batch → receive prompts
        └── POST /scored_data → receive (tokens, scores, advantages)
```

## Requirements

- verl installed from source
- Atropos installed (`pip install atropos`)
- vLLM for inference
- Ray for distributed training

## Quick Start

### 1. Start Atropos API Server

```bash
# Follow Atropos setup instructions
# https://github.com/NousResearch/atropos
python -m atroposlib.api.server
```

### 2. Run GRPO Training

```bash
cd examples/atropos_integration
bash run_atropos_grpo.sh
```

## Integration Components

### Core Files

- `atropos_api_client.py` - Client for Atropos API communication
- `vllm_server_manager.py` - vLLM server lifecycle management
- `verl_integration/atropos_async_rollout.py` - Async rollout manager for Ray
- `verl_integration/atropos_worker.py` - Ray worker for Atropos rollouts

### API Client

The `AtroposAPIClient` handles:
- Server health checks
- Trainer registration
- Batch retrieval
- Scored data submission

### vLLM Server Manager

Manages:
- Server launch and cleanup
- Health monitoring
- Weight updates (via server restart)
- Port management

## Usage with verl Trainer

After the integration is merged into verl, you can run:

```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.rollout.name=atropos \
    actor_rollout_ref.rollout.atropos_url=http://localhost:8000 \
    actor_rollout_ref.rollout.vllm_port=8100 \
    data.train_files=YOUR_DATA.parquet \
    data.train_batch_size=16 \
    trainer.total_epochs=10
```

## Key Differences from Standard verl

1. **Rollout Generation**: Instead of verl generating responses internally, Atropos environments generate rollouts and send back scored data
2. **Reward Signal**: Token-level advantages come from Atropos (optional override)
3. **vLLM Management**: verl is responsible for spinning up/down vLLM servers and updating weights

## References

- [verl repository](https://github.com/verl-project/verl)
- [Atropos repository](https://github.com/NousResearch/atropos)
- [Atropos API spec](https://github.com/NousResearch/atropos/tree/main/atroposlib/api)
- [Original bounty issue](https://github.com/verl-project/verl/issues/1782)
