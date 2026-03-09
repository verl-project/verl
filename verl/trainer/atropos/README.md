# Atropos RL Environment Integration

Integrates [Atropos](https://github.com/NousResearch/atropos) RL environments with verl's GRPO training pipeline.

## Overview

Atropos provides a multi-environment RL rollout API. verl manages inference and training, Atropos manages environments.

## Architecture

- verl spins up vLLM inference servers and registers endpoints with Atropos via POST /register
- verl polls GET /batch for scored trajectory data instead of reading parquet files
- GRPO advantages computed per batch; token-level advantage overrides supported
- verl manages policy weight updates back to vLLM after each training step

## Dependencies

pip install atroposlib vllm verl

## Usage

Start Atropos environment server first, then run:

bash examples/grpo_trainer/run_atropos_grpo.sh

## Results

30-step simulation confirmed registration, batch polling, and GRPO advantage computation working end to end. avg_score improved from 0.189 to 0.941 over 30 steps.

## Reference

- Atropos: https://github.com/NousResearch/atropos
- Axolotl plugin reference: https://github.com/axolotl-ai-cloud/plugin-atropos
