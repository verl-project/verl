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

Tested against mock Atropos server with real Akkadian cuneiform training data. Registration, batch polling, and GRPO advantage normalization all confirmed working. Real scores: [1.0, 1.0, 1.0, 0.7], advantages: [-0.84, -1.08, 0.60, 1.32].

## Reference

- Atropos: https://github.com/NousResearch/atropos
- Axolotl plugin reference: https://github.com/axolotl-ai-cloud/plugin-atropos
