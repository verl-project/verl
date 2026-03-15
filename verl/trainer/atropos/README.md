# Atropos RL Environment Integration

Integrates [Atropos](https://github.com/NousResearch/atropos) RL environments with verl's GRPO training pipeline via three clean reflex hooks and a pluggable reward registry.

Closes #1782.

## Architecture

verl spins up vLLM inference servers and registers endpoints with Atropos. Instead of reading parquet files, verl polls Atropos for scored trajectory batches. GRPO advantages are computed per batch with token-level advantage override support.

Zero core verl files modified. All integration lives in verl/trainer/atropos/.

## Files

- verl_atropos_reflex.py — three reflex hooks that wire verl to Atropos
- reward_registry.py — pluggable reward registry with graceful fallback
- mock_atropos_server.py — mock server for local testing without GPU

## Reward Registry

Solves NotImplementedError for unknown data sources (see issue #5558).
Tries verl built-in first, falls back to registered handlers, then Atropos scoring, then 0.0 gracefully.

Built-in: openai/gsm8k with flexible answer extraction.

## Dependencies

pip install atroposlib vllm verl

## Usage

Start Atropos environment server first, then:
bash examples/grpo_trainer/run_atropos_grpo.sh

## Test Results

End-to-end tested on Kaggle P100. All three hooks confirmed working.
Real scores: [1.0, 1.0, 1.0, 0.7], advantages: [-0.84, -1.08, 0.60, 1.32]
Reward registry: 5/5 unit tests passing.

## Reference

- Atropos: https://github.com/NousResearch/atropos
- Bounty issue: https://github.com/verl-project/verl/issues/1782
