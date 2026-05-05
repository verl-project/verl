# REINFORCE++

REINFORCE++ is a simple, critic-free PG variant that extends REINFORCE with token-level KL penalties and advantage whitening.

Reference: [REINFORCE++: A Simple and Efficient Approach for Aligning Large Language Models](https://arxiv.org/abs/2501.03262).

## Canonical Scripts

| Script                               | Infer | Train | Platform |
|--------------------------------------|-------|-------|----------|
| `run_qwen3_8b_fsdp.sh`          | vLLM  | FSDP  | NVIDIA   |

Switch to the baseline variant by setting `ADV_ESTIMATOR=reinforce_plus_plus_baseline` when running the script.
