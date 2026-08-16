# CISPO

CISPO (Clipped IS-weight Policy Optimization) is a policy-loss variant that decouples the lower/upper clip ratios to stabilize IS-ratio-weighted updates, used in MiniMax-M1.

Reference: [MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585).

## Canonical Scripts

| Script                               | Infer | Train | Platform |
|--------------------------------------|-------|-------|----------|
| `run_qwen3_8b_fsdp.sh`               | vLLM        | FSDP     | NVIDIA    |
| `run_qwen2_5_0_5b_megatron.sh`       | vLLM-Ascend | Megatron | Ascend NPU |

## Key Flags

- `actor_rollout_ref.actor.policy_loss.loss_mode=cispo`
- `actor_rollout_ref.actor.clip_ratio_low=10` (effectively unclamped on lower side)
- `actor_rollout_ref.actor.clip_ratio_high=0.2`

## Megatron + vLLM-Ascend

```bash
MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct \
DATA_ROOT=/path/to/data \
NPUS_PER_NODE=4 \
bash examples/cispo_trainer/run_qwen2_5_0_5b_megatron.sh
```

Ensure the container provides enough `/dev/shm` capacity for the configured weight-transfer bucket.
See [verl-ascend-recipe issue #17](https://github.com/verl-project/verl-ascend-recipe/issues/17)
for the validated environment, training logs, and 100-step results.
