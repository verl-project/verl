# Multi-Token-Prediction (MTP) Training

MTP uses an auxiliary token-prediction head (speculative / draft head) during training. Currently supported on MiMo-7B-RL with Megatron backend.

## Canonical Scripts

| Script                                                                    | Infer  | Train    | Mode                              | Platform |
|---------------------------------------------------------------------------|--------|----------|-----------------------------------|----------|
| `run_mimo_7b_mtp_megatron.sh`                                             | SGLang | Megatron | Sync hybrid-engine                | NVIDIA   |
| `run_mimo_7b_mtp_rl_vllm_sgl_megatron.sh`                                 | SGLang / vLLM | Megatron | Sync hybrid-engine, slime-aligned RL/EAGLE setup | NVIDIA |

IMPORTANT: after downloading MiMo-7B-RL, set `max_position_embeddings: 32768` in its `config.json`.

## Key Flags

- `actor_rollout_ref.model.mtp.enable=True`
- `actor_rollout_ref.model.mtp.enable_train=True`
- `actor_rollout_ref.model.mtp.mtp_loss_scaling_factor=0.1`
- `actor_rollout_ref.model.mtp.detach_encoder=True`

## Multi-node fully-async layout

The `*_multinode.sh` variant uses the fully-async one-step-off trainer
(`recipe.fully_async_policy.fully_async_main`), pinned to verl v0.9.1. Scale it via:

```bash
TRAIN_NNODES=4 TRAIN_NGPUS_PER_NODE=8 \
ROLLOUT_NNODES=4 ROLLOUT_NGPUS_PER_NODE=8 \
bash recipe/fully_async_policy/shell/run_mimo_7b_mtp_fully_async_megatron_multinode.sh
```

Defaults to a single-node 4+4 split (trainer + rollout) for a smoke-test,
matching the historical `..._math_megatron_4_4.sh` layout.
