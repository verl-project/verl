# Multi-Token-Prediction (MTP) Training

MTP uses an auxiliary token-prediction head (speculative / draft head) during training. Currently supported on MiMo-7B-RL with Megatron backend.

## Canonical Scripts

| Script                                                                    | Infer  | Train    | Mode                              | Platform |
|---------------------------------------------------------------------------|--------|----------|-----------------------------------|----------|
| `run_mimo_7b_mtp_megatron.sh`                                             | SGLang | Megatron | Sync hybrid-engine                | NVIDIA / Ascend |
| `run_mimo_7b_mtp_rl_vllm_sgl_megatron.sh`                                 | SGLang / vLLM | Megatron | Sync hybrid-engine, slime-aligned RL/EAGLE setup | NVIDIA |
| `run_mimo_7b_mtp_fully_async_megatron_multinode.sh`                       | SGLang | Megatron | Fully-async split-placement (DAPO)| NVIDIA   |

IMPORTANT: after downloading MiMo-7B-RL, set `max_position_embeddings: 32768` in its `config.json`.

## Key Flags

- `actor_rollout_ref.model.mtp.enable=True`
- `actor_rollout_ref.model.mtp.enable_train=True`
- `actor_rollout_ref.model.mtp.mtp_loss_scaling_factor=0.1`
- `actor_rollout_ref.model.mtp.detach_encoder=True`

## Ascend synchronous training

Use the same canonical script with `DEVICE=npu`. The Ascend mode defaults to four NPUs,
Megatron TP 2, SGLang TP 1, two rollouts per prompt, 100 training steps, graph-mode
rollout, and direct paired HCCL weight synchronization.

```bash
DEVICE=npu \
NPUS_PER_NODE=4 \
MODEL_PATH=/path/to/MiMo-7B-RL \
DATA_ROOT=/path/to/math \
bash examples/mtp_trainer/run_mimo_7b_mtp_megatron.sh
```

`DATA_ROOT` must contain `train.parquet` and `test.parquet`. Checkpoints are written to
`OUTPUT_DIR` and timestamped console logs to `LOG_DIR`. See
[verl-ascend-recipe issue #20](https://github.com/verl-project/verl-ascend-recipe/issues/20)
for the validated environment, training log, and 100-step results.

## Multi-node fully-async layout

The `*_multinode.sh` variant uses the fully-async one-step-off trainer
(`verl.experimental.fully_async_policy.fully_async_main`). Scale it via:

```bash
TRAIN_NNODES=4 TRAIN_NGPUS_PER_NODE=8 \
ROLLOUT_NNODES=4 ROLLOUT_NGPUS_PER_NODE=8 \
bash examples/mtp_trainer/run_mimo_7b_mtp_fully_async_megatron_multinode.sh
```

Defaults to a single-node 4+4 split (trainer + rollout) for a smoke-test,
matching the historical `..._math_megatron_4_4.sh` layout.
