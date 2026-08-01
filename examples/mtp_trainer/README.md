# Multi-Token-Prediction (MTP) Training

MTP uses an auxiliary token-prediction head (speculative / draft head) during training. It is supported on MiMo-7B-RL and on NVIDIA Nemotron 3 Super through Megatron-Core's native `HybridModel` MTP implementation.

## Canonical Scripts

| Script                                                                    | Infer         | Train    | Mode                                                | Platform |
|---------------------------------------------------------------------------|---------------|----------|-----------------------------------------------------|----------|
| `run_mimo_7b_mtp_megatron.sh`                                             | SGLang        | Megatron | Sync hybrid-engine                                  | NVIDIA   |
| `run_mimo_7b_mtp_rl_vllm_sgl_megatron.sh`                                 | SGLang / vLLM | Megatron | Sync hybrid-engine, slime-aligned RL/EAGLE setup    | NVIDIA   |
| `run_mimo_7b_mtp_fully_async_megatron_multinode.sh`                       | SGLang        | Megatron | Fully-async split-placement (DAPO)                  | NVIDIA   |
| `examples/sft/gsm8k/run_nemotron_3_super_megatron.sh`                      | N/A           | Megatron | SFT with native Megatron-Core `HybridModel` MTP     | NVIDIA   |

IMPORTANT: after downloading MiMo-7B-RL, set `max_position_embeddings: 32768` in its `config.json`.

## NVIDIA Nemotron 3 Super dependencies

The native `HybridModel` SFT launcher requires this exact dependency snapshot:

- Megatron Bridge commit [`1f12931e2f34ec26f578a4cffe15adc06f71a5a2`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/1f12931e2f34ec26f578a4cffe15adc06f71a5a2)

- Megatron Core **0.19.0** at commit [`cd4afffa648426a959dc7cb1e24b5ce7d0c3ff54`](https://github.com/NVIDIA/Megatron-LM/commit/cd4afffa648426a959dc7cb1e24b5ce7d0c3ff54). A Git checkout reports `0.19.0+cd4afff`; pin the commit instead of relying on the version number alone.

The launcher is `examples/sft/gsm8k/run_nemotron_3_super_megatron.sh`. It selects Megatron Bridge with `engine.use_mbridge=True` and `engine.vanilla_mbridge=False`.

## Key Flags

- `actor_rollout_ref.model.mtp.enable=True`
- `actor_rollout_ref.model.mtp.enable_train=True`
- `actor_rollout_ref.model.mtp.mtp_loss_scaling_factor=0.1`
- `actor_rollout_ref.model.mtp.detach_encoder=True`

For SFT, use the equivalent `model.mtp.*` keys. Native Megatron-Core `HybridModel` MTP requires both `model.mtp.enable=True` and `model.mtp.enable_train=True`; enabling the module without training it is not supported by this path.

`detach_encoder=True` detaches the base-model hidden state only for the auxiliary MTP loss. It therefore isolates auxiliary MTP gradients to the MTP module, while the normal SFT language-model loss continues to train the base model.

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
