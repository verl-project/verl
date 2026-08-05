# SAPO (Smooth Advantage Policy Optimization)

SAPO replaces PPO's ratio clipping with a smooth, `tau`-parameterized surrogate objective.

Reference: [Revisiting Policy Gradient Methods for Large Language Models](https://arxiv.org/pdf/2511.20347).

## Canonical Scripts

| Script                          | Infer | Train    | Platform        |
|---------------------------------|-------|----------|-----------------|
| `run_qwen3_8b_fsdp.sh`          | vLLM  | FSDP2    | Ascend          |
| `run_qwen3_30b_a3b_fsdp.sh`     | vLLM  | FSDP2    | NVIDIA          |
| `run_qwen3_30b_a3b_megatron.sh` | vLLM  | Megatron | NVIDIA / Ascend |

Platform and inference backend are selected at runtime via the `DEVICE` and
`INFER_BACKEND` env vars, not by separate per-platform scripts.

## Key Flags

- `actor_rollout_ref.actor.policy_loss.loss_mode=sapo`
- `actor_rollout_ref.actor.tau_pos=1.0`
- `actor_rollout_ref.actor.tau_neg=1.05`

Note: `tau_pos`/`tau_neg` live on the actor config, not under `policy_loss` --
`compute_policy_loss_sapo` reads `config.tau_pos` off `ActorConfig`. Overriding
them under `policy_loss` silently has no effect.

Note: SAPO disables ratio clipping; no `clip_ratio_low/high` needed.
