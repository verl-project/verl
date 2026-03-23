# Self-Distillation Policy Optimization (SDPO)

Last updated: 03/03/2026.

SDPO is a policy optimization variant that augments actor updates with self-distillation from successful trajectories in the same rollout batch.

Paper: [Self-Distillation Policy Optimization](https://arxiv.org/abs/2601.20802).

## Core Idea

At each training step:

1. Rollout responses are grouped by sample `uid`.
2. Successful responses (above `success_reward_threshold`) are reused as demonstrations.
3. Optionally, environment feedback is included in the reprompt.
4. A teacher prompt is built per sample and concatenated with the original response tokens.
5. The actor is updated with PPO-style optimization where policy loss is replaced by SDPO distillation loss (`loss_mode=sdpo`).
6. A colocated teacher is updated with EMA from the student weights.

## Key Configs

- `actor_rollout_ref.actor.policy_loss.loss_mode: sdpo`
- `actor_rollout_ref.actor.self_distillation.full_logit_distillation`
- `actor_rollout_ref.actor.self_distillation.distillation_topk`
- `actor_rollout_ref.actor.self_distillation.alpha`
- `actor_rollout_ref.actor.self_distillation.success_reward_threshold`
- `actor_rollout_ref.actor.self_distillation.teacher_regularization` (`ema`, `trust_region`, or `none`)
- `actor_rollout_ref.actor.self_distillation.teacher_update_rate` (or `ema_update_rate` alias)
- `actor_rollout_ref.actor.self_distillation.include_environment_feedback`
- `actor_rollout_ref.actor.self_distillation.environment_feedback_only_without_solution`

## Current Constraints in verl

- SDPO is only supported with legacy worker mode (not model-engine worker mode).
- SDPO requires `fsdp` / `fsdp2` actor strategy.
- SDPO cannot be combined with a separate KL reference policy (`use_kl_in_reward` or `use_kl_loss` reference path).
- Distillation with multimodal actor inputs is currently not supported.
- Trust-region teacher regularization requires `actor_rollout_ref.actor.use_fused_kernels=false`.

## Minimal Usage

Use the preset config:

```bash
python3 -m verl.trainer.main_ppo --config-name sdpo
```

Or override from `ppo_trainer`:

```bash
python3 -m verl.trainer.main_ppo \
  actor_rollout_ref.actor.policy_loss.loss_mode=sdpo \
  actor_rollout_ref.actor.self_distillation.ema_update_rate=0.05
```
