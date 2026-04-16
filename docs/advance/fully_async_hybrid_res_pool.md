# Fully Async Hybrid Resource Pool

`fully_async` hybrid resource pool is an opt-in path for the experimental fully async trainer. It does not change the
default behavior of existing async training or `main_ppo`. Unless
`async_training.hybrid_res_pool=True` is set, verl keeps the previous resource-pool behavior unchanged.

This path is only for:

```bash
python3 -m verl.experimental.fully_async_policy.fully_async_main
```

It is not a change to `main_ppo`.

## When To Use It

Enable this mode when you want the fully async trainer to derive logical trainer and rollout pools from
`actor_rollout_ref.partition`. This is useful when the physical allocation is shared but the fully async path should
normalize it into separate logical chunks for trainer and rollout.

## Why It Exists

The original fully async path assumes a uniform `nnodes * n_gpus_per_node` resource view. That is sufficient for
balanced allocations such as `8:8`, but it is awkward for asymmetric cross-machine splits such as `12:4` or `4:12`.
`hybrid_res_pool` keeps the old path unchanged by default and only adds an opt-in normalization step for the
experimental fully async entrypoint.

## Requirements

- `async_training.hybrid_res_pool` defaults to `False`.
- Old behavior stays unchanged unless `async_training.hybrid_res_pool=True`.
- This path is async-only and requires `actor_rollout_ref.rollout.mode=async`.
- `actor_rollout_ref.hybrid_engine` must stay `False`.
- `actor_rollout_ref.partition` is required when this mode is enabled.
- Non-`naive` checkpoint backends are unsupported on this path. Set
  `actor_rollout_ref.rollout.checkpoint_engine.backend=naive`.

## Partition Format

In the fully async path, `actor_rollout_ref.partition` is interpreted as a trainer/rollout split.

- `8_4-4`: logical trainer pool `[4,4,4]`, logical rollout pool `[4]`
- `4-4_8`: logical trainer pool `[4]`, logical rollout pool `[4,4,4]`

Use the variant that matches whether you want more logical chunks assigned to trainer or rollout.

## Normalization Rule

The partition is parsed on the physical topology and then expanded into equal logical chunks by GCD.

| physical partition | logical trainer pool | logical rollout pool |
| --- | --- | --- |
| `8_4-4` | `[4, 4, 4]` | `[4]` |
| `4-4_8` | `[4]` | `[4, 4, 4]` |

## Minimal Example

The example below shows only the inputs that matter for the hybrid resource pool switch. Keep the rest of your fully
async config unchanged.

```bash
python3 -m verl.experimental.fully_async_policy.fully_async_main \
  --config-path=config \
  --config-name=fully_async_ppo_trainer.yaml \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.hybrid_engine=False \
  actor_rollout_ref.rollout.checkpoint_engine.backend=naive \
  trainer.nnodes=2 \
  trainer.n_gpus_per_node=8 \
  rollout.nnodes=1 \
  rollout.n_gpus_per_node=8 \
  async_training.hybrid_res_pool=True \
  +actor_rollout_ref.partition=8_4-4
```

To flip the logical layout, use:

```bash
+actor_rollout_ref.partition=4-4_8
```

## Notes

- This switch only affects the experimental fully async entrypoint.
- If you do not enable the switch, existing fully async runs continue to behave as before.
- If you need a checkpoint backend other than `naive`, this path is not supported yet.
- Enabling `hybrid_res_pool` without `actor_rollout_ref.partition` is an error.
