# Muon optimizer (Megatron backend)

This PR exposes Megatron-Core's `TensorParallelMuon` (via `emerging_optimizers`)
in verl's native Megatron backend. Muon is applied to 2D weight matrices; all
other parameters (embeddings, norms, biases, router, lm_head) keep AdamW.

## How to enable

Add the following Hydra overrides to any Megatron GRPO/PPO run (e.g. one of the
scripts under `examples/grpo_trainer/*_megatron.sh`). Values below are the
recommended starting point and mirror the defaults documented in
`verl/trainer/config/optim/megatron.yaml`.

```bash
python3 -m verl.trainer.main_ppo \
    ... \
    actor_rollout_ref.actor.optim.optimizer=muon \
    actor_rollout_ref.actor.optim.use_layer_wise_distributed_optimizer=True \
    actor_rollout_ref.actor.optim.muon_momentum=0.95 \
    actor_rollout_ref.actor.optim.muon_nesterov=False \
    actor_rollout_ref.actor.optim.muon_split_qkv=True \
    actor_rollout_ref.actor.optim.muon_scale_mode=spectral \
    actor_rollout_ref.actor.optim.muon_coefficient_type=quintic \
    actor_rollout_ref.actor.optim.muon_num_ns_steps=5 \
    actor_rollout_ref.actor.optim.muon_tp_mode=blockwise
```

## Key knobs

| Field | Recommended | Meaning |
| --- | --- | --- |
| `optimizer` | `muon` | selects the Muon (emerging) optimizer; Muon on matrices, AdamW fallback on the rest. |
| `use_layer_wise_distributed_optimizer` | `True` | build Megatron's LayerWise distributed optimizer path so Muon's per-layer buffers are distributed (avoids the extra fp32 master clone; keeps memory below AdamW). |
| `muon_momentum` | `0.95` | Muon momentum; tuning it rarely helps. |
| `muon_nesterov` | `False` | Nesterov momentum for the Muon update. |
| `muon_split_qkv` | `True` | orthogonalize per-head QKV projections independently. |
| `muon_scale_mode` | `spectral` | update-scaling mode. |
| `muon_num_ns_steps` | `5` | Newton–Schulz iteration steps for the orthogonalization. |
| `muon_tp_mode` | `blockwise` | tensor-parallel sharding mode for the Muon update. |

`lr` / `weight_decay` are reused from the AdamW settings — Muon does not need a
separate learning rate. A `muon_*` field that the installed Megatron-Core build
does not declare raises at build time instead of being silently ignored.

## Notes

- Requires a Megatron-Core build with `emerging_optimizers` support.
- `use_layer_wise_distributed_optimizer=True` is what keeps Muon's peak optimizer
  memory below AdamW at 30B scale; without it the layer-wise buffers are not
  distributed.
