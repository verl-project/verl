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
    actor_rollout_ref.actor.optim.muon_tp_mode=blockwise \
    actor_rollout_ref.actor.optim.muon_match_adamw_update_rms=True
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
| `muon_match_adamw_update_rms` | `True` | derive `muon_extra_scale_factor` from `betas[0]` so Muon's update RMS matches AdamW's — see below. |

A `muon_*` field that the installed Megatron-Core build does not declare raises at
build time instead of being silently ignored.

### Learning rate: Muon's effective step is `lr × muon_extra_scale_factor`

An AdamW learning rate is **not** directly reusable. Megatron-Core's default
`muon_extra_scale_factor = 1.0` is *not* AdamW-comparable: carrying an AdamW `lr`
over unchanged yields roughly a **4.4×** larger effective step (`1 / 0.2294`, see
below), which is a common cause of unstable or non-converging Muon runs.

`emerging_optimizers` gives the closed form for the factor that matches AdamW's
update RMS norm:

```
muon_extra_scale_factor = sqrt((1 - beta1) / (1 + beta1))
```

where `beta1` is AdamW's first-moment coefficient — `0.229416` at the default
`beta1 = 0.9` — quoted for orientation only. **Configure the switch, not the
number**: set `muon_match_adamw_update_rms=True` and verl derives the factor from
your `optim.betas[0]` and logs the resolved value on rank 0. A hand-entered
constant is the thing to avoid regardless of how it was obtained — anyone reading
the config can re-derive the switch's value from `beta1`, but not a pasted-in
literal. Setting both `muon_match_adamw_update_rms=True` and an explicit
`muon_extra_scale_factor` raises.

Note that `muon_scale_mode` and `muon_extra_scale_factor` are orthogonal and both
matter: the former normalizes for parameter *shape*, the latter for the
*momentum/EMA*. Sources: emerging_optimizers 0.3.0
`orthogonalized_optimizers/muon.py::get_muon_scale_factor` docstring,
<https://kexue.fm/archives/11267>, <https://arxiv.org/abs/2502.16982>. The often
quoted `0.2` is the value of the *factor* at `beta1 = 0.9`, not a target for any
measured update RMS.

## Notes

- Requires a Megatron-Core build with `emerging_optimizers` support.
- `use_layer_wise_distributed_optimizer=True` is what keeps Muon's peak optimizer
  memory below AdamW at 30B scale; without it the layer-wise buffers are not
  distributed.
