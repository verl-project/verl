# PR 1 — Placement search + Ray pool wiring + `main_ppo` init

## Goal

Deliver rack-aware **placement search** end-to-end for the trainer resource pool:

- Discover rack → node layout from Ray (`get_cluster_placement_racks`).
- Optionally pin placement groups with `RACK_*` bundle resources and/or Ray NodeIDs (`_soft_target_node_id`).
- Build `resource_pool_spec` and optional `selected_nodes` via `init_pool_degrees_and_spec` when `trainer.enable_placement=true`.
- Optional: copy pins to `reward_pool` when `placement_reward_pool_follows_trainer=true` and shapes match.

**Out of scope for this PR:** analytical auto-parallel (`verl/utils/auto_parallelization.py`), NPU-specific worker/device tweaks, extra launch scripts, and non-placement docs (those are PR 2 / PR 3).

## Files in this PR

| File | Change |
|------|--------|
| `verl/utils/placement.py` | **New** — placement search, validators, `init_pool_degrees_and_spec`, `ParallelismDegrees` (for degree validation when Megatron config exists). |
| `verl/single_controller/ray/base.py` | `RayResourcePool` / `ResourcePoolManager` — rack labels, node ID hints, `validate_rack_pinned_pools_feasible`. |
| `verl/trainer/main_ppo.py` | `init_resource_pool_mgr` — reads `trainer.enable_placement`, `placement_pin_*`, `placement_rack_bundle_resource_map`, `placement_reward_pool_follows_trainer`. |
| `verl/trainer/config/ppo_trainer.yaml` | Trainer keys under `trainer:` for placement. |
| `verl/trainer/config/ppo_megatron_trainer.yaml` | Same keys (Megatron entry config). |

## Config keys (reference)

- `trainer.enable_placement`
- `trainer.placement_pin_rack_resources`
- `trainer.placement_pin_node_affinity`
- `trainer.placement_rack_bundle_resource_map`
- `trainer.placement_reward_pool_follows_trainer`

## How to build this branch locally

From a clean branch off upstream `main` (or your fork default):

```bash
git checkout -b pr1/placement-ray-main-ppo
git add \
  verl/utils/placement.py \
  verl/single_controller/ray/base.py \
  verl/trainer/main_ppo.py \
  verl/trainer/config/ppo_trainer.yaml \
  verl/trainer/config/ppo_megatron_trainer.yaml
git status   # review
git commit -m "feat(planner): rack-aware placement search and Ray PG wiring"
```

## Smoke / review checklist

- [ ] `trainer.enable_placement=false` (default) — behavior matches vanilla verl (flat `resource_pool_spec`).
- [ ] Single-node with `enable_placement=true` and Ray `RACK_*` on head — job starts; logs show placement selection or flat fallback if no racks.
- [ ] Multi-node (optional): `placement_pin_rack_resources` / `placement_pin_node_affinity` match your Ray cluster layout.

## PR title (suggested)

`feat: rack-aware placement search for Ray resource pools (placement.py + main_ppo + base)`
