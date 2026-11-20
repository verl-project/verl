# Native FSDP2 Muon implementation plan

Last updated: 11/09/2026

Historical 2026-11-06 plan. The initial direct Dion transport was revised after
GPU numerical failures on 2026-11-09. See [the implementation guide](fsdp2_muon.md)
and [GPU validation](fsdp2_muon_gpu_validation.md) for the current full-matrix
gather path and measured evidence.

Goal: optional native FSDP2 Muon with AdamW fallback; preserve the RL training pipeline.

Baseline: verl main 689fa1a2a1a35fbfee6cdd8cabdec0c15688ccaf, initially clean checkout.
The supplied workspace was empty; the official repository was cloned here without modifying other checkouts.

Architecture: reuse pinned Microsoft Dion Muon full-matrix distributed implementation, not Megatron.
Keep FP32 stored parameters/optimizer states; allow FP32 or BF16 forward/reduction policies.
Dion's orthogonalization communicates/computes BF16 even with FP32 stored parameters.
Use one standard optimizer, explicit mixed groups, and validation hooks; keep existing clipping,
scheduling, checkpoint manager and rollout sync. Unknown/special parameters conservatively use AdamW.

- [x] Write CPU tests for config defaults/overrides, dependency isolation, grouping, rejected combinations,
  real dependency updates, scheduler and state restore. Observe failure before implementing.
- [x] Add configuration fields to dataclass/YAML and regenerate reference YAML via repository generator.
- [x] Add grouping/validation/optional Dion builder; wire native FSDP engine context into generic builder.
- [x] Supply GPU-only fully_shard acceptance, optimizer and training comparisons, process restart checkpoint test.
- [x] Run available CPU tests and static checks, inspect diff, document exact commands/results and limitations.

The initial scope excluded GPU rental and execution. On 2026-11-09 the user
provided an instance and authorized synthetic GPU acceptance, followed by a
small-model GRPO comparison and restart validation. See the
[dated GRPO validation record](fsdp2_muon_grpo_validation.md) for results and
submission integration with upstream main.
