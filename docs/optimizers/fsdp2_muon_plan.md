# Native FSDP2 Muon implementation plan

Last updated: 11/06/2026.

This document outlines proposed Muon support for verl's native FSDP2 backend.
It does not add an optimizer implementation or change training behavior.
Related discussion: [#8549](https://github.com/verl-project/verl/issues/8549).

## Proposed scope

- Make Muon opt-in through the native FSDP2 optimizer configuration; retain AdamW as the default.
- Evaluate Microsoft Dion's distributed Muon implementation as an optional dependency, with a pinned version.
- Apply Muon to supported matrix parameters and AdamW to embeddings, output heads, norms, biases, and unsupported parameter roles.
- Use global parameter shapes and FSDP2 sharding metadata; do not independently orthogonalize local parameter shards.
- Preserve the existing gradient clipping, LR scheduling, checkpoint, and rollout weight synchronization paths.
- Define supported dtypes, layouts, and offload settings explicitly, and reject unsupported combinations.

## Implementation and validation checklist

- [ ] Add optimizer configuration and dependency isolation.
- [ ] Implement parameter grouping and native FSDP2 optimizer construction.
- [ ] Add CPU tests for configuration, grouping, invalid combinations, and the default AdamW path.
- [ ] Validate real FSDP2 updates on GPU against an unsharded reference over multiple steps.
- [ ] Validate mixed precision, distributed layouts, and checkpoint save/resume.
- [ ] Run a short end-to-end training check through the existing workflow.
- [ ] Document supported configurations, commands, results, and remaining limitations.

Implementation and CPU/GPU validation are not included in this draft.
