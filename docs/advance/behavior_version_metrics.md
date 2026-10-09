# Behavior version metrics

Last updated: 10/04/2026.

Enable token-weighted behavior age and provenance coverage for the v1 trainer:

```text
actor_rollout_ref.rollout.collect_behavior_version_metrics=true
```

The default is `false`: generation does not record version segments and the trainer does not request additional metadata for these metrics. Enabling the flag does not change sampling or training loss.

The fully asynchronous rollout client records `[generation_version, new_token_count]` for every nonempty generation attempt. Aborting and resuming under a newer weight version produces multiple segments; empty aborts contribute no tokens. Versions come from the backend's existing `global_steps` metadata. Missing backend versions remain unknown.

The v1 trainer measures age against the weights used for the current update (`global_steps - 1`). `training/off_policy/token_staleness/mean` weights every token equally; `min`, `max`, and `stale_fraction` cover the same tokens. `cross_version_response_fraction` counts covered responses whose tokens span multiple generation versions. Padding is excluded.

`behavior_version/response_coverage` and `behavior_version/token_coverage` report how much consumed data has complete, valid provenance. A missing field, future or invalid version, or token-count mismatch reduces coverage and never becomes age zero. Responses with incomplete provenance do not enter age statistics. Older queue data can be consumed without this field; mixed batches retain coverage from compatible rows. No age statistics are emitted when coverage is empty.

These statistics describe consumed trajectories, not evicted or discarded generation work. Collection currently covers the fully asynchronous client; other clients or agent loops may have lower coverage. The feature does not require checkpoint persistence or other rollout observability features.

Only training batches are aggregated, using the consumed batch's TransferQueue partition and excluding padding. Statistics use model-version units, matching existing trajectory staleness. Provenance metrics live under training/off_policy/behavior_version; cross-version fractions use covered responses as their denominator, while coverage uses all consumed nonpadding responses or tokens. Validation does not emit these training metrics.
