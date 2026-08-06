# Optimal Token Baseline (OTB)

OTB uses a token-wise optimal variance-reduction baseline derived from the per-prompt sample group. See `tir_optimal_token_baseline` variant for a TIR-specific form.

## Canonical Scripts

| Script                               | Infer | Train | Platform |
|--------------------------------------|-------|-------|----------|
| `run_qwen3_8b_fsdp.sh`          | vLLM  | FSDP  | NVIDIA   |

Set `ADV_ESTIMATOR=tir_optimal_token_baseline` for the TIR variant.

## Key Flags

- `algorithm.adv_estimator=optimal_token_baseline`
- `actor_rollout_ref.actor.calculate_sum_pi_squared=True`
