# RLOO (REINFORCE Leave-One-Out)

RLOO is a simple policy gradient baseline that avoids a critic. Advantage for each sample is computed against the average of its siblings (leave-one-out), which acts as a per-prompt variance-reduction baseline.

Reference: [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740).

## Canonical Scripts

| Script                               | Infer | Train | Platform |
|--------------------------------------|-------|-------|----------|
| `run_qwen3_8b_fsdp.sh`          | vLLM  | FSDP  | NVIDIA   |

Override any argument via env vars at the top of the script (e.g. `MODEL_PATH=Qwen/Qwen3-14B bash run_qwen3_8b_fsdp.sh`).

## Key Flags

- `algorithm.adv_estimator=rloo`
- `actor_rollout_ref.rollout.n=5` — RLOO needs ≥2 samples per prompt; 5 is a common default.
- `actor_rollout_ref.actor.use_kl_loss=False` and `algorithm.use_kl_in_reward=True` — RLOO typically uses reward-side KL, not loss-side KL.
