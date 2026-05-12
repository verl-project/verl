# ReMax

ReMax is a lightweight policy-gradient method that uses a single greedy-decoded baseline response per prompt to reduce variance without a critic.

Reference: [ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models](https://arxiv.org/abs/2310.10505).

## Canonical Scripts

| Script                               | Infer | Train | Platform |
|--------------------------------------|-------|-------|----------|
| `run_qwen3_8b_fsdp.sh`          | vLLM  | FSDP  | NVIDIA   |

Override any argument via env vars at the top of the script.

## Key Flags

- `algorithm.adv_estimator=remax`
- `actor_rollout_ref.actor.use_kl_loss=False` and `algorithm.use_kl_in_reward=True`
