# GDPO

GDPO is a multi-reward, rubric-style variant whose advantage estimator aggregates several reward signals (accuracy, format, etc.). It uses a custom reward manager and a custom scoring function.

## Canonical Scripts

| Script                               | Infer | Train | Platform |
|--------------------------------------|-------|-------|----------|
| `run_qwen3_8b_fsdp.sh`          | vLLM  | FSDP  | NVIDIA   |

Prepare a rubric-style dataset (e.g. `rlla_4k`) and point `DATA_DIR` to it.

## Key Flags

- `algorithm.adv_estimator=gdpo`
- `+algorithm.gdpo_reward_keys='["accuracy_reward", "format_reward"]'`
- `reward.reward_manager.name=gdpo`
- `reward.custom_reward_function.path=$REPO_ROOT/verl/utils/reward_score/rlla.py`
