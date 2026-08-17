# GDPO

GDPO is a multi-reward, rubric-style variant whose advantage estimator aggregates several reward signals (accuracy, format, etc.). It uses a custom reward manager and a custom scoring function.

## Canonical Scripts

| Script                               | Infer | Train | Platform |
|--------------------------------------|-------|-------|----------|
| `run_qwen3_8b_fsdp.sh`               | vLLM        | FSDP     | NVIDIA    |
| `run_qwen2_5_1_5b_megatron.sh`       | vLLM-Ascend | Megatron | Ascend NPU |

Prepare a rubric-style dataset (e.g. `rlla_4k`) and point `DATA_ROOT` to it.

## Key Flags

- `algorithm.adv_estimator=gdpo`
- `+algorithm.gdpo_reward_keys='["accuracy_reward", "format_reward"]'`
- `reward.reward_manager.name=gdpo`
- `reward.custom_reward_function.path=$REPO_ROOT/verl/utils/reward_score/rlla.py`

## Megatron + vLLM-Ascend

```bash
MODEL_PATH=/path/to/Qwen2.5-1.5B-Instruct \
DATA_ROOT=/path/to/data \
NPUS_PER_NODE=4 \
bash examples/gdpo_trainer/run_qwen2_5_1_5b_megatron.sh
```

Ensure the container provides enough `/dev/shm` capacity for the configured weight-transfer bucket.
See [verl-ascend-recipe issue #18](https://github.com/verl-project/verl-ascend-recipe/issues/18)
for the validated environment, training logs, and 100-step results.
