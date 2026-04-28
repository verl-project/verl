# Tuning Examples

Examples that focus on *tuning-related* features rather than on a specific RL algorithm. Everything in here trains with `verl.trainer.main_ppo` and the current Hydra API.

## Subdirectories

### `lora/` — LoRA fine-tuning

Canonical LoRA GRPO scripts (training only adapters, rollout still serves the adapter via `load_format=safetensors + layered_summon`).

| Script                                         | Model             | Infer | Train    | Notes                       |
|------------------------------------------------|-------------------|-------|----------|-----------------------------|
| `run_qwen3_8b_fsdp.sh`               | Qwen3-8B          | vLLM  | FSDP     | text, GSM8K                 |
| `run_qwen3_8b_from_adapter_fsdp.sh`  | Qwen3-8B          | vLLM  | FSDP     | start from existing adapter |
| `run_qwen3_8b_merge_fsdp.sh`         | Qwen3-8B          | vLLM  | FSDP     | merge adapter into base     |
| `run_qwen2_5_vl_7b_fsdp.sh`          | Qwen2.5-VL-7B     | vLLM  | FSDP     | vision, Geo3K               |
| `run_qwen3_30b_a3b_megatron.sh`      | Qwen3-30B-A3B     | vLLM  | Megatron | MoE                         |

Key flags:
- `actor_rollout_ref.model.lora_rank`, `actor_rollout_ref.model.lora_alpha`
- `actor_rollout_ref.rollout.load_format=safetensors`
- `actor_rollout_ref.rollout.layered_summon=True`

### `scaling/` — Large-model scale demos

Single/multi-node tuning recipes for large dense models; geared to practitioners trying to fit and run these models out of the box with GRPO + GSM8K/MATH.

| Script                                    | Model           | Infer | Train    | Hardware                 |
|-------------------------------------------|-----------------|-------|----------|--------------------------|
| `run_qwen2_5_32b_megatron.sh`        | Qwen2.5-32B     | vLLM  | Megatron | 1×8 GPUs (TP=8)          |
| `run_qwen2_5_72b_fsdp.sh`            | Qwen2.5-72B     | vLLM  | FSDP     | 4×8 GPUs (TP=16, offload)|

## Conventions

- All scripts expose `MODEL_PATH`, `NNODES`, `NGPUS_PER_NODE`, batch sizes, learning rates, `ROLLOUT_TP`, `ROLLOUT_N`, etc. via `VAR=${VAR:-default}`.
- Dynamic batch size and `trainer.balance_batch=True` are enabled by default.
- No deprecated knobs (`ppo_micro_batch_size`, `data.val_batch_size`, top-level `reward_model.*`, `actor.ulysses_sequence_parallel_size`, `ppo_megatron_trainer.yaml`).
