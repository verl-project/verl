## Qwen3-Omni Thinker Assets

Prepare these assets on the cluster before running `Qwen3-Omni Thinker` GSPO jobs.

### Required paths

| Variable | Description |
| --- | --- |
| `MODEL_PATH` | Local checkpoint directory for `Qwen3-Omni-30B-A3B-Thinking`. |
| `TRAIN_FILE` | Training parquet with `<audio>` placeholders and an `audios` column. |
| `TEST_FILE` | Validation parquet with the same schema as the training parquet. |
| `RAY_DATA_HOME` | Root directory for models, data, and checkpoints on the cluster. |

### Required runtime

- `transformers` version that includes `Qwen3OmniMoeProcessor` and `Qwen3OmniMoeThinkerForConditionalGeneration`
- `vllm` version with multimodal audio input support and `mm_processor_kwargs`
- `qwen_omni_utils` when `use_audio_in_video=True`
- Ray cluster with `ray` CLI available in `PATH`

### Dataset schema expectations

- `prompt` contains chat messages with `<audio>` placeholders
- `audios` contains the audio payloads referenced by those placeholders
- Optional `images` / `videos` columns may coexist for mixed-modality samples

### Suggested validation flow

1. Prepare `MODEL_PATH`, `TRAIN_FILE`, and `TEST_FILE`.
2. Run a short smoke test:

```bash
MODEL_PATH=/path/to/Qwen3-Omni-30B-A3B-Thinking \
TRAIN_FILE=/path/to/omniinstruct/train.parquet \
TEST_FILE=/path/to/omniinstruct/test.parquet \
TOTAL_TRAINING_STEPS=5 \
bash examples/gspo_trainer/run_qwen3_omni_thinker_fsdp2.sh
```

3. For cluster execution, submit the same command through your Ray job launcher:

```bash
ray job submit --runtime-env=verl/trainer/runtime_env.yaml -- \
  bash examples/gspo_trainer/run_qwen3_omni_thinker_fsdp2.sh \
  trainer.total_training_steps=120
```
