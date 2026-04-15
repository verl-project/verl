# FlowGRPO Trainer

This example shows how to post-train `Qwen-Image` with FlowGRPO on an OCR-style image generation task using `vllm-omni` rollout and a visual generative reward model (`Qwen3-VL-8B-Instruct` in this example).

For the full installation and quickstart guide, see `docs/start/flowgrpo_quickstart.rst`. For algorithm details, see `docs/algo/flowgrpo.md`.

## Installation

First, follow the standard `verl` installation guide in `docs/start/install.rst`.

Then install the FlowGRPO example-specific dependencies in the same environment:

```bash
pip install "vllm==0.18" "vllm-omni==0.18" python-Levenshtein
```

The provided script is configured for a single node with `4` GPUs.

## Prepare the dataset

Obtain the raw OCR dataset from the original Flow-GRPO repository:

- https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr

Then preprocess it into parquet files:

```bash
python3 examples/flowgrpo_trainer/data_process/qwenimage_ocr.py \
  --local_dataset_path ~/dataset/ocr \
  --local_save_dir ~/data/ocr
```

This produces:

- `~/data/ocr/train.parquet`
- `~/data/ocr/test.parquet`

## Prepare the models

The example script expects the following local model paths:

```bash
$HOME/models/Qwen/Qwen-Image
$HOME/models/Qwen/Qwen-Image/tokenizer
$HOME/models/Qwen/Qwen3-VL-8B-Instruct
```

If your models live elsewhere, update the path overrides in `examples/flowgrpo_trainer/run_qwen_image_ocr_lora.sh`.

## Run training

Launch the example from the repository root:

```bash
bash examples/flowgrpo_trainer/run_qwen_image_ocr_lora.sh
```

The script runs `python3 -m verl.trainer.main_flowgrpo` with:

- `algorithm.adv_estimator=flow_grpo`
- `actor_rollout_ref.rollout.name=vllm_omni`
- `reward.reward_manager.name=visual`
- `reward.custom_reward_function.name=compute_score_ocr`
- LoRA fine-tuning for `Qwen-Image`
- `trainer.n_gpus_per_node=4`

## Logging

W&B logging is enabled by default in the example script:

```bash
export WANDB_API_KEY=<your_wandb_api_key>
```

The script sets:

```bash
trainer.logger='["console", "wandb"]'
trainer.project_name=flow_grpo
trainer.experiment_name=qwen_image_ocr_lora
```

Override these values on the command line if you want to log under a different project or run name.
