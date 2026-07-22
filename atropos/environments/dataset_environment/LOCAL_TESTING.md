# Dataset Environment Local Testing Guide

This document explains how to run the Dataset Environment locally for testing purposes.

## Prerequisites

1. Make sure you have the repository cloned and dependencies installed
2. Ensure you have a compatible model available (local or API)

## Option 1: Single Script End-to-End Execution

The easiest way to test the Dataset Environment is to use the unified launcher script:

```bash
python -m environments.dataset_environment.launch_local_dataset_run
```

This script:
1. Starts the Trajectory Handler API server via uvicorn
2. Launches the Dataset Environment in serve mode (connected to the API)
3. Runs the example GRPO trainer directly

The script has environment defaults configured for:
- Using a small LLM (Qwen2.5-1.5B) running on localhost:9001
- A test subset of a public HF dataset
- Basic length-based rewards

## Option 2: Step-by-step Manual Testing

### 1. Start the API Server

```bash
uvicorn atroposlib.api.server:app --host 127.0.0.1 --port 8000
```

### 2. Launch the Environment

```bash
python -m environments.dataset_environment.dataset_env serve \
  --group_size 4 \
  --max_num_workers 2 \
  --rollout_server_url http://127.0.0.1:8000 \
  --tokenizer_name Qwen/Qwen2.5-1.5B-Instruct \
  --use_wandb --wandb_name dataset_env_local_test \
  --max_token_length 512 \
  --ensure_scores_are_not_same \
  --dataset_name HuggingFaceH4/testing_self_instruct_process_essays \
  --split train[:100] \
  --prompt_field prompt --answer_field answer \
  --reward_functions length \
  --max_tokens 128 --temperature 0.7 \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --base_url http://127.0.0.1:9001 \
  --slurm --testing
```

### 3. Launch the Trainer

In a separate terminal:

```bash
python -m example_trainer.grpo.train \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --training_steps 20 \
  --batch_size 2 \
  --gradient_accumulation_steps 2 \
  --seq_len 512
```

## Option N: Use the Dataset Local Server

For easier configuration via YAML files, you can use the local server script:

```bash
# This command will look for environments/dataset_environment/configs/gsm8k.yaml
python environments/dataset_environment/dataset_local_server.py --config gsm8k

# You can also provide a full path:
# python environments/dataset_environment/dataset_local_server.py --config /path/to/your/custom_config.yaml
```

This will load the specified config and run the environment accordingly.

## Debugging

To check if requests are properly sent to and received by the API server, you can inspect the logs from both the environment and the API server. Look for:

- API logs showing incoming requests
- Environment logs showing completions being generated and scored

For model-specific issues, check:
- Ensure your model server is running at the specified URL
- Check model server logs for any errors related to generation

## Configuration Structure

Configuration files placed in `environments/dataset_environment/configs/` typically contain:

```yaml
# Example: environments/dataset_environment/configs/my_config.yaml

# Base environment parameters (can be overridden by dataset specifics)
tokenizer_name: "NousResearch/DeepHermes-3-Llama-3-8B-Preview"
group_size: 1
use_wandb: false
# ... other base parameters

# Dataset specific configuration
dataset:
  # Dataset parameters
  dataset_name: "databricks/databricks-dolly-15k"
  prompt_field: "instruction"
  # ... other dataset parameters
  reward_functions:
    - type: "accuracy"
      weight: 1.0
    - type: "repetition_penalty"
      weight: 0.2

# Optional Server configuration (if not using CLI flags in dataset_env)
server_configs:
  - model_name: "gpt-4.1-nano"
    api_key: ${OPENAI_API_KEY}
    timeout: 600
```

### Important Configuration Parameters

#### Base Parameters

- `tokenizer_name`: The tokenizer to use for encoding/decoding text
- `group_size`: Number of responses to collect per prompt
- `max_token_length`: Maximum token length for generation
- `steps_per_eval`: How often to run evaluations

#### Dataset Specific Parameters (`dataset:` section)

- `dataset_name`: HuggingFace dataset name (required)
- `dataset_config`: Dataset configuration name (optional)
- `prompt_field`: Field in dataset to use as prompt (required)
- `answer_field`: Field in dataset to use as answer (optional)
- `system_prompt`: System prompt to use (optional)
- `reward_functions`: List of reward functions to apply (optional)

#### Server Configuration (`server_configs:` section, optional in local server)

- `model_name`: LLM model to use
- `api_key`: API key for the model (can use environment variables with ${VAR_NAME} syntax)
- `timeout`: Request timeout in seconds

## Troubleshooting

If you encounter issues with reward functions, make sure they are properly registered in the registry.

For dataset-related issues, verify that the dataset exists on HuggingFace and that the specified fields exist in the dataset.
