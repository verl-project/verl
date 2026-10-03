## Quick Start

### Option A: Unified End-to-End Launcher

```bash
python -m environments.dataset_environment.launch_local_dataset_run
```
This single command spins up:
1. The Trajectory Handler API server (`uvicorn atroposlib.api.server:app`)
2. The DatasetEnv in serve mode (connected to the API)
3. The example GRPO trainer (via `example_trainer.grpo.train`)

### Option B: Manual Steps

1. **Start the API server**

   ```bash
   uvicorn atroposlib.api.server:app --host 127.0.0.1 --port 8000
   ```

2. **Launch the Dataset Environment**

   - **Using CLI flags**:
     (These flags override any config file settings)
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

   - **Using YAML config files**:

     Place a dataset config under `environments/dataset_environment/configs/<name>.yaml`:
     ```yaml
     # Example: environments/dataset_environment/configs/gsm8k.yaml
     dataset:
       dataset_name: "gsm8k"
       dataset_config: "main"
       split: "train"
       prompt_field: "question"
       answer_field: "answer"
       system_prompt: "You are a mathematical problem solver..."

     generation:
       temperature: 0.7
       top_p: 0.95

     reward_functions:
       - type: "accuracy"
         weight: 1.0
     ```

     Then run the local test server:
     ```bash
     # Will look for environments/dataset_environment/configs/gsm8k.yaml
     python environments/dataset_environment/dataset_local_server.py --config gsm8k
     ```

3. **Launch the Trainer**

   ```bash
   python -m example_trainer.grpo
   ```

## Configuration Files Directory

Dataset environment specific configurations now live in `environments/dataset_environment/configs/`.
Shared configurations (like agents) might still reside in the project's root `configs/` directory.

- `environments/dataset_environment/configs/` for dataset-specific configs (used by `dataset_local_server.py`).
- You can reference any `<name>.yaml` within this directory via the `--config` flag in the local server script.

## Reward Function Registry & Customization

Reward functions are managed by a centralized registry (see `atroposlib/envs/reward_fns/reward_function.py`). Built-in types include:

- `accuracy`: exact match to ground truth (tolerance, split_on_think_tag)
- `format`: checks for specific tags (preferred_tags)
- `reasoning_steps`: quality of step-by-step reasoning
- `repetition_penalty`: penalizes repetition
- `cosine_scaled`: semantic similarity scaled from embeddings
- `crossword_format`: crossword-specific penalty
- `r1`: combined accuracy + format

To preview all available functions:
```python
from atroposlib.envs.reward_fns import registry
print(registry.list())
```

### Creating Custom Reward Functions

1. Create a new file under `atroposlib/envs/reward_fns/my_reward.py`.
2. Subclass `RewardFunction` and register it:

   ```python
   from atroposlib.envs.reward_fns import registry, RewardFunction

   @registry.register
   class MyCustomReward(RewardFunction):
       def __init__(self, custom_param=1.0, weight=1.0, **kwargs):
           super().__init__(weight=weight, **kwargs)
           self.custom_param = custom_param

       def compute(self, completions, **kwargs):
           return [1.0 if "good answer" in self.get_content(c) else 0.0 for c in completions]
   ```

3. Reference it in your YAML config:

   ```yaml
   reward_functions:
     - type: "my_custom"
       weight: 1.0
       params:
         custom_param: 2.0
   ```

### Dataset Environments

Dataset environments load data from HuggingFace datasets and evaluate LLM responses against ground truth. They're ideal for academic benchmarks and datasets with clear evaluation criteria.

Example configuration:
```yaml
dataset:
  dataset_name: "gsm8k"
  dataset_config: "main"
  split: "train"
  prompt_field: "question"
  answer_field: "answer"
  system_prompt: "You are a mathematical problem solver..."
  reward_functions:
    - type: "accuracy"
      weight: 1.0
```

## Reward Functions

The system features a flexible reward function architecture for evaluating model outputs.

### Basic Usage

In your environment config, specify reward functions:

```yaml
reward_functions:
  - type: "accuracy"
    weight: 1.0
  - type: "format"
    weight: 0.5
```

### Combining Reward Functions

Combine multiple reward functions with weights:

```yaml
reward_functions:
  - type: "combined"
    params:
      normalization: "sum"
      rewards:
        - type: "accuracy"
          weight: 1.5
        - type: "format"
          weight: 0.5
```

### Available Reward Functions

#### `accuracy`
Evaluates if completions match ground truth answers.

```yaml
type: "accuracy"
weight: 1.0
params:
  tolerance: 1e-6
  split_on_think_tag: true
  max_boxed_threshold: 6
```

#### `format`
Checks if completions include specific XML-style tags.

```yaml
type: "format"
weight: 1.0
params:
  preferred_tags: ["think", "reasoning"]
  require_all_tags: false
  case_sensitive: false
```

#### `reasoning_steps`
Evaluates step-by-step reasoning quality.

```yaml
type: "reasoning_steps"
weight: 1.0
params:
  min_words: 10
  min_steps: 3
  base_score: 0.1
```

#### `repetition_penalty`
Penalizes repetitive content.

```yaml
type: "repetition_penalty"
weight: 0.5
params:
  threshold: 0.05
  min_words: 10
  min_sentences: 2
```

#### `cosine_scaled`
Measures semantic similarity between completions and solutions.

```yaml
type: "cosine_scaled"
weight: 0.8
params:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  scale_factor: 1.0
  min_reward: -1.0
  max_reward: 1.0
```

#### `crossword_format`
Game-specific reward for crossword puzzles.

```yaml
type: "crossword_format"
weight: 1.0
params:
  reward_value: 1.0
  penalize_invalid_chars: true
```

#### `r1`
Combined reward using both reasoning format and accuracy.

```yaml
type: "r1"
weight: 1.0
params:
  format_weight: 0.5
  accuracy_weight: 1.0
```

### Creating Custom Reward Functions

To create a custom reward function:

1. Create a new file in `atroposlib/envs/reward_fns/my_reward.py`

2. Define your reward function class:

```python
from typing import Any, List
from atroposlib.envs.reward_fns import registry, RewardFunction

@registry.register
class MyCustomReward(RewardFunction):
    def __init__(self, custom_param=1.0, weight=1.0, **kwargs):
        super().__init__(weight=weight, **kwargs)
        self.custom_param = custom_param

    def compute(self, completions: List[Any], **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            content = self.get_content(completion)
            # Implement your reward logic
            reward = 1.0 if "good answer" in content else 0.0
            rewards.append(reward)
        return rewards
```

3. Use it in your config:

```yaml
reward_functions:
  - type: "my_custom"
    weight: 1.0
    params:
      custom_param: 2.0
```

### Dataset Environment Debugger

The dataset environment debugger allows you to run a dataset environment locally with a Hugging Face model, providing enhanced visibility into reward function performance and model responses.

```bash
# Run with default settings
python -m atroposlib.cli.dataset_env_debugger --env gsm8k_debug --agent nous_hermes_8b

# List available environments and agents
python -m atroposlib.cli.dataset_env_debugger --list-configs

# Interactive mode with debugging information
python -m atroposlib.cli.dataset_env_debugger --env gsm8k_debug --agent nous_hermes_8b --interactive --debug

# Run with custom generation parameters
python -m atroposlib.cli.dataset_env_debugger --env gsm8k_debug --agent nous_hermes_8b --temperature 0.5 --top-p 0.95

# Run with detailed logging
python -m atroposlib.cli.dataset_env_debugger --env gsm8k_debug --agent nous_hermes_8b --verbose
```

## Environment Overview

This environment demonstrates how to use a standard dataset (e.g., from Hugging Face Datasets) as a source for generating prompts and evaluating LLM responses. It allows for testing and training models on established benchmarks or custom datasets where prompts and expected answers/ground truth are available.

**Demonstrates:**
- Loading and processing data from Hugging Face Datasets.
- Configuring system prompts, prompt/answer fields.
- Applying various reward functions (accuracy, format, semantic similarity, etc.) to evaluate generations.
- Integrating with the `atroposlib` framework for data collection and scoring.

**Training Goal:**
- To train LLMs to follow instructions and generate responses that align with the format and content specified by the dataset and reward functions.
- To improve performance on specific tasks defined by datasets (e.g., math problem solving, code generation, question answering).

## Local Testing

To test this environment locally, you can run the provided local server. This server simulates the interaction flow without needing the full distributed setup.

First, ensure you have the necessary dependencies installed.

Then, run the local server script from the root of the repository:

```bash
python environments/dataset_environment/dataset_local_server.py --config-path path/to/your/dataset_config.yaml
```

Replace `path/to/your/dataset_config.yaml` with the actual path to your environment configuration file (e.g., `configs/envs/gsm8k.yaml`). The server will load the dataset specified in the config, process items, and simulate generating responses.


FOR RELEASE - FIX
