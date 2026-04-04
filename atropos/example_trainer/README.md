# GRPO Example Trainer

This directory contains an example script (`grpo.py`) demonstrating how to integrate a custom training loop with the Atropos API for reinforcement learning using the GRPO (Group Relative Policy Optimization) algorithm.

**Note: Example trainer does not support multimodal training out of the box. As other trainers add support for Atropos, we will list them in the main readme, some of which may support multimodal RL - please check the main repo readme for any updates.**

This example uses `vLLM` for efficient inference during the (simulated) data generation phase and `transformers` for the training phase.

**Note:** This script is intended as a *reference example* for API integration and basic training setup. It is not optimized for large-scale, efficient training.

## Prerequisites

1.  **Python:** Python 3.8 or higher is recommended.
2.  **Atropos API Server:** The Atropos API server must be running and accessible (defaults to `http://localhost:8000` in the script).
3.  **Python Packages:** You need to install the required Python libraries:
    *   `torch` (with CUDA support recommended)
    *   `transformers`
    *   `vllm`
    *   `pydantic`
    *   `numpy`
    *   `requests`
    *   `tenacity`
    *   `wandb` (optional, for logging)

## Setup

1.  **Clone the Repository:** Ensure you have the repository containing this example.
2.  **Install Dependencies:** `pip install -r requirements.txt`
3.  **Ensure Atropos API is Running:** `run-api` in a new window
4.  **Run an env:** `python environments/gsm8k_server.py serve --slurm False`

## Configuration

The training configuration is managed within the `grpo.py` script using the `TrainingConfig` Pydantic model (found near the top of the file).

Key parameters you might want to adjust include:

*   `model_name`: The Hugging Face model identifier to use for training (e.g., `"gpt2"`, `"Qwen/Qwen2.5-1.5B-Instruct"`).
*   `training_steps`: The total number of optimization steps to perform.
*   `batch_size` / `gradient_accumulation_steps`: Control the effective batch size.
*   `lr`: Learning rate.
*   `save_path`: Directory where model checkpoints will be saved.
*   `vllm_port`: The port used by the vLLM server instance launched by this script.
*   `vllm_restart_interval`: How often (in steps) to save a checkpoint and restart the vLLM server with the new weights.
*   `use_wandb`: Set to `True` to enable logging to Weights & Biases.
*   `wandb_project`: Your W&B project name (required if `use_wandb=True`).
*   `wandb_group`: Optional W&B group name.

**API Endpoints:** The script currently assumes the Atropos API is available at `http://localhost:8000/register` and `http://localhost:8000/batch`. If your API runs elsewhere, you'll need to modify the `register_trainer` and `get_batch` functions accordingly.

## Running the Example

Once the prerequisites are met and configuration is set:

1.  Navigate to the root directory of the project in your terminal.
2.  Run the script:

    ```bash
    python example_trainer/grpo.py
    ```

## Output

*   **Logs:** Training progress, loss, logp, and vLLM status will be printed to the console.
*   **Checkpoints:** Model checkpoints will be saved periodically in the directory specified by `save_path` (default: `./trained_model_checkpoints`). A `final_model` directory will be created upon completion.
*   **WandB:** If `use_wandb` is `True`, logs will be sent to Weights & Biases. A link to the run page will be printed in the console.
*   `temp.json`: Contains the raw data from the last fetched batch (used for debugging/manual inspection).

```bash
# Install dependencies
pip install -r example_trainer/requirements.txt

# Run the trainer directly (basic test)
python example_trainer/grpo.py
