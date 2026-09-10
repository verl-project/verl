#!/usr/bin/env python3
"""
Local dataset training launcher.

Usage:
    python -m environments.dataset_environment.launch_local_dataset_run

This script does:
  1) Starts the Trajectory Handler API server via uvicorn
  2) Launches the DatasetEnv in local serve mode
  3) Imports and runs the example trainer (GRPO) directly

Requirements:
  - Run from project root so example_trainer is on PYTHONPATH
  - example_trainer/ is a valid Python package (with __init__.py)
"""
import atexit
import os
import signal
import subprocess
import sys
import time
import traceback

# Ensure project root is on PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import trainer via standard module import
try:
    from example_trainer.grpo import TrainingConfig, train
except ImportError as e:
    print(f"Error importing example_trainer.grpo: {e}")
    print(
        "Ensure you're running from project root and that example_trainer/ is a package."
    )
    sys.exit(1)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
API_HOST = "127.0.0.1"
API_PORT = 8000

VLLM_HOST = "127.0.0.1"
VLLM_PORT = 9001

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TOKENIZER_NAME = MODEL_NAME

TRAINER_CONFIG = {
    "model_name": MODEL_NAME,
    "training_steps": 20,
    "batch_size": 2,
    "gradient_accumulation_steps": 2,
    "seq_len": 512,
    "vllm_port": VLLM_PORT,
    "vllm_restart_interval": 10,
    "use_wandb": False,
    "wandb_project": "",
    "wandb_group": "",
    "save_path": "./trained_model_checkpoints_local_test",
}

# Flags for launching DatasetEnv serve
DATASET_FLAGS = [
    "--group_size",
    "4",
    "--max_num_workers",
    "2",
    "--rollout_server_url",
    f"http://{API_HOST}:{API_PORT}",
    "--tokenizer_name",
    TOKENIZER_NAME,
    "--use_wandb",
    "--wandb_name",
    "dataset_env_local_test",
    "--max_token_length",
    str(TRAINER_CONFIG["seq_len"]),
    "--ensure_scores_are_not_same",
    "--dataset_name",
    "HuggingFaceH4/testing_self_instruct_process_essays",
    "--split",
    "train[:100]",
    "--prompt_field",
    "prompt",
    "--answer_field",
    "answer",
    "--reward_functions",
    "length",
    "--max_tokens",
    "128",
    "--temperature",
    "0.7",
    "--model_name",
    MODEL_NAME,
    "--base_url",
    f"http://{VLLM_HOST}:{VLLM_PORT}",
    "--slurm",
    "--testing",
]

# Track background processes for cleanup
processes = []


def cleanup_processes():
    print("\nCleaning up background processes...")
    for p in reversed(processes):
        if p.poll() is None:
            print(f"Terminating PID {p.pid}...")
            p.terminate()
            try:
                p.wait(timeout=5)
                print(f"PID {p.pid} terminated.")
            except subprocess.TimeoutExpired:
                print(f"PID {p.pid} did not terminate; killing.")
                p.kill()
                p.wait()
                print(f"PID {p.pid} killed.")
        else:
            print(f"PID {p.pid} already exited.")
    print("Cleanup complete.")


atexit.register(cleanup_processes)


def handle_signal(sig, frame):
    print(f"\nSignal {sig} received; exiting.")
    sys.exit(0)


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


def main():
    # 1) Start the API server
    print("--- Starting Trajectory Handler API Server ---")
    api_cmd = [
        "uvicorn",
        "atroposlib.api.server:app",
        "--host",
        API_HOST,
        "--port",
        str(API_PORT),
    ]
    print(f"$ {' '.join(api_cmd)}")
    api_proc = subprocess.Popen(api_cmd)
    processes.append(api_proc)
    time.sleep(3)

    # 2) Start the dataset environment
    print("\n--- Starting Dataset Environment ---")
    env_cmd = [
        "python",
        "-m",
        "environments.dataset_environment.dataset_env",
        "serve",
    ] + DATASET_FLAGS
    print(f"$ {' '.join(env_cmd)}")
    env_proc = subprocess.Popen(env_cmd)
    processes.append(env_proc)
    time.sleep(3)

    # 3) Run the example trainer
    print("\n--- Running Example Trainer (GRPO) ---")
    config = TrainingConfig(**TRAINER_CONFIG)
    try:
        train(config)
    except Exception:
        print("Error during training:")
        traceback.print_exc()
    print("--- Training complete ---")


if __name__ == "__main__":
    main()
