# ADAPTED FROM THE SAMPLE TRAINER

import atexit
import json
import math
import os
import random
import shutil
import string
import subprocess
import time
from typing import List, Optional, Tuple

import numpy as np
import requests
import torch
import torch.nn.functional as F
import wandb  # Added for logging
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

# Global variable to keep track of the vLLM process
vllm_process = None


def cleanup_vllm():
    global vllm_process
    if vllm_process:
        print("\nTerminating vLLM process...")
        vllm_process.terminate()
        try:
            vllm_process.wait(timeout=5)  # Wait a bit for graceful shutdown
            print("vLLM process terminated.")
        except subprocess.TimeoutExpired:
            print("vLLM process did not terminate gracefully, killing.")
            vllm_process.kill()
            vllm_process.wait()
            print("vLLM process killed.")
        vllm_process = None


# Register the cleanup function to be called on script exit
atexit.register(cleanup_vllm)


class TrainingConfig(BaseModel):
    """
    Training details, model, etc
    """

    model_name: str = Field(..., description="Name of the base model to train")
    lr: float = Field(1e-5, description="Learning rate for the optimizer")
    training_steps: int = Field(
        10, description="Number of training steps"
    )  # Renamed from epochs
    batch_size: int = Field(
        2, description="Batch size for training (will be handled by get_data)"
    )
    seq_len: int = Field(2048, description="Sequence length for training")
    gradient_accumulation_steps: int = Field(
        32, description="Number of gradient accumulation steps"
    )
    device: str = Field(
        "cuda" if torch.cuda.is_available() else "cpu", description="Device to train on"
    )
    save_path: str = Field(
        "trained_model_checkpoints", description="Base path to save model checkpoints"
    )
    vllm_restart_interval: int = Field(
        3, description="Restart vLLM every N training steps"
    )
    vllm_port: int = Field(9001, description="Port for the vLLM server")

    # Wandb configuration
    use_wandb: bool = Field(
        False, description="Whether to use Weights & Biases for logging"
    )
    wandb_project: Optional[str] = Field(None, description="Wandb project name")
    wandb_group: Optional[str] = Field(None, description="Wandb group name")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def register_trainer(config: TrainingConfig):
    """
    Register the trainer with the Atropos API
    """
    requests.post(
        "http://localhost:8000/register",
        json={
            "wandb_group": config.wandb_group,
            "wandb_project": config.wandb_project,
            "batch_size": config.batch_size * config.gradient_accumulation_steps,
            "max_token_len": config.seq_len,
            "starting_step": 0,
            "checkpoint_dir": config.save_path,
            "save_checkpoint_interval": config.training_steps,
            "num_steps": config.training_steps,
        },
        timeout=10,
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def get_batch():
    data = requests.get("http://localhost:8000/batch", timeout=10).json()
    return data


def pad_data_to_good_offset(data, batch_size: int):
    max_token_len = max(
        [max([len(x) for x in item["tokens"]]) for item in data["batch"]]
    )
    # usually 64 is a good choice to ensure nonweird scaling behavior on GPUS
    # so we pad to the nearest multiple of 64
    good_multiple = 64
    if (max_token_len - 1) % (good_multiple) != 0:
        max_token_len = math.ceil((max_token_len - 1) / (good_multiple)) * good_multiple
        token_setup_len = (
            max_token_len + 1
        )  # add 1 so we can make it causal at the proper length
    else:
        token_setup_len = max_token_len
        max_token_len = (
            max_token_len - 1
        )  # since it's causal we need to remove the last bit...
    # pad all tokens to max_token_len and add to lists
    input_ids = list()
    labels = list()
    advantages = list()
    lengths = list()
    for item in data["batch"]:
        scores = item["scores"]
        scores = np.array(scores)
        # check if we have more than 1 score...
        if len(scores) > 1:
            scores = scores - scores.mean()
            scores = scores / max(scores.std(), 1e-8)
        item["scores"] = scores
        if item["overrides"] is not None:
            for i in range(len(item["overrides"])):
                if item["overrides"][i].get("set_advantage_to_zero", False):
                    item["scores"][i] = 0
        for i in range(len(item["tokens"])):
            lengths.append(
                math.ceil((len(item["tokens"][i]) - 1) / (good_multiple))
                * good_multiple
            )
            label_item = np.concatenate(
                [
                    np.array(item["masks"][i]),
                    np.full(
                        max(0, token_setup_len - len(item["tokens"][i])),
                        -100,
                        dtype=np.int32,
                    ),
                ]
            )
            item["tokens"][i] = np.concatenate(
                [
                    np.array(item["tokens"][i]),
                    np.zeros(
                        max(0, token_setup_len - len(item["tokens"][i])), dtype=np.int32
                    ),
                ]
            )
            input_ids.append(item["tokens"][i][:-1])
            labels.append(label_item[1:])
            advantages.append(item["scores"][i])
    # combine all lists into tensors
    token_batches = []
    label_batches = []
    advantage_batches = []
    for i in range(len(input_ids) // batch_size):
        token_batches.append(
            torch.tensor(
                np.stack(input_ids[i * batch_size : (i + 1) * batch_size], axis=0)
            )
        )
        label_batches.append(
            torch.tensor(
                np.stack(labels[i * batch_size : (i + 1) * batch_size], axis=0)
            )
        )
        advantage_batches.append(
            torch.tensor(
                np.stack(advantages[i * batch_size : (i + 1) * batch_size], axis=0)
            ).view(-1, 1)
        )
    return token_batches, label_batches, advantage_batches


def get_data(
    batch_size: int, seq_len: int
) -> List[Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]]:
    """
    getting data from the api
    """
    batches = []
    while True:
        data = get_batch()
        if data["batch"] is not None:
            # Save the batch
            with open("temp.json", "w", encoding="utf-8") as f:
                json.dump(data, f)
            # In case the inference runs ahead of the training, we loop until we don't have any more data
            batches.append(pad_data_to_good_offset(data, batch_size))
        elif len(batches) > 0:
            # Return the batches
            return batches
        else:
            time.sleep(1)


def train(config: TrainingConfig):
    """
    Setups and runs GRPO training, restarting vLLM periodically, with wandb logging.
    """
    global vllm_process  # Declare intention to modify the global variable

    # --- Wandb Setup ---
    if config.use_wandb:
        if not config.wandb_project:
            print("Warning: wandb_project not set, disabling wandb.")
            config.use_wandb = False
        else:
            if not config.wandb_group:
                # Set group to random 8 character string
                config.wandb_group = "".join(
                    random.choices(string.ascii_letters + string.digits, k=8)
                )
            try:
                wandb.init(
                    project=config.wandb_project,
                    group=config.wandb_group,
                    config=config.dict(),  # Log config parameters
                )
                print(
                    f"Wandb logging enabled. Run: {wandb.run.name} (Project: {config.wandb_project}) "
                )
            except Exception as e:
                print(f"Error initializing wandb: {e}. Disabling wandb.")
                config.use_wandb = False
    # --- End Wandb Setup ---

    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=torch.bfloat16
    )

    model.to(config.device)
    model.gradient_checkpointing_enable()
    model.train()

    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=config.lr)

    print(
        f"Starting training for {config.training_steps} steps on device: {config.device}"
    )
    print(
        f"vLLM will be restarted every {config.vllm_restart_interval} steps on port {config.vllm_port}"
    )

    os.makedirs(config.save_path, exist_ok=True)  # Ensure base save directory exists
    register_trainer(config)

    # Init vllm
    vllm_command = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        config.model_name,
        "--port",
        str(config.vllm_port),
        "--dtype",
        "auto",
        "--gpu-memory-utilization",
        "0.45",
        "--disable-log-requests",
    ]
    print(f"  Launching vLLM server: {' '.join(vllm_command)}")
    try:
        vllm_process = subprocess.Popen(vllm_command)
        print(f"  vLLM server launched with PID: {vllm_process.pid}")
        # Check immediate errors
        try:
            stdout, stderr = vllm_process.communicate(timeout=2)
            if vllm_process.returncode is not None and vllm_process.returncode != 0:
                print(f"  Error starting vLLM: {stderr.decode()}")
                vllm_process = None
                # Maybe raise error or just warn?
                print("  WARNING: Failed to start vLLM server after checkpoint.")
        except subprocess.TimeoutExpired:
            print("  vLLM process started (check logs for details).")
    except FileNotFoundError:
        print(
            "\n *** ERROR: 'python -m vllm...' command not found. Make sure vLLM is installed and accessible. ***\n"
        )
        # Potentially stop training or just disable further vLLM restarts
        print("  Disabling further vLLM restarts.")
        config.vllm_restart_interval = (
            config.training_steps + 1
        )  # Prevent further restarts
    except Exception as e:
        print(f"\n *** ERROR: Failed to launch vLLM: {e} ***\n")
        print("  Disabling further vLLM restarts.")
        config.vllm_restart_interval = (
            config.training_steps + 1
        )  # Prevent further restarts

    batches = list()
    for step in range(config.training_steps):
        total_loss = 0
        print(f"Step {step+1}/{config.training_steps}")
        total_pos_logp = 0
        total_neg_logp = 0
        total_logp = 0
        total_pos = 0
        total_neg = 0
        if len(batches) == 0:
            batches = get_data(config.batch_size, config.seq_len)
        token_batches, label_batches, advantage_batches = batches.pop(0)
        # Terminate existing vLLM process if running
        if (
            step + 1
        ) % config.vllm_restart_interval == 0 or step == config.training_steps - 1:  # Also restart/save on last step
            # Terminate existing vLLM process if running
            if vllm_process:
                print("  Terminating existing vLLM process...")
                vllm_process.terminate()
                try:
                    vllm_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(
                        "  Existing vLLM process did not terminate gracefully, killing."
                    )
                    vllm_process.kill()
                    vllm_process.wait()
                vllm_process = None
        for tokens, labels, advantages in zip(
            token_batches, label_batches, advantage_batches
        ):

            tokens, labels, advantages = (
                tokens.to(config.device),
                labels.to(config.device),
                advantages.to(config.device),
            )

            # Forward pass
            # User specified that tokens/labels are already prepared by get_data
            outputs = model(tokens)  # Assuming model just needs tokens
            logits = outputs.logits  # Assuming this is the structure

            # Calculate GRPO loss (reverting to user's previous logic)
            # User stated ignore_index is -100 and tokens/labels are aligned by get_data
            # Assuming logits correspond directly to labels indices (no shift needed here)
            logp_per_token = -F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # Flatten logits
                labels.view(-1),  # Flatten labels
                reduction="none",
                ignore_index=-100,  # User specified ignore index
            ).view(
                labels.shape
            )  # Reshape back to (batch, seq_len)

            # Masking based on labels != -100
            mask = (labels != -100).float()
            with torch.no_grad():
                pos = (advantages > 0).float()
                neg = (advantages <= 0).float()
                avg_logp = (logp_per_token * mask).sum(-1) / mask.sum(-1)
                pos_logp = (logp_per_token * pos).mean().item()
                neg_logp = (logp_per_token * neg).mean().item()
                total_pos_logp += pos_logp
                total_neg_logp += neg_logp
                total_logp += avg_logp
                total_pos += pos.sum().item()
                total_neg += neg.sum().item()

            grpo_loss_term = torch.exp(logp_per_token - logp_per_token.detach())
            grpo_loss = (
                ((-grpo_loss_term * mask).sum(-1) / mask.sum(-1))
                * advantages.to(logp_per_token.device)
            ).mean() / config.gradient_accumulation_steps
            grpo_loss.backward()
            total_loss += grpo_loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        if total_pos > 0:
            total_pos_logp /= total_pos
        if total_neg > 0:
            total_neg_logp /= total_neg
        # --- Wandb Logging ---
        if config.use_wandb:
            wandb.log(
                {
                    "train/loss": total_loss,
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/grad_norm": grad_norm.item(),
                    "train/pos_logp": total_pos_logp,
                    "train/neg_logp": total_neg_logp,
                    "train/logp": total_logp,
                },
                step=step + 1,
            )
        # --- End Wandb Logging ---

        print(f"  Step Loss: {grpo_loss.item():.4f}")

        # --- vLLM Restart Logic (Moved AFTER optimizer step) ---
        # Note: There are much better ways of updating the policy, this is just a very simple example
        if (
            step + 1
        ) % config.vllm_restart_interval == 0 or step == config.training_steps - 1:  # Also restart/save on last step
            checkpoint_path = os.path.join(
                config.save_path, f"step_{step+1}"
            )  # Save as step+1 since it's after step completion
            print(f"  Saving checkpoint to {checkpoint_path}...")
            # Ensure fresh directory for saving
            if os.path.exists(checkpoint_path):
                shutil.rmtree(checkpoint_path)  # Remove old checkpoint if it exists
            os.makedirs(checkpoint_path, exist_ok=True)
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            print("  Checkpoint saved.")

            # Terminate existing vLLM process if running
            if vllm_process:
                print("  Terminating existing vLLM process...")
                vllm_process.terminate()
                try:
                    vllm_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(
                        "  Existing vLLM process did not terminate gracefully, killing."
                    )
                    vllm_process.kill()
                    vllm_process.wait()
                vllm_process = None

            # Launch new vLLM process (only if not the very last step, maybe? depends on use case)
            # Let's still launch it on the last step for consistency, cleanup will handle it.
            vllm_command = [
                "python",
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                os.path.join(config.save_path, f"step_{step+1}"),
                "--port",
                str(config.vllm_port),
                "--dtype",
                "auto",
                "--gpu-memory-utilization",
                "0.45",
                "--disable-log-requests",
                "--served-model-name",
                config.model_name,
            ]
            print(f"  Launching vLLM server: {' '.join(vllm_command)}")
            torch.cuda.empty_cache()
            try:
                vllm_process = subprocess.Popen(vllm_command)
                print(f"  vLLM server launched with PID: {vllm_process.pid}")
                # Check immediate errors
                try:
                    stdout, stderr = vllm_process.communicate(timeout=2)
                    if (
                        vllm_process.returncode is not None
                        and vllm_process.returncode != 0
                    ):
                        print(f"  Error starting vLLM: {stderr.decode()}")
                        vllm_process = None
                        # Maybe raise error or just warn?
                        print(
                            "  WARNING: Failed to start vLLM server after checkpoint."
                        )
                except subprocess.TimeoutExpired:
                    print("  vLLM process started (check logs for details).")
            except FileNotFoundError:
                print(
                    "\n *** ERROR: 'python -m vllm...' command not found. ",
                    "Make sure vLLM is installed and accessible. ***\n",
                )
                # Potentially stop training or just disable further vLLM restarts
                print("  Disabling further vLLM restarts.")
                config.vllm_restart_interval = (
                    config.training_steps + 1
                )  # Prevent further restarts
            except Exception as e:
                print(f"\n *** ERROR: Failed to launch vLLM: {e} ***\n")
                print("  Disabling further vLLM restarts.")
                config.vllm_restart_interval = (
                    config.training_steps + 1
                )  # Prevent further restarts
        # --- End vLLM Restart Logic ---

        # Basic check if vLLM process terminated unexpectedly (outside interval check)
        if vllm_process and vllm_process.poll() is not None:
            print(
                f"\n *** WARNING: vLLM process terminated unexpectedly (return code: {vllm_process.returncode}). ",
                "Check vLLM logs. ***\n",
            )
            stderr_output = (
                vllm_process.stderr.read().decode()
                if vllm_process.stderr
                else "No stderr"
            )
            print(f"vLLM stderr: {stderr_output}")
            vllm_process = None  # Reset so it relaunches next interval

    print("Training finished.")
    # --- Wandb Finish ---
    if config.use_wandb:
        wandb.finish()
    # --- End Wandb Finish ---
    # Final cleanup (vLLM termination) is handled by atexit

    # --- Placeholder for final model save ---
    final_save_path = os.path.join(config.save_path, "final_model")
    print(f"Saving final model to {final_save_path}")
    if os.path.exists(final_save_path):
        shutil.rmtree(final_save_path)
    os.makedirs(final_save_path, exist_ok=True)
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print("Final model saved.")


if __name__ == "__main__":
    training_config = TrainingConfig(
        model_name="google/gemma-3-27b-it",
        training_steps=20,
        vllm_restart_interval=3,
        use_wandb=True,
        wandb_project="grpo-physical-trainer",
    )

    train(training_config)
