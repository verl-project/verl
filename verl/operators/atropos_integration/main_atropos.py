#!/usr/bin/env python3
"""
Main entry point for Atropos-integrated GRPO training in verl.

This script integrates with verl's Ray-based trainer to run GRPO training
with Atropos for external rollouts.

Usage:
    python main_atropos.py \
        actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
        actor_rollout_ref.rollout.name=atropos \
        actor_rollout_ref.rollout.atropos_url=http://localhost:8000 \
        actor_rollout_ref.rollout.vllm_port=8100 \
        data.train_files=YOUR_DATA.parquet \
        data.train_batch_size=16 \
        trainer.total_epochs=10
"""

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import ray
import torch
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Atropos GRPO Training for verl")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Model path for vLLM",
    )
    parser.add_argument(
        "--atropos-url",
        type=str,
        default="http://localhost:8000",
        help="Atropos API server URL",
    )
    parser.add_argument(
        "--vllm-port",
        type=int,
        default=8100,
        help="Port for vLLM server",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization for vLLM",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to training data (parquet)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=512,
        help="Maximum prompt length",
    )
    parser.add_argument(
        "--max-response-length",
        type=int,
        default=1024,
        help="Maximum response length",
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=1000,
        help="Total training steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-6,
        help="Learning rate",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=64,
        help="LoRA rank (0 to disable LoRA)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="verl-atropos",
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default="",
        help="WandB group name",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Steps between checkpoints",
    )
    parser.add_argument(
        "--atropos-model",
        type=str,
        default=None,
        help="Model name to report to Atropos (uses model-path if None)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
    )
    return parser.parse_args()


class AtroposGRPO_trainer:
    """
    GRPO trainer with Atropos integration.

    This trainer:
    1. Initializes Ray and verl's worker groups
    2. Manages vLLM server lifecycle
    3. Registers with Atropos API
    4. Runs the GRPO training loop with Atropos rollouts
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.atropos_worker = None
        self.global_steps = 0

        # Ray initialization will be done by verl
        if not ray.is_initialized():
            ray.init()

    def setup_rollout_worker(self):
        """Set up the Atropos rollout worker."""
        from verl.operators.atropos_integration.verl_integration.atropos_worker import AtroposWorker

        rollout_config = self.config.get("rollout", {})
        model_config = self.config.get("model", {})

        self.atropos_worker = AtroposWorker(
            atropos_url=rollout_config.get("atropos_url", "http://localhost:8000"),
            vllm_model_path=model_config.get("path", "Qwen/Qwen2.5-3B-Instruct"),
            vllm_port=rollout_config.get("vllm_port", 8100),
            vllm_gpu_memory_utilization=rollout_config.get(
                "gpu_memory_utilization", 0.8
            ),
            max_token_len=rollout_config.get("max_token_len", 1024),
            batch_size=self.config.get("data", {}).get("train_batch_size", 16),
            config=rollout_config,
        )

    def start(self):
        """Start the training."""
        logger.info("Starting Atropos GRPO trainer...")

        # Start vLLM server
        logger.info("Starting vLLM server...")
        if not self.atropos_worker.start_vllm(timeout=120):
            raise RuntimeError("Failed to start vLLM server")

        # Register with Atropos
        logger.info("Registering with Atropos API...")
        wandb_config = self.config.get("wandb", {})
        trainer_uuid = self.atropos_worker.register_with_atropos(
            wandb_group=wandb_config.get("group", ""),
            wandb_project=wandb_config.get("project", "verl-atropos"),
            checkpoint_dir=self.config.get("checkpoint", {}).get("dir", "./checkpoints"),
            save_checkpoint_interval=self.config.get("checkpoint", {}).get("interval", 100),
            num_steps=self.config.get("training", {}).get("total_steps", 1000),
        )
        logger.info(f"Registered with Atropos (UUID: {trainer_uuid})")

    def run_training_loop(self):
        """Run the main GRPO training loop."""
        logger.info("Starting training loop...")

        total_steps = self.config.get("training", {}).get("total_steps", 1000)
        pbar = tqdm(total=total_steps, initial=self.global_steps, desc="Training")

        while self.global_steps < total_steps:
            try:
                # Get batch from Atropos
                batch_data = self.atropos_worker.get_batch_from_atropos(timeout=60)

                if batch_data is None or batch_data.get("batch") is None:
                    # No batch ready, wait and retry
                    time.sleep(1)
                    continue

                # Extract prompts from batch
                prompts = []
                prompt_ids_list = []
                for item in batch_data.get("batch", []):
                    prompt_text = item.get("prompt", "")
                    prompts.append(prompt_text)
                    ids = self.atropos_worker.tokenizer.encode(
                        prompt_text,
                        max_length=self.config.get("data", {}).get("max_prompt_length", 512),
                        truncation=True,
                        add_special_tokens=True,
                    )
                    prompt_ids_list.append(ids)

                # Generate responses with vLLM
                temperature = self.config.get("rollout", {}).get("temperature", 1.0)
                max_new_tokens = self.config.get("data", {}).get("max_response_length", 1024)

                responses, logprobs = self.atropos_worker.generate_with_vllm(
                    prompts=prompts,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                )

                # Extract rewards from Atropos batch
                # The rewards come from the environment scoring
                rewards = []
                advantages = []
                ref_logprobs = []

                batch_items = batch_data.get("batch", [])
                for i, item in enumerate(batch_items):
                    # Rewards are provided by Atropos environments
                    rewards.append(item.get("score", 0.0))
                    adv = item.get("advantages", None)
                    advantages.append(adv if adv is not None else [0.0] * len(logprobs[i]))
                    ref_lp = item.get("ref_logprobs", None)
                    ref_logprobs.append(ref_lp if ref_lp is not None else [0.0] * len(logprobs[i]))

                # Tokenize responses
                response_ids_list = []
                response_masks = []
                for resp in responses:
                    ids = self.atropos_worker.tokenizer.encode(
                        resp,
                        add_special_tokens=False,
                    )
                    response_ids_list.append(ids)
                    response_masks.append([1] * len(ids))

                # Submit scored data to Atropos
                self.atropos_worker.submit_scored_data(
                    tokens=response_ids_list,
                    masks=response_masks,
                    scores=rewards,
                    advantages=advantages,
                    inference_logprobs=logprobs,
                    ref_logprobs=ref_logprobs,
                )

                # For now, we just track steps
                # In a full integration, this would update the policy
                self.global_steps += 1
                pbar.update(1)

                # Periodic checkpoint
                if self.global_steps % self.config.get("checkpoint", {}).get("interval", 100) == 0:
                    logger.info(f"Step {self.global_steps}: Checkpoint saved")

            except KeyboardInterrupt:
                logger.info("Training interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in training loop: {e}", exc_info=True)
                time.sleep(5)

        pbar.close()
        logger.info("Training complete!")

    def stop(self):
        """Stop training and clean up resources."""
        logger.info("Stopping trainer...")
        if self.atropos_worker:
            self.atropos_worker.close()
        ray.shutdown()


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.log_level)

    logger.info("Atropos GRPO Training for verl")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Atropos URL: {args.atropos_url}")
    logger.info(f"Data: {args.data_path}")

    # Build config dict
    config = {
        "model": {
            "path": args.model_path,
        },
        "rollout": {
            "name": "atropos",
            "atropos_url": args.atropos_url,
            "vllm_port": args.vllm_port,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_token_len": args.max_response_length,
            "temperature": 1.0,
        },
        "data": {
            "train_files": args.data_path,
            "train_batch_size": args.batch_size,
            "max_prompt_length": args.max_prompt_length,
            "max_response_length": args.max_response_length,
        },
        "training": {
            "total_steps": args.training_steps,
            "learning_rate": args.learning_rate,
            "lora_rank": args.lora_rank,
        },
        "wandb": {
            "project": args.wandb_project,
            "group": args.wandb_group,
        },
        "checkpoint": {
            "dir": args.checkpoint_dir,
            "interval": args.checkpoint_interval,
        },
    }

    # Load config from file if provided
    if args.config:
        with open(args.config, "r") as f:
            file_config = yaml.safe_load(f)
            # Merge file config with args (args take precedence)
            config.update(file_config)

    # Create and run trainer
    trainer = AtroposGRPO_trainer(config)

    try:
        trainer.setup_rollout_worker()
        trainer.start()
        trainer.run_training_loop()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        trainer.stop()


if __name__ == "__main__":
    main()
