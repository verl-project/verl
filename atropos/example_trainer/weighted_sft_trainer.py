"""
Example Weighted SFT Trainer using Atropos Interface

This demonstrates how to integrate the WeightedSFTInterface with Atropos
for loss-masked weighted supervised fine-tuning.
"""

import json
import math
import os
import random
import requests
import string
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from tenacity import retry, stop_after_attempt, wait_exponential

# Import our weighted SFT interface
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from atropos_sft_interface import WeightedSFTInterface, AtroposBatchProcessor


@dataclass
class WeightedSFTConfig:
    """Configuration for weighted SFT training."""
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    learning_rate: float = 1e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    training_steps: int = 100
    seq_len: int = 512
    save_path: str = "./checkpoints/weighted_sft"
    
    # Atropos API settings
    api_url: str = "http://localhost:8000"
    
    # Weighted SFT settings
    loss_reduction: str = "mean"
    ignore_index: int = -100
    advantage_normalization: str = "batch"
    temperature: float = 1.0
    
    # Logging
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_group: Optional[str] = None
    log_interval: int = 10


class WeightedSFTTrainer:
    """
    Trainer that uses the WeightedSFTInterface for Atropos integration.
    """
    
    def __init__(self, config: WeightedSFTConfig):
        self.config = config
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name, 
            torch_dtype=torch.bfloat16
        )
        self.model.to(config.device)
        self.model.train()
        
        # Initialize optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate)
        
        # Initialize weighted SFT interface
        sft_config = {
            "loss_reduction": config.loss_reduction,
            "ignore_index": config.ignore_index,
            "advantage_normalization": config.advantage_normalization,
            "temperature": config.temperature
        }
        self.sft_interface = WeightedSFTInterface(sft_config)
        
        # Initialize batch processor
        self.batch_processor = AtroposBatchProcessor(
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=config.seq_len
        )
        
        # Setup logging
        if config.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=config.wandb_project,
                    group=config.wandb_group,
                    config=config.__dict__
                )
                self.wandb = wandb
            except ImportError:
                print("wandb not available, disabling logging")
                self.wandb = None
        else:
            self.wandb = None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
    def register_with_atropos(self):
        """Register this trainer with the Atropos API."""
        response = requests.post(
            f"{self.config.api_url}/register",
            json={
                "wandb_group": self.config.wandb_group or "weighted_sft",
                "wandb_project": self.config.wandb_project or "weighted_sft",
                "batch_size": self.config.batch_size * self.config.gradient_accumulation_steps,
                "max_token_len": self.config.seq_len,
                "starting_step": 0,
                "checkpoint_dir": self.config.save_path,
                "save_checkpoint_interval": self.config.training_steps,
                "num_steps": self.config.training_steps,
            },
            timeout=10,
        )
        response.raise_for_status()
        print("Successfully registered with Atropos API")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
    def get_batch_from_atropos(self) -> Optional[List[Dict]]:
        """Get a batch from the Atropos API."""
        response = requests.get(f"{self.config.api_url}/batch", timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("batch")
    
    def train_step(self, batch_data: List[Dict]) -> Dict[str, float]:
        """
        Perform a single training step with weighted SFT loss.
        
        Args:
            batch_data: Batch from Atropos API
            
        Returns:
            Dictionary of metrics
        """
        # Process Atropos batch into our format
        processed_batch = self.batch_processor.process_atropos_batch(batch_data)
        
        # Convert to tensors
        tensors = self.batch_processor.to_tensors(processed_batch, self.config.device)
        
        tokens = tensors["tokens"]
        loss_masks = tensors["loss_masks"]
        advantages = tensors["advantages"]
        
        # Forward pass
        outputs = self.model(tokens)
        logits = outputs.logits
        
        # Compute weighted SFT loss
        loss_results = self.sft_interface.compute_weighted_loss(
            logits=logits,
            tokens=tokens,
            loss_masks=loss_masks,
            advantages=advantages
        )
        
        loss = loss_results["loss"]
        
        # Backward pass
        loss.backward()
        
        # Compute metrics
        with torch.no_grad():
            effective_mask = loss_results["effective_mask"]
            token_losses = loss_results["token_losses"]
            weighted_losses = loss_results["weighted_losses"]
            
            # Basic metrics
            num_tokens = effective_mask.sum().item()
            avg_token_loss = (token_losses * effective_mask).sum().item() / max(num_tokens, 1)
            avg_weighted_loss = (weighted_losses).sum().item() / max(num_tokens, 1)
            
            # Advantage statistics
            valid_advantages = advantages[effective_mask > 0]
            if len(valid_advantages) > 0:
                avg_advantage = valid_advantages.mean().item()
                std_advantage = valid_advantages.std().item()
                pos_advantages = (valid_advantages > 0).float().mean().item()
            else:
                avg_advantage = std_advantage = pos_advantages = 0.0
        
        return {
            "loss": loss.item(),
            "avg_token_loss": avg_token_loss,
            "avg_weighted_loss": avg_weighted_loss,
            "num_tokens": num_tokens,
            "avg_advantage": avg_advantage,
            "std_advantage": std_advantage,
            "pos_advantage_ratio": pos_advantages,
        }
    
    def train(self):
        """Main training loop."""
        print(f"Starting weighted SFT training for {self.config.training_steps} steps")
        print(f"Model: {self.config.model_name}")
        print(f"Device: {self.config.device}")
        
        # Register with Atropos
        self.register_with_atropos()
        
        # Create save directory
        os.makedirs(self.config.save_path, exist_ok=True)
        
        step = 0
        accumulated_metrics = {}
        
        while step < self.config.training_steps:
            # Get batch from Atropos
            batch_data = self.get_batch_from_atropos()
            
            if batch_data is None:
                print("No batch available, waiting...")
                time.sleep(1)
                continue
            
            # Perform training step
            metrics = self.train_step(batch_data)
            
            # Accumulate metrics
            for key, value in metrics.items():
                if key not in accumulated_metrics:
                    accumulated_metrics[key] = []
                accumulated_metrics[key].append(value)
            
            # Update parameters every gradient_accumulation_steps
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            step += 1
            
            # Logging
            if step % self.config.log_interval == 0:
                # Average accumulated metrics
                avg_metrics = {
                    key: sum(values) / len(values) 
                    for key, values in accumulated_metrics.items()
                }
                
                print(f"Step {step}/{self.config.training_steps}")
                print(f"  Loss: {avg_metrics['loss']:.4f}")
                print(f"  Avg Token Loss: {avg_metrics['avg_token_loss']:.4f}")
                print(f"  Avg Weighted Loss: {avg_metrics['avg_weighted_loss']:.4f}")
                print(f"  Avg Advantage: {avg_metrics['avg_advantage']:.4f}")
                print(f"  Pos Advantage Ratio: {avg_metrics['pos_advantage_ratio']:.3f}")
                
                if self.wandb:
                    self.wandb.log(avg_metrics, step=step)
                
                # Reset accumulated metrics
                accumulated_metrics = {}
            
            # Save checkpoint periodically
            if step % 100 == 0 and step > 0:
                checkpoint_path = os.path.join(self.config.save_path, f"step_{step}")
                self.save_checkpoint(checkpoint_path)
        
        # Final save
        final_path = os.path.join(self.config.save_path, "final")
        self.save_checkpoint(final_path)
        
        if self.wandb:
            self.wandb.finish()
        
        print("Training completed!")
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Checkpoint saved to {path}")


def main():
    """Example usage of the weighted SFT trainer."""
    config = WeightedSFTConfig(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        training_steps=200,
        batch_size=2,
        learning_rate=1e-5,
        advantage_normalization="batch",
        use_wandb=False,  # Set to True to enable wandb logging
        wandb_project="atropos-weighted-sft",
    )
    
    trainer = WeightedSFTTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
