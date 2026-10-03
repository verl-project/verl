"""
Weighted SFT Environment for Atropos

This environment generates synthetic data with token-level advantages
for testing the weighted SFT interface. It can be used to validate
the integration and test different advantage patterns.
"""

import asyncio
import json
import random
import time
from typing import Dict, List, Optional

import aiohttp
from transformers import AutoTokenizer

from atroposlib.envs.base import ScoredDataGroup
from atroposlib.type_definitions import Message


class WeightedSFTEnvironment:
    """
    Environment that generates synthetic training data with token-level advantages.
    
    This environment creates various patterns of advantages to test different
    scenarios in weighted SFT training.
    """
    
    def __init__(
        self,
        tokenizer_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        api_url: str = "http://localhost:8000",
        env_name: str = "weighted_sft_env",
        max_seq_length: int = 256,
        advantage_patterns: Optional[List[str]] = None
    ):
        """
        Initialize the weighted SFT environment.
        
        Args:
            tokenizer_name: Name of the tokenizer to use
            api_url: URL of the Atropos API
            env_name: Name for this environment
            max_seq_length: Maximum sequence length
            advantage_patterns: List of advantage patterns to use
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.api_url = api_url
        self.env_name = env_name
        self.max_seq_length = max_seq_length
        
        # Define advantage patterns
        self.advantage_patterns = advantage_patterns or [
            "uniform_positive",
            "uniform_negative", 
            "increasing",
            "decreasing",
            "alternating",
            "sparse_positive",
            "end_weighted",
            "beginning_weighted"
        ]
        
        # Sample texts for training
        self.sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language for data science.",
            "The weather today is sunny and warm.",
            "Books are a great source of knowledge and entertainment.",
            "Cooking is both an art and a science.",
            "Music has the power to evoke strong emotions.",
            "Exercise is important for maintaining good health.",
            "Technology continues to advance at a rapid pace.",
            "Friendship is one of life's greatest treasures."
        ]
    
    async def register_environment(self):
        """Register this environment with the Atropos API."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_url}/register-env",
                json={
                    "max_token_length": self.max_seq_length,
                    "desired_name": self.env_name,
                    "weight": 1.0
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    self.env_id = result.get("env_id")
                    print(f"Registered environment {self.env_name} with ID {self.env_id}")
                else:
                    raise Exception(f"Failed to register environment: {response.status}")
    
    def generate_advantage_pattern(self, pattern_name: str, length: int) -> List[float]:
        """
        Generate advantage values according to the specified pattern.
        
        Args:
            pattern_name: Name of the advantage pattern
            length: Length of the sequence
            
        Returns:
            List of advantage values
        """
        if pattern_name == "uniform_positive":
            return [1.0] * length
        
        elif pattern_name == "uniform_negative":
            return [-0.5] * length
        
        elif pattern_name == "increasing":
            return [i / (length - 1) for i in range(length)]
        
        elif pattern_name == "decreasing":
            return [(length - 1 - i) / (length - 1) for i in range(length)]
        
        elif pattern_name == "alternating":
            return [1.0 if i % 2 == 0 else -0.5 for i in range(length)]
        
        elif pattern_name == "sparse_positive":
            advantages = [0.0] * length
            # Set random 20% of positions to positive
            num_positive = max(1, length // 5)
            positive_indices = random.sample(range(length), num_positive)
            for idx in positive_indices:
                advantages[idx] = 2.0
            return advantages
        
        elif pattern_name == "end_weighted":
            # Higher advantages towards the end
            return [0.1 + 1.9 * (i / (length - 1))**2 for i in range(length)]
        
        elif pattern_name == "beginning_weighted":
            # Higher advantages at the beginning
            return [2.0 - 1.9 * (i / (length - 1))**2 for i in range(length)]
        
        else:
            # Default to uniform positive
            return [1.0] * length
    
    def create_training_sample(self) -> ScoredDataGroup:
        """
        Create a single training sample with token-level advantages.
        
        Returns:
            ScoredDataGroup ready to send to Atropos API
        """
        # Select random text and pattern
        text = random.choice(self.sample_texts)
        pattern = random.choice(self.advantage_patterns)
        
        # Tokenize the text
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Truncate if too long
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]
        
        # Create loss mask (all tokens are valid for training)
        mask = tokens.copy()  # Use tokens as mask (non-zero values)
        
        # Generate advantages
        advantages = self.generate_advantage_pattern(pattern, len(tokens))
        
        # Create sequence-level score (average of token advantages)
        score = sum(advantages) / len(advantages)
        
        # Create the data group
        group = ScoredDataGroup()
        group["tokens"] = [tokens]
        group["masks"] = [mask]
        group["scores"] = [score]
        group["advantages"] = [advantages]
        group["messages"] = [[{"role": "assistant", "content": text, "reward": None}]]
        group["overrides"] = [{"pattern": pattern, "weighted_sft": True}]
        group["group_overrides"] = {"environment": self.env_name, "pattern_type": pattern}
        
        return group
    
    async def send_data_to_api(self, data_group: ScoredDataGroup):
        """Send a data group to the Atropos API."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_url}/scored_data",
                json={
                    "tokens": data_group["tokens"],
                    "masks": data_group["masks"],
                    "scores": data_group["scores"],
                    "advantages": data_group.get("advantages"),
                    "ref_logprobs": data_group.get("ref_logprobs"),
                    "overrides": data_group.get("overrides"),
                    "group_overrides": data_group.get("group_overrides"),
                    "messages": data_group.get("messages")
                }
            ) as response:
                if response.status != 200:
                    print(f"Failed to send data: {response.status}")
                    print(await response.text())
    
    async def run_environment(self, num_samples: int = 1000, delay: float = 0.1):
        """
        Run the environment, generating and sending training samples.
        
        Args:
            num_samples: Number of samples to generate
            delay: Delay between samples in seconds
        """
        print(f"Starting weighted SFT environment: {self.env_name}")
        print(f"Will generate {num_samples} samples with {delay}s delay")
        
        # Register with API
        await self.register_environment()
        
        # Generate and send samples
        for i in range(num_samples):
            try:
                # Create training sample
                sample = self.create_training_sample()
                
                # Send to API
                await self.send_data_to_api(sample)
                
                if (i + 1) % 50 == 0:
                    print(f"Sent {i + 1}/{num_samples} samples")
                
                # Wait before next sample
                await asyncio.sleep(delay)
                
            except Exception as e:
                print(f"Error generating sample {i}: {e}")
                continue
        
        print(f"Environment {self.env_name} completed {num_samples} samples")


async def main():
    """Example usage of the weighted SFT environment."""
    env = WeightedSFTEnvironment(
        tokenizer_name="Qwen/Qwen2.5-1.5B-Instruct",
        api_url="http://localhost:8000",
        env_name="weighted_sft_test",
        max_seq_length=128,
        advantage_patterns=[
            "uniform_positive",
            "increasing", 
            "end_weighted",
            "sparse_positive"
        ]
    )
    
    # Run for 500 samples with 0.05s delay
    await env.run_environment(num_samples=500, delay=0.05)


if __name__ == "__main__":
    asyncio.run(main())
