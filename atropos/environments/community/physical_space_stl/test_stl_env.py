#!/usr/bin/env python3
"""
Test script for the PhysicalEnv that loads and processes STL files
"""
import asyncio

from physical_env import APIServerConfig, BaseEnvConfig, EvalHandlingEnum, PhysicalEnv


async def test_render_stl():
    """Test loading and rendering an STL file"""
    # Create a test environment
    env_config = BaseEnvConfig(
        tokenizer_name="google/gemma-3-27b-it",
        group_size=8,
        use_wandb=False,
        rollout_server_url="http://localhost:8000",
        total_steps=1000,
        batch_size=12,
        steps_per_eval=100,
        eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
        max_token_length=2048,
        wandb_name="physical_test",
    )

    server_configs = [
        APIServerConfig(
            model_name="google/gemma-3-27b-it",
            base_url="http://localhost:9001/v1",
            api_key="x",
            num_requests_for_eval=256,
        ),
    ]

    print("Creating test environment")
    env = PhysicalEnv(env_config, server_configs, slurm=False, testing=True)

    print("Setting up environment")
    await env.setup()

    # Test get_next_item
    print("Testing get_next_item")
    try:
        item = await env.get_next_item()
        print(f"Got item: {item['prompt']}")
        print(f"Image shape: {item['image'].shape}")
        print(f"STL path: {item['stl_path']}")
    except Exception as e:
        print(f"Error getting next item: {e}")

    print("Test completed successfully")


if __name__ == "__main__":
    asyncio.run(test_render_stl())
