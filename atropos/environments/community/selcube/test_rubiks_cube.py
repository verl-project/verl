#!/usr/bin/env python3
"""
Test script for the Rubik's Cube environment
"""

import asyncio
import random

from rubiks_cube_environment import RubiksCubeEnv, RubiksCubeEnvConfig
from rubiks_cube_visualizer import save_cube_visualization
from simple_cube import Cube

from atroposlib.envs.server_handling.server_manager import APIServerConfig


async def test_cube_visualization():
    """Test the cube visualization functionality"""
    # Create a cube
    cube = Cube()

    # Scramble it with some random moves
    moves = [
        "U",
        "D",
        "L",
        "R",
        "F",
        "B",
        "U'",
        "D'",
        "L'",
        "R'",
        "F'",
        "B'",
        "U2",
        "D2",
        "L2",
        "R2",
        "F2",
        "B2",
    ]

    move_history = []
    for _ in range(5):
        move = random.choice(moves)
        move_history.append(move)
        cube.rotate(move)

    # Visualize the scrambled cube
    cube_state = str(cube)
    html_path = save_cube_visualization(
        cube_state, move_history, "test_scrambled_cube.html"
    )

    print(f"Scrambled cube visualization saved to {html_path}")
    print(f"Moves applied: {move_history}")
    print(f"Is solved: {cube.is_solved()}")


async def test_environment():
    """Test the basic functionality of the environment"""
    # Create the environment configuration
    config = RubiksCubeEnvConfig(
        tokenizer_name="gpt2",  # Use a simple tokenizer for testing
        group_size=2,  # Small group size for testing
        use_wandb=False,
        max_steps=5,
        scramble_moves=3,
        debug_mode=True,
    )

    # Create server configuration
    server_configs = [
        APIServerConfig(
            model_name="gpt2",
            base_url="http://localhost:9004/v1",
            api_key="x",
        )
    ]

    # Create the environment
    env = RubiksCubeEnv(config, server_configs, slurm=False, testing=True)

    # Test creating an episode
    seed = 12345
    episode = env._get_or_create_episode(seed)

    # Print initial state
    print(f"Initial cube state (seed {seed}):")
    print(episode.get_cube_state_visualization())

    # Test visualization
    html_path = save_cube_visualization(
        episode.get_cube_state_visualization(), [], "test_initial_cube.html"
    )
    print(f"Initial cube visualization saved to {html_path}")

    # Test applying moves
    test_moves = ["U", "R", "F'"]
    for move in test_moves:
        success = episode.apply_move(move)
        print(f"Applied move {move}: {'Success' if success else 'Failed'}")

    # Check if solved
    print(f"Is solved: {episode.is_solved()}")

    # Test final state visualization
    html_path = save_cube_visualization(
        episode.get_cube_state_visualization(),
        episode.actions,
        "test_after_moves_cube.html",
    )
    print(f"Final cube visualization saved to {html_path}")


if __name__ == "__main__":
    # Run the tests
    print("Running Rubik's Cube environment tests...")
    asyncio.run(test_cube_visualization())
    asyncio.run(test_environment())
    print("Tests completed.")
