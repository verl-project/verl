import os
import random
import sys

import numpy as np
from dotenv import load_dotenv

# Add the parent directory to the path so we can import our modules
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # noqa: E402

# Import our TutorEnv
from tutor_env import TutorEnv  # noqa: E402


# This would be replaced with Atropos imports in a full implementation
# For now, we'll simulate the RL loop
class SimpleAgent:
    """
    A simple agent that selects actions for the TutorEnv.
    This is a placeholder for an actual Atropos policy.
    """

    def __init__(self, action_space):
        """Initialize with the action space of the environment."""
        self.action_space = action_space
        self.last_rewards = []
        self.action_values = np.ones(action_space.n) * 0.5  # Initialize values

    def select_action(self, observation):
        """
        Select an action based on the current observation.
        Uses simple epsilon-greedy strategy.
        """
        # Exploration-exploitation trade-off
        epsilon = 0.2

        if random.random() < epsilon:
            # Explore: random action
            return self.action_space.sample()
        else:
            # Exploit: best action based on current values
            return np.argmax(self.action_values)

    def update(self, action, reward):
        """Update action values based on reward."""
        # Simple update rule
        learning_rate = 0.1
        self.action_values[action] = (1 - learning_rate) * self.action_values[
            action
        ] + learning_rate * reward
        self.last_rewards.append(reward)


def run_episode(env, agent, max_steps=10):
    """Run a single episode of the environment."""
    observation, info = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    while not done and step_count < max_steps:
        # Select action
        action = agent.select_action(observation)

        # Take step in environment
        next_observation, reward, done, truncated, info = env.step(action)

        # Render environment (for human readability)
        env.render()

        # Update agent
        agent.update(action, reward)

        # Update tracking variables
        observation = next_observation
        total_reward += reward
        step_count += 1

    return total_reward


def main():
    """Main function to run the tutoring environment."""
    # Load environment variables
    load_dotenv()

    # Check for API key
    api_key = os.getenv("NOUS_API_KEY")
    if not api_key:
        print("Warning: No NOUS_API_KEY found in environment variables.")
        print("Please set this key in your .env file.")
        return

    # Path to student profile
    profile_path = "config/example_profile.json"

    # Create environment
    env = TutorEnv(profile_path=profile_path, render_mode="human")

    # Create agent
    agent = SimpleAgent(env.action_space)

    # Run multiple episodes
    num_episodes = 5
    episode_rewards = []

    print("\n=== Starting Training ===")
    for episode in range(num_episodes):
        print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
        episode_reward = run_episode(env, agent)
        episode_rewards.append(episode_reward)
        print(
            f"\nEpisode {episode + 1} completed with total reward: {episode_reward:.2f}"
        )

    # Print training summary
    print("\n=== Training Summary ===")
    print(f"Average episode reward: {np.mean(episode_rewards):.2f}")
    print(f"Action values learned: {agent.action_values}")

    # Close environment
    env.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
