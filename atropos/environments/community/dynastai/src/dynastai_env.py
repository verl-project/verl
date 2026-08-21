"""
DynastAI Environment - Atropos-compatible Python RL environment

This module implements the DynastAI game environment as an Atropos environment.
It integrates with FastAPI for the web interface and OpenRouter for card generation.
"""

import os
import uuid
from typing import Dict, List, Tuple, Union

from dotenv import load_dotenv

# Try to import from atroposlib, but provide fallbacks for standalone mode
try:
    from atroposlib.envs.base import BaseEnv, BaseEnvConfig
    from atroposlib.envs.server_handling.server_baseline import ServerBaseline
    from atroposlib.envs.server_handling.server_manager import APIServerConfig

    HAS_ATROPOSLIB = True
except ImportError:
    # Create minimal stub classes for standalone mode
    class BaseEnvConfig:
        pass

    class BaseEnv:
        def __init__(self, *args, **kwargs):
            pass

    class ServerBaseline:
        pass

    class APIServerConfig:
        pass

    HAS_ATROPOSLIB = False
    print("Running in standalone mode without atroposlib")

from .game_logic import GameState, apply_choice_effects

# Load environment variables from .env file
load_dotenv()


class DynastAIEnvConfig(BaseEnvConfig):
    """
    Configuration class for DynastAI Environment
    """

    card_template_count: int = 400  # Number of base card templates
    api_host: str = "localhost"
    api_port: int = 9001
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    llm_model: str = "qwen/Qwen1.5-7B"  # Default model to use
    initial_category_weights: Dict[str, int] = {
        "power": 50,
        "stability": 50,
        "piety": 50,
        "wealth": 50,
    }

    # Optional web UI configuration
    web_ui: bool = True
    web_port: int = 3000


class DynastAIEnv(BaseEnv):
    """
    DynastAI Atropos Environment
    """

    name = "dynastai"
    env_config_cls = DynastAIEnvConfig

    def __init__(
        self,
        config: DynastAIEnvConfig = None,
        server_configs: Union[ServerBaseline, List[APIServerConfig]] = None,
        slurm=False,
        testing=False,
    ):
        if HAS_ATROPOSLIB:
            super().__init__(config, server_configs, slurm, testing)

        # In standalone mode, initialize with default config if none provided
        if config is None:
            config = DynastAIEnvConfig()

        self.config = config

        # Game state storage (in-memory keyed by session_id)
        self.game_states: Dict[str, GameState] = {}

        # Category weights for adaptive card selection
        self.category_weights = config.initial_category_weights.copy()

        # Replay storage
        self.replay_history = []

    @classmethod
    def config_init(
        cls,
    ) -> Tuple[DynastAIEnvConfig, Union[ServerBaseline, List[APIServerConfig]]]:
        """
        Initialize the environment configuration
        """
        return cls.env_config_cls(), ServerBaseline()

    async def reset(self):
        """
        Reset the environment
        """
        # Create a new game session
        session_id = str(uuid.uuid4())
        self.game_states[session_id] = GameState()
        return {
            "session_id": session_id,
            "metrics": self.game_states[session_id].get_metrics(),
        }

    async def step(self, action):
        """
        Take a step in the environment based on the action

        Parameters:
        - action: dict containing action information including:
            - session_id: string
            - choice: "yes" or "no"

        Returns:
        - observation: dict containing new state
        - reward: float
        - done: boolean
        - info: dict with additional info
        """
        session_id = action.get("session_id")
        choice = action.get("choice")

        if session_id not in self.game_states:
            raise ValueError(f"Invalid session ID: {session_id}")

        game_state = self.game_states[session_id]

        # Apply the effects of the choice
        is_game_over, metrics, effects = apply_choice_effects(game_state, choice)

        # Generate observation
        observation = {"metrics": metrics, "current_card": game_state.current_card}

        # Calculate reward (0 unless game is over)
        reward = 0.0
        done = is_game_over

        # If game is over, calculate final reward based on the adaptive reward mechanism
        if done:
            reward = self._calculate_adaptive_reward(game_state)
            # Update category weights
            self._update_category_weights(game_state)

        # Additional info
        info = {"effects": effects}

        return observation, reward, done, info

    def _calculate_adaptive_reward(self, game_state):
        """
        Calculate the adaptive reward based on the game state

        R = power_final * P + stability_final * S + piety_final * Pi + wealth_final * W
        Where:
        - P = number of Power cards played this reign
        - S = number of Stability cards played this reign
        - Pi = number of Piety cards played this reign
        - W = number of Wealth cards played this reign
        """
        # Get final metrics
        metrics = game_state.get_metrics()

        # Count cards by category
        category_counts = game_state.get_category_counts()

        # Calculate reward
        reward = (
            metrics["power"] * category_counts["power"]
            + metrics["stability"] * category_counts["stability"]
            + metrics["piety"] * category_counts["piety"]
            + metrics["wealth"] * category_counts["wealth"]
        )

        return reward

    def _update_category_weights(self, game_state):
        """
        Update category weights using exponential moving average (EMA)

        weights["power"]     = 0.9 * weights["power"]     + 0.1 * (power_final     * P_last)
        weights["stability"] = 0.9 * weights["stability"] + 0.1 * (stability_final * S_last)
        weights["piety"]     = 0.9 * weights["piety"]     + 0.1 * (piety_final     * Pi_last)
        weights["wealth"]    = 0.9 * weights["wealth"]    + 0.1 * (wealth_final    * W_last)
        """
        alpha = 0.9  # Weight for the old value
        beta = 0.1  # Weight for the new value

        # Get final metrics
        metrics = game_state.get_metrics()

        # Count cards by category
        category_counts = game_state.get_category_counts()

        # Update weights
        for category in self.category_weights:
            self.category_weights[category] = alpha * self.category_weights[
                category
            ] + beta * (metrics[category] * category_counts[category])

        # Log the updated weights
        print(f"Updated category weights: {self.category_weights}")
