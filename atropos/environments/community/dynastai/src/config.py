"""
Configuration for DynastAI

This module holds configuration parameters for the DynastAI game environment.
"""

import os
from typing import Dict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


class DynastAIConfig(BaseModel):
    """
    Configuration class for DynastAI game
    """

    # API configuration
    api_host: str = Field(default="localhost", description="API server host")
    api_port: int = Field(default=9001, description="API server port")

    # Web UI configuration
    web_enabled: bool = Field(default=True, description="Enable web UI")
    web_port: int = Field(default=3000, description="Web UI port")

    # Game configuration
    initial_metrics: Dict[str, int] = Field(
        default={
            "power": 50,
            "stability": 50,
            "piety": 50,
            "wealth": 50,
            "reign_year": 1,
        },
        description="Initial game metrics",
    )

    initial_category_weights: Dict[str, float] = Field(
        default={"power": 50.0, "stability": 50.0, "piety": 50.0, "wealth": 50.0},
        description="Initial category weights for card selection",
    )

    # OpenRouter configuration
    openrouter_api_key: str = Field(
        default=os.getenv("OPENROUTER_API_KEY", ""), description="OpenRouter API key"
    )

    openrouter_model: str = Field(
        default="qwen/Qwen1.5-7B", description="OpenRouter model to use"
    )

    # Data storage configuration
    data_dir: str = Field(default="data", description="Directory for storing game data")

    # Card configuration
    cards_file: str = Field(
        default="cards.json", description="Filename for storing cards"
    )

    use_local_cards: bool = Field(
        default=True, description="Whether to use cards from local JSON file"
    )

    use_api_cards: bool = Field(
        default=True, description="Whether to generate cards using OpenRouter API"
    )

    # Game difficulty settings
    min_effect_value: int = Field(default=-20, description="Minimum effect value")
    max_effect_value: int = Field(default=20, description="Maximum effect value")
    normal_effect_range: tuple = Field(
        default=(-10, 10), description="Normal range for effects"
    )

    # Adaptive weighting parameters
    weight_alpha: float = Field(
        default=0.9, description="Alpha parameter for EMA weight update"
    )
    weight_beta: float = Field(
        default=0.1, description="Beta parameter for EMA weight update"
    )


def get_config() -> DynastAIConfig:
    """
    Returns the configuration object with values from environment variables
    """
    return DynastAIConfig()
