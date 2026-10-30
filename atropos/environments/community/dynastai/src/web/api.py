# FastAPI endpoints for DynastAI game
"""
This module provides the REST API endpoints for the DynastAI game:
- GET /state: Get current game state
- POST /generate_card: Generate a new card
- POST /card_choice: Submit player choice
- POST /end_reign: End a reign and compute rewards
"""

import json
import os
import subprocess
import uuid
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import the game logic
from ..game_logic import GameState, apply_choice_effects, generate_card

# Try to import the environment if running in standalone mode
try:
    from ..dynastai_env import HAS_ATROPOSLIB, DynastAIEnv

    # Create a standalone environment instance if running without atroposlib
    standalone_env = None
    if not HAS_ATROPOSLIB:
        standalone_env = DynastAIEnv()
        # Use the environment's game states and category weights
        game_sessions = standalone_env.game_states
        category_weights = standalone_env.category_weights
    else:
        # In-memory store for game sessions
        game_sessions: Dict[str, GameState] = {}
        # In-memory store for category weights across reigns
        category_weights: Dict[str, float] = {
            "power": 50,
            "stability": 50,
            "piety": 50,
            "wealth": 50,
        }
except ImportError:
    # Fallback if import fails
    game_sessions: Dict[str, GameState] = {}
    category_weights: Dict[str, float] = {
        "power": 50,
        "stability": 50,
        "piety": 50,
        "wealth": 50,
    }

# Path to save reign trajectories
trajectories_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "trajectories.json"
)

# Make sure the data directory exists
os.makedirs(os.path.dirname(trajectories_path), exist_ok=True)


# Define the API models using Pydantic
class NewGameRequest(BaseModel):
    """Request model for creating a new game"""

    session_id: Optional[str] = None  # Optional custom session ID


class GameStateResponse(BaseModel):
    """Response model for game state"""

    session_id: str
    metrics: Dict[str, int]
    current_card: Optional[Dict[str, Any]] = None


class GenerateCardRequest(BaseModel):
    """Request model for generating a new card"""

    session_id: str


class CardChoiceRequest(BaseModel):
    """Request model for submitting a card choice"""

    session_id: str
    choice: str  # "yes" or "no"


class TrajectoryItem(BaseModel):
    """Model for a single trajectory item"""

    card_id: str
    category: str
    choice: str
    effects: Dict[str, Any]
    post_metrics: Dict[str, int]

    class Config:
        extra = "ignore"  # Allow extra fields


class EndReignRequest(BaseModel):
    """Request model for ending a reign"""

    session_id: str
    trajectory: List[TrajectoryItem]
    final_metrics: Dict[str, int]
    reign_length: int
    cause_of_end: Optional[str] = None

    class Config:
        extra = "ignore"  # Allow extra fields


class EndReignResponse(BaseModel):
    """Response model for ending a reign"""

    reward: float
    session_id: str
    new_weights: Dict[str, float]


# Create FastAPI instance
api = FastAPI(
    title="DynastAI API",
    description="REST API for the DynastAI medieval kingdom management game",
    version="1.0.0",
)

# Add CORS middleware to allow requests from any origin
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


@api.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to the DynastAI API"}


@api.post("/new_game", response_model=GameStateResponse)
async def new_game(request: NewGameRequest = None):
    """Create a new game session"""
    # Generate a session ID if not provided
    session_id = (
        request.session_id if request and request.session_id else str(uuid.uuid4())
    )

    # Create a new game state
    game_sessions[session_id] = GameState()

    # Return the initial game state
    return GameStateResponse(
        session_id=session_id,
        metrics=game_sessions[session_id].get_metrics(),
        current_card=None,
    )


@api.get("/state/{session_id}", response_model=GameStateResponse)
async def get_state(session_id: str):
    """Get the current state of a game session"""
    if session_id not in game_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return GameStateResponse(
        session_id=session_id,
        metrics=game_sessions[session_id].get_metrics(),
        current_card=game_sessions[session_id].current_card,
    )


@api.post("/generate_card", response_model=Dict[str, Any])
async def generate_new_card(request: GenerateCardRequest):
    """Generate a new card for the game session"""
    if request.session_id not in game_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    game_state = game_sessions[request.session_id]

    # Check if this session has a history of previous reigns
    # This would be used to adapt card generation based on previous outcomes
    previous_reigns = (
        game_state.previous_reigns if hasattr(game_state, "previous_reigns") else []
    )

    # Generate a new card using the current metrics, category weights, and reign history
    card = generate_card(
        game_state.get_metrics(), category_weights, previous_reigns=previous_reigns
    )

    # Store the card in the game state
    game_state.current_card = card

    # Return the card
    return card


@api.post("/card_choice", response_model=GameStateResponse)
async def process_card_choice(request: CardChoiceRequest):
    """Process a player's card choice"""
    if request.session_id not in game_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    if request.choice not in ["yes", "no"]:
        raise HTTPException(status_code=400, detail="Choice must be 'yes' or 'no'")

    game_state = game_sessions[request.session_id]

    if not game_state.current_card:
        raise HTTPException(status_code=400, detail="No active card for this session")

    # Apply the effects of the choice
    is_game_over, metrics, effects = apply_choice_effects(game_state, request.choice)

    # Return the updated game state along with game_over flag
    return {
        "session_id": request.session_id,
        "metrics": metrics,
        "current_card": game_state.current_card,
        "game_over": is_game_over,
    }


@api.post("/end_reign", response_model=EndReignResponse)
async def end_reign(request: EndReignRequest, background_tasks: BackgroundTasks):
    """
    End a reign and compute reward
    This endpoint receives the entire trajectory of a reign
    """
    if request.session_id not in game_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    # Log the received trajectory for debugging
    print(f"Received trajectory with {len(request.trajectory)} items")

    try:
        # Calculate the adaptive reward
        reward = calculate_adaptive_reward(request.final_metrics, request.trajectory)

        # Update category weights
        update_category_weights(request.final_metrics, request.trajectory)

        # Log the trajectory
        log_trajectory(request, reward)

        # Store previous reign data before resetting
        previous_reign = {
            "final_metrics": request.final_metrics,
            "reign_length": request.reign_length,
            "cause_of_end": request.cause_of_end,
            "reward": reward,
        }

        # Create new game state while preserving reign history
        new_state = GameState()

        # Initialize previous_reigns if needed
        new_state.previous_reigns = []

        # If session already exists, get previous reigns history
        if request.session_id in game_sessions:
            if hasattr(game_sessions[request.session_id], "previous_reigns"):
                new_state.previous_reigns = game_sessions[
                    request.session_id
                ].previous_reigns

        # Add this reign to history
        new_state.previous_reigns.append(previous_reign)

        # Update the session with new state
        game_sessions[request.session_id] = new_state

        # Execute the dynastai_server.py script in the background
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        stdout_file = os.path.join(
            log_dir, f"dynastai_server_stdout_{request.session_id}.log"
        )
        stderr_file = os.path.join(
            log_dir, f"dynastai_server_stderr_{request.session_id}.log"
        )

        background_tasks.add_task(
            subprocess.run,
            [
                "python",
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    "dynastai_server.py",
                ),
                "process",
                "--openai.model_name",
                "Qwen/Qwen3-1.7B",
                "--openai.api_key",
                "x",
                "--openai.base_url",
                "http://localhost:9002/v1",
                "--slurm",
                "false",
                "--env.tokenizer_name",
                "Qwen/Qwen3-1.7B",
                "--env.ensure_scores_are_not_same",
                "false",
                "--env.data_path_to_save_groups",
                "dynastai_rollouts.jsonl",
                "--env.group_size",
                "8",
                "--env.max_num_workers=-1",
                "--env.max_eval_workers=16",
                "--env.max_num_workers_per_node=8",
                "--env.batch_size=-1",
                "--env.max_token_length=2048",
            ],
            stdout=open(stdout_file, "w"),
            stderr=open(stderr_file, "w"),
        )

        return EndReignResponse(
            reward=reward, session_id=request.session_id, new_weights=category_weights
        )

    except Exception as e:
        # Log the error for debugging
        print(f"Error processing end_reign: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Error calculating reward: {str(e)}"
        )


def calculate_adaptive_reward(
    final_metrics: Dict[str, int], trajectory: List[TrajectoryItem]
) -> float:
    """
    Calculate the adaptive reward based on the final metrics and trajectory.

    Following the formula: R = power_final * P + stability_final * S + piety_final * Pi + wealth_final * W
    Where P, S, Pi, W are the counts of cards played in each category
    """
    # Initialize category counts
    category_counts = {"power": 0, "stability": 0, "piety": 0, "wealth": 0}

    # Count cards played in each category
    for item in trajectory:
        try:
            category = item.category.lower()
            if category in category_counts:
                category_counts[category] += 1
        except Exception as e:
            print(f"Error processing trajectory item: {str(e)}, item: {item}")
            continue

    # Calculate reward using the formula from README.md
    try:
        # For each category, multiply final metric value by the count of cards in that category
        reward = 0.0
        reward += final_metrics.get("power", 50) * category_counts["power"]
        reward += final_metrics.get("stability", 50) * category_counts["stability"]
        reward += final_metrics.get("piety", 50) * category_counts["piety"]
        reward += final_metrics.get("wealth", 50) * category_counts["wealth"]

        # If no cards were played, use the average of final metrics as reward
        if sum(category_counts.values()) == 0:
            total = sum(
                final_metrics.get(key, 50)
                for key in ["power", "stability", "piety", "wealth"]
            )
            reward = total / 4.0

        print(f"Calculated reward: {reward} based on:")
        print(f"Final metrics: {final_metrics}")
        print(f"Category counts: {category_counts}")

        return float(reward)

    except Exception as e:
        print(f"Error in adaptive reward calculation: {str(e)}")
        # Provide a fallback reward calculation
        return float(
            sum(
                final_metrics.get(key, 50)
                for key in ["power", "stability", "piety", "wealth"]
            )
            / 4
        )


def update_category_weights(
    final_metrics: Dict[str, int], trajectory: List[TrajectoryItem]
):
    """
    Update category weights using exponential moving average (EMA) based on
    the average per-card adaptive rewards value of its associated metric.
    """
    # Initialize tracking variables
    category_totals = {"power": 0, "stability": 0, "piety": 0, "wealth": 0}
    category_counts = {"power": 0, "stability": 0, "piety": 0, "wealth": 0}

    # Calculate total reward for each category
    for item in trajectory:
        try:
            category = item.category.lower()
            if category in category_totals:
                category_totals[category] += final_metrics.get(category, 50)
                category_counts[category] += 1
        except Exception as e:
            print(f"Error processing category weight for item: {e}")
            continue

    # Update weights using EMA
    alpha = 0.9  # Weight for the old value
    beta = 0.1  # Weight for the new value

    for category in category_weights:
        # Calculate average reward for this category (use current weight if no cards in this category)
        avg_reward = final_metrics.get(category, 50)
        if category_counts[category] > 0:
            avg_reward = category_totals[category] / category_counts[category]

        # Update weight using EMA
        category_weights[category] = (
            alpha * category_weights[category] + beta * avg_reward
        )

        # Ensure weights stay in a reasonable range
        category_weights[category] = max(1, min(100, category_weights[category]))

    print(f"Updated category weights: {category_weights}")


def log_trajectory(request: EndReignRequest, reward: float):
    """Log the trajectory to a JSON file"""
    trajectory_data = {
        "session_id": request.session_id,
        "trajectory": [item.dict() for item in request.trajectory],
        "final_metrics": request.final_metrics,
        "reign_length": request.reign_length,
        "cause_of_end": request.cause_of_end,
        "reward": reward,
        "weights": category_weights,
    }

    try:
        # Load existing trajectories
        trajectories = []
        if os.path.exists(trajectories_path):
            with open(trajectories_path, "r") as f:
                trajectories = json.load(f)

        # Add new trajectory
        trajectories.append(trajectory_data)

        # Save back to file
        with open(trajectories_path, "w") as f:
            json.dump(trajectories, f, indent=2)

    except Exception as e:
        print(f"Error logging trajectory: {e}")
        # Continue without failing the request
