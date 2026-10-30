"""
Utility functions for DynastAI

This module provides utility functions for the DynastAI game.
"""

import json
import os
import random
import uuid
from datetime import datetime
from typing import Any, Dict, List


def ensure_dir_exists(path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary
    """
    os.makedirs(path, exist_ok=True)


def save_json(data: Any, filepath: str) -> None:
    """
    Save data to a JSON file
    """
    # Ensure the directory exists
    ensure_dir_exists(os.path.dirname(filepath))

    # Write the data
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Any:
    """
    Load data from a JSON file
    """
    if not os.path.exists(filepath):
        return None

    with open(filepath, "r") as f:
        return json.load(f)


def append_to_json_list(item: Any, filepath: str) -> None:
    """
    Append an item to a JSON list file
    """
    data = []

    # Load existing data if file exists
    if os.path.exists(filepath):
        data = load_json(filepath) or []

    # Ensure it's a list
    if not isinstance(data, list):
        data = [data]

    # Append the new item
    data.append(item)

    # Save back to the file
    save_json(data, filepath)


def generate_id() -> str:
    """
    Generate a unique ID
    """
    return str(uuid.uuid4())


def timestamp() -> str:
    """
    Get a formatted timestamp
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize weights so they sum to 1.0
    """
    total = sum(weights.values())
    if total == 0:
        # Handle case where all weights are 0
        n = len(weights)
        return {k: 1.0 / n for k in weights}
    return {k: v / total for k, v in weights.items()}


def weighted_choice(options: List[str], weights: Dict[str, float]) -> str:
    """
    Make a weighted random choice from a list of options
    """
    # Extract weights for the given options
    option_weights = [weights.get(option, 1.0) for option in options]

    # Ensure no negative weights
    option_weights = [max(0.001, w) for w in option_weights]

    # Make the weighted choice
    return random.choices(options, weights=option_weights, k=1)[0]


def clamp(value: float, minimum: float, maximum: float) -> float:
    """
    Clamp a value between minimum and maximum
    """
    return max(minimum, min(maximum, value))


class MetricsTracker:
    """
    Track and analyze metrics over time
    """

    def __init__(self):
        self.metrics_history = {}

    def add_metrics(self, session_id: str, metrics: Dict[str, Any]) -> None:
        """Add metrics for a session"""
        if session_id not in self.metrics_history:
            self.metrics_history[session_id] = []

        # Add timestamp
        metrics_with_time = metrics.copy()
        metrics_with_time["timestamp"] = timestamp()

        self.metrics_history[session_id].append(metrics_with_time)

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get metrics history for a session"""
        return self.metrics_history.get(session_id, [])

    def get_average_metrics(self, session_id: str) -> Dict[str, float]:
        """Get average metrics for a session"""
        history = self.get_history(session_id)
        if not history:
            return {}

        # Initialize with keys from the first metrics entry
        result = {
            k: 0
            for k in history[0]
            if isinstance(history[0][k], (int, float)) and k != "timestamp"
        }

        # Sum up values
        for metrics in history:
            for k in result:
                if k in metrics and isinstance(metrics[k], (int, float)):
                    result[k] += metrics[k]

        # Calculate averages
        count = len(history)
        for k in result:
            result[k] /= count

        return result
