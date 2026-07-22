import numpy as np


def get_std_min_max_avg(name: str, data: list, metrics_dict: dict) -> dict:
    """
    Calculate the standard deviation, minimum, maximum, and average of a list of numbers.
    Adds it to the wandb dict for logging.

    Args:
        data (list): A list of numbers.

    Returns:
        dict: A dictionary containing the standard deviation, minimum, maximum, and average.
    """
    metrics_dict[f"{name}_mean"] = np.mean(data)
    metrics_dict[f"{name}_std"] = np.std(data)
    metrics_dict[f"{name}_max"] = np.max(data)
    metrics_dict[f"{name}_min"] = np.min(data)
    return metrics_dict
