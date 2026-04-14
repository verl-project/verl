"""
Data metrics computation module
Calculates statistical and quality metrics for dataset splits
"""

import logging
from typing import Dict, List, Any, Callable, Optional
import numpy as np

logger = logging.getLogger(__name__)


def compute_data_statistics(data: List[Dict]) -> Dict[str, float]:
    """
    Compute basic statistical metrics for a dataset

    Args:
        data: List of data samples (each sample is a dict)

    Returns:
        Dictionary with statistical metrics
    """
    if not data:
        return {}

    metrics = {}
    metrics['num_samples'] = len(data)

    # Compute prompt length statistics
    prompt_lengths = []
    for item in data:
        if 'prompt' in item:
            prompt = item['prompt']
            if isinstance(prompt, list):
                # Assume list of dicts with 'content' field
                content_length = sum(len(str(p.get('content', ''))) for p in prompt)
            else:
                content_length = len(str(prompt))
            prompt_lengths.append(content_length)

    if prompt_lengths:
        metrics['avg_prompt_length'] = float(np.mean(prompt_lengths))
        metrics['std_prompt_length'] = float(np.std(prompt_lengths))
        metrics['min_prompt_length'] = float(np.min(prompt_lengths))
        metrics['max_prompt_length'] = float(np.max(prompt_lengths))

    # Compute response length statistics (if available)
    response_lengths = []
    for item in data:
        if 'extra_info' in item and 'answer' in item['extra_info']:
            response_lengths.append(len(str(item['extra_info']['answer'])))

    if response_lengths:
        metrics['avg_response_length'] = float(np.mean(response_lengths))
        metrics['std_response_length'] = float(np.std(response_lengths))
        metrics['min_response_length'] = float(np.min(response_lengths))
        metrics['max_response_length'] = float(np.max(response_lengths))

    # Count unique data sources
    data_sources = set()
    for item in data:
        if 'data_source' in item:
            data_sources.add(item['data_source'])
    metrics['num_unique_data_sources'] = len(data_sources)

    # Count unique abilities
    abilities = set()
    for item in data:
        if 'ability' in item:
            abilities.add(item['ability'])
    metrics['num_unique_abilities'] = len(abilities)

    logger.info(f"Computed statistics for {len(data)} samples")
    return metrics


def compute_data_quality_metrics(data: List[Dict]) -> Dict[str, float]:
    """
    Compute data quality metrics

    Args:
        data: List of data samples

    Returns:
        Dictionary with quality metrics
    """
    if not data:
        return {}

    metrics = {}

    # Check answer coverage
    has_answer = 0
    for item in data:
        if 'extra_info' in item and 'answer' in item['extra_info']:
            answer = item['extra_info']['answer']
            if answer and str(answer).strip():
                has_answer += 1

    metrics['answer_coverage'] = float(has_answer / len(data)) if data else 0.0

    # Check format validity
    valid_format = 0
    for item in data:
        has_prompt = 'prompt' in item and item['prompt']
        has_reward_model = 'reward_model' in item and item['reward_model']
        if has_prompt and has_reward_model:
            valid_format += 1

    metrics['format_validity'] = float(valid_format / len(data)) if data else 0.0

    # Compute text diversity (simple: unique samples / total samples)
    unique_prompts = set()
    for item in data:
        if 'prompt' in item:
            prompt_str = str(item['prompt'])
            unique_prompts.add(prompt_str)

    metrics['prompt_uniqueness'] = float(len(unique_prompts) / len(data)) if data else 0.0

    logger.info(f"Computed quality metrics for {len(data)} samples")
    return metrics


def compute_difficulty_metrics(
    data: List[Dict],
    reward_func: Callable,
    sample_size: int = 100
) -> Dict[str, float]:
    """
    Compute data difficulty metrics using a reward function

    Args:
        data: List of data samples
        reward_func: Function that takes (solution, ground_truth) and returns score [0, 1]
        sample_size: Number of samples to evaluate (for efficiency)

    Returns:
        Dictionary with difficulty metrics
    """
    if not data:
        return {}

    # Sample data for efficiency
    sample_indices = np.random.choice(len(data), min(sample_size, len(data)), replace=False)
    sampled_data = [data[i] for i in sample_indices]

    difficulties = []
    for item in sampled_data:
        try:
            if 'reward_model' in item and 'ground_truth' in item['reward_model']:
                ground_truth = item['reward_model']['ground_truth']
                # For difficulty, we use 1 - score (higher score = easier)
                # This is a placeholder; actual implementation depends on reward function
                difficulty = 0.5  # Default neutral difficulty
                difficulties.append(difficulty)
        except Exception as e:
            logger.warning(f"Failed to compute difficulty: {e}")

    metrics = {}
    if difficulties:
        metrics['avg_difficulty'] = float(np.mean(difficulties))
        metrics['std_difficulty'] = float(np.std(difficulties))
        metrics['min_difficulty'] = float(np.min(difficulties))
        metrics['max_difficulty'] = float(np.max(difficulties))
    else:
        logger.warning("No valid difficulties computed")

    logger.info(f"Computed difficulty metrics for {len(sampled_data)} samples")
    return metrics


def compute_all_data_metrics(
    data: List[Dict],
    include_difficulty: bool = False,
    reward_func: Optional[Callable] = None
) -> Dict[str, float]:
    """
    Compute all data metrics

    Args:
        data: List of data samples
        include_difficulty: Whether to compute difficulty metrics
        reward_func: Reward function for difficulty computation

    Returns:
        Dictionary with all metrics
    """
    all_metrics = {}

    # Compute statistics
    stats = compute_data_statistics(data)
    all_metrics.update(stats)

    # Compute quality metrics
    quality = compute_data_quality_metrics(data)
    all_metrics.update(quality)

    # Compute difficulty metrics if requested
    if include_difficulty and reward_func:
        difficulty = compute_difficulty_metrics(data, reward_func)
        all_metrics.update(difficulty)

    return all_metrics
