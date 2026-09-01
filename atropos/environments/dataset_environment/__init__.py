"""
Dataset Environment for training models with Hugging Face datasets.

This environment provides a flexible way to train models using a variety of datasets
from Hugging Face or other sources, evaluating generations against reference answers
with customizable reward functions.
"""

from environments.dataset_environment.dataset_env import DatasetEnv, DatasetEnvConfig

__all__ = ["DatasetEnv", "DatasetEnvConfig"]
