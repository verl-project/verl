"""Reward function for evaluating semantic similarity between completions and solutions."""

import logging
from typing import Any, List, Optional, Union

import scipy

try:
    import torch
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(
        "torch not installed, please install atroposlib[rewardfns] to use this reward function"
    )
    raise e


from transformers import AutoModel, AutoTokenizer

from .registry import registry
from .reward_function import RewardFunction

logger = logging.getLogger(__name__)


@registry.register
class CosineScaledReward(RewardFunction):
    """
    Reward function that measures semantic similarity between completions and solutions.

    Uses sentence embeddings to compute cosine similarity, providing higher rewards
    for completions that are semantically similar to the reference solution.
    """

    # Class-level variables for model caching
    _model = None
    _tokenizer = None
    _model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: Optional[str] = None,
        scale_factor: float = 1.0,
        min_reward: float = -1.0,
        max_reward: float = 1.0,
        default_reward: float = 0.0,
        weight: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the cosine similarity reward function.

        Args:
            model_name: Name of embedding model to use (default: "sentence-transformers/all-MiniLM-L6-v2")
            scale_factor: Factor to scale similarity by (default: 1.0)
            min_reward: Minimum reward value (default: -1.0)
            max_reward: Maximum reward value (default: 1.0)
            default_reward: Default reward when similarity can't be calculated (default: 0.0)
            weight: Weight for this reward
            **kwargs: Additional configuration
        """
        super().__init__(weight=weight, **kwargs)
        self.model_name = model_name or self._model_name
        self.scale_factor = scale_factor
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.default_reward = default_reward

        # Initialize model and tokenizer if needed
        self._ensure_model_loaded()

    def _ensure_model_loaded(self):
        """Ensure the model and tokenizer are loaded, loading them if needed."""
        # Check if we need to load a different model than what's cached
        if self.model_name != self._model_name or CosineScaledReward._model is None:
            try:
                CosineScaledReward._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name
                )
                CosineScaledReward._model = AutoModel.from_pretrained(self.model_name)
                CosineScaledReward._model_name = self.model_name
                logger.info(
                    f"Loaded model and tokenizer for cosine similarity: {self.model_name}"
                )
            except Exception as e:
                logger.error(f"Error loading model for cosine similarity: {e}")
                logger.exception(e)
                CosineScaledReward._tokenizer = None
                CosineScaledReward._model = None

    def _mean_pooling(self, model_output, attention_mask):
        """Mean Pooling - Take attention mask into account for correct averaging"""
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def _get_embeddings(self, text):
        """Get embeddings for text using the model"""
        if CosineScaledReward._model is None or CosineScaledReward._tokenizer is None:
            logger.error("Model or tokenizer not available for embeddings")
            return None

        try:
            # Tokenize and prepare for the model
            encoded_input = CosineScaledReward._tokenizer(
                text, padding=True, truncation=True, return_tensors="pt"
            )

            # Get model output
            with torch.no_grad():
                model_output = CosineScaledReward._model(**encoded_input)

            # Perform mean pooling
            sentence_embeddings = self._mean_pooling(
                model_output, encoded_input["attention_mask"]
            )

            return sentence_embeddings.numpy()
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            logger.exception(e)
            return None

    def compute(
        self,
        completions: List[Any],
        solution: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> List[float]:
        """
        Calculate reward based on cosine similarity between completion and solution.

        Args:
            completions: List of completions to evaluate
            solution: The reference solution to compare against
            **kwargs: Additional context

        Returns:
            List of rewards based on cosine similarity, scaled to [min_reward, max_reward]
        """
        # Extract content from different possible formats
        completion_contents = [
            self.get_content(completion) for completion in completions
        ]

        # If no solution provided, can't calculate similarity
        if not solution:
            logger.warning("No solution provided for cosine similarity calculation")
            return [self.default_reward] * len(completion_contents)

        solution_text = (
            solution if isinstance(solution, str) else self.get_content(solution)
        )

        rewards = []
        for content in completion_contents:
            try:
                # Get embeddings
                solution_embedding = self._get_embeddings(solution_text)
                completion_embedding = self._get_embeddings(content)

                if solution_embedding is None or completion_embedding is None:
                    logger.warning("Could not get embeddings for cosine similarity")
                    rewards.append(self.default_reward)
                    continue

                # Calculate cosine similarity
                similarity = scipy.spatial.distance.cosine(
                    solution_embedding.flatten(), completion_embedding.flatten()
                )

                # Scale similarity to a reward between min_reward and max_reward
                # Cosine distance ranges from 0 (similar) to 2 (dissimilar)
                # We want to map 0 → max_reward (good) and 2 → min_reward (bad)
                normalized_similarity = 1.0 - similarity * self.scale_factor
                reward = min(
                    self.max_reward, max(self.min_reward, normalized_similarity)
                )

                logger.info(f"Cosine similarity: {similarity}, scaled reward: {reward}")
                rewards.append(reward)

            except Exception as e:
                logger.error(f"Error in cosine similarity calculation: {e}")
                logger.exception(e)
                rewards.append(self.default_reward)

        return rewards


# Legacy function for backward compatibility
def cosine_scaled_reward(
    completions: List[Any], solution=None, **kwargs
) -> List[float]:
    """
    Legacy function wrapper for CosineScaledReward.

    Args:
        completions: List of completions to evaluate
        solution: The reference solution to compare against
        **kwargs: Additional parameters

    Returns:
        List of rewards based on cosine similarity, scaled to [-1, 1]
    """
    reward_fn = CosineScaledReward()
    return reward_fn.compute(completions, solution=solution, **kwargs)
