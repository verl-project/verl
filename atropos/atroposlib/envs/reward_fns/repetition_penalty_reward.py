"""Reward function for penalizing repetitive content in completions."""

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set

from .registry import registry
from .reward_function import RewardFunction

logger = logging.getLogger(__name__)


@registry.register
class RepetitionPenaltyReward(RewardFunction):
    """
    Reward function that penalizes repetitive content in completions.

    Analyzes various types of repetition:
    1. Repeated sentences/paragraphs
    2. Repeated words beyond natural frequency
    3. Repeated phrases (n-grams)
    4. Consecutive word repetition ("stuttering")
    5. Repeated sentence beginnings
    """

    def __init__(
        self,
        threshold: float = 0.05,
        min_words: int = 10,
        min_sentences: int = 2,
        short_text_penalty: float = -0.1,
        paragraph_repetition_base_penalty: float = -0.6,
        component_weights: Optional[Dict[str, float]] = None,
        stopwords: Optional[Set[str]] = None,
        weight: float = 1.0,
        **kwargs,
    ):
        """
        Initialize repetition penalty reward function.

        Args:
            threshold: Maximum acceptable repetition rate
            min_words: Minimum words required for full analysis
            min_sentences: Minimum sentences required for full analysis
            short_text_penalty: Penalty to apply for very short texts
            paragraph_repetition_base_penalty: Base penalty for repeated paragraphs
            component_weights: Custom weights for each repetition component
            stopwords: Set of common words to ignore in word repetition check
            weight: Weight for this reward
            **kwargs: Additional configuration
        """
        super().__init__(weight=weight, **kwargs)
        self.threshold = threshold
        self.min_words = min_words
        self.min_sentences = min_sentences
        self.short_text_penalty = short_text_penalty
        self.paragraph_repetition_base_penalty = paragraph_repetition_base_penalty

        # Default component weights
        self.component_weights = {
            "word_repetition": 0.3,
            "phrase_repetition": 0.4,
            "consecutive_repetition": 0.6,
            "beginning_repetition": 0.5,
        }

        # Override with custom weights if provided
        if component_weights:
            self.component_weights.update(component_weights)

        # Default stopwords (common words to ignore)
        self.stopwords = stopwords or {
            "this",
            "that",
            "with",
            "from",
            "what",
            "when",
            "where",
            "which",
            "who",
            "whom",
            "whose",
            "will",
            "shall",
            "should",
            "would",
            "could",
            "have",
            "has",
            "had",
            "been",
            "being",
            "than",
            "then",
            "there",
            "these",
            "those",
            "their",
        }

    def compute(self, completions: List[Any], **kwargs) -> List[float]:
        """
        Calculate penalties for repetitive content.

        Args:
            completions: List of completions to evaluate
            **kwargs: Additional context

        Returns:
            List of penalty scores between -1.0 and 0.0 (0.0 = no repetition)
        """
        # Extract content from different possible formats
        completion_contents = [
            self.get_content(completion) for completion in completions
        ]

        rewards = []
        for content in completion_contents:
            try:
                # Split text into words and sentences
                words = re.findall(r"\b\w+\b", content.lower())
                sentences = re.split(r"[.!?]", content)
                sentences = [s.strip() for s in sentences if s.strip()]

                # For very short responses, apply a small penalty to be safe
                if len(words) < self.min_words or len(sentences) < self.min_sentences:
                    logger.info("Text too short for detailed repetition analysis")
                    rewards.append(self.short_text_penalty)
                    continue

                # Check for identical sentences (paragraph repetition)
                sentence_counts = Counter(sentences)
                repeated_sentences = sum(
                    count - 1
                    for sentence, count in sentence_counts.items()
                    if count > 1
                    and len(sentence.split()) > 3  # Only count substantial sentences
                )

                # Severe penalty for repeated paragraphs or sentences
                if repeated_sentences > 0:
                    paragraph_repetition_score = min(
                        1.0, repeated_sentences / len(sentences)
                    )
                    # Apply a strong penalty for paragraph repetition
                    penalty = self.paragraph_repetition_base_penalty - (
                        paragraph_repetition_score * 0.4
                    )
                    logger.info(
                        f"Paragraph repetition detected: {repeated_sentences} instances"
                    )
                    rewards.append(penalty)
                    continue

                # Check for word repetition
                word_counts = Counter(words)
                content_words = [
                    w
                    for w in word_counts.keys()
                    if len(w) > 3 and w not in self.stopwords
                ]
                repeated_words = sum(
                    count - 1  # Count repetitions beyond the first occurrence
                    for word, count in word_counts.items()
                    if count > 2
                    and word in content_words  # Only count meaningful words
                )
                word_repetition_rate = repeated_words / len(words) if words else 0

                # Check for phrase repetition (n-grams)
                phrase_repetition = 0
                if len(words) >= 5:
                    for n in range(3, 6):  # Check 3, 4, and 5-grams
                        if len(words) >= n:
                            ngrams = [
                                " ".join(words[i : i + n])
                                for i in range(len(words) - n + 1)
                            ]
                            ngram_counts = Counter(ngrams)
                            repeated_ngrams = sum(
                                count - 1
                                for phrase, count in ngram_counts.items()
                                if count > 1
                            )
                            phrase_repetition += repeated_ngrams * (
                                n / 3
                            )  # Weight by n-gram size

                phrase_repetition_rate = phrase_repetition / len(words) if words else 0

                # Check for consecutive word repetition (stuttering)
                consecutive_repeats = 0
                for i in range(1, len(words)):
                    if (
                        words[i] == words[i - 1] and len(words[i]) > 2
                    ):  # Ignore short words
                        consecutive_repeats += 1
                consecutive_repetition_rate = (
                    consecutive_repeats / len(words) if words else 0
                )

                # Check for beginning of sentence repetition
                sentence_beginnings = []
                for sentence in sentences:
                    words = sentence.split()
                    if len(words) >= 3:
                        sentence_beginnings.append(" ".join(words[:3]))

                beginning_counts = Counter(sentence_beginnings)
                repeated_beginnings = sum(
                    count - 1
                    for beginning, count in beginning_counts.items()
                    if count > 1
                )
                beginning_repetition_rate = (
                    repeated_beginnings / len(sentences) if sentences else 0
                )

                # Calculate overall repetition score with weights
                repetition_score = (
                    (
                        word_repetition_rate
                        * self.component_weights.get("word_repetition", 0.3)
                    )
                    + (
                        phrase_repetition_rate
                        * self.component_weights.get("phrase_repetition", 0.4)
                    )
                    + (
                        consecutive_repetition_rate
                        * self.component_weights.get("consecutive_repetition", 0.6)
                    )
                    + (
                        beginning_repetition_rate
                        * self.component_weights.get("beginning_repetition", 0.5)
                    )
                )

                # Calculate penalty: 0.0 for no repetition, up to -1.0 for high repetition
                if repetition_score <= self.threshold:
                    penalty = 0.0
                else:
                    # Scale penalty from 0 to -1 based on how much repetition exceeds threshold
                    penalty = -min(
                        1.0, (repetition_score - self.threshold) / (1 - self.threshold)
                    )

                    # Make sure we have at least some penalty for any repetition
                    penalty = min(penalty, -0.1)

                logger.info(
                    f"Word rep: {word_repetition_rate:.3f}, Phrase rep: {phrase_repetition_rate:.3f}, "
                    f"Consecutive rep: {consecutive_repetition_rate:.3f}, ",
                    f"Sentence rep: {beginning_repetition_rate:.3f}, "
                    f"Overall score: {repetition_score:.3f}, penalty: {penalty:.3f}",
                )
                rewards.append(penalty)

            except Exception as e:
                logger.error(f"Error in repetition penalty calculation: {e}")
                logger.exception(e)
                rewards.append(-0.2)  # Apply small penalty on error

        return rewards


# Legacy function for backward compatibility
def repetition_penalty_reward(
    completions: List[Any], threshold: float = 0.05, **kwargs
) -> List[float]:
    """
    Legacy function wrapper for RepetitionPenaltyReward.

    Args:
        completions: List of completions to evaluate
        threshold: Maximum acceptable repetition rate (default 0.05)
                  Lower threshold means stricter penalties for repetition
        **kwargs: Additional parameters

    Returns:
        List of rewards between -1.0 and 0.0, where 0.0 means no repetition
    """
    reward_fn = RepetitionPenaltyReward(threshold=threshold)
    return reward_fn.compute(completions, **kwargs)
