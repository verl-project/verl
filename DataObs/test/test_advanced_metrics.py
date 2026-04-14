#!/usr/bin/env python3
"""
Test script for advanced metrics: text similarity, entropy, and multi-reward difficulty
"""

import sys
import json
from pathlib import Path
from dataclasses import dataclass

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent / "lib"))

from advanced_metrics import (
    compute_text_similarity_matrix,
    compute_dataset_diversity,
    compute_dataset_entropy,
    MultiRewardDifficultyAssessor,
    RewardFunctionConfig,
    compute_advanced_data_metrics,
    reward_exact_match,
    reward_partial_match,
    reward_contains,
    reward_similarity,
)


def create_test_data(n_samples=50):
    """Create test dataset"""
    data = []
    questions = [
        "What is 2 + 2?",
        "What is the capital of France?",
        "What is 5 * 3?",
        "What is the largest planet?",
        "What is the chemical symbol for gold?",
    ]

    for i in range(n_samples):
        q_idx = i % len(questions)
        data.append({
            "data_source": "test/dataset",
            "prompt": [{"role": "user", "content": questions[q_idx]}],
            "ability": "qa",
            "reward_model": {
                "style": "rule",
                "ground_truth": ["4", "Paris", "15", "Jupiter", "Au"][q_idx]
            },
            "extra_info": {
                "split": "train",
                "index": i,
                "answer": ["4", "Paris", "15", "Jupiter", "Au"][q_idx],
                "question": questions[q_idx]
            }
        })

    return data


def test_text_similarity():
    """Test text similarity computation"""
    print("\n" + "="*60)
    print("Test 1: Text Similarity and Diversity")
    print("="*60)

    data = create_test_data(50)

    # Compute diversity
    diversity = compute_dataset_diversity(data, sample_size=30)
    print(f"✓ Diversity metrics computed:")
    for key, value in diversity.items():
        print(f"  - {key}: {value:.4f}")


def test_entropy():
    """Test entropy computation"""
    print("\n" + "="*60)
    print("Test 2: Information Entropy")
    print("="*60)

    data = create_test_data(50)

    # Compute entropy
    entropy = compute_dataset_entropy(data)
    print(f"✓ Entropy metrics computed:")
    for key, value in entropy.items():
        print(f"  - {key}: {value:.4f}")


def test_multi_reward_difficulty():
    """Test multi-reward difficulty assessment"""
    print("\n" + "="*60)
    print("Test 3: Multi-Reward Difficulty Assessment")
    print("="*60)

    data = create_test_data(50)

    # Define reward functions
    reward_configs = [
        RewardFunctionConfig(
            name="exact_match",
            func=reward_exact_match,
            weight=0.5,
            description="Exact string match"
        ),
        RewardFunctionConfig(
            name="partial_match",
            func=reward_partial_match,
            weight=0.3,
            description="Word overlap"
        ),
        RewardFunctionConfig(
            name="contains",
            func=reward_contains,
            weight=0.2,
            description="Contains ground truth"
        ),
    ]

    # Create assessor
    assessor = MultiRewardDifficultyAssessor(reward_configs)

    # Test single sample
    print("\n1. Single sample difficulty:")
    solution = "The answer is 4"
    ground_truth = "4"
    scores = assessor.compute_sample_difficulty(solution, ground_truth)
    print(f"  Solution: '{solution}'")
    print(f"  Ground truth: '{ground_truth}'")
    for name, score in scores.items():
        print(f"    - {name}: {score:.4f}")

    ensemble_score = assessor.compute_ensemble_difficulty(solution, ground_truth)
    print(f"  Ensemble score: {ensemble_score:.4f}")

    # Test dataset difficulty
    print("\n2. Dataset difficulty distribution:")
    difficulty_metrics = assessor.compute_dataset_difficulty(data, sample_size=30)
    for key, value in difficulty_metrics.items():
        if isinstance(value, float):
            print(f"  - {key}: {value:.4f}")


def test_advanced_metrics():
    """Test integrated advanced metrics"""
    print("\n" + "="*60)
    print("Test 4: Integrated Advanced Metrics")
    print("="*60)

    data = create_test_data(50)

    # Define reward functions
    reward_configs = [
        RewardFunctionConfig(
            name="exact_match",
            func=reward_exact_match,
            weight=0.5
        ),
        RewardFunctionConfig(
            name="partial_match",
            func=reward_partial_match,
            weight=0.3
        ),
        RewardFunctionConfig(
            name="similarity",
            func=reward_similarity,
            weight=0.2
        ),
    ]

    # Compute all advanced metrics
    all_metrics = compute_advanced_data_metrics(
        data,
        reward_functions=reward_configs,
        include_similarity=True,
        include_entropy=True,
        include_difficulty=True,
        sample_size=30
    )

    print(f"✓ Computed {len(all_metrics)} advanced metrics:")
    for key, value in sorted(all_metrics.items()):
        if isinstance(value, float):
            print(f"  - {key}: {value:.4f}")


def test_similarity_matrix():
    """Test similarity matrix computation"""
    print("\n" + "="*60)
    print("Test 5: Text Similarity Matrix")
    print("="*60)

    texts = [
        "What is the capital of France?",
        "What is the capital of Germany?",
        "What is 2 + 2?",
        "What is 3 + 3?",
    ]

    # Compute similarity matrix
    similarity_matrix = compute_text_similarity_matrix(texts, method='jaccard')

    print(f"✓ Computed {len(texts)}x{len(texts)} similarity matrix:")
    print("\nTexts:")
    for i, text in enumerate(texts):
        print(f"  {i}: {text}")

    print("\nSimilarity matrix (Jaccard):")
    print("     ", end="")
    for i in range(len(texts)):
        print(f"  {i:6.2f}", end="")
    print()

    for i in range(len(texts)):
        print(f"  {i}: ", end="")
        for j in range(len(texts)):
            print(f"  {similarity_matrix[i, j]:6.2f}", end="")
        print()


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Advanced Metrics Tests")
    print("="*60)

    try:
        test_text_similarity()
        test_entropy()
        test_similarity_matrix()
        test_multi_reward_difficulty()
        test_advanced_metrics()

        print("\n" + "="*60)
        print("✓ All advanced metrics tests passed!")
        print("="*60 + "\n")
        return 0

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
