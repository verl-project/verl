#!/usr/bin/env python3
"""
Simple test script to verify DataObs core functionality
"""

import sys
import json
import tempfile
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent / "lib"))

from data_obs import DataSplitter, GPUAllocator, ResultCollector, TrainingResult
from data_metrics import compute_data_statistics, compute_data_quality_metrics


def create_dummy_data(n_samples=100):
    """Create dummy dataset for testing"""
    data = []
    for i in range(n_samples):
        data.append({
            "data_source": "test/dataset",
            "prompt": [
                {
                    "role": "user",
                    "content": f"Question {i}: What is {i} + 1?"
                }
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": str(i + 1)
            },
            "extra_info": {
                "split": "train",
                "index": i,
                "answer": str(i + 1),
                "question": f"What is {i} + 1?"
            }
        })
    return data


def test_data_splitter():
    """Test DataSplitter functionality"""
    print("\n" + "="*50)
    print("Test 1: DataSplitter")
    print("="*50)

    with tempfile.TemporaryDirectory() as tmpdir:
        data = create_dummy_data(100)
        splitter = DataSplitter(n_splits=5, output_dir=tmpdir)

        # Test splitting
        splits = splitter.split_dataset(data, seed=42)
        assert len(splits) == 5, f"Expected 5 splits, got {len(splits)}"
        assert sum(len(s) for s in splits) == 100, "Total samples mismatch"
        print(f"✓ Split 100 samples into 5 parts: {[len(s) for s in splits]}")

        # Test saving splits
        for split_id, split_data in enumerate(splits):
            splitter.save_split(split_data, split_id, format='jsonl')
        print(f"✓ Saved all splits to {tmpdir}/splits/")

        # Test metric computation
        metric_funcs = {
            'num_samples': lambda d: len(d),
            'avg_prompt_length': lambda d: sum(len(str(item['prompt'])) for item in d) / len(d),
        }

        for split_id, split_data in enumerate(splits):
            metrics = splitter.compute_metrics(split_data, metric_funcs, split_id)
            splitter.save_metrics(metrics)
        print(f"✓ Computed and saved metrics for all splits")

        # Test loading metrics
        metrics_df = splitter.load_metrics()
        assert len(metrics_df) == 5, f"Expected 5 metric rows, got {len(metrics_df)}"
        print(f"✓ Loaded metrics: {len(metrics_df)} rows, {len(metrics_df.columns)} columns")


def test_gpu_allocator():
    """Test GPUAllocator functionality"""
    print("\n" + "="*50)
    print("Test 2: GPUAllocator")
    print("="*50)

    # Test with specified GPUs
    allocator = GPUAllocator(available_gpus=[0, 1, 2, 3], gpus_per_split=1)
    allocations = allocator.allocate(n_splits=10)
    assert len(allocations) == 10, f"Expected 10 allocations, got {len(allocations)}"
    print(f"✓ Allocated GPUs for 10 splits: {allocations}")

    # Test with multiple GPUs per split
    allocator = GPUAllocator(available_gpus=[0, 1, 2, 3], gpus_per_split=2)
    allocations = allocator.allocate(n_splits=5)
    assert all(len(a) == 2 for a in allocations), "Expected 2 GPUs per split"
    print(f"✓ Allocated 2 GPUs per split: {allocations}")


def test_result_collector():
    """Test ResultCollector functionality"""
    print("\n" + "="*50)
    print("Test 3: ResultCollector")
    print("="*50)

    with tempfile.TemporaryDirectory() as tmpdir:
        collector = ResultCollector(tmpdir)

        # Add results
        for split_id in range(5):
            result = TrainingResult(
                split_id=split_id,
                metrics={'accuracy': 0.8 + 0.01 * split_id, 'loss': 0.5 - 0.05 * split_id},
                checkpoint_path=f'/path/to/checkpoint_{split_id}'
            )
            collector.add_result(result)

        print(f"✓ Added 5 training results")

        # Load results
        results_df = collector.load_results()
        assert len(results_df) == 5, f"Expected 5 results, got {len(results_df)}"
        print(f"✓ Loaded results: {len(results_df)} rows")

        # Get results dict
        results_dict = collector.get_results_dict()
        assert len(results_dict) == 5, f"Expected 5 result dicts, got {len(results_dict)}"
        print(f"✓ Got results dict with {len(results_dict)} entries")


def test_data_metrics():
    """Test data metrics computation"""
    print("\n" + "="*50)
    print("Test 4: Data Metrics")
    print("="*50)

    data = create_dummy_data(50)

    # Test statistics
    stats = compute_data_statistics(data)
    assert 'num_samples' in stats, "Missing num_samples"
    assert stats['num_samples'] == 50, f"Expected 50 samples, got {stats['num_samples']}"
    print(f"✓ Computed statistics: {len(stats)} metrics")
    print(f"  - num_samples: {stats['num_samples']}")
    print(f"  - avg_prompt_length: {stats.get('avg_prompt_length', 'N/A')}")

    # Test quality metrics
    quality = compute_data_quality_metrics(data)
    assert 'answer_coverage' in quality, "Missing answer_coverage"
    assert 'format_validity' in quality, "Missing format_validity"
    print(f"✓ Computed quality metrics: {len(quality)} metrics")
    print(f"  - answer_coverage: {quality['answer_coverage']:.2%}")
    print(f"  - format_validity: {quality['format_validity']:.2%}")


def main():
    """Run all tests"""
    print("\n" + "="*50)
    print("DataObs Core Functionality Tests")
    print("="*50)

    try:
        test_data_splitter()
        test_gpu_allocator()
        test_result_collector()
        test_data_metrics()

        print("\n" + "="*50)
        print("✓ All tests passed!")
        print("="*50 + "\n")
        return 0

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
