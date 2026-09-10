#!/usr/bin/env python3
"""
Script to prepare the Polaris dataset for VERL training.
Converts POLARIS-Project/Polaris-Dataset-53K to the format expected by VERL.
"""

import os
import random

import pandas as pd
from datasets import load_dataset


def convert_sample(sample):
    """Convert a Polaris sample to VERL format."""
    return {
        'data_source': 'openai/gsm8k',  # Use GSM8K reward function
        'prompt': [
            {
                'content': sample['problem'] + ' Let\'s think step by step and output the final answer after "####".',
                'role': 'user'
            }
        ],
        'ability': 'math',
        'reward_model': {
            'ground_truth': str(sample['answer']),
            'style': 'rule'
        },
        'extra_info': {
            'answer': f"#### {sample['answer']}",
            'question': sample['problem'],
            'difficulty': sample['difficulty']
        }
    }

def main():
    print("Loading Polaris dataset...")
    ds = load_dataset('POLARIS-Project/Polaris-Dataset-53K')
    polaris_data = ds['train']
    
    print(f"Loaded {len(polaris_data)} samples")
    
    # Convert all samples
    print("Converting samples to training format...")
    converted_data = []
    for i, sample in enumerate(polaris_data):
        converted_data.append(convert_sample(sample))
        if i % 10000 == 0:
            print(f"Processed {i} samples...")
    
    # Create DataFrame
    df = pd.DataFrame(converted_data)
    
    # Split into train/test (80/20 split)
    random.seed(42)
    indices = list(range(len(df)))
    random.shuffle(indices)
    split_idx = int(0.8 * len(df))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # Create directory
    output_dir = '/home/sam/data/polaris'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as parquet files
    train_df.to_parquet(f'{output_dir}/train.parquet', index=False)
    test_df.to_parquet(f'{output_dir}/test.parquet', index=False)
    
    print(f"Saved to {output_dir}/")
    print(f"Train file: {output_dir}/train.parquet")
    print(f"Test file: {output_dir}/test.parquet")
    
    # Show sample
    print("\nSample converted data:")
    print(train_df.iloc[0])

if __name__ == "__main__":
    main() 