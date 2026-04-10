#!/usr/bin/env python3
"""
Compute metrics for existing splits
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.data_obs import DataSplitter, DatasetMetrics
from lib.data_metrics import compute_all_data_metrics
from lib.advanced_metrics import (
    compute_dataset_diversity,
    compute_dataset_entropy,
    compute_ppl_metrics,
    compute_ifd_metrics
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_metrics(output_dir: str, split_ids: list = None, model: str = None):
    """Compute metrics for existing splits"""
    output_dir = Path(output_dir)
    splits_dir = output_dir / "splits"

    if not splits_dir.exists():
        logger.error(f"Splits directory not found: {splits_dir}")
        return

    # Find all split files
    if split_ids is None:
        split_files = sorted(splits_dir.glob("split_*.jsonl"))
    else:
        split_files = [splits_dir / f"split_{i}.jsonl" for i in split_ids]

    logger.info(f"Found {len(split_files)} splits to compute metrics")

    for split_file in split_files:
        split_id = int(split_file.stem.split("_")[1])
        logger.info(f"Processing split {split_id}...")

        # Load data
        data = []
        with open(split_file) as f:
            for line in f:
                data.append(json.loads(line))

        # Compute all metrics
        metrics = compute_all_data_metrics(data, include_difficulty=False)
        metrics.update(compute_dataset_diversity(data))
        metrics.update(compute_dataset_entropy(data))

        # Compute PPL and IFD if model specified
        if model:
            logger.info(f"Computing PPL and IFD with model: {model}")
            ppl_metrics = compute_ppl_metrics(data, model_name=model)
            ifd_metrics = compute_ifd_metrics(data, model_name=model)
            metrics.update(ppl_metrics)
            metrics.update(ifd_metrics)

        # Save metrics
        dataset_metrics = DatasetMetrics(
            split_id=split_id,
            num_samples=len(data),
            metrics=metrics
        )

        metrics_dir = output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_dir / f"split_{split_id}_metrics.json"

        with open(metrics_file, 'w') as f:
            json.dump(dataset_metrics.to_dict(), f, indent=2)

        logger.info(f"Saved metrics for split {split_id}: {len(metrics)} metrics")


def main():
    parser = argparse.ArgumentParser(description="Compute metrics for existing splits")
    parser.add_argument('--output_dir', required=True, help='Output directory containing splits')
    parser.add_argument('--data_name', default=None, help='Dataset name (subdirectory)')
    parser.add_argument('--split_ids', type=int, nargs='+', default=None,
                        help='Specific split IDs to compute (default: all)')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name/path for PPL and IFD computation')

    args = parser.parse_args()

    # Append data_name to output_dir if provided
    if args.data_name:
        output_dir = Path(args.output_dir) / args.data_name
    else:
        output_dir = Path(args.output_dir)

    if not output_dir.exists():
        logger.error(f"Output directory not found: {output_dir}")
        return

    compute_metrics(str(output_dir), args.split_ids, args.model)


if __name__ == '__main__':
    main()