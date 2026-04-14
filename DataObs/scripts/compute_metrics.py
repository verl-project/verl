#!/usr/bin/env python3
"""
Compute metrics for existing splits
支持单个或全部 metrics 计算，diversity 支持多种相似度函数
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict, Callable

import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline_lib.data_obs import DataSplitter, DatasetMetrics
from pipeline_lib.data_metrics import (
    compute_data_statistics,
    compute_data_quality_metrics,
    compute_all_data_metrics
)
from pipeline_lib.advanced_metrics import (
    compute_dataset_diversity,
    compute_dataset_entropy,
    compute_ppl_metrics,
    compute_ifd_metrics,
    SimilarityType,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Metrics 注册表
METRICS_REGISTRY: Dict[str, Callable] = {
    "statistics": compute_data_statistics,
    "quality": compute_data_quality_metrics,
    "diversity": compute_dataset_diversity,
    "entropy": compute_dataset_entropy,
    "ppl": compute_ppl_metrics,
    "ifd": compute_ifd_metrics,
}


def compute_metrics(
    output_dir: str,
    split_ids: Optional[List[int]] = None,
    metrics: Optional[List[str]] = None,
    model: Optional[str] = None,
    similarity_type: str = "jaccard",
    compute_on: str = "prompt"
):
    """Compute metrics for existing splits

    Args:
        output_dir: Output directory
        split_ids: Specific split IDs to compute (None = all)
        metrics: List of metrics to compute (None = all)
        model: Model name for PPL/IFD computation
        similarity_type: Similarity type for diversity
        compute_on: Compute diversity on "prompt", "answer", "both"
    """
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

    # Convert similarity type
    try:
        sim_type = SimilarityType[similarity_type.upper()]
    except KeyError:
        logger.warning(f"Unknown similarity type: {similarity_type}, using jaccard")
        sim_type = SimilarityType.JACCARD

    # Determine which metrics to compute
    if metrics is None:
        # 没有指定则计算全部
        metrics_to_compute = list(METRICS_REGISTRY.keys())
    else:
        metrics_to_compute = [m.lower() for m in metrics]

    logger.info(f"Computing metrics: {metrics_to_compute}")

    for split_file in split_files:
        split_id = int(split_file.stem.split("_")[1])
        logger.info(f"Processing split {split_id}...")

        # Load data
        data = []
        with open(split_file) as f:
            for line in f:
                data.append(json.loads(line))

        all_metrics = {}

        # Compute each metric
        for metric_name in metrics_to_compute:
            if metric_name not in METRICS_REGISTRY:
                logger.warning(f"Unknown metric: {metric_name}, skipping")
                continue

            func = METRICS_REGISTRY[metric_name]

            try:
                if metric_name == "diversity":
                    result = func(
                        data,
                        sample_size=100,
                        similarity_type=sim_type,
                        compute_on=compute_on
                    )
                elif metric_name in ["ppl", "ifd"]:
                    if model is None:
                        logger.warning(f"Metric '{metric_name}' requires --model, skipping")
                        continue
                    result = func(data, model_name=model)
                else:
                    result = func(data)

                all_metrics.update(result)
                logger.info(f"  {metric_name}: computed {len(result)} metrics")

            except Exception as e:
                logger.error(f"  Failed to compute {metric_name}: {e}")

        # Save metrics
        dataset_metrics = DatasetMetrics(
            split_id=split_id,
            num_samples=len(data),
            metrics=all_metrics
        )

        metrics_dir = output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_dir / f"split_{split_id}_metrics.json"

        with open(metrics_file, 'w') as f:
            json.dump(dataset_metrics.to_dict(), f, indent=2)

        logger.info(f"Saved metrics for split {split_id}: {len(all_metrics)} metrics")


def main():
    parser = argparse.ArgumentParser(description="Compute metrics for existing splits")
    parser.add_argument('--output_dir', required=True, help='Output directory containing splits')
    parser.add_argument('--data_name', default=None, help='Dataset name (subdirectory)')
    parser.add_argument('--split_ids', type=int, nargs='+', default=None,
                        help='Specific split IDs to compute (default: all)')
    parser.add_argument('--metrics', type=str, nargs='+', default=None,
                        help=f'Metrics to compute (default: all). Available: {list(METRICS_REGISTRY.keys())}')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name/path for PPL and IFD computation')
    parser.add_argument('--similarity_type', type=str, default='jaccard',
                        help=f'Similarity type for diversity (default: jaccard). Available: {[s.value for s in SimilarityType]}')
    parser.add_argument('--compute_on', type=str, default='both',
                        choices=['prompt', 'answer', 'both'],
                        help='What to compute diversity on (default: prompt)')

    args = parser.parse_args()

    # Append data_name to output_dir if provided
    if args.data_name:
        output_dir = Path(args.output_dir) / args.data_name
    else:
        output_dir = Path(args.output_dir)

    if not output_dir.exists():
        logger.error(f"Output directory not found: {output_dir}")
        return

    compute_metrics(
        str(output_dir),
        args.split_ids,
        args.metrics,
        args.model,
        args.similarity_type,
        args.compute_on
    )


if __name__ == '__main__':
    main()