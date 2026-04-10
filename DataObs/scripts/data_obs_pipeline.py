"""
DataObs Pipeline: End-to-end data analysis workflow
Splits dataset, computes metrics, trains models, and analyzes correlations
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, List

import pandas as pd
import os
# Add lib to path
# sys.path.insert(0, str(Path(__file__).parent / "lib"))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.data_obs import DataSplitter, GPUAllocator, ResultCollector, DatasetMetrics
from lib.data_metrics import compute_all_data_metrics
from lib.advanced_metrics import compute_dataset_diversity, compute_dataset_entropy
from lib.training_pipeline import TrainingPipeline
from lib.analysis_pipeline import CorrelationAnalyzer, AnalysisVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(data_path: str) -> List[dict]:
    """Load data from parquet or jsonl file"""
    data_path = Path(data_path)

    if data_path.suffix == '.parquet':
        import pyarrow.parquet as pq
        table = pq.read_table(data_path)
        data = table.to_pandas().to_dict('records')
    elif data_path.suffix == '.jsonl':
        data = []
        with open(data_path) as f:
            for line in f:
                data.append(json.loads(line))
    elif data_path.suffix == '.json':
        with open(data_path) as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")

    logger.info(f"Loaded {len(data)} samples from {data_path}")
    return data


def main():
    parser = argparse.ArgumentParser(
        description='DataObs Pipeline: Data analysis and training workflow'
    )
    parser.add_argument('--data_path', required=True, help='Path to dataset (parquet/jsonl/json)')
    parser.add_argument('--data_name', default=None, help='Dataset name for output subdirectory')
    parser.add_argument('--n_splits', type=int, default=10, help='Number of data splits')
    parser.add_argument('--output_dir', required=True, help='Output directory for experiment')
    parser.add_argument('--gpu_ids', default='0,1,2,3,4,5,6,7', help='Available GPU IDs (comma-separated)')
    parser.add_argument('--gpus_per_split', type=int, default=1, help='GPUs per training split')
    parser.add_argument('--train_script', default='scripts/sft.sh', help='Training script path')
    parser.add_argument('--model_id', default=None, help='Model ID for training')
    parser.add_argument('--cot_datasynth_dir', default="/home/hrh/COT-DataSynth", help='CoT-DataSynth directory')
    parser.add_argument('--run_training', action='store_true', help='Run training phase')
    parser.add_argument('--run_analysis', action='store_true', help='Run analysis phase')
    parser.add_argument('--skip_split', action='store_true', help='Skip data splitting (use existing splits)')
    parser.add_argument('--skip_metrics', action='store_true', help='Skip metric computation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Setup paths
    output_dir = Path(args.output_dir)
    if args.data_name is None:
        args.data_name = Path(args.data_path).stem
    output_dir = output_dir / args.data_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cot_datasynth_dir is None:
        # Try to find CoT-DataSynth directory
        current_dir = Path(__file__).parent
        if (current_dir.parent / 'verl').exists():
            args.cot_datasynth_dir = str(current_dir.parent)
        else:
            raise ValueError("Could not find CoT-DataSynth directory. Please specify --cot_datasynth_dir")

    logger.info(f"Output directory: {output_dir}") 
    logger.info(f"CoT-DataSynth directory: {args.cot_datasynth_dir}")

    # Phase 1: Data Splitting
    if not args.skip_split:
        logger.info("=" * 50)
        logger.info("Phase 1: Data Splitting")
        logger.info("=" * 50)

        data = load_data(args.data_path)
        splitter = DataSplitter(args.n_splits, str(output_dir))
        splits = splitter.split_dataset(data, seed=args.seed)

        for split_id, split_data in enumerate(splits):
            splitter.save_split(split_data, split_id, format='jsonl')
 
        logger.info(f"Saved {args.n_splits} splits to {output_dir}/splits/")
    else:
        logger.info("Skipping data splitting (using existing splits)")
        splitter = DataSplitter(args.n_splits, str(output_dir))

    # Phase 2: Compute Data Metrics
    if not args.skip_metrics:
        logger.info("=" * 50)
        logger.info("Phase 2: Computing Data Metrics")
        logger.info("=" * 50)

        metrics_list = []
        for split_id in range(args.n_splits):
            split_file = output_dir / "splits" / f"split_{split_id}.jsonl"
            if not split_file.exists():
                logger.warning(f"Split file not found: {split_file}")
                continue

            # Load split data
            split_data = []
            with open(split_file) as f:
                for line in f:
                    split_data.append(json.loads(line))

            # Compute metrics
            metrics = compute_all_data_metrics(split_data, include_difficulty=False)

            # Compute advanced metrics
            diversity_metrics = compute_dataset_diversity(split_data)
            entropy_metrics = compute_dataset_entropy(split_data)
            metrics.update(diversity_metrics)
            metrics.update(entropy_metrics)

            dataset_metrics = DatasetMetrics(
                split_id=split_id,
                num_samples=len(split_data),
                metrics=metrics
            )
            splitter.save_metrics(dataset_metrics)
            metrics_list.append(dataset_metrics.to_dict())

        # Save summary
        metrics_df = pd.DataFrame(metrics_list)
        metrics_csv = output_dir / "data_metrics_summary.csv"
        metrics_df.to_csv(metrics_csv, index=False)
        logger.info(f"Saved metrics summary to {metrics_csv}")

    # Phase 3: Training
    if args.run_training:
        logger.info("=" * 50)
        logger.info("Phase 3: Training")
        logger.info("=" * 50)

        # Setup GPU allocation
        gpu_ids = [int(g) for g in args.gpu_ids.split(',')]
        allocator = GPUAllocator(gpu_ids, args.gpus_per_split)
        gpu_allocations = allocator.allocate(args.n_splits)

        # Setup training pipeline
        training_pipeline = TrainingPipeline(str(output_dir), args.cot_datasynth_dir)

        # Prepare training configs
        base_config = {}
        if args.model_id:
            base_config['model.partial_pretrain'] = args.model_id

        configs = training_pipeline.prepare_training_configs(
            str(output_dir / "splits"),
            gpu_allocations,
            base_config,
            args.n_splits
        )

        # Run trainings
        logger.info(f"Running {len(configs)} trainings...")
        results = training_pipeline.run_all_trainings(
            configs,
            args.train_script,
            parallel=False,
            timeout=None
        )

        logger.info(f"Training results: {results}")

    # Phase 4: Analysis
    if args.run_analysis:
        logger.info("=" * 50)
        logger.info("Phase 4: Analysis and Visualization")
        logger.info("=" * 50)

        # Load data metrics
        metrics_csv = output_dir / "data_metrics_summary.csv"
        if not metrics_csv.exists():
            logger.warning(f"Metrics file not found: {metrics_csv}")
            return

        data_metrics_df = pd.read_csv(metrics_csv)

        # Load training results (placeholder)
        # In real scenario, this would be loaded from training outputs
        logger.info("Note: Training results loading not implemented yet")
        logger.info("Please manually collect training results and save to training_results.json")

        # Example: Create dummy training results for demonstration
        training_results_df = pd.DataFrame({
            'split_id': range(args.n_splits),
            'accuracy': [0.5 + 0.05 * i for i in range(args.n_splits)],
            'loss': [1.0 - 0.05 * i for i in range(args.n_splits)]
        })

        # Compute correlations
        analyzer = CorrelationAnalyzer(str(output_dir / "results"))
        correlations = analyzer.compute_correlations(data_metrics_df, training_results_df)
        correlations.to_csv(output_dir / "results" / "correlations.csv", index=False)
        logger.info(f"Saved correlations to {output_dir}/results/correlations.csv")

        # Generate visualizations
        visualizer = AnalysisVisualizer(str(output_dir / "results"))
        visualizer.plot_correlation_heatmap(data_metrics_df, training_results_df)
        visualizer.plot_scatter_matrix(data_metrics_df, training_results_df)

        logger.info("Analysis completed!")

    logger.info("=" * 50)
    logger.info("Pipeline completed!")
    logger.info(f"Results saved to {output_dir}")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()
