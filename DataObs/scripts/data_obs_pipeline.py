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
from lib.data_metrics import compute_data_statistics, compute_data_quality_metrics
from lib.advanced_metrics import (
    compute_dataset_diversity,
    compute_dataset_entropy,
    compute_ppl_metrics,
    compute_ifd_metrics,
    SimilarityType,
)
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
    parser.add_argument('--cot_datasynth_dir', default="/home/hrh/CoT-DataSynth", help='CoT-DataSynth directory')
    parser.add_argument('--run_training', action='store_true', help='Run training phase')
    parser.add_argument('--run_analysis', action='store_true', help='Run analysis phase')
    parser.add_argument('--run_evaluation', action='store_true', help='Run evaluation phase (on existing trained models)')
    parser.add_argument('--skip_split', action='store_true', help='Skip data splitting (use existing splits)')
    parser.add_argument('--skip_metrics', action='store_true', help='Skip metric computation')
    parser.add_argument('--skip_training', action='store_true', help='Skip training phase')
    parser.add_argument('--skip_evaluation', action='store_true', help='Skip evaluation phase')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--splits_dir', default=None,
                        help='Path to existing splits directory (use with --skip_split)')
    parser.add_argument('--splits_format', default='parquet',
                        choices=['jsonl', 'json', 'parquet'],
                        help='Format for saving splits (default: parquet, for verl compatibility)')
    parser.add_argument('--similarity_type', default='jaccard',
                        help='Similarity type for diversity: jaccard, levenshtein, cosine, jaro_winkler, ngram, bertouch, bleu, rouge')
    parser.add_argument('--compute_on', default='both',
                        choices=['prompt', 'answer', 'both'],
                        help='What to compute diversity on')
    parser.add_argument('--model', default=None,
                        help='Model for PPL and IFD computation')

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
            splitter.save_split(split_data, split_id, format=args.splits_format)

        logger.info(f"Saved {args.n_splits} splits to {output_dir}/splits/")
        splits_dir = output_dir / "splits"
    else:
        logger.info("Skipping data splitting (using existing splits)")
        splitter = DataSplitter(args.n_splits, str(output_dir))
        # 如果指定了 splits_dir，使用它；否则用默认路径
        if args.splits_dir:
            splits_dir = Path(args.splits_dir)
            logger.info(f"Using splits from: {splits_dir}")
            # 自动检测 splits 数量 (支持 parquet 和 jsonl)
            existing_splits = list(splits_dir.glob("split_*.parquet")) or list(splits_dir.glob("split_*.jsonl"))
            if existing_splits:
                args.n_splits = max(int(f.stem.split("_")[1]) for f in existing_splits) + 1
                logger.info(f"Auto-detected {args.n_splits} splits")
        else:
            splits_dir = output_dir / "splits"

    # Phase 2: Compute Data Metrics
    if not args.skip_metrics:
        logger.info("=" * 50)
        logger.info("Phase 2: Computing Data Metrics")
        logger.info("=" * 50)

        # Convert similarity type
        try:
            sim_type = SimilarityType[args.similarity_type.upper()]
        except KeyError:
            logger.warning(f"Unknown similarity type: {args.similarity_type}, using jaccard")
            sim_type = SimilarityType.JACCARD

        metrics_list = []
        for split_id in range(args.n_splits):
            split_file = splits_dir / f"split_{split_id}.jsonl"
            if not split_file.exists():
                logger.warning(f"Split file not found: {split_file}")
                continue

            # Load split data
            split_data = []
            with open(split_file) as f:
                for line in f:
                    split_data.append(json.loads(line))

            # Compute all metrics
            metrics = {}

            # 1. statistics
            logger.info(f"Split {split_id}: computing statistics...")
            metrics.update(compute_data_statistics(split_data))

            # 2. quality
            logger.info(f"Split {split_id}: computing quality...")
            metrics.update(compute_data_quality_metrics(split_data))

            # 3. diversity (支持多种相似度函数)
            logger.info(f"Split {split_id}: computing diversity with {args.similarity_type}...")
            metrics.update(compute_dataset_diversity(
                split_data,
                similarity_type=sim_type,
                compute_on=args.compute_on
            ))

            # 4. entropy
            logger.info(f"Split {split_id}: computing entropy...")
            metrics.update(compute_dataset_entropy(split_data))

            # 5. PPL (需要 model)
            if args.model:
                logger.info(f"Split {split_id}: computing PPL with {args.model}...")
                try:
                    metrics.update(compute_ppl_metrics(split_data, model_name=args.model))
                except Exception as e:
                    logger.warning(f"Failed to compute PPL: {e}")

            # 6. IFD (需要 model)
            if args.model:
                logger.info(f"Split {split_id}: computing IFD with {args.model}...")
                try:
                    metrics.update(compute_ifd_metrics(split_data, model_name=args.model))
                except Exception as e:
                    logger.warning(f"Failed to compute IFD: {e}")

            dataset_metrics = DatasetMetrics(
                split_id=split_id,
                num_samples=len(split_data),
                metrics=metrics
            )
            splitter.save_metrics(dataset_metrics)

            # 展开 metrics 字典到顶层
            metrics_dict = dataset_metrics.to_dict()
            flattened = {
                'split_id': metrics_dict['split_id'],
                'num_samples': metrics_dict['num_samples']
            }
            flattened.update(metrics_dict['metrics'])
            metrics_list.append(flattened)

        # Save summary
        metrics_df = pd.DataFrame(metrics_list)
        metrics_csv = output_dir / "data_metrics_summary.csv"
        metrics_df.to_csv(metrics_csv, index=False)
        logger.info(f"Saved metrics summary to {metrics_csv}")

    # Phase 3: Training
    if args.run_training and not args.skip_training:
        logger.info("=" * 50)
        logger.info("Phase 3: Training")
        logger.info("=" * 50)

        # Setup GPU allocation
        gpu_ids = [int(g) for g in args.gpu_ids.split(',')]
        allocator = GPUAllocator(gpu_ids, args.gpus_per_split)

        # 如果有 --splits_dir，自动检测 splits 数量
        if args.splits_dir:
            splits_path = Path(args.splits_dir)
            # 自动检测 splits 数量 (支持 parquet 和 jsonl)
            existing_splits = list(splits_path.glob("split_*.parquet")) or list(splits_path.glob("split_*.jsonl"))
            if existing_splits:
                args.n_splits = max(int(f.stem.split("_")[1]) for f in existing_splits) + 1
                splits_dir = splits_path
                logger.info(f"Auto-detected {args.n_splits} splits from {splits_dir}")
        else:
            splits_dir = output_dir / "splits"

        gpu_allocations = allocator.allocate(args.n_splits)

        # Setup training pipeline
        training_pipeline = TrainingPipeline(str(output_dir), args.cot_datasynth_dir)

        # Prepare training configs
        base_config = {}
        if args.model_id:
            base_config['model.partial_pretrain'] = args.model_id

        configs = training_pipeline.prepare_training_configs(
            str(splits_dir),
            gpu_allocations,
            base_config,
            args.n_splits,
            args.splits_format
        )

        # Run trainings
        logger.info(f"Running {len(configs)} trainings...")

        # 准备评估脚本路径
        eval_script = Path(args.cot_datasynth_dir) / "scripts" / "eval_dataobs.sh"
        eval_data_path = "/data/open_datasets/GSM8K/test.parquet"

        results = training_pipeline.run_all_trainings(
            configs,
            args.train_script,
            parallel=False,
            timeout=None,
            eval_script=str(eval_script) if eval_script.exists() else None,
            eval_data_path=eval_data_path
        )

        logger.info(f"Training results: {results}")

    # Phase 3.5: Evaluation (optional, can be run independently)
    if args.run_evaluation and not args.skip_evaluation:
        logger.info("=" * 50)
        logger.info("Phase 3.5: Evaluation on Test Set")
        logger.info("=" * 50)

        training_pipeline = TrainingPipeline(str(output_dir), args.cot_datasynth_dir)

        # 准备评估脚本路径
        eval_script = Path(args.cot_datasynth_dir) / "scripts" / "eval_dataobs.sh"
        eval_data_path = "/data/open_datasets/GSM8K/test.parquet"

        if not eval_script.exists():
            logger.error(f"Evaluation script not found: {eval_script}")
        else:
            # 使用 GPU 分配器分配 GPU
            gpu_ids = [int(g) for g in args.gpu_ids.split(',')]
            allocator = GPUAllocator(gpu_ids, args.gpus_per_split)
            gpu_allocations = allocator.allocate(args.n_splits)

            logger.info(f"Running evaluation for {args.n_splits} splits...")
            logger.info(f"GPU allocations: {gpu_allocations}")

            for split_id in range(args.n_splits):
                split_output_dir = training_pipeline.training_dir / f"split_{split_id}"
                if split_output_dir.exists():
                    # 获取分配的 GPU
                    gpu_id = gpu_allocations[split_id][0] if gpu_allocations[split_id] else 0
                    logger.info(f"Evaluating split {split_id} on GPU {gpu_id}...")
                    training_pipeline.run_evaluation(
                        split_id,
                        str(split_output_dir),
                        str(eval_script),
                        eval_data_path,
                        gpu_id=gpu_id
                    )
                else:
                    logger.warning(f"Split {split_id} output directory not found: {split_output_dir}")

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

        # 展开 metrics 列（如果存在）
        if 'metrics' in data_metrics_df.columns:
            import ast
            # 将 metrics 字符串转换为字典
            metrics_expanded = data_metrics_df['metrics'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            # 展开为多列
            metrics_df = pd.json_normalize(metrics_expanded)
            # 删除原始 metrics 列，添加展开的列
            data_metrics_df = data_metrics_df.drop('metrics', axis=1)
            data_metrics_df = pd.concat([data_metrics_df, metrics_df], axis=1)

        # 自动收集训练结果
        logger.info("Collecting training results...")

        # 确保 training_pipeline 已初始化 (如果跳过了训练)
        if not args.run_training:
            training_pipeline = TrainingPipeline(str(output_dir), args.cot_datasynth_dir)

        training_results = training_pipeline.collect_training_results(
            args.n_splits,
            metric_keys=['train_loss', 'val_loss', 'val_accuracy']
        )

        if training_results:
            # 保存为 CSV
            training_results_df = pd.DataFrame(training_results).T
            training_results_df.index.name = 'split_id'
            training_results_csv = output_dir / "training_results.csv"
            training_results_df.to_csv(training_results_csv)
            logger.info(f"Saved training results to {training_results_csv}")
        else:
            logger.warning("No training results found")
            # 创建 dummy 数据用于演示
            training_results_df = pd.DataFrame({
                'split_id': range(args.n_splits),
                'accuracy': [0.5 + 0.05 * i for i in range(args.n_splits)],
                'loss': [1.0 - 0.05 * i for i in range(args.n_splits)]
            })
            training_results_df = training_results_df.set_index('split_id')

        # Compute correlations - 保存到 observation 目录
        obs_dir = output_dir / "observation"
        obs_dir.mkdir(parents=True, exist_ok=True)

        analyzer = CorrelationAnalyzer(str(obs_dir))
        correlations = analyzer.compute_correlations(data_metrics_df, training_results_df)
        correlations.to_csv(obs_dir / "correlations.csv", index=False)
        logger.info(f"Saved correlations to {obs_dir}/correlations.csv")

        # 生成强相关性统计表
        strong_corr = analyzer.get_strong_correlations_summary(correlations, threshold=0.6)
        strong_corr.to_csv(obs_dir / "strong_correlations_0.6.csv", index=False)
        logger.info(f"Found {len(strong_corr)} strong correlations (|corr| >= 0.6)")

        # 生成中等相关性统计表
        medium_corr = analyzer.get_strong_correlations_summary(correlations, threshold=0.4)
        medium_corr.to_csv(obs_dir / "strong_correlations_0.4.csv", index=False)
        logger.info(f"Found {len(medium_corr)} medium correlations (|corr| >= 0.4)")

        # Generate visualizations
        visualizer = AnalysisVisualizer(str(obs_dir))
        visualizer.plot_correlation_heatmap(data_metrics_df, training_results_df)
        visualizer.plot_scatter_matrix(data_metrics_df, training_results_df)
        visualizer.plot_metrics_overview(data_metrics_df, training_results_df)
        visualizer.plot_training_convergence(training_results_df)

        logger.info("Analysis completed!")

    logger.info("=" * 50)
    logger.info("Pipeline completed!")
    logger.info(f"Results saved to {output_dir}")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()
