"""
DataObs: Data Observation and Analysis Pipeline
Handles data splitting, metric computation, and result management
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _make_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, bytes):
        return obj.decode('utf-8')
    return obj


def jsonl_to_parquet(jsonl_path: str, parquet_path: str = None) -> str:
    """
    Convert jsonl file to parquet format

    Args:
        jsonl_path: Path to input jsonl file
        parquet_path: Path to output parquet file (optional, auto-generated if None)

    Returns:
        Path to the parquet file
    """
    jsonl_path = Path(jsonl_path)
    if parquet_path is None:
        parquet_path = jsonl_path.with_suffix('.parquet')
    else:
        parquet_path = Path(parquet_path)

    # Read jsonl
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    # Convert to dataframe and save as parquet
    df = pd.DataFrame(data)
    df.to_parquet(parquet_path, index=False)

    logger.info(f"Converted {len(data)} samples from {jsonl_path} to {parquet_path}")
    return str(parquet_path)


def batch_jsonl_to_parquet(splits_dir: str, output_dir: str = None) -> List[str]:
    """
    Convert all jsonl splits to parquet format

    Args:
        splits_dir: Directory containing split_*.jsonl files
        output_dir: Output directory for parquet files (default: same as splits_dir)

    Returns:
        List of parquet file paths
    """
    splits_dir = Path(splits_dir)
    if output_dir is None:
        output_dir = splits_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = []
    for jsonl_file in sorted(splits_dir.glob("split_*.jsonl")):
        parquet_file = output_dir / jsonl_file.with_suffix('.parquet').name
        jsonl_to_parquet(jsonl_file, parquet_file)
        parquet_files.append(str(parquet_file))

    logger.info(f"Converted {len(parquet_files)} files to parquet format")
    return parquet_files


@dataclass
class DatasetMetrics:
    """Metrics for a dataset split"""
    split_id: int
    num_samples: int
    metrics: Dict[str, float]

    def to_dict(self):
        return asdict(self)


class DataSplitter:
    """Split dataset into n parts and compute metrics"""

    def __init__(self, n_splits: int, output_dir: str):
        self.n_splits = n_splits
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = self.output_dir / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)
        self.splits_dir = self.output_dir / "splits"
        self.splits_dir.mkdir(exist_ok=True)
        logger.info(f"DataSplitter initialized: {self.n_splits} splits, output_dir={self.output_dir}")

    def split_dataset(self, data: List[Dict], seed: int = 42) -> List[List[Dict]]:
        """Split data into n_splits parts"""
        np.random.seed(seed)
        indices = np.arange(len(data))
        np.random.shuffle(indices)

        split_indices = np.array_split(indices, self.n_splits)
        splits = [[data[i] for i in split_idx] for split_idx in split_indices]

        logger.info(f"Split {len(data)} samples into {self.n_splits} parts: {[len(s) for s in splits]}")
        return splits

    def compute_metrics(
        self,
        split_data: List[Dict],
        metric_funcs: Dict[str, Callable],
        split_id: int
    ) -> DatasetMetrics:
        """Compute metrics for a dataset split"""
        metrics = {}
        for metric_name, metric_func in metric_funcs.items():
            try:
                metrics[metric_name] = float(metric_func(split_data))
            except Exception as e:
                logger.warning(f"Failed to compute {metric_name} for split {split_id}: {e}")
                metrics[metric_name] = None

        dataset_metrics = DatasetMetrics(
            split_id=split_id,
            num_samples=len(split_data),
            metrics=metrics
        )
        return dataset_metrics

    def save_split(self, split_data: List[Dict], split_id: int, format: str = "parquet"):
        """Save a split to disk"""
        serializable_data = [_make_serializable(item) for item in split_data]

        if format == "jsonl":
            output_file = self.splits_dir / f"split_{split_id}.jsonl"
            with open(output_file, 'w') as f:
                for item in serializable_data:
                    f.write(json.dumps(item) + '\n')
        elif format == "json":
            output_file = self.splits_dir / f"split_{split_id}.json"
            with open(output_file, 'w') as f:
                json.dump(serializable_data, f)
        elif format == "parquet":
            output_file = self.splits_dir / f"split_{split_id}.parquet"
            df = pd.DataFrame(serializable_data)
            df.to_parquet(output_file, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Saved split {split_id} to {output_file}")
        return output_file

    def save_metrics(self, dataset_metrics: DatasetMetrics):
        """Save metrics for a split"""
        metrics_file = self.metrics_dir / f"split_{dataset_metrics.split_id}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(dataset_metrics.to_dict(), f, indent=2)
        logger.info(f"Saved metrics for split {dataset_metrics.split_id}")
        return metrics_file

    def load_metrics(self) -> pd.DataFrame:
        """Load all metrics into a DataFrame"""
        metrics_list = []
        for metrics_file in sorted(self.metrics_dir.glob("split_*_metrics.json")):
            with open(metrics_file) as f:
                metrics_list.append(json.load(f))

        if not metrics_list:
            logger.warning("No metrics files found")
            return pd.DataFrame()

        df = pd.DataFrame(metrics_list)
        return df


@dataclass
class TrainingConfig:
    """Configuration for training a split"""

    def __init__(
        self,
        split_id: int,
        data_path: str,
        output_dir: str,
        gpu_ids: Optional[List[int]] = None,
        **kwargs
    ):
        self.split_id = split_id
        self.data_path = data_path
        self.output_dir = output_dir
        self.gpu_ids = gpu_ids or []
        self.extra_config = kwargs

    def to_dict(self):
        return {
            'split_id': self.split_id,
            'data_path': self.data_path,
            'output_dir': self.output_dir,
            'gpu_ids': self.gpu_ids,
            **self.extra_config
        }


class GPUAllocator:
    """Allocate GPUs for training splits"""

    def __init__(self, available_gpus: Optional[List[int]] = None, gpus_per_split: int = 1):
        if available_gpus is None:
            # Auto-detect available GPUs
            try:
                import torch
                self.available_gpus = list(range(torch.cuda.device_count()))
            except Exception as e:
                logger.warning(f"Failed to detect GPUs: {e}, using CPU only")
                self.available_gpus = []
        else:
            self.available_gpus = available_gpus

        self.gpus_per_split = gpus_per_split
        logger.info(f"Available GPUs: {self.available_gpus}, gpus_per_split: {gpus_per_split}")

    def allocate(self, n_splits: int) -> List[List[int]]:
        """Allocate GPUs for n_splits"""
        if not self.available_gpus:
            logger.warning("No GPUs available, all splits will use CPU")
            return [[] for _ in range(n_splits)]

        allocations = []
        for i in range(n_splits):
            start_idx = (i * self.gpus_per_split) % len(self.available_gpus)
            gpu_ids = [
                self.available_gpus[(start_idx + j) % len(self.available_gpus)]
                for j in range(self.gpus_per_split)
            ]
            allocations.append(gpu_ids)

        logger.info(f"GPU allocation for {n_splits} splits: {allocations}")
        return allocations


@dataclass
class TrainingResult:
    """Result from training a split"""
    split_id: int
    metrics: Dict[str, float]
    checkpoint_path: str

    def to_dict(self):
        return asdict(self)


class ResultCollector:
    """Collect and manage training results"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.output_dir / "training_results.json"
        self.results = []
        logger.info(f"ResultCollector initialized: {self.output_dir}")

    def add_result(self, result: TrainingResult):
        """Add a training result"""
        self.results.append(result.to_dict())
        self._save()
        logger.info(f"Added result for split {result.split_id}")

    def _save(self):
        """Save results to disk"""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

    def load_results(self) -> pd.DataFrame:
        """Load results as DataFrame"""
        if self.results_file.exists():
            with open(self.results_file) as f:
                results = json.load(f)
            return pd.DataFrame(results)
        return pd.DataFrame()

    def get_results_dict(self) -> Dict[int, Dict]:
        """Get results as dict keyed by split_id"""
        df = self.load_results()
        if df.empty:
            return {}
        return {int(row['split_id']): row['metrics'] for _, row in df.iterrows()}
