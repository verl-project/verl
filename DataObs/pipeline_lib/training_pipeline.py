"""
Training pipeline management
Handles training execution, GPU allocation, and result collection
"""

import logging
import subprocess
import json
import yaml
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Manage training execution for multiple splits"""

    def __init__(self, output_dir: str, cot_datasynth_dir: str):
        """
        Initialize training pipeline

        Args:
            output_dir: Directory to save training results
            cot_datasynth_dir: Path to CoT-DataSynth project directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cot_datasynth_dir = Path(cot_datasynth_dir)
        self.training_dir = self.output_dir / "training"
        self.training_dir.mkdir(exist_ok=True)
        self.eval_dir = self.output_dir / "eval"
        self.eval_dir.mkdir(exist_ok=True)
        self.config_file = self.output_dir / "training_configs.json"
        self.log_file = self.training_dir / "training_log.json"
        logger.info(f"TrainingPipeline initialized: {self.output_dir}")

    def prepare_training_configs(
        self,
        splits_dir: str,
        gpu_allocations: List[List[int]],
        base_config: Dict[str, Any],
        n_splits: int,
        base_model_id: str
    ) -> List[Dict[str, Any]]:
        """
        Prepare training configurations for all splits

        Args:
            splits_dir: Directory containing split data files
            gpu_allocations: GPU allocation for each split
            base_config: Base training configuration
            n_splits: Number of splits

        Returns:
            List of training configurations
        """
        configs = []
        for split_id in range(n_splits):
            split_data_path = Path(splits_dir) / f"split_{split_id}.parquet"
            if not split_data_path.exists():
                logger.warning(f"Split data not found: {split_data_path}")
                continue

            split_output_dir = self.training_dir / f"split_{split_id}"
            split_output_dir.mkdir(exist_ok=True)

            eval_output_dir = self.eval_dir / f"split_{split_id}"
            eval_output_dir.mkdir(exist_ok=True)

            config = {
                'split_id': split_id,
                'data_path': str(split_data_path),
                'output_dir': str(split_output_dir),
                'eval_output_dir': str(eval_output_dir),
                'gpu_ids': gpu_allocations[split_id] if split_id < len(gpu_allocations) else [],
                'base_model_id': base_model_id,
                **base_config
            }
            configs.append(config)

        # Save configs
        with open(self.config_file, 'w') as f:
            json.dump(configs, f, indent=2)

        logger.info(f"Prepared {len(configs)} training configurations")
        return configs

    def run_training(
        self,
        config: Dict[str, Any],
        script_path: str,
        eval_data_path: Optional[str] = '/data/open_datasets/GSM8K/test.parquet',
        timeout: Optional[int] = None
    ) -> bool:
        """
        Execute a single training

        Args:
            config: Training configuration
            script_path: Path to training script (e.g., scripts/sft.sh)
            timeout: Timeout in seconds

        Returns:
            True if training succeeded, False otherwise
        """
        split_id = config['split_id']
        gpu_ids = config['gpu_ids']
        data_path = config['data_path']
        output_dir = config['output_dir']
        base_model_id = config['base_model_id']

        # Prepare GPU string
        gpu_str = ','.join(map(str, gpu_ids)) if gpu_ids else '0'

        # Prepare command
        script_path = Path(script_path)
        if not script_path.is_absolute():
            script_path = self.cot_datasynth_dir / script_path

        cmd = [
            'bash',
            str(script_path),
            str(base_model_id), # model id (param 1)
            str(data_path),     # data path (param 2)
            eval_data_path,     # val data path(param 3)
            str(output_dir),    # save path (param 4)
            gpu_str,            # gpu_id (param 5)
        ]

        # Add extra config parameters
        for key, value in config.items():
            if key not in ['split_id', 'gpu_ids', 'data_path', 'output_dir', 'eval_output_dir', 'base_model_id']:
                cmd.append(f'{key}={value}')

        logger.info(f"Running training for split {split_id}: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.cot_datasynth_dir),
                timeout=timeout,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                # 尝试从 stdout 解析验证指标
                training_metrics = self._parse_training_metrics(result.stdout, result.stderr)
                self._save_training_results(split_id, output_dir, training_metrics)
                logger.info(f"Training for split {split_id} completed successfully")
                return True
            else:
                logger.error(f"Training for split {split_id} failed with return code {result.returncode}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"Training for split {split_id} timed out after {timeout} seconds")
            return False
        except Exception as e:
            logger.error(f"Training for split {split_id} failed with exception: {e}")
            return False

    def run_all_trainings(
        self,
        configs: List[Dict[str, Any]],
        script_path: str,
        parallel: bool = False,
        timeout: Optional[int] = None,
        skip_completed: bool = True,
        eval_script_path: Optional[str] = None,
        eval_data_path: Optional[str] = None
    ) -> Dict[int, bool]:
        """
        Run all trainings

        Args:
            configs: List of training configurations
            script_path: Path to training script
            parallel: Whether to run trainings in parallel (not implemented yet)
            timeout: Timeout per training in seconds
            skip_completed: Skip splits that already have training_results.json
            eval_script_path: Path to evaluation script (optional)
            eval_data_path: Path to evaluation data (optional)

        Returns:
            Dictionary mapping split_id to success status
        """
        results = {}
        skipped = []

        if parallel:
            logger.warning("Parallel training not implemented yet, running sequentially")

        for config in configs:
            split_id = config['split_id']

            # Check if already completed
            if skip_completed and self.is_split_completed(split_id):
                logger.info(f"Split {split_id}: already completed, skipping")
                skipped.append(split_id)
                self._update_split_log(split_id, "skipped", "Already completed")
                results[split_id] = True
                continue

            # Update status to running
            self._update_split_log(split_id, "running")
            success = self.run_training(config, script_path, eval_data_path, timeout)
            results[split_id] = success

            # Update status based on result
            if success:
                self._update_split_log(split_id, "completed", "Training succeeded")

                # Run evaluation if script provided
                if eval_script_path and eval_data_path:
                    logger.info(f"Running evaluation for split {split_id}...")
                    eval_success = self.run_evaluation(
                        config,
                        eval_script_path,
                        eval_data_path
                    )
                    if eval_success:
                        logger.info(f"Evaluation for split {split_id} succeeded")
                    else:
                        logger.warning(f"Evaluation for split {split_id} failed")
            else:
                self._update_split_log(split_id, "failed", "Training failed")

            # Add delay between trainings to avoid resource conflicts
            if split_id < len(configs) - 1:
                time.sleep(5)

        logger.info(f"Training completed: {sum(results.values())}/{len(results)} successful, {len(skipped)} skipped")
        return results

    def run_evaluation(
        self,
        config: Dict[str, Any],
        eval_script_path: str,
        eval_data_path: str,
    ) -> bool:
        """
        Run evaluation on a trained split

        Args:
            config: Training configuration
            eval_script_path: Path to evaluation script
            eval_data_path: Path to evaluation data

        Returns:
            True if evaluation succeeded, False otherwise
        """
        gpu_ids = config['gpu_ids']
        output_dir = config['output_dir']
        eval_output_dir = config['eval_output_dir']
        base_model_id = config['base_model_id']

        # 查找最新的 checkpoint (global_step_*)
        output_path = Path(output_dir)
        checkpoints = list(output_path.glob("global_step_*"))
        if not checkpoints:
            logger.warning(f"No checkpoints found in {output_dir}")
            return False

        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.name.split('_')[2]))[-1]
        logger.info(f"Using checkpoint: {latest_checkpoint}")

        # Prepare GPU string
        gpu_str = ','.join(map(str, gpu_ids)) if gpu_ids else '0'
        
        # Prepare command
        eval_script_path = Path(eval_script_path)
        if not eval_script_path.is_absolute():
            eval_script_path = self.cot_datasynth_dir / eval_script_path

        cmd = [
            'bash',
            str(eval_script_path),
            str(latest_checkpoint),     # checkpoint path (param 1)
            str(base_model_id),         # base model (param 2)
            str(eval_data_path),        # eval data path (param 3)
            str(eval_output_dir),       # eval output dir (param 4)
            str(gpu_str),               # gpu_id (param 5)
        ]

        logger.info(f"Running evaluation: {' '.join(cmd)}")
        logger.info(f"Working directory: {self.cot_datasynth_dir}")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.cot_datasynth_dir),
                timeout=3600,  # 1 hour timeout for evaluation
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                # 尝试从评估输出中提取准确率
                with open(f'{eval_output_dir}/logs/evaluation.log', 'r', encoding='utf-8') as f:
                    output = f.read()
                import re

                # 尝试多种模式匹配准确率
                acc_patterns = [
                    r'accuracy[:\s]+([0-9.]+)',
                    r'test_score[:\s]+([0-9.]+)',
                    r'pass@1[:\s]+([0-9.]+)',
                ]

                accuracy = None
                for pattern in acc_patterns:
                    acc_match = re.search(pattern, output, re.IGNORECASE)
                    if acc_match:
                        accuracy = float(acc_match.group(1))
                        logger.info(f"Evaluation accuracy: {accuracy}")
                        break

                if accuracy is not None:
                    # 保存到 training_results.json
                    results_file = Path(output_dir) / "training_results.json"
                    if results_file.exists():
                        with open(results_file) as f:
                            results = json.load(f)
                    else:
                        results = {}

                    results['test_accuracy'] = accuracy
                    with open(results_file, 'w') as f:
                        json.dump(results, f, indent=2)

                    logger.info(f"Saved test_accuracy to {results_file}")
                else:
                    logger.warning("Could not extract accuracy from evaluation output")

                return True
            else:
                logger.error(f"Evaluation failed with return code {result.returncode}")
                logger.error(f"STDOUT: {result.stdout[-1000:]}")  # 最后 1000 字符
                logger.error(f"STDERR: {result.stderr[-1000:]}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"Evaluation timed out after 3600 seconds")
            return False
        except Exception as e:
            logger.error(f"Evaluation failed with exception: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _parse_training_metrics(self, stdout: str, stderr: str) -> Dict[str, float]:
        """从训练输出中解析 metrics"""
        metrics = {}
        output = stdout + stderr

        # 尝试匹配常见的 metric 模式
        import re

        # 匹配 "train_loss: 0.123" 或 "train/loss: 0.123"
        loss_patterns = [
            r'train[/_]loss[:\s]+([0-9.]+)',
            r'loss[:\s]+([0-9.]+)',
            r'Final.*loss[:\s]+([0-9.]+)',
        ]
        for pattern in loss_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                metrics['train_loss'] = float(match.group(1))
                break

        # 匹配验证准确率
        acc_patterns = [
            r'val[/_]acc(?:uracy)?[:\s]+([0-9.]+)',
            r'accuracy[:\s]+([0-9.]+)',
            r'eval[/_]acc(?:uracy)?[:\s]+([0-9.]+)',
            r'Final.*acc(?:uracy)?[:\s]+([0-9.]+)',
        ]
        for pattern in acc_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                metrics['val_accuracy'] = float(match.group(1))
                break

        # 匹配验证 loss
        val_loss_patterns = [
            r'val[/_]loss[:\s]+([0-9.]+)',
            r'eval[/_]loss[:\s]+([0-9.]+)',
        ]
        for pattern in val_loss_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                metrics['val_loss'] = float(match.group(1))
                break

        return metrics

    def _save_training_results(self, split_id: int, output_dir: str, metrics: Dict[str, float]):
        """保存训练结果到 JSON 文件"""
        output_path = Path(output_dir) / "training_results.json"

        # 如果文件已存在，先读取
        existing = {}
        if output_path.exists():
            try:
                with open(output_path) as f:
                    existing = json.load(f)
            except:
                pass

        # 合并 metrics
        existing.update(metrics)
        existing['split_id'] = split_id
        existing['status'] = 'completed'
        existing['timestamp'] = datetime.now().isoformat()

        with open(output_path, 'w') as f:
            json.dump(existing, f, indent=2)

        logger.info(f"Saved training metrics to {output_path}")

    def collect_training_results(
        self,
        n_splits: int,
        metric_keys: Optional[List[str]] = None
    ) -> Dict[int, Dict[str, float]]:
        """
        Collect training results from output directories

        Args:
            n_splits: Number of splits
            metric_keys: Keys to extract from results (e.g., ['accuracy', 'loss'])

        Returns:
            Dictionary mapping split_id to metrics
        """
        results = {}

        for split_id in range(n_splits):
            split_output_dir = self.training_dir / f"split_{split_id}"
            results_file = split_output_dir / "training_results.json"

            if results_file.exists():
                try:
                    with open(results_file) as f:
                        split_results = json.load(f)

                    # 只保留数值类型的 metrics，排除 split_id
                    filtered_results = {}
                    for key, value in split_results.items():
                        if key != 'split_id' and isinstance(value, (int, float)):
                            filtered_results[key] = value

                    results[split_id] = filtered_results
                    logger.info(f"Loaded results for split {split_id}")
                except Exception as e:
                    logger.warning(f"Failed to load results for split {split_id}: {e}")
            else:
                logger.warning(f"Results file not found for split {split_id}: {results_file}")

        return results

    def _load_log(self) -> Dict[str, Any]:
        """加载训练日志"""
        if self.log_file.exists():
            try:
                with open(self.log_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load training log: {e}")
        return {"splits": {}, "created_at": datetime.now().isoformat()}

    def _save_log(self, log_data: Dict[str, Any]):
        """保存训练日志"""
        log_data["updated_at"] = datetime.now().isoformat()
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

    def _update_split_log(self, split_id: int, status: str, message: str = ""):
        """更新单个 split 的日志"""
        log_data = self._load_log()
        if str(split_id) not in log_data["splits"]:
            log_data["splits"][str(split_id)] = {}
        log_data["splits"][str(split_id)]["status"] = status
        log_data["splits"][str(split_id)]["updated_at"] = datetime.now().isoformat()
        if message:
            log_data["splits"][str(split_id)]["message"] = message
        self._save_log(log_data)

    def is_split_completed(self, split_id: int) -> bool:
        """检查 split 是否已完成"""
        split_output_dir = self.training_dir / f"split_{split_id}"
        results_file = split_output_dir / "training_results.json"
        return results_file.exists()

    def get_training_status(self) -> Dict[int, str]:
        """
        Get training status for all splits

        Returns:
            Dictionary mapping split_id to status ('pending', 'running', 'completed', 'failed')
        """
        status = {}

        for split_dir in sorted(self.training_dir.glob("split_*")):
            split_id = int(split_dir.name.split('_')[1])
            results_file = split_dir / "training_results.json"

            if results_file.exists():
                status[split_id] = 'completed'
            elif (split_dir / "training.log").exists():
                status[split_id] = 'running'
            else:
                status[split_id] = 'pending'

        return status
