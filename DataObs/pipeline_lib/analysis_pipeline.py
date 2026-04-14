"""
Analysis and visualization pipeline
Computes correlations and generates visualizations
"""

import logging
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """Analyze correlations between data metrics and training performance"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"CorrelationAnalyzer initialized: {self.output_dir}")

    def compute_correlations(
        self,
        data_metrics_df: pd.DataFrame,
        training_results_df: pd.DataFrame,
        method: str = 'pearson'
    ) -> pd.DataFrame:
        """
        Compute correlations between data metrics and training results

        Args:
            data_metrics_df: DataFrame with data metrics (columns: metric names, index: split_id)
            training_results_df: DataFrame with training results (columns: metric names, index: split_id)
            method: Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
            DataFrame with correlation coefficients
        """
        # Ensure both DataFrames have split_id as index
        if 'split_id' in data_metrics_df.columns:
            data_metrics_df = data_metrics_df.set_index('split_id')
        if 'split_id' in training_results_df.columns:
            training_results_df = training_results_df.set_index('split_id')

        # Align indices
        common_splits = data_metrics_df.index.intersection(training_results_df.index)
        data_metrics_df = data_metrics_df.loc[common_splits]
        training_results_df = training_results_df.loc[common_splits]

        logger.info(f"Computing {method} correlations for {len(common_splits)} splits")

        # Compute correlations
        correlations = {}
        for data_col in data_metrics_df.columns:
            for result_col in training_results_df.columns:
                try:
                    # Remove NaN values and align indices
                    data_vals = data_metrics_df[data_col].dropna()
                    result_vals = training_results_df[result_col].dropna()

                    # Find common indices
                    common_idx = data_vals.index.intersection(result_vals.index)
                    if len(common_idx) < 3:
                        continue

                    data_vals = data_vals.loc[common_idx]
                    result_vals = result_vals.loc[common_idx]

                    if method == 'pearson':
                        corr = data_vals.corr(result_vals)
                    elif method == 'spearman':
                        corr = data_vals.corr(result_vals, method='spearman')
                    elif method == 'kendall':
                        corr = data_vals.corr(result_vals, method='kendall')
                    else:
                        raise ValueError(f"Unknown method: {method}")

                    correlations[f"{data_col} vs {result_col}"] = corr
                except Exception as e:
                    logger.warning(f"Failed to compute correlation for {data_col} vs {result_col}: {e}")

        corr_df = pd.DataFrame(list(correlations.items()), columns=['pair', 'correlation'])
        corr_df = corr_df.sort_values('correlation', key=abs, ascending=False)

        logger.info(f"Computed {len(corr_df)} correlations")
        return corr_df

    def compute_pvalues(
        self,
        data_metrics_df: pd.DataFrame,
        training_results_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute p-values for correlations

        Args:
            data_metrics_df: DataFrame with data metrics
            training_results_df: DataFrame with training results

        Returns:
            DataFrame with p-values
        """
        try:
            from scipy.stats import pearsonr
        except ImportError:
            logger.warning("scipy not available, skipping p-value computation")
            return pd.DataFrame()

        # Ensure both DataFrames have split_id as index
        if 'split_id' in data_metrics_df.columns:
            data_metrics_df = data_metrics_df.set_index('split_id')
        if 'split_id' in training_results_df.columns:
            training_results_df = training_results_df.set_index('split_id')

        # Align indices
        common_splits = data_metrics_df.index.intersection(training_results_df.index)
        data_metrics_df = data_metrics_df.loc[common_splits]
        training_results_df = training_results_df.loc[common_splits]

        pvalues = {}
        for data_col in data_metrics_df.columns:
            for result_col in training_results_df.columns:
                try:
                    data_vals = data_metrics_df[data_col].dropna().values
                    result_vals = training_results_df[result_col].loc[data_metrics_df[data_col].dropna().index].dropna().values

                    if len(data_vals) < 3:
                        continue

                    _, pval = pearsonr(data_vals, result_vals)
                    pvalues[f"{data_col} vs {result_col}"] = pval
                except Exception as e:
                    logger.warning(f"Failed to compute p-value for {data_col} vs {result_col}: {e}")

        pval_df = pd.DataFrame(list(pvalues.items()), columns=['pair', 'pvalue'])
        pval_df = pval_df.sort_values('pvalue')

        logger.info(f"Computed {len(pval_df)} p-values")
        return pval_df

    def get_significant_correlations(
        self,
        correlations_df: pd.DataFrame,
        pvalues_df: pd.DataFrame,
        threshold: float = 0.05
    ) -> pd.DataFrame:
        """
        Filter significant correlations

        Args:
            correlations_df: DataFrame with correlations
            pvalues_df: DataFrame with p-values
            threshold: Significance threshold

        Returns:
            DataFrame with significant correlations
        """
        merged = correlations_df.merge(pvalues_df, on='pair', how='inner')
        significant = merged[merged['pvalue'] < threshold]
        logger.info(f"Found {len(significant)} significant correlations (p < {threshold})")
        return significant

    def get_strong_correlations_summary(
        self,
        correlations_df: pd.DataFrame,
        threshold: float = 0.6,
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get summary of strong correlations (|corr| >= threshold)

        Args:
            correlations_df: DataFrame with correlations
            threshold: Correlation threshold (default: 0.6)
            output_file: Output file path (optional)

        Returns:
            DataFrame with strong correlations
        """
        # 过滤相关性大于阈值的对
        strong = correlations_df[correlations_df['correlation'].abs() >= threshold].copy()
        strong = strong.sort_values('correlation', key=abs, ascending=False)

        logger.info(f"Found {len(strong)} strong correlations (|corr| >= {threshold})")

        if output_file:
            strong.to_csv(output_file, index=False)
            logger.info(f"Saved strong correlations to {output_file}")

        return strong


class AnalysisVisualizer:
    """Generate visualizations for analysis results"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"AnalysisVisualizer initialized: {self.output_dir}")

    def plot_correlation_heatmap(
        self,
        data_metrics_df: pd.DataFrame,
        training_results_df: pd.DataFrame,
        output_file: Optional[str] = None
    ):
        """
        Plot correlation heatmap (split into multiple figures by metric groups)

        Args:
            data_metrics_df: DataFrame with data metrics
            training_results_df: DataFrame with training results
            output_file: Output file path (optional)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib/seaborn not available, skipping heatmap")
            return

        # Ensure both DataFrames have split_id as index
        if 'split_id' in data_metrics_df.columns:
            data_metrics_df = data_metrics_df.set_index('split_id')
        if 'split_id' in training_results_df.columns:
            training_results_df = training_results_df.set_index('split_id')

        # Align indices
        common_splits = data_metrics_df.index.intersection(training_results_df.index)
        data_metrics_df = data_metrics_df.loc[common_splits]
        training_results_df = training_results_df.loc[common_splits]

        # Compute correlation matrix - only numeric columns
        combined_df = pd.concat([data_metrics_df, training_results_df], axis=1)
        numeric_cols = combined_df.select_dtypes(include=['number']).columns
        corr_matrix = combined_df[numeric_cols].corr()

        # 分组绘制：按 metric 类型分组
        metric_groups = {
            'length': [c for c in numeric_cols if 'length' in c.lower()],
            'diversity': [c for c in numeric_cols if 'diversity' in c.lower() or 'similarity' in c.lower()],
            'entropy': [c for c in numeric_cols if 'entropy' in c.lower()],
            'ppl': [c for c in numeric_cols if 'ppl' in c.lower()],
            'ifd': [c for c in numeric_cols if 'ifd' in c.lower()],
            'quality': [c for c in numeric_cols if 'coverage' in c.lower() or 'validity' in c.lower() or 'uniqueness' in c.lower()],
            'training': [c for c in numeric_cols if 'loss' in c.lower() or 'accuracy' in c.lower()],
        }

        # 过滤空组
        metric_groups = {k: v for k, v in metric_groups.items() if v}

        # 为每个组绘制热力图
        for group_name, cols in metric_groups.items():
            if not cols:
                continue

            # 获取这些列与训练结果的相关性
            training_cols = training_results_df.select_dtypes(include=['number']).columns.tolist()
            plot_cols = list(cols) + training_cols
            plot_cols = [c for c in plot_cols if c in corr_matrix.columns]

            if len(plot_cols) < 2:
                continue

            subset_corr = corr_matrix.loc[plot_cols, plot_cols]

            # 绘制
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(subset_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, cbar_kws={'label': 'Correlation'})
            ax.set_title(f'Correlation Heatmap: {group_name.capitalize()} Metrics')
            plt.tight_layout()

            if output_file:
                output_path = str(output_file).replace('.png', f'_{group_name}.png')
            else:
                output_path = self.output_dir / f"correlation_heatmap_{group_name}.png"

            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved {group_name} heatmap to {output_path}")
            plt.close()

        # 也绘制完整的热力图（但更小的字体）
        fig, ax = plt.subplots(figsize=(16, 14))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax,
                    cbar_kws={'label': 'Correlation'}, annot_kws={'size': 8})
        ax.set_title('Full Correlation Heatmap: All Metrics')
        plt.tight_layout()

        if output_file:
            output_path = output_file
        else:
            output_path = self.output_dir / "correlation_heatmap_full.png"

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved full heatmap to {output_path}")
        plt.close()

    def plot_scatter_matrix(
        self,
        data_metrics_df: pd.DataFrame,
        training_results_df: pd.DataFrame,
        output_file: Optional[str] = None
    ):
        """
        Plot scatter matrix

        Args:
            data_metrics_df: DataFrame with data metrics
            training_results_df: DataFrame with training results
            output_file: Output file path (optional)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping scatter matrix")
            return

        # Ensure both DataFrames have split_id as index
        if 'split_id' in data_metrics_df.columns:
            data_metrics_df = data_metrics_df.set_index('split_id')
        if 'split_id' in training_results_df.columns:
            training_results_df = training_results_df.set_index('split_id')

        # Align indices
        common_splits = data_metrics_df.index.intersection(training_results_df.index)
        if len(common_splits) == 0:
            logger.warning(f"No common splits: data_metrics index={list(data_metrics_df.index)}, training_results index={list(training_results_df.index)}")
            return
        data_metrics_df = data_metrics_df.loc[common_splits]
        training_results_df = training_results_df.loc[common_splits]

        # Select numeric columns
        data_cols = data_metrics_df.select_dtypes(include=[np.number]).columns[:3]
        result_cols = training_results_df.select_dtypes(include=[np.number]).columns[:3]

        n_data = len(data_cols)
        n_result = len(result_cols)

        if n_data == 0 or n_result == 0:
            logger.warning(f"No numeric columns found for scatter matrix: data_cols={list(data_cols)}, result_cols={list(result_cols)}, data_dtypes={data_metrics_df.dtypes.tolist()}, result_dtypes={training_results_df.dtypes.tolist()}")
            return

        fig, axes = plt.subplots(n_result, n_data, figsize=(12, 10))
        if n_result == 1 or n_data == 1:
            axes = axes.reshape(n_result, n_data)

        for i, result_col in enumerate(result_cols):
            for j, data_col in enumerate(data_cols):
                ax = axes[i, j]
                # 确保数据对齐
                x_data = data_metrics_df.loc[common_splits, data_col]
                y_data = training_results_df.loc[common_splits, result_col]
                ax.scatter(x_data, y_data)
                ax.set_xlabel(data_col)
                ax.set_ylabel(result_col)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150)
            logger.info(f"Saved scatter matrix to {output_file}")
        else:
            output_file = self.output_dir / "scatter_matrix.png"
            plt.savefig(output_file, dpi=150)
            logger.info(f"Saved scatter matrix to {output_file}")

        plt.close()

    def plot_metrics_overview(
        self,
        data_metrics_df: pd.DataFrame,
        training_results_df: pd.DataFrame,
        output_file: Optional[str] = None
    ):
        """绘制 metrics 概览图"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib not available, skipping overview")
            return

        # 设置 split_id 为索引
        if 'split_id' in data_metrics_df.columns:
            data_metrics_df = data_metrics_df.set_index('split_id')
        if 'split_id' in training_results_df.columns:
            training_results_df = training_results_df.set_index('split_id')

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 各 split 的 train_loss 和 val_loss
        ax1 = axes[0, 0]
        if 'train_loss' in training_results_df.columns and 'val_loss' in training_results_df.columns:
            ax1.plot(training_results_df.index, training_results_df['train_loss'], 'o-', label='train_loss')
            ax1.plot(training_results_df.index, training_results_df['val_loss'], 's-', label='val_loss')
            ax1.set_xlabel('Split ID')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training & Validation Loss per Split')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # 2. 关键 data metrics 分布
        ax2 = axes[0, 1]
        key_metrics = ['num_samples', 'jaccard_diversity_score', 'jaccard_avg_similarity']
        key_metrics = [m for m in key_metrics if m in data_metrics_df.columns]
        if key_metrics:
            data_metrics_df[key_metrics].plot(kind='bar', ax=ax2)
            ax2.set_title('Data Metrics per Split')
            ax2.set_xlabel('Split ID')
            ax2.legend(loc='upper right', fontsize=8)
            ax2.tick_params(axis='x', rotation=45)

        # 3. diversity vs val_loss 散点图
        ax3 = axes[1, 0]
        div_cols = [c for c in data_metrics_df.columns if 'diversity_score' in c]
        if div_cols and 'val_loss' in training_results_df.columns:
            for div_col in div_cols[:1]:  # 只画一个
                ax3.scatter(data_metrics_df[div_col], training_results_df['val_loss'])
                ax3.set_xlabel(div_col)
                ax3.set_ylabel('val_loss')
                ax3.set_title(f'{div_col} vs val_loss')
                ax3.grid(True, alpha=0.3)

        # 4. 关键指标的热力图
        ax4 = axes[1, 1]
        key_cols = ['num_samples', 'jaccard_diversity_score', 'jaccard_avg_similarity',
                    'jaccard_avg_similarity', 'train_loss', 'val_loss']
        key_cols = [c for c in key_cols if c in data_metrics_df.columns or c in training_results_df.columns]
        if key_cols:
            combined = pd.concat([data_metrics_df, training_results_df], axis=1)
            available = [c for c in key_cols if c in combined.columns]
            available = [c for c in available if combined[c].dtype.name in ['float64', 'int64']]
            if available:
                sns.heatmap(combined[available].T, annot=True, fmt='.2f', cmap='viridis', ax=ax4)
                ax4.set_title('Key Metrics Heatmap')

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150)
        else:
            output_file = self.output_dir / "metrics_overview.png"
            plt.savefig(output_file, dpi=150)
            logger.info(f"Saved metrics overview to {output_file}")

        plt.close()

    def plot_training_convergence(
        self,
        training_results_df: pd.DataFrame,
        output_file: Optional[str] = None
    ):
        """绘制训练收敛曲线"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping convergence plot")
            return

        if 'split_id' in training_results_df.columns:
            training_results_df = training_results_df.set_index('split_id')

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 1. Loss 曲线
        ax1 = axes[0]
        if 'train_loss' in training_results_df.columns:
            ax1.plot(training_results_df.index, training_results_df['train_loss'], 'o-', label='Train Loss', color='blue')
        if 'val_loss' in training_results_df.columns:
            ax1.plot(training_results_df.index, training_results_df['val_loss'], 's-', label='Val Loss', color='red')
        ax1.set_xlabel('Split ID')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Comparison across Splits')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Loss 差异 (过拟合指标)
        ax2 = axes[1]
        if 'train_loss' in training_results_df.columns and 'val_loss' in training_results_df.columns:
            gap = training_results_df['val_loss'] - training_results_df['train_loss']
            ax2.bar(training_results_df.index, gap, color='orange', alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
            ax2.set_xlabel('Split ID')
            ax2.set_ylabel('Val Loss - Train Loss')
            ax2.set_title('Generalization Gap (Overfitting Indicator)')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150)
        else:
            output_file = self.output_dir / "training_convergence.png"
            plt.savefig(output_file, dpi=150)
            logger.info(f"Saved convergence plot to {output_file}")

        plt.close()

    def plot_metric_vs_performance(
        self,
        data_metric_col: str,
        performance_col: str,
        data_metrics_df: pd.DataFrame,
        training_results_df: pd.DataFrame,
        output_file: Optional[str] = None
    ):
        """
        Plot single metric vs performance

        Args:
            data_metric_col: Column name in data_metrics_df
            performance_col: Column name in training_results_df
            data_metrics_df: DataFrame with data metrics
            training_results_df: DataFrame with training results
            output_file: Output file path (optional)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping plot")
            return

        # Ensure both DataFrames have split_id as index
        if 'split_id' in data_metrics_df.columns:
            data_metrics_df = data_metrics_df.set_index('split_id')
        if 'split_id' in training_results_df.columns:
            training_results_df = training_results_df.set_index('split_id')

        # Align indices
        common_splits = data_metrics_df.index.intersection(training_results_df.index)
        data_metrics_df = data_metrics_df.loc[common_splits]
        training_results_df = training_results_df.loc[common_splits]

        plt.figure(figsize=(8, 6))
        plt.scatter(data_metrics_df[data_metric_col], training_results_df[performance_col])
        plt.xlabel(data_metric_col)
        plt.ylabel(performance_col)
        plt.title(f'{data_metric_col} vs {performance_col}')
        plt.grid(True, alpha=0.3)

        if output_file:
            plt.savefig(output_file, dpi=150)
            logger.info(f"Saved plot to {output_file}")
        else:
            output_file = self.output_dir / f"{data_metric_col}_vs_{performance_col}.png"
            plt.savefig(output_file, dpi=150)
            logger.info(f"Saved plot to {output_file}")

        plt.close()
