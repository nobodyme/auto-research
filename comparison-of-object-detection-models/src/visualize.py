"""
Visualization utilities for model comparison results.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ResultsVisualizer:
    """Create visualizations for model comparison results."""

    def __init__(self, output_dir: Path):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_latency_comparison(
        self,
        results: Dict[str, Dict[str, Any]],
        save_path: str = "latency_comparison.png"
    ) -> None:
        """
        Plot latency comparison across models.

        Args:
            results: Model comparison results
            save_path: Filename to save plot
        """
        logger.info("Creating latency comparison plot...")

        # Extract data
        models = list(results.keys())
        latencies = [results[m].get('latency_ms', 0) for m in models]
        fps_values = [results[m].get('fps', 0) for m in models]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Latency plot
        colors = sns.color_palette("husl", len(models))
        bars1 = ax1.bar(models, latencies, color=colors)
        ax1.set_xlabel('Model', fontsize=12)
        ax1.set_ylabel('Latency (ms)', fontsize=12)
        ax1.set_title('Inference Latency Comparison', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)

        # FPS plot
        bars2 = ax2.bar(models, fps_values, color=colors)
        ax2.set_xlabel('Model', fontsize=12)
        ax2.set_ylabel('FPS', fontsize=12)
        ax2.set_title('Frames Per Second (FPS) Comparison', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        save_full_path = self.output_dir / save_path
        plt.savefig(save_full_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved latency comparison to {save_full_path}")

    def plot_accuracy_comparison(
        self,
        results: Dict[str, Dict[str, Any]],
        save_path: str = "accuracy_comparison.png"
    ) -> None:
        """
        Plot accuracy (mAP) comparison across models.

        Args:
            results: Model comparison results
            save_path: Filename to save plot
        """
        logger.info("Creating accuracy comparison plot...")

        # Extract data
        models = list(results.keys())
        mAP_values = [results[m].get('mAP', 0) * 100 for m in models]
        mAP50_values = [results[m].get('mAP50', 0) * 100 for m in models]
        mAP75_values = [results[m].get('mAP75', 0) * 100 for m in models]

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(models))
        width = 0.25

        # Create grouped bars
        bars1 = ax.bar(x - width, mAP_values, width, label='mAP@0.5:0.95', color='#3498db')
        bars2 = ax.bar(x, mAP50_values, width, label='mAP@0.5', color='#2ecc71')
        bars3 = ax.bar(x + width, mAP75_values, width, label='mAP@0.75', color='#e74c3c')

        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('mAP (%)', fontsize=12)
        ax.set_title('Accuracy Comparison (COCO Benchmark)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}',
                           ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        save_full_path = self.output_dir / save_path
        plt.savefig(save_full_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved accuracy comparison to {save_full_path}")

    def plot_efficiency_frontier(
        self,
        results: Dict[str, Dict[str, Any]],
        save_path: str = "efficiency_frontier.png"
    ) -> None:
        """
        Plot accuracy vs latency trade-off (Pareto frontier).

        Args:
            results: Model comparison results
            save_path: Filename to save plot
        """
        logger.info("Creating efficiency frontier plot...")

        fig, ax = plt.subplots(figsize=(10, 8))

        # Extract data
        for model_name, model_results in results.items():
            mAP = model_results.get('mAP', 0) * 100
            latency = model_results.get('latency_ms', 0)

            if mAP > 0 and latency > 0:
                # Determine model family for color
                if 'yolov8' in model_name.lower():
                    color = '#3498db'
                    marker = 'o'
                elif 'yolov10' in model_name.lower():
                    color = '#2ecc71'
                    marker = 's'
                elif 'yolov11' in model_name.lower():
                    color = '#9b59b6'
                    marker = '^'
                elif 'rtdetr' in model_name.lower():
                    color = '#e74c3c'
                    marker = 'D'
                else:
                    color = '#95a5a6'
                    marker = 'v'

                ax.scatter(latency, mAP, s=200, alpha=0.7, color=color, marker=marker,
                          edgecolors='black', linewidth=1.5, label=model_name)

                # Add labels
                ax.annotate(model_name, (latency, mAP),
                           textcoords="offset points", xytext=(0, 10),
                           ha='center', fontsize=9, fontweight='bold')

        ax.set_xlabel('Latency (ms) - Lower is Better', fontsize=12, fontweight='bold')
        ax.set_ylabel('mAP@0.5:0.95 (%) - Higher is Better', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy vs Latency Trade-off\n(Pareto Frontier)',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add diagonal reference lines (efficiency levels)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax.axvline(x=10, color='gray', linestyle='--', alpha=0.3, linewidth=1)

        # Remove duplicate labels in legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=9)

        plt.tight_layout()
        save_full_path = self.output_dir / save_path
        plt.savefig(save_full_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved efficiency frontier to {save_full_path}")

    def plot_model_size_comparison(
        self,
        results: Dict[str, Dict[str, Any]],
        save_path: str = "model_size_comparison.png"
    ) -> None:
        """
        Plot model size comparison.

        Args:
            results: Model comparison results
            save_path: Filename to save plot
        """
        logger.info("Creating model size comparison plot...")

        # Extract data
        models = list(results.keys())
        sizes = [results[m].get('model_size_mb', 0) for m in models]
        params = [results[m].get('parameters', 0) / 1e6 for m in models]  # Convert to millions

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        colors = sns.color_palette("viridis", len(models))

        # Model size plot
        bars1 = ax1.bar(models, sizes, color=colors)
        ax1.set_xlabel('Model', fontsize=12)
        ax1.set_ylabel('Model Size (MB)', fontsize=12)
        ax1.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)

        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9)

        # Parameters plot
        bars2 = ax2.bar(models, params, color=colors)
        ax2.set_xlabel('Model', fontsize=12)
        ax2.set_ylabel('Parameters (Millions)', fontsize=12)
        ax2.set_title('Model Parameters Comparison', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)

        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}M',
                    ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        save_full_path = self.output_dir / save_path
        plt.savefig(save_full_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved model size comparison to {save_full_path}")

    def create_summary_table(
        self,
        results: Dict[str, Dict[str, Any]],
        save_path: str = "summary_table.png"
    ) -> pd.DataFrame:
        """
        Create and save a summary table of results.

        Args:
            results: Model comparison results
            save_path: Filename to save table image

        Returns:
            DataFrame containing summary
        """
        logger.info("Creating summary table...")

        # Create DataFrame
        data = []
        for model_name, model_results in results.items():
            data.append({
                'Model': model_name,
                'mAP (%)': f"{model_results.get('mAP', 0) * 100:.2f}",
                'mAP@50 (%)': f"{model_results.get('mAP50', 0) * 100:.2f}",
                'Latency (ms)': f"{model_results.get('latency_ms', 0):.2f}",
                'FPS': f"{model_results.get('fps', 0):.1f}",
                'Parameters (M)': f"{model_results.get('parameters', 0) / 1e6:.1f}",
                'Size (MB)': f"{model_results.get('model_size_mb', 0):.1f}"
            })

        df = pd.DataFrame(data)

        # Save as CSV
        csv_path = self.output_dir / "summary_table.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved summary table CSV to {csv_path}")

        # Create table visualization
        fig, ax = plt.subplots(figsize=(14, len(data) * 0.5 + 2))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center',
                        colColours=['#3498db'] * len(df.columns))

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(data) + 1):
            color = '#ecf0f1' if i % 2 == 0 else 'white'
            for j in range(len(df.columns)):
                table[(i, j)].set_facecolor(color)

        plt.title('Model Comparison Summary', fontsize=16, fontweight='bold', pad=20)
        save_full_path = self.output_dir / save_path
        plt.savefig(save_full_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved summary table image to {save_full_path}")

        return df

    def create_all_plots(self, results: Dict[str, Dict[str, Any]]) -> None:
        """
        Create all comparison plots.

        Args:
            results: Model comparison results
        """
        logger.info("Creating all visualization plots...")

        self.plot_latency_comparison(results)
        self.plot_accuracy_comparison(results)
        self.plot_efficiency_frontier(results)
        self.plot_model_size_comparison(results)
        self.create_summary_table(results)

        logger.info(f"All plots saved to {self.output_dir}")
