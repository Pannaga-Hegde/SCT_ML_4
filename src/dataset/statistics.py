"""Dataset statistics and EDA visualizer module for GestureFlow.

Computes class and subject distributions, resolution and channel profiling, file size statistics,
and generates publication-quality EDA plot artifacts adhering to docs/design.md visual guidelines.
"""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.config.config import DatasetConfig, config


@dataclass
class StatisticsSummary:
    """Dataclass storing dataset statistics summary metrics."""

    total_images: int = 0
    total_subjects: int = 0
    total_classes: int = 0
    subject_counts: Dict[str, int] = field(default_factory=dict)
    class_counts: Dict[str, int] = field(default_factory=dict)
    resolution_counts: Dict[str, int] = field(default_factory=dict)
    channel_counts: Dict[str, int] = field(default_factory=dict)
    min_resolution: Tuple[int, int] = (0, 0)
    max_resolution: Tuple[int, int] = (0, 0)
    mean_file_size_bytes: float = 0.0
    total_dataset_size_mb: float = 0.0

    def to_dict(self) -> Dict:
        """Convert statistics to dictionary representation."""
        return {
            "total_images": self.total_images,
            "total_subjects": self.total_subjects,
            "total_classes": self.total_classes,
            "subject_counts": self.subject_counts,
            "class_counts": self.class_counts,
            "resolution_counts": self.resolution_counts,
            "channel_counts": self.channel_counts,
            "min_resolution": f"{self.min_resolution[0]}x{self.min_resolution[1]}",
            "max_resolution": f"{self.max_resolution[0]}x{self.max_resolution[1]}",
            "mean_file_size_kb": round(self.mean_file_size_bytes / 1024, 2),
            "total_dataset_size_mb": round(self.total_dataset_size_mb, 2),
        }


class DatasetStatistics:
    """Calculator and visualizer for EDA dataset statistics."""

    def __init__(self, cfg: DatasetConfig = config) -> None:
        """Initialize DatasetStatistics engine.

        Args:
            cfg: Dataset configuration object.
        """
        self.cfg = cfg
        self.dataset_root = Path(cfg.dataset_root)
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def set_matplotlib_dark_theme(self) -> None:
        """Apply dark slate palette tokens per docs/design.md."""
        plt.style.use("dark_background")
        plt.rcParams.update(
            {
                "figure.facecolor": "#080b11",
                "axes.facecolor": "#0f172a",
                "axes.edgecolor": "#1e293b",
                "axes.labelcolor": "#f8fafc",
                "text.color": "#f8fafc",
                "xtick.color": "#94a3b8",
                "ytick.color": "#94a3b8",
                "grid.color": "#1e293b",
                "font.family": "sans-serif",
            }
        )

    def compute_statistics(self) -> StatisticsSummary:
        """Scan dataset and compute summary statistics metrics.

        Returns:
            StatisticsSummary dataclass instance.
        """
        summary = StatisticsSummary()
        if not self.dataset_root.exists():
            return summary

        image_files = []
        for subj in sorted(self.cfg.all_subjects):
            subj_dir = self.dataset_root / subj
            if subj_dir.exists():
                image_files.extend(sorted(list(subj_dir.glob("*/*.png")) + list(subj_dir.glob("*/*.jpg"))))

        summary.total_images = len(image_files)
        file_sizes = []
        resolutions = []
        channels_list = []
        subj_counter: Counter = Counter()
        class_counter: Counter = Counter()

        for img_path in image_files:
            rel_parts = img_path.relative_to(self.dataset_root).parts
            if len(rel_parts) >= 2:
                subj = rel_parts[0]
                cls = rel_parts[1]
                subj_counter[subj] += 1
                class_counter[cls] += 1

            file_sizes.append(img_path.stat().st_size)

            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    mode = img.mode
                    resolutions.append((width, height))
                    channels_list.append(mode)
            except Exception:
                continue

        summary.subject_counts = dict(sorted(subj_counter.items()))
        summary.class_counts = dict(sorted(class_counter.items()))
        summary.total_subjects = len(summary.subject_counts)
        summary.total_classes = len(summary.class_counts)

        res_str_counter = Counter([f"{w}x{h}" for w, h in resolutions])
        summary.resolution_counts = dict(res_str_counter)
        summary.channel_counts = dict(Counter(channels_list))

        if file_sizes:
            summary.mean_file_size_bytes = float(np.mean(file_sizes))
            summary.total_dataset_size_mb = float(np.sum(file_sizes)) / (1024 * 1024)

        if resolutions:
            sorted_res = sorted(resolutions, key=lambda r: r[0] * r[1])
            summary.min_resolution = sorted_res[0]
            summary.max_resolution = sorted_res[-1]

        return summary

    def generate_plots(self, summary: StatisticsSummary) -> List[Path]:
        """Generate and save all required EDA visualization plots.

        Args:
            summary: Computed StatisticsSummary metrics.

        Returns:
            List of generated plot image file paths.
        """
        self.set_matplotlib_dark_theme()
        generated_files = []

        # 1. Class Distribution Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        classes = list(summary.class_counts.keys())
        counts = list(summary.class_counts.values())
        bars = ax.bar(classes, counts, color="#00f2fe", edgecolor="#0f172a", linewidth=1.2)
        ax.set_title("Gesture Class Distribution (LeapGestRecog)", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Gesture Class", fontsize=11, labelpad=8)
        ax.set_ylabel("Image Count", fontsize=11, labelpad=8)
        plt.xticks(rotation=30, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{int(height)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#f8fafc",
            )
        plt.tight_layout()
        class_plot_path = self.output_dir / "class_distribution.png"
        fig.savefig(class_plot_path, dpi=300)
        plt.close(fig)
        generated_files.append(class_plot_path)

        # 2. Subject Distribution Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        subjects = list(summary.subject_counts.keys())
        subj_counts = list(summary.subject_counts.values())
        bars = ax.bar(subjects, subj_counts, color="#4facfe", edgecolor="#0f172a", linewidth=1.2)
        ax.set_title("Subject Sample Distribution (LeapGestRecog)", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Subject ID", fontsize=11, labelpad=8)
        ax.set_ylabel("Image Count", fontsize=11, labelpad=8)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{int(height)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#f8fafc",
            )
        plt.tight_layout()
        subj_plot_path = self.output_dir / "subject_distribution.png"
        fig.savefig(subj_plot_path, dpi=300)
        plt.close(fig)
        generated_files.append(subj_plot_path)

        # 3. Resolution Distribution Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        res_labels = list(summary.resolution_counts.keys())
        res_values = list(summary.resolution_counts.values())
        bars = ax.bar(res_labels, res_values, color="#f59e0b", edgecolor="#0f172a", linewidth=1.2, width=0.4)
        ax.set_title("Dataset Image Resolution Distribution", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Resolution (Width x Height)", fontsize=11, labelpad=8)
        ax.set_ylabel("Image Count", fontsize=11, labelpad=8)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{int(height)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#f8fafc",
            )
        plt.tight_layout()
        res_plot_path = self.output_dir / "resolution_distribution.png"
        fig.savefig(res_plot_path, dpi=300)
        plt.close(fig)
        generated_files.append(res_plot_path)

        # 4. Sample Image Grid (10 classes grid sample)
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle("LeapGestRecog 10-Class Sample Preview", fontsize=16, fontweight="bold", y=0.98)
        axes_flat = axes.flatten()

        image_files = []
        for subj in sorted(self.cfg.all_subjects):
            subj_dir = self.dataset_root / subj
            if subj_dir.exists():
                image_files.extend(sorted(list(subj_dir.glob("*/*.png"))))
        class_samples: Dict[str, Path] = {}
        for img_path in image_files:
            rel_parts = img_path.relative_to(self.dataset_root).parts
            if len(rel_parts) >= 2:
                cls_name = rel_parts[1]
                if cls_name not in class_samples:
                    class_samples[cls_name] = img_path
                if len(class_samples) == 10:
                    break

        sorted_classes = sorted(list(class_samples.keys()))
        for idx, cls_name in enumerate(sorted_classes):
            ax = axes_flat[idx]
            img_p = class_samples[cls_name]
            try:
                img = Image.open(img_p)
                ax.imshow(img, cmap="gray")
            except Exception:
                pass
            ax.set_title(cls_name, fontsize=10, pad=4)
            ax.axis("off")

        plt.tight_layout()
        grid_plot_path = self.output_dir / "sample_grid.png"
        fig.savefig(grid_plot_path, dpi=300)
        plt.close(fig)
        generated_files.append(grid_plot_path)

        return generated_files
